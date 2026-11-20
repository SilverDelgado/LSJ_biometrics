import contextlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, get_worker_info

from lsj_biometrics.modules.extractors.ssm import SSMConfig, SSMFaceModel, save_ssm_face_model
from lsj_biometrics.training.augment import Perturber, TrainingAugmenter
from lsj_biometrics.training.casia import CasiaDataset, CasiaIndex, IdentitySplit, holdout_record, images_of, read_index, split_identities, subset_of
from lsj_biometrics.training.heads import AdaFace
from lsj_biometrics.training.verification import evaluate_sets

PATH_KEYS = ("casia", "eval_dir", "augment_specs", "teacher_weights", "teacher_embeddings", "imagenet_weights", "output_dir")
NULLABLE_PATHS = ("teacher_weights", "teacher_embeddings", "imagenet_weights")
TOP_KEYS = (
    "paths",
    "model",
    "batch_size",
    "accumulation",
    "checkpointing",
    "epochs",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "distill_weight",
    "perturb_probability",
    "workers",
    "device",
    "eval_sets",
    "eval_limit",
    "log_every",
    "seed",
    "holdout_identities",
    "subset",
)
BENCHMARK_WARMUP = 5
CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)\.pt$")


@dataclass(frozen=True)
class Paths:
    casia: Path
    eval_dir: Path
    augment_specs: Path
    teacher_weights: Path | None
    teacher_embeddings: Path | None
    imagenet_weights: Path | None
    output_dir: Path


@dataclass(frozen=True)
class TrainConfig:
    paths: Paths
    model: dict
    batch_size: int
    accumulation: int
    checkpointing: bool
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    distill_weight: float
    perturb_probability: float
    workers: int
    device: str
    eval_sets: tuple[str, ...]
    eval_limit: int | None
    log_every: int
    seed: int
    holdout_identities: int
    subset: int | None
    raw: dict


def _exact_keys(mapping, keys: tuple[str, ...], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a mapping")
    for key in keys:
        if key not in mapping:
            raise ValueError(f"{where} is missing '{key}'")
    for key in mapping:
        if key not in keys:
            raise ValueError(f"{where} has an unknown key '{key}'")


def _integer(raw: dict, key: str, minimum: int, where: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where}: {key} must be an integer >= {minimum}, got {value!r}")
    return value


def _number(raw: dict, key: str, minimum: float, maximum: float, where: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum or value > maximum:
        raise ValueError(f"{where}: {key} must be a number in {minimum}..{maximum}, got {value!r}")
    return float(value)


def _optional_integer(raw: dict, key: str, minimum: int, where: str) -> int | None:
    if raw[key] is None:
        return None
    return _integer(raw, key, minimum, where)


def load_config(path: Path) -> TrainConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    where = str(path)
    _exact_keys(raw, TOP_KEYS, where)
    _exact_keys(raw["paths"], PATH_KEYS, f"{where}: paths")
    paths = {}
    for key in PATH_KEYS:
        value = raw["paths"][key]
        if value is None and key in NULLABLE_PATHS:
            paths[key] = None
        elif isinstance(value, str) and value:
            paths[key] = Path(value)
        else:
            raise ValueError(f"{where}: paths.{key} must be a path, got {value!r}")
    if not isinstance(raw["model"], dict):
        raise ValueError(f"{where}: model must be a mapping of SSMConfig fields, empty for the defaults")
    if not isinstance(raw["checkpointing"], bool):
        raise ValueError(f"{where}: checkpointing must be true or false")
    if not isinstance(raw["device"], str) or not raw["device"]:
        raise ValueError(f"{where}: device must be cpu, cuda or cuda:<n>")
    eval_sets = raw["eval_sets"]
    if not isinstance(eval_sets, list) or not eval_sets:
        raise ValueError(f"{where}: eval_sets must list at least one .bin name: best.pt is chosen by their mean accuracy")
    names = []
    for name in eval_sets:
        if not isinstance(name, str):
            raise ValueError(f"{where}: eval_sets holds {name!r}, not a name")
        names.append(name)
    epochs = _integer(raw, "epochs", 1, where)
    warmup_epochs = _integer(raw, "warmup_epochs", 0, where)
    if warmup_epochs >= epochs:
        raise ValueError(f"{where}: warmup_epochs {warmup_epochs} leaves no epoch of {epochs} for the cosine")
    batch_size = _integer(raw, "batch_size", 2, where)
    accumulation = _integer(raw, "accumulation", 1, where)
    lr = _number(raw, "lr", 1e-8, 1.0, where)
    weight_decay = _number(raw, "weight_decay", 0.0, 1.0, where)
    distill_weight = _number(raw, "distill_weight", 0.0, 1000.0, where)
    perturb_probability = _number(raw, "perturb_probability", 0.0, 1.0, where)
    workers = _integer(raw, "workers", 0, where)
    eval_limit = _optional_integer(raw, "eval_limit", 20, where)
    log_every = _integer(raw, "log_every", 1, where)
    seed = _integer(raw, "seed", 0, where)
    holdout_identities = _integer(raw, "holdout_identities", 0, where)
    subset = _optional_integer(raw, "subset", 1, where)
    return TrainConfig(paths=Paths(**paths), model=raw["model"], batch_size=batch_size, accumulation=accumulation, checkpointing=raw["checkpointing"], epochs=epochs, lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs, distill_weight=distill_weight, perturb_probability=perturb_probability, workers=workers, device=raw["device"], eval_sets=tuple(names), eval_limit=eval_limit, log_every=log_every, seed=seed, holdout_identities=holdout_identities, subset=subset, raw=raw)


def as_tuples(value):
    if not isinstance(value, list):
        return value
    items = []
    for item in value:
        items.append(as_tuples(item))
    return tuple(items)


def build_model(config: TrainConfig) -> SSMFaceModel:
    fields = {}
    for key, value in config.model.items():
        fields[key] = as_tuples(value)
    return SSMFaceModel(SSMConfig(**fields))


def check_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"device {device} asked for, but torch sees no cuda device")


def autocast(device: str):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def prepare_worker(worker_id: int) -> None:
    cv2.setNumThreads(1)
    get_worker_info().dataset.augmenter.prepare()


def epoch_loader(dataset: CasiaDataset, config: TrainConfig, epoch: int) -> DataLoader:
    dataset.augmenter.epoch = epoch
    generator = torch.Generator()
    generator.manual_seed(config.seed * 100003 + epoch)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=config.workers, worker_init_fn=prepare_worker, drop_last=True, pin_memory=config.device.startswith("cuda"))


def training_indices(config: TrainConfig, index: CasiaIndex, split: IdentitySplit) -> np.ndarray:
    return subset_of(images_of(index, split.trained), config.subset)


def teacher_rows(config: TrainConfig, index: CasiaIndex) -> int:
    if config.subset is None:
        return index.count
    split = split_identities(index, config.holdout_identities, config.seed)
    return int(training_indices(config, index, split).max()) + 1


def load_teacher_embeddings(path: Path | None, rows_needed: int, dimension: int) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise ValueError(f"teacher embeddings {path} missing: run precompute-teacher first, or set paths.teacher_embeddings to null")
    rows = np.load(path)
    if rows.ndim != 2 or rows.shape[1] != dimension or rows.shape[0] < rows_needed:
        raise ValueError(f"{path} is {rows.shape}, the run needs at least {rows_needed} x {dimension}")
    return rows


def parameter_groups(modules: list[torch.nn.Module], weight_decay: float) -> list[dict]:
    decayed = []
    kept = []
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim < 2:
                kept.append(parameter)
            else:
                decayed.append(parameter)
    return [{"params": decayed, "weight_decay": weight_decay}, {"params": kept, "weight_decay": 0.0}]


class WarmupCosine:

    def __init__(self, warmup_steps: int, total_steps: int) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@dataclass(frozen=True)
class Session:
    config: TrainConfig
    dataset: CasiaDataset
    model: SSMFaceModel
    head: AdaFace
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    teacher: np.ndarray | None
    steps_per_epoch: int
    index: CasiaIndex
    split: IdentitySplit
    dense_labels: torch.Tensor


def build_session(config: TrainConfig) -> Session:
    check_device(config.device)
    torch.manual_seed(config.seed)
    index = read_index(config.paths.casia)
    split = split_identities(index, config.holdout_identities, config.seed)
    indices = training_indices(config, index, split)
    augmenter = TrainingAugmenter(Perturber(config.paths.augment_specs), config.perturb_probability, config.seed)
    dataset = CasiaDataset(config.paths.casia, augmenter, indices)
    if config.workers == 0:
        augmenter.prepare()
    model = build_model(config)
    if config.checkpointing:
        model.set_checkpointing(True)
    teacher = load_teacher_embeddings(config.paths.teacher_embeddings, int(indices.max()) + 1, model.config.embedding_dim)
    head = AdaFace(model.config.embedding_dim, int(split.trained.size))
    model.to(config.device)
    head.to(config.device)
    optimizer = torch.optim.AdamW(parameter_groups([model, head], config.weight_decay), lr=config.lr)
    steps_per_epoch = len(dataset) // config.batch_size // config.accumulation
    if steps_per_epoch == 0:
        raise ValueError(f"{len(dataset)} images do not fill one step of {config.batch_size} x {config.accumulation}")
    schedule = WarmupCosine(config.warmup_epochs * steps_per_epoch, config.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    dense_labels = torch.from_numpy(split.dense).to(config.device)
    return Session(config=config, dataset=dataset, model=model, head=head, optimizer=optimizer, scheduler=scheduler, teacher=teacher, steps_per_epoch=steps_per_epoch, index=index, split=split, dense_labels=dense_labels)


def forward_backward(session: Session, images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> tuple[float, float]:
    config = session.config
    x = images.to(config.device, non_blocking=True).permute(0, 3, 1, 2).float()
    labels = session.dense_labels[labels.to(config.device, non_blocking=True)]
    with autocast(config.device):
        features = session.model(x)
    features = features.float()
    classification = F.cross_entropy(session.head(features, labels), labels)
    loss = classification
    distillation = torch.zeros((), device=features.device)
    if session.teacher is not None:
        targets = torch.from_numpy(session.teacher[indices.numpy()].astype(np.float32)).to(config.device, non_blocking=True)
        distillation = (1.0 - F.cosine_similarity(features, targets, dim=1)).mean()
        loss = loss + config.distill_weight * distillation
    (loss / config.accumulation).backward()
    return float(classification.detach()), float(distillation.detach())


def optimizer_step(session: Session) -> None:
    session.optimizer.step()
    session.optimizer.zero_grad(set_to_none=True)
    session.scheduler.step()


class Log:

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def __call__(self, line: str) -> None:
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def run_epoch(session: Session, epoch: int, log: Log) -> dict:
    config = session.config
    session.model.train()
    session.head.train()
    session.optimizer.zero_grad(set_to_none=True)
    iterations = session.steps_per_epoch * config.accumulation
    classification_sum = 0.0
    distillation_sum = 0.0
    window_started = time.perf_counter()
    started = window_started
    seen = 0
    for iteration, (images, labels, indices) in enumerate(epoch_loader(session.dataset, config, epoch)):
        if iteration >= iterations:
            break
        classification, distillation = forward_backward(session, images, labels, indices)
        classification_sum += classification
        distillation_sum += distillation
        seen += images.shape[0]
        if (iteration + 1) % config.accumulation == 0:
            optimizer_step(session)
        if (iteration + 1) % config.log_every == 0:
            elapsed = time.perf_counter() - window_started
            rate = config.log_every * config.batch_size / elapsed
            lr = session.optimizer.param_groups[0]["lr"]
            log(f"[OK] epoch {epoch} iteration {iteration + 1}/{iterations} adaface {classification:.4f} distill {distillation:.4f} lr {lr:.2e} {rate:.1f} images/s {elapsed / config.log_every:.3f} s/iteration")
            window_started = time.perf_counter()
    seconds = time.perf_counter() - started
    return {
        "epoch": epoch,
        "adaface": classification_sum / iterations,
        "distill": distillation_sum / iterations,
        "lr": session.optimizer.param_groups[0]["lr"],
        "seconds": seconds,
        "images_per_second": seen / seconds,
    }


def latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    latest = None
    latest_epoch = 0
    for path in output_dir.iterdir():
        match = CHECKPOINT_PATTERN.match(path.name)
        if match and int(match.group(1)) > latest_epoch:
            latest_epoch = int(match.group(1))
            latest = path
    return latest


def save_atomic(state: dict, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    partial.replace(path)


def write_metrics(history: list[dict], path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(history, indent=2), encoding="utf-8")
    partial.replace(path)


def check_holdout(path: Path, record: dict, checkpoint: Path | None) -> None:
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        if stored != record:
            raise ValueError(f"{path} holds another hold-out (seed {stored['seed']}, {stored['count']} identities) than this config draws (seed {record['seed']}, {record['count']}): use another output_dir")
        return
    if checkpoint is not None:
        raise ValueError(f"{checkpoint} has no holdout.json next to it: its held-out identities are unknown")
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    partial.replace(path)
    print(f"[OK] {record['count']} identities held out of training: {path}")


def train(config: TrainConfig) -> None:
    session = build_session(config)
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log = Log(output_dir / "train.log")
    history = []
    best_mean = -1.0
    start_epoch = 1
    checkpoint = latest_checkpoint(output_dir)
    check_holdout(output_dir / "holdout.json", holdout_record(session.index, session.split), checkpoint)
    if checkpoint is None:
        if config.paths.imagenet_weights is not None:
            from lsj_biometrics.modules.extractors.ssm.model import load_imagenet_backbone

            load_imagenet_backbone(session.model, config.paths.imagenet_weights)
        log(f"[OK] fresh run in {output_dir}: {len(session.dataset)} images, {session.head.kernel.shape[1]} identities ({session.split.holdout.size} held out), {session.steps_per_epoch} steps per epoch, distillation {'on' if session.teacher is not None else 'off'}")
    else:
        state = torch.load(checkpoint, map_location=config.device, weights_only=True)
        session.model.load_state_dict(state["model"])
        session.head.load_state_dict(state["head"])
        session.optimizer.load_state_dict(state["optimizer"])
        session.scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_mean = state["best_mean"]
        start_epoch = state["epoch"] + 1
        log(f"[OK] resumed from {checkpoint}: epoch {state['epoch']} done, best mean accuracy {best_mean:.4f}")
    for epoch in range(start_epoch, config.epochs + 1):
        metrics = run_epoch(session, epoch, log)
        results = evaluate_sets(session.model, config.paths.eval_dir, config.eval_sets, config.eval_limit, config.batch_size, config.device, True, None)
        sets = {}
        total = 0.0
        for result in results:
            sets[result.name] = result.as_dict()
            total += result.accuracy
        metrics["sets"] = sets
        metrics["mean_accuracy"] = total / len(results)
        history.append(metrics)
        summary = []
        for result in results:
            summary.append(f"{result.name} {result.accuracy:.4f}")
        log(f"[OK] epoch {epoch}: adaface {metrics['adaface']:.4f} distill {metrics['distill']:.4f} {metrics['images_per_second']:.1f} images/s, {', '.join(summary)}, mean {metrics['mean_accuracy']:.4f}")
        if metrics["mean_accuracy"] > best_mean:
            best_mean = metrics["mean_accuracy"]
            partial = output_dir / "best.pt.partial"
            save_ssm_face_model(session.model, partial)
            partial.replace(output_dir / "best.pt")
            log(f"[OK] best.pt is epoch {epoch}")
        state = {
            "model": session.model.state_dict(),
            "head": session.head.state_dict(),
            "optimizer": session.optimizer.state_dict(),
            "scheduler": session.scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "history": history,
            "best_mean": best_mean,
            "config": config.raw,
        }
        save_atomic(state, output_dir / f"epoch_{epoch}.pt")
        write_metrics(history, output_dir / "metrics.json")
    log(f"[OK] training done: best mean accuracy {best_mean:.4f}")


def benchmark(config: TrainConfig, steps: int) -> None:
    session = build_session(config)
    if config.paths.imagenet_weights is not None:
        from lsj_biometrics.modules.extractors.ssm.model import load_imagenet_backbone

        load_imagenet_backbone(session.model, config.paths.imagenet_weights)
    cuda = config.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    session.model.train()
    session.head.train()
    session.optimizer.zero_grad(set_to_none=True)
    done = 0
    epoch = 1
    started = time.perf_counter()
    timed_from = 0
    while done < steps:
        for images, labels, indices in epoch_loader(session.dataset, config, epoch):
            classification, distillation = forward_backward(session, images, labels, indices)
            done += 1
            if done % config.accumulation == 0:
                optimizer_step(session)
            if done == BENCHMARK_WARMUP:
                if cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                timed_from = done
            if done % config.log_every == 0:
                print(f"[OK] iteration {done}/{steps} adaface {classification:.4f} distill {distillation:.4f}", flush=True)
            if done >= steps:
                break
        epoch += 1
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    timed = done - timed_from
    print(f"[OK] {timed} iterations of {config.batch_size} images in {elapsed:.1f} s: {timed * config.batch_size / elapsed:.1f} images/s, {elapsed / timed:.3f} s/iteration")
    if cuda:
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[OK] peak VRAM {peak:.2f} GiB allocated by torch (batch {config.batch_size}, checkpointing {config.checkpointing})")
    epoch_hours = session.steps_per_epoch * config.accumulation * elapsed / timed / 3600
    print(f"[OK] at this rate one epoch of {len(session.dataset)} images takes {epoch_hours:.2f} h, {config.epochs} epochs {epoch_hours * config.epochs:.1f} h plus evaluation")
