import argparse
import json
from pathlib import Path

SHEET_CROPS = 3


def register(subparsers) -> None:
    command = subparsers.add_parser("train", help="train the SSM extractor; resumes from the last epoch_<k>.pt in output_dir")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--steps", type=int, default=None, help="run this many iterations, print peak VRAM and images/s, save nothing")
    command.set_defaults(func=run_train)

    command = subparsers.add_parser("precompute-teacher", help="ArcFace R50 embeddings of every CASIA image, flip-averaged, as float16 .npy")
    command.add_argument("--config", type=Path, required=True)
    command.set_defaults(func=run_precompute_teacher)

    command = subparsers.add_parser("evaluate", help="10-fold accuracy and TAR@FAR=1e-3 on the .bin pair sets, clean and perturbed")
    model = command.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=Path, help="an SSM checkpoint (best.pt)")
    model.add_argument("--teacher", type=Path, help="the ArcFace R50 weights, measured as the reference")
    command.add_argument("--eval-dir", type=Path, required=True)
    command.add_argument("--sets", nargs="+", required=True)
    command.add_argument("--out", type=Path, required=True, help="directory of the results JSON")
    command.add_argument("--no-flip", action="store_true", help="embed the face alone, without adding its mirror")
    command.add_argument("--perturb", type=Path, default=None, help="catalogue YAML: every transform at every level on the second image of each pair")
    command.add_argument("--limit", type=int, default=None, help="pairs per set, evenly spaced, for quick checks")
    command.add_argument("--device", default="cuda")
    command.add_argument("--batch-size", type=int, default=64)
    command.set_defaults(func=run_evaluate)

    command = subparsers.add_parser("calibrate", help="threshold at a target FPIR: held-out CASIA identities searched against the real gallery")
    command.add_argument("--operating-point", type=Path, required=True, help="model, detector and device; --write puts the threshold here")
    command.add_argument("--gallery", type=Path, required=True, help="folder of the photos to enrol, one person each")
    command.add_argument("--holdout", type=Path, required=True, help="holdout.json of the training run")
    command.add_argument("--casia", type=Path, required=True, help="the CASIA-WebFace the model was trained on")
    command.add_argument("--fpir", type=float, required=True, help="target FPIR, e.g. 1e-3")
    command.add_argument("--out", type=Path, required=True, help="directory of the calibration gallery (gallery.csv, crops), the sweep JSON and the false-positive sheet")
    command.add_argument("--galleries", type=int, default=10, help="simulated open-set galleries for the FNIR")
    command.add_argument("--limit", type=int, default=None, help="only the first N held-out identities, for quick checks")
    command.add_argument("--perturb", type=Path, nargs="+", default=None, help="catalogue YAMLs: FPIR and FNIR at the threshold with the probes perturbed, report only")
    command.add_argument("--write", action="store_true", help="write threshold and threshold_origin into the operating point")
    command.add_argument("--batch-size", type=int, default=64)
    command.set_defaults(func=run_calibrate)

    command = subparsers.add_parser("augment-sheet", help="PNG of CASIA crops under every transform and level of the training catalogue")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(func=run_augment_sheet)


def run_train(args: argparse.Namespace) -> None:
    from lsj_biometrics.training.train import benchmark, load_config, train

    config = load_config(args.config)
    if args.steps is None:
        train(config)
        return
    if args.steps < 1:
        raise ValueError(f"--steps {args.steps}: at least one")
    benchmark(config, args.steps)


def run_precompute_teacher(args: argparse.Namespace) -> None:
    import numpy as np

    from lsj_biometrics.training.casia import CasiaDataset, read_index
    from lsj_biometrics.training.teacher import load_teacher, precompute
    from lsj_biometrics.training.train import check_device, load_config, teacher_rows

    config = load_config(args.config)
    if config.paths.teacher_weights is None or config.paths.teacher_embeddings is None:
        raise ValueError(f"{args.config}: precompute-teacher needs paths.teacher_weights and paths.teacher_embeddings")
    check_device(config.device)
    rows = teacher_rows(config, read_index(config.paths.casia))
    dataset = CasiaDataset(config.paths.casia, None, np.arange(rows, dtype=np.int64))
    precompute(dataset, load_teacher(config.paths.teacher_weights), config.paths.teacher_embeddings, config.batch_size, config.device, config.workers)


def run_evaluate(args: argparse.Namespace) -> None:
    from lsj_biometrics.training.augment import Perturber
    from lsj_biometrics.training.train import check_device
    from lsj_biometrics.training.verification import evaluate_sets, print_table

    check_device(args.device)
    if args.model is not None:
        from lsj_biometrics.modules.extractors.ssm import load_ssm_face_model

        model = load_ssm_face_model(args.model)
        source = args.model
    else:
        from lsj_biometrics.training.teacher import load_teacher

        model = load_teacher(args.teacher)
        source = args.teacher
    model = model.to(args.device).eval()
    perturber = None
    if args.perturb is not None:
        perturber = Perturber(args.perturb)
        perturber.prepare()
    flip = not args.no_flip
    results = evaluate_sets(model, args.eval_dir, tuple(args.sets), args.limit, args.batch_size, args.device, flip, perturber)
    print_table(results)
    rows = []
    for result in results:
        rows.append(result.as_dict())
    condition = "clean"
    if args.perturb is not None:
        condition = args.perturb.stem
    name = f"{source.stem}_{condition}_{'flip' if flip else 'noflip'}"
    if args.limit is not None:
        name = f"{name}_limit{args.limit}"
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{name}.json"
    report = {"model": str(source), "flip": flip, "catalogue": None if args.perturb is None else str(args.perturb), "limit": args.limit, "results": rows}
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_path}")


def run_calibrate(args: argparse.Namespace) -> None:
    import tempfile

    import numpy as np

    from lsj_biometrics.config.operating_point import load_operating_point
    from lsj_biometrics.modules.extractors.ssm import SSMFeatureExtractor, load_ssm_face_model
    from lsj_biometrics.training.calibration import RELIABLE_FALSE_POSITIVES, embed_images, enrol_gallery, enrolled_crops, false_positive_sheet, fpir_threshold, measure, perturbed_rates, probe_identities, simulated_galleries, simulated_outcomes, simulated_rates, sweep, threshold_table, write_threshold
    from lsj_biometrics.training.casia import CasiaDataset, images_of, load_holdout, read_index
    from lsj_biometrics.training.train import check_device

    if args.galleries < 1:
        raise ValueError(f"--galleries {args.galleries}: at least one")
    point = load_operating_point(args.operating_point)
    check_device(point.device)
    index = read_index(args.casia)
    split = load_holdout(args.holdout, index)
    held = split.holdout
    if args.limit is not None:
        if args.limit < 1 or args.limit > held.size:
            raise ValueError(f"--limit {args.limit} outside 1..{held.size} held-out identities")
        held = held[:args.limit]
    if held.size == 0:
        raise ValueError(f"{args.holdout} holds no identity out: train with holdout_identities > 0")
    args.out.mkdir(parents=True, exist_ok=True)
    real, results = enrol_gallery(point, args.gallery, args.out)
    enrolled = []
    for result in results:
        enrolled.append(result.identity.identity_id)
    print(f"[OK] enrolled {len(enrolled)} photos of {args.gallery} with {point.model} ({point.detector}, {point.device}) into {args.out / 'gallery.csv'}")
    extractor = SSMFeatureExtractor(load_ssm_face_model(point.model), device=point.device, batch_size=args.batch_size)
    dataset = CasiaDataset(args.casia, None, images_of(index, held))
    labels = index.labels[dataset.indices]
    identities = probe_identities(labels)
    print(f"[OK] embedding {len(dataset)} images of {held.size} held-out identities")
    clean = embed_images(dataset, extractor, args.batch_size, None, None, None)
    outcome = measure(real, clean, identities)
    with tempfile.TemporaryDirectory() as scratch:
        galleries = simulated_galleries(point, clean, labels, len(enrolled), args.galleries, split.seed, Path(scratch))
        simulated = simulated_outcomes(galleries, clean, identities)
        mated = 0
        for result in simulated:
            mated += int(np.sum(result.mated))
        chosen = fpir_threshold(outcome, args.fpir)
        if chosen.false_positives < RELIABLE_FALSE_POSITIVES:
            print(f"[WARN] FPIR {args.fpir:g} rests on {chosen.false_positives} false positives out of {chosen.probes} probes")
        table = threshold_table(outcome, simulated)
        rates = simulated_rates(simulated, chosen.threshold)
        print(f"[OK] threshold {chosen.threshold:.6f} at FPIR {args.fpir:g}: {chosen.false_positives}/{chosen.probes} non-mated probes above it, simulated FNIR {rates['fnir_mean']:.4f} +- {rates['fnir_std']:.4f}, simulated FPIR {rates['simulated_fpir_mean']:.2e} ({args.galleries} galleries of {len(enrolled)}, {mated} mated probes)")
        perturbed = []
        if args.perturb is not None:
            perturbed = perturbed_rates(dataset, extractor, args.batch_size, real, galleries, identities, tuple(args.perturb), chosen.threshold)
        swept = sweep(outcome, simulated)
    name = f"{point.model.stem}_fpir{args.fpir:g}"
    if args.limit is not None:
        name = f"{name}_limit{args.limit}"
    sheet = args.out / f"false_positives_{name}.png"
    false_positives = false_positive_sheet(dataset, outcome, enrolled_crops(results), sheet)
    print(f"[OK] top {len(false_positives)} false positives drawn in {sheet}: look for a held-out celebrity who is someone of the gallery")
    calibration = {"target": args.fpir, "threshold": chosen.threshold, "false_positives": chosen.false_positives, "probes": chosen.probes, "fpir": chosen.fpir, "fnir_mean": rates["fnir_mean"], "fnir_std": rates["fnir_std"], "simulated_fpir_mean": rates["simulated_fpir_mean"]}
    report = {
        "model": str(point.model),
        "detector": point.detector,
        "gallery": str(args.gallery),
        "enrolled": enrolled,
        "holdout": str(args.holdout),
        "holdout_identities": int(held.size),
        "non_mated_probes": chosen.probes,
        "simulation": {"galleries": args.galleries, "gallery_size": len(enrolled), "mated_probes": mated, "seed": split.seed},
        "calibration": calibration,
        "table": table,
        "sweep": swept,
        "perturbed": perturbed,
        "false_positives": false_positives,
    }
    out_path = args.out / f"calibration_{name}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_path}")
    if args.write:
        origin = f"FPIR {args.fpir:g} on {args.gallery} ({len(enrolled)} enrolled) against {chosen.probes} held-out CASIA probes of {held.size} identities, {point.model}"
        write_threshold(args.operating_point, chosen.threshold, origin)
        print(f"[OK] threshold {chosen.threshold:.6f} written into {args.operating_point}")


def run_augment_sheet(args: argparse.Namespace) -> None:
    from lsj_biometrics.training.augment import Perturber, render_sheet
    from lsj_biometrics.training.casia import CasiaDataset, read_index
    from lsj_biometrics.training.train import load_config

    config = load_config(args.config)
    index = read_index(config.paths.casia)
    positions = []
    for crop in range(SHEET_CROPS):
        positions.append(crop * index.count // SHEET_CROPS)
    dataset = CasiaDataset(config.paths.casia, None, positions)
    crops = []
    for position in range(len(dataset)):
        crops.append(dataset[position][0])
    render_sheet(crops, Perturber(config.paths.augment_specs), config.seed, args.out)
