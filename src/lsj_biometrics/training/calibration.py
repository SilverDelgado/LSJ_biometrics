from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.config.operating_point import OperatingPoint
from lsj_biometrics.gallery.decision import accepts
from lsj_biometrics.gallery.enrollment import EnrollmentResult
from lsj_biometrics.image.loading import load_rgb
from lsj_biometrics.modules.extractors.ssm import SSMFeatureExtractor
from lsj_biometrics.system import BiometricSystem, build_system, enroll_folder
from lsj_biometrics.training.augment import Perturber
from lsj_biometrics.training.casia import IMAGE_SIZE, CasiaDataset

TABLE_TARGETS = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4)
RELIABLE_FALSE_POSITIVES = 10
GRID = np.round(np.arange(-1.0, 1.0001, 0.005), 3)
SHEET_ROWS = 20
TEXT_WIDTH = 260


@dataclass(frozen=True)
class Threshold:
    target: float
    threshold: float
    false_positives: int
    probes: int
    fpir: float


@dataclass(frozen=True)
class Outcome:
    identities: np.ndarray
    mated: np.ndarray
    correct: np.ndarray
    hits: np.ndarray
    top: np.ndarray


@dataclass(frozen=True)
class SimulatedGallery:
    system: BiometricSystem
    probes: np.ndarray


def probe_identities(labels: np.ndarray) -> np.ndarray:
    identities = []
    for label in labels:
        identities.append(f"casia:{int(label)}")
    return np.asarray(identities)


def gallery_system(point: OperatingPoint, database: Path, crops: Path) -> BiometricSystem:
    return build_system(replace(point, database=database, crops=crops))


def enrol_gallery(point: OperatingPoint, folder: Path, directory: Path) -> tuple[BiometricSystem, list[EnrollmentResult]]:
    database = directory / "gallery.csv"
    if database.exists():
        database.unlink()
    system = gallery_system(point, database, directory / "crops")
    results, failed = enroll_folder(system, folder)
    if not results:
        raise ValueError(f"no face enrolled from {folder}")
    return system, results


def enrolled_crops(results: list[EnrollmentResult]) -> dict[str, np.ndarray]:
    crops = {}
    for result in results:
        for template in result.templates:
            crops[result.identity.identity_id] = load_rgb(template.face.image)
    return crops


def embed_images(dataset: CasiaDataset, extractor: SSMFeatureExtractor, batch_size: int, perturber: Perturber | None, transform: str | None, level: int | None) -> np.ndarray:
    vectors = []
    faces = []
    for position in range(len(dataset)):
        image, label, index = dataset[position]
        if perturber is not None:
            image = perturber.apply(image, transform, level, index)
        face_id = f"casia:{index}"
        faces.append(CroppedFaceImage(face_id=face_id, image=Image(image_id=face_id, pixels=image), layout=extractor.layout))
        if len(faces) == batch_size or position == len(dataset) - 1:
            for template in extractor.extract_batch(faces):
                vectors.append(np.asarray(template.vector, dtype=np.float32))
            faces = []
    return np.stack(vectors)


def measure(system: BiometricSystem, vectors: np.ndarray, identities: np.ndarray) -> Outcome:
    result = system.identify_batch(vectors)
    mated = np.isin(identities, np.asarray(result.enrolled))
    correct = result.identities == identities
    return Outcome(identities=identities, mated=mated, correct=correct, hits=result.identities, top=result.scores)


def fpir(outcome: Outcome, threshold: float) -> float:
    if outcome.mated.all():
        raise ValueError("every probe is enrolled in this gallery: no non-mated probe to measure FPIR")
    return float(np.mean(accepts(outcome.top[~outcome.mated], threshold)))


def fnir(outcome: Outcome, threshold: float) -> float:
    mated = outcome.mated
    return float(np.mean(~outcome.correct[mated] | ~accepts(outcome.top[mated], threshold)))


def fpir_threshold(outcome: Outcome, target: float) -> Threshold:
    ordered = np.sort(outcome.top[~outcome.mated])[::-1]
    allowed = int(np.floor(target * ordered.size))
    if allowed == 0:
        raise ValueError(f"FPIR {target:g} is below one false positive in {ordered.size} non-mated probes: it cannot be measured here")
    threshold = float(np.nextafter(ordered[allowed], np.inf))
    false_positives = int(np.sum(accepts(ordered, threshold)))
    return Threshold(target=target, threshold=threshold, false_positives=false_positives, probes=int(ordered.size), fpir=false_positives / ordered.size)


def measurable(target: float, probes: int) -> bool:
    return int(np.floor(target * probes)) > 0


def simulated_galleries(point: OperatingPoint, vectors: np.ndarray, labels: np.ndarray, size: int, count: int, seed: int, directory: Path) -> list[SimulatedGallery]:
    identities, first, counts = np.unique(labels, return_index=True, return_counts=True)
    eligible = identities[counts >= 2]
    if eligible.size < size:
        raise ValueError(f"{eligible.size} held-out identities have a second image, a simulated gallery needs {size}")
    if identities.size <= size:
        raise ValueError(f"{identities.size} held-out identities leave none outside a simulated gallery of {size} to measure its FPIR")
    first_of = {}
    for identity, position in zip(identities, first):
        first_of[int(identity)] = int(position)
    galleries = []
    for gallery in range(count):
        chosen = np.sort(np.random.default_rng([seed, gallery]).choice(eligible, size=size, replace=False))
        system = gallery_system(point, directory / f"gallery_{gallery}.csv", directory / "crops")
        probes = np.ones(labels.size, dtype=bool)
        for identity in chosen:
            position = first_of[int(identity)]
            system.inscribir_t(f"casia:{int(identity)}", vectors[position])
            probes[position] = False
        galleries.append(SimulatedGallery(system=system, probes=np.flatnonzero(probes)))
    return galleries


def simulated_outcomes(galleries: list[SimulatedGallery], vectors: np.ndarray, identities: np.ndarray) -> list[Outcome]:
    outcomes = []
    for gallery in galleries:
        outcomes.append(measure(gallery.system, vectors[gallery.probes], identities[gallery.probes]))
    return outcomes


def simulated_rates(outcomes: list[Outcome], threshold: float) -> dict:
    misses = []
    false_positives = []
    for outcome in outcomes:
        misses.append(fnir(outcome, threshold))
        false_positives.append(fpir(outcome, threshold))
    return {"fnir_mean": float(np.mean(misses)), "fnir_std": float(np.std(misses)), "simulated_fpir_mean": float(np.mean(false_positives))}


def threshold_table(real: Outcome, simulated: list[Outcome]) -> list[dict]:
    rows = []
    probes = int(np.sum(~real.mated))
    print(f"{'target FPIR':>11} {'threshold':>10} {'false pos':>9} {'FPIR':>9} {'FNIR mean':>9} {'std':>7} {'sim FPIR':>9}")
    for target in TABLE_TARGETS:
        if not measurable(target, probes):
            continue
        row = fpir_threshold(real, target)
        rates = simulated_rates(simulated, row.threshold)
        rows.append({"target": target, "threshold": row.threshold, "false_positives": row.false_positives, "fpir": row.fpir, "fnir_mean": rates["fnir_mean"], "fnir_std": rates["fnir_std"], "simulated_fpir_mean": rates["simulated_fpir_mean"]})
        print(f"{target:>11g} {row.threshold:>10.4f} {row.false_positives:>9} {row.fpir:>9.2e} {rates['fnir_mean']:>9.4f} {rates['fnir_std']:>7.4f} {rates['simulated_fpir_mean']:>9.2e}")
    return rows


def perturbed_rates(dataset: CasiaDataset, extractor: SSMFeatureExtractor, batch_size: int, real: BiometricSystem, galleries: list[SimulatedGallery], identities: np.ndarray, catalogues: tuple[Path, ...], threshold: float) -> list[dict]:
    rows = []
    print(f"{'catalogue':<16} {'condition':<24} {'FPIR':>9} {'FNIR mean':>9} {'std':>7} {'sim FPIR':>9}")
    for catalogue in catalogues:
        perturber = Perturber(catalogue)
        perturber.prepare()
        for transform in sorted(perturber.levels):
            for level in perturber.levels[transform]:
                probed = embed_images(dataset, extractor, batch_size, perturber, transform, level)
                rate = fpir(measure(real, probed, identities), threshold)
                rates = simulated_rates(simulated_outcomes(galleries, probed, identities), threshold)
                rows.append({"catalogue": str(catalogue), "transform": transform, "level": level, "fpir": rate, "fnir_mean": rates["fnir_mean"], "fnir_std": rates["fnir_std"], "simulated_fpir_mean": rates["simulated_fpir_mean"]})
                print(f"{Path(catalogue).stem:<16} {transform + '_' + str(level):<24} {rate:>9.2e} {rates['fnir_mean']:>9.4f} {rates['fnir_std']:>7.4f} {rates['simulated_fpir_mean']:>9.2e}")
    return rows


def sweep(real: Outcome, simulated: list[Outcome]) -> list[dict]:
    rows = []
    for threshold in GRID:
        rates = simulated_rates(simulated, float(threshold))
        rows.append({"threshold": float(threshold), "fpir": fpir(real, float(threshold)), "fnir_mean": rates["fnir_mean"], "fnir_std": rates["fnir_std"], "simulated_fpir_mean": rates["simulated_fpir_mean"]})
    return rows


def _write(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


def false_positive_sheet(dataset: CasiaDataset, real: Outcome, crops: dict[str, np.ndarray], out_path: Path) -> list[dict]:
    non_mated = np.flatnonzero(~real.mated)
    order = non_mated[np.argsort(-real.top[non_mated], kind="stable")[:SHEET_ROWS]]
    size = IMAGE_SIZE
    canvas = np.full((size * len(order), 2 * size + TEXT_WIDTH, 3), 255, dtype=np.uint8)
    rows = []
    for row, position in enumerate(order):
        image, label, index = dataset[int(position)]
        enrolled = str(real.hits[position])
        y = row * size
        canvas[y:y + size, 0:size] = image
        canvas[y:y + size, size:2 * size] = crops[enrolled]
        _write(canvas, f"#{row + 1} score {real.top[position]:.4f}", 2 * size + 6, y + 30)
        _write(canvas, f"casia label {label}, image {index}", 2 * size + 6, y + 55)
        _write(canvas, enrolled[:34], 2 * size + 6, y + 80)
        rows.append({"rank": row + 1, "score": float(real.top[position]), "casia_label": int(label), "casia_index": int(index), "gallery_identity": enrolled})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise ValueError(f"could not write {out_path}")
    return rows


def write_threshold(operating_point: Path, threshold: float, origin: str) -> None:
    lines = operating_point.read_text(encoding="utf-8").splitlines()
    found = {"threshold": 0, "threshold_origin": 0}
    written = []
    for line in lines:
        if line.startswith("threshold:"):
            found["threshold"] += 1
            written.append(f"threshold: {threshold!r}")
        elif line.startswith("threshold_origin:"):
            found["threshold_origin"] += 1
            written.append(f"threshold_origin: \"{origin}\"")
        else:
            written.append(line)
    for key in found:
        if found[key] != 1:
            raise ValueError(f"{operating_point} has {found[key]} '{key}:' lines, expected exactly one")
    operating_point.write_text("\n".join(written) + "\n", encoding="utf-8")
