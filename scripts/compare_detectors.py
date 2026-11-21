import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ai_dataset_transformation import TransformationRegistry, register_all_transforms

from lsj_biometrics.campaign.face import BoundingBox, Face, FaceLayout, Landmarks
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.engine.face_detector import FaceDetector
from lsj_biometrics.image.alignment import align_face
from lsj_biometrics.image.loading import load_rgb
from lsj_biometrics.modules.detectors.mtcnn import MTCNNFaceDetector
from lsj_biometrics.modules.detectors.yunet import MODEL, YuNetFaceDetector
from lsj_biometrics.modules.extractors.ssm import SSMConfig
from lsj_biometrics.system import largest_face

TRANSFORMS = ("darkening", "motion_blur", "sensor_noise", "compression", "occlusion", "night_ir")

LEVELS = (1, 2, 3, 4, 5)

CAMERA_SIDE = 1280

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

COLUMNS = 7

LABEL_HEIGHT = 26


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True, help="the transform_specs.yaml the perturbations come from")
    parser.add_argument("--output", type=Path, required=True, help="where the crop sheets are written")
    args = parser.parse_args()

    TransformationRegistry.load_specs(str(args.specs), verbose=False, force_reload=True)
    register_all_transforms(verbose=False)
    config = SSMConfig()
    layout = FaceLayout(size=config.image_size, landmarks=Landmarks(*config.reference_landmarks))
    detectors = {
        "yunet": YuNetFaceDetector(MODEL),
        "mtcnn": MTCNNFaceDetector(),
    }

    originals = []
    photos = []
    for path in sorted(args.photos.iterdir()):
        if path.suffix.lower() in PHOTO_SUFFIXES:
            pixels = load_rgb(path)
            originals.append((path.stem, pixels))
            photos.append((path.stem, camera_size(pixels)))
    print(f"[OK] {len(photos)} photos")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, detector in detectors.items():
        started = time.perf_counter()
        faces = []
        for stem, pixels in originals:
            faces.append(selected(detector, Image(image_id=stem, pixels=pixels)))
        elapsed = (time.perf_counter() - started) / len(originals)
        sheet = crop_sheet(originals, faces, layout)
        path = args.output / f"crops_{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        found = 0
        for face in faces:
            if face is not None:
                found += 1
        print(f"[OK] {name}: {found} of {len(photos)} clean photos with a face, {elapsed * 1000:.0f} ms per photo, sheet {path}")
        for (stem, _), face in zip(originals, faces):
            if face is None:
                print(f"[WARN] {name}: no face in {stem}")

    clean: dict[str, list[Face | None]] = {}
    for name, detector in detectors.items():
        faces = []
        for stem, pixels in photos:
            faces.append(selected(detector, Image(image_id=stem, pixels=pixels)))
        clean[name] = faces
    print(f"[OK] perturbations on the photos at most {CAMERA_SIDE} px")

    detected: dict[tuple[str, str, int], int] = {}
    for transform in TRANSFORMS:
        transform_function = TransformationRegistry.get_transform(transform)
        follows_time = TransformationRegistry.follows_time(transform)
        for level in LEVELS:
            params = TransformationRegistry.get_level_params(transform, level)
            for name in detectors:
                detected[(transform, name, level)] = 0
            for index, (stem, pixels) in enumerate(photos):
                bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                if follows_time:
                    perturbed = transform_function(bgr, _seed=index, _t=0.0, **params)
                else:
                    perturbed = transform_function(bgr, _seed=index, **params)
                image = Image(image_id=f"{stem}:{transform}_{level}", pixels=cv2.cvtColor(perturbed, cv2.COLOR_BGR2RGB))
                for name, detector in detectors.items():
                    reference = clean[name][index]
                    face = selected(detector, image)
                    if reference is not None and face is not None and overlap(reference.bounding_box, face.bounding_box) > 0.0:
                        detected[(transform, name, level)] += 1
        print(f"[OK] {transform} done")

    print()
    header = f"{'transform':<14}{'detector':<9}"
    for level in LEVELS:
        header += f"{'L' + str(level):>8}"
    header += f"{'mean':>8}"
    print(header)
    totals = {}
    for name in detectors:
        totals[name] = 0
    for transform in TRANSFORMS:
        for name in detectors:
            line = f"{transform:<14}{name:<9}"
            total = 0
            for level in LEVELS:
                count = detected[(transform, name, level)]
                total += count
                line += f"{100 * count / len(photos):>7.0f}%"
            line += f"{100 * total / (len(photos) * len(LEVELS)):>7.1f}%"
            totals[name] += total
            print(line)
    print()
    for name in detectors:
        print(f"[OK] {name}: {100 * totals[name] / (len(photos) * len(LEVELS) * len(TRANSFORMS)):.1f}% of the perturbed photos still detected at the clean face")


def camera_size(pixels: np.ndarray) -> np.ndarray:
    height, width = pixels.shape[:2]
    scale = CAMERA_SIDE / max(height, width)
    if scale >= 1.0:
        return pixels
    return cv2.resize(pixels, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def selected(detector: FaceDetector, image: Image) -> Face | None:
    faces = detector.detect(image).faces
    if not faces:
        return None
    return largest_face(faces)


def overlap(first: BoundingBox, second: BoundingBox) -> float:
    width = min(first.x + first.width, second.x + second.width) - max(first.x, second.x)
    height = min(first.y + first.height, second.y + second.height) - max(first.y, second.y)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def crop_sheet(photos: list[tuple[str, np.ndarray]], faces: list[Face | None], layout: FaceLayout) -> np.ndarray:
    rows = (len(photos) + COLUMNS - 1) // COLUMNS
    cell = layout.size
    sheet = np.full((rows * (cell + LABEL_HEIGHT), COLUMNS * cell, 3), 255, dtype=np.uint8)
    for index, ((stem, pixels), face) in enumerate(zip(photos, faces)):
        top = (index // COLUMNS) * (cell + LABEL_HEIGHT)
        left = (index % COLUMNS) * cell
        if face is None:
            sheet[top : top + cell, left : left + cell] = 160
            cv2.putText(sheet, "NO FACE", (left + 22, top + cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1, cv2.LINE_AA)
        else:
            sheet[top : top + cell, left : left + cell] = align_face(pixels, face.landmarks, layout)
        cv2.putText(sheet, stem[:20], (left + 2, top + cell + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        if len(stem) > 20:
            cv2.putText(sheet, stem[20:40], (left + 2, top + cell + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


if __name__ == "__main__":
    main()
