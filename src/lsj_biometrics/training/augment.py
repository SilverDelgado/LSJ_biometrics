import os
from pathlib import Path

import cv2
import numpy as np
import yaml

SCALE_RANGE = (0.85, 1.0)
ROTATION_DEGREES = 10.0
SHIFT_FRACTION = 0.06
FLIP_PROBABILITY = 0.5
SEED_LIMIT = 2**31 - 1

_loaded_catalogue = None


def read_levels(catalogue: Path) -> dict[str, tuple[int, ...]]:
    with open(catalogue, encoding="utf-8") as handle:
        specs = yaml.safe_load(handle)
    levels = {}
    for name, spec in specs.items():
        numbers = []
        for level in spec["levels"]:
            numbers.append(int(level))
        if not numbers:
            raise ValueError(f"{name} in {catalogue} has no levels")
        levels[name] = tuple(sorted(numbers))
    return levels


class Perturber:

    def __init__(self, catalogue: Path) -> None:
        self.catalogue = Path(catalogue).resolve()
        self.levels = read_levels(self.catalogue)

    def prepare(self) -> None:
        global _loaded_catalogue
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        from ai_dataset_transformation import TransformationRegistry, register_all_transforms

        TransformationRegistry.load_specs(str(self.catalogue), verbose=False, force_reload=True)
        register_all_transforms(verbose=False)
        _loaded_catalogue = self.catalogue

    def apply(self, rgb: np.ndarray, transform: str, level: int, seed: int) -> np.ndarray:
        if _loaded_catalogue != self.catalogue:
            raise RuntimeError(f"{self.catalogue} is not loaded in this process: call prepare() first")
        from ai_dataset_transformation import TransformationRegistry

        params = TransformationRegistry.get_level_params(transform, level)
        function = TransformationRegistry.get_transform(transform)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if TransformationRegistry.follows_time(transform):
            out = function(bgr, _seed=int(seed), _t=0.0, **params)
        else:
            out = function(bgr, _seed=int(seed), **params)
        if out.shape != rgb.shape or out.dtype != np.uint8:
            raise ValueError(f"{transform} level {level} returned {out.shape} {out.dtype} for {rgb.shape} uint8")
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = image.shape[0]
    scale = rng.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
    angle = rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
    shift_x = rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION) * size
    shift_y = rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION) * size
    flip = rng.random() < FLIP_PROBABILITY
    centre = (size - 1) / 2.0
    matrix = cv2.getRotationMatrix2D((centre, centre), angle, scale)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    out = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if flip:
        out = np.ascontiguousarray(out[:, ::-1])
    return out


class TrainingAugmenter:

    def __init__(self, perturber: Perturber, probability: float, seed: int) -> None:
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"perturb probability {probability} outside 0..1")
        self.perturber = perturber
        self.probability = probability
        self.seed = seed
        self.epoch = 0
        self.names = tuple(sorted(perturber.levels))

    def prepare(self) -> None:
        self.perturber.prepare()

    def __call__(self, image: np.ndarray, index: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch, index])
        if rng.random() < self.probability:
            name = self.names[int(rng.integers(len(self.names)))]
            levels = self.perturber.levels[name]
            level = levels[int(rng.integers(len(levels)))]
            image = self.perturber.apply(image, name, level, int(rng.integers(SEED_LIMIT)))
        return jitter(image, rng)


LABEL_WIDTH = 150
BAND_HEIGHT = 18
TEXT_COLOUR = (0, 0, 0)


def _write(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOUR, 1, cv2.LINE_AA)


def render_sheet(crops: list[np.ndarray], perturber: Perturber, seed: int, out_path: Path) -> None:
    perturber.prepare()
    size = crops[0].shape[0]
    columns = 1
    for name in perturber.levels:
        columns = max(columns, 1 + len(perturber.levels[name]))
    blocks = []
    for name in sorted(perturber.levels):
        blocks.append(name)
    blocks.append("jitter")
    block_height = BAND_HEIGHT + size * len(crops)
    canvas = np.full((block_height * len(blocks), LABEL_WIDTH + size * columns, 3), 255, dtype=np.uint8)
    for block, name in enumerate(blocks):
        top = block * block_height
        _write(canvas, name, 4, top + 13)
        _write(canvas, "clean", LABEL_WIDTH + 4, top + 13)
        if name == "jitter":
            levels = tuple(range(1, columns))
        else:
            levels = perturber.levels[name]
        for column, level in enumerate(levels, start=1):
            label = f"level {level}"
            if name == "jitter":
                label = f"draw {level}"
            _write(canvas, label, LABEL_WIDTH + column * size + 4, top + 13)
        for row, crop in enumerate(crops):
            y = top + BAND_HEIGHT + row * size
            canvas[y:y + size, LABEL_WIDTH:LABEL_WIDTH + size] = crop
            for column, level in enumerate(levels, start=1):
                if name == "jitter":
                    cell = jitter(crop, np.random.default_rng([seed, row, level]))
                else:
                    cell = perturber.apply(crop, name, level, seed + row)
                x = LABEL_WIDTH + column * size
                canvas[y:y + size, x:x + size] = cell
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise ValueError(f"could not write {out_path}")
    print(f"[OK] augmentation sheet {out_path}: {len(crops)} crops x {len(blocks)} blocks")
