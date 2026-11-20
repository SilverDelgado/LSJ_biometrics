import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

RECORD_MAGIC = 0xCED7230A
HEADER_FORMAT = "<IfQQ"
HEADER_SIZE = 24
LENGTH_BITS = 29
IMAGE_SIZE = 112


class Augmenter(Protocol):
    def __call__(self, image: np.ndarray, index: int) -> np.ndarray: ...


@dataclass(frozen=True)
class CasiaIndex:
    root: Path
    offsets: np.ndarray
    labels: np.ndarray
    num_classes: int

    @property
    def count(self) -> int:
        return int(self.labels.shape[0])


def read_index(root: Path) -> CasiaIndex:
    root = Path(root)
    for name in ("train.rec", "train.idx", "property"):
        if not (root / name).exists():
            raise ValueError(f"{name} missing in {root}")
    records = {}
    with open(root / "train.idx", encoding="utf-8") as handle:
        for line in handle:
            number, offset = line.strip().split("\t")
            records[int(number)] = int(offset)
    offsets = np.zeros(len(records), dtype=np.int64)
    for number, offset in records.items():
        if number >= len(records):
            raise ValueError(f"train.idx in {root} skips records: {number} of {len(records)}")
        offsets[number] = offset
    with open(root / "train.rec", "rb") as handle:
        flag, label, extra, payload = read_record(handle, int(offsets[0]))
        if flag != 2:
            raise ValueError(f"record 0 of {root} is not the insightface header: flag {flag}")
        images_end = int(extra[0])
        identities_end = int(extra[1])
        if identities_end != len(offsets):
            raise ValueError(f"record 0 says {identities_end} records, train.idx has {len(offsets)}")
        labels = np.full(images_end - 1, -1, dtype=np.int64)
        expected_start = 1
        for number in range(images_end, identities_end):
            flag, label, extra, payload = read_record(handle, int(offsets[number]))
            start = int(extra[0])
            end = int(extra[1])
            if start != expected_start or end <= start:
                raise ValueError(f"identity record {number} covers [{start}, {end}), expected it to start at {expected_start}")
            labels[start - 1:end - 1] = number - images_end
            expected_start = end
    if expected_start != images_end:
        raise ValueError(f"the identity records end at image {expected_start}, the images at {images_end}")
    num_classes = identities_end - images_end
    stated = (root / "property").read_text(encoding="utf-8").strip().split(",")
    if int(stated[0]) != num_classes or int(stated[1]) != IMAGE_SIZE or int(stated[2]) != IMAGE_SIZE:
        raise ValueError(f"property of {root} says {stated}, the .rec has {num_classes} identities of {IMAGE_SIZE}x{IMAGE_SIZE}")
    return CasiaIndex(root=root, offsets=offsets, labels=labels, num_classes=num_classes)


def read_record(handle, offset: int) -> tuple[int, float, tuple, bytes]:
    handle.seek(offset)
    magic, length = struct.unpack("<II", handle.read(8))
    if magic != RECORD_MAGIC:
        raise ValueError(f"wrong magic at offset {offset}: the .rec is corrupt")
    if length >> LENGTH_BITS != 0:
        raise ValueError(f"record at offset {offset} is split in parts, which this reader does not join")
    data = handle.read(length)
    flag, label, first_id, second_id = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    extra = ()
    start = HEADER_SIZE
    if flag > 0:
        extra = struct.unpack(f"<{flag}f", data[HEADER_SIZE:HEADER_SIZE + 4 * flag])
        start = HEADER_SIZE + 4 * flag
    return flag, label, extra, data[start:]


def decode_rgb(payload: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("a CASIA record does not decode as an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class CasiaDataset:

    def __init__(self, root: Path, augmenter: Augmenter | None, indices: np.ndarray) -> None:
        self.index = read_index(root)
        self.augmenter = augmenter
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= self.index.count):
            raise ValueError(f"indices out of range: the .rec has {self.index.count} images")
        self._handle = None
        self._pid = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, position: int) -> tuple[np.ndarray, int, int]:
        index = int(self.indices[position])
        image = self.read(index)
        if self.augmenter is not None:
            image = self.augmenter(image, index)
        return image, int(self.index.labels[index]), index

    def read(self, index: int) -> np.ndarray:
        flag, label, extra, payload = read_record(self._file(), int(self.index.offsets[index + 1]))
        expected = int(self.index.labels[index])
        if flag != 0 or int(label) != expected:
            raise ValueError(f"image {index} carries label {label}, its identity record says {expected}")
        image = decode_rgb(payload)
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"image {index} is {image.shape}, expected {IMAGE_SIZE}x{IMAGE_SIZE}x3")
        return image

    def _file(self):
        if self._handle is None or self._pid != os.getpid():
            self._handle = open(self.index.root / "train.rec", "rb")
            self._pid = os.getpid()
        return self._handle

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_pid"] = None
        return state


def subset_of(indices: np.ndarray, subset: int | None) -> np.ndarray:
    if subset is None:
        return indices
    if subset < 1 or subset > indices.size:
        raise ValueError(f"subset {subset} outside 1..{indices.size}")
    return indices[:subset]


@dataclass(frozen=True)
class IdentitySplit:
    seed: int
    holdout: np.ndarray
    trained: np.ndarray
    dense: np.ndarray


def split_identities(index: CasiaIndex, holdout_count: int, seed: int) -> IdentitySplit:
    if holdout_count < 0 or holdout_count >= index.num_classes:
        raise ValueError(f"holdout_identities {holdout_count} outside 0..{index.num_classes - 1}")
    order = np.random.default_rng(seed).permutation(index.num_classes)
    holdout = np.sort(order[:holdout_count])
    held = np.zeros(index.num_classes, dtype=bool)
    held[holdout] = True
    trained = np.flatnonzero(~held)
    dense = np.full(index.num_classes, -1, dtype=np.int64)
    dense[trained] = np.arange(trained.size)
    return IdentitySplit(seed=seed, holdout=holdout, trained=trained, dense=dense)


def images_of(index: CasiaIndex, labels: np.ndarray) -> np.ndarray:
    member = np.zeros(index.num_classes, dtype=bool)
    member[labels] = True
    return np.flatnonzero(member[index.labels]).astype(np.int64)


HOLDOUT_KEYS = ("seed", "count", "classes", "images", "labels")


def holdout_record(index: CasiaIndex, split: IdentitySplit) -> dict:
    labels = []
    for label in split.holdout:
        labels.append(int(label))
    return {"seed": split.seed, "count": len(labels), "classes": index.num_classes, "images": index.count, "labels": labels}


def load_holdout(path: Path, index: CasiaIndex) -> IdentitySplit:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError(f"{path} must be a mapping")
    for key in HOLDOUT_KEYS:
        if key not in record:
            raise ValueError(f"{path} is missing '{key}'")
    for key in record:
        if key not in HOLDOUT_KEYS:
            raise ValueError(f"{path} has an unknown key '{key}'")
    split = split_identities(index, record["count"], record["seed"])
    if holdout_record(index, split) != record:
        raise ValueError(f"{path} does not match the split seed {record['seed']} draws on {index.root} ({index.num_classes} identities, {index.count} images)")
    return split
