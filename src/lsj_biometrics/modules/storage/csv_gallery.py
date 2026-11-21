import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import numpy as np

from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.matcher import Match
from lsj_biometrics.gallery.repository import probe_vector

COLUMNS = ("identity_id", "template_id", "model_key", "created_at", "vector")

SCORE_BATCH = 4096


class CsvGalleryRepository:
    def __init__(self, path: Path, model_key: str) -> None:
        self.model_key = model_key
        self._path = path
        self._model_name, self._model_version = split_model_key(model_key)
        self._rows: list[Template] = []
        if path.exists():
            self._rows = self._read()
        self._rebuild()

    def save_templates(self, templates: Sequence[Template]) -> None:
        rows = list(self._rows)
        known = set(self._template_ids)
        dimension = None
        if self._rows:
            dimension = self._matrix.shape[1]
        for template in templates:
            self._check(template)
            if template.template_id in known:
                raise ValueError(f"template {template.template_id} is already stored")
            if dimension is None:
                dimension = template.dimension
            if template.dimension != dimension:
                raise ValueError(f"template {template.template_id} has dimension {template.dimension}, the gallery {dimension}")
            known.add(template.template_id)
            rows.append(template)
        self._write(rows)
        self._rows = rows
        self._rebuild()

    def templates(self, identity_id: str) -> list[Template]:
        found = []
        for row in self._rows:
            if row.identity_id == identity_id:
                found.append(row)
        return found

    def scores(self, vectors: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        best, chosen = self._best(vectors)
        return best, self._identities

    def search(self, template: Template, rank: int) -> list[Match]:
        if template.model_key != self.model_key:
            raise ValueError(f"template from {template.model_key} cannot search a {self.model_key} gallery")
        if not self._rows:
            return []
        best, chosen = self._best(probe_vector(template))
        matches = []
        for column in np.argsort(-best[0], kind="stable")[:rank]:
            matches.append(Match(identity_id=self._identities[column], template_id=self._template_ids[chosen[0, column]], score=float(best[0, column])))
        return matches

    def _best(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"probes must be a P x D matrix, got shape {vectors.shape}")
        best = np.zeros((vectors.shape[0], len(self._identities)), dtype=np.float32)
        chosen = np.zeros((vectors.shape[0], len(self._identities)), dtype=np.int64)
        if not self._rows:
            return best, chosen
        if vectors.shape[1] != self._matrix.shape[1]:
            raise ValueError(f"probes of dimension {vectors.shape[1]} cannot search a gallery of dimension {self._matrix.shape[1]}")
        for start in range(0, vectors.shape[0], SCORE_BATCH):
            raw = vectors[start:start + SCORE_BATCH] @ self._matrix.T
            for column, members in enumerate(self._members):
                block = raw[:, members]
                position = block.argmax(axis=1)
                best[start:start + SCORE_BATCH, column] = block[np.arange(block.shape[0]), position]
                chosen[start:start + SCORE_BATCH, column] = members[position]
        return best, chosen

    def _check(self, template: Template) -> None:
        if template.model_key != self.model_key:
            raise ValueError(f"template from {template.model_key} cannot be stored in a {self.model_key} gallery")
        if template.template_id is None or template.identity_id is None or template.created_at is None:
            raise ValueError("a stored template needs its template_id, identity_id and created_at")
        if template.created_at.tzinfo is None:
            raise ValueError(f"template {template.template_id} created_at must be timezone-aware")
        if not template.normalized:
            raise ValueError(f"template {template.template_id} is not normalized")
        for value in template.vector:
            if not math.isfinite(value):
                raise ValueError(f"template {template.template_id} has a non-finite value")

    def _rebuild(self) -> None:
        vectors = []
        identity_ids = []
        template_ids = []
        for row in self._rows:
            vectors.append(row.vector)
            identity_ids.append(row.identity_id)
            template_ids.append(row.template_id)
        self._matrix = np.asarray(vectors, dtype=np.float32)
        self._template_ids = template_ids
        identities = []
        members = {}
        for row, identity_id in enumerate(identity_ids):
            if identity_id not in members:
                identities.append(identity_id)
                members[identity_id] = []
            members[identity_id].append(row)
        self._identities = tuple(identities)
        self._members = []
        for identity_id in identities:
            self._members.append(np.asarray(members[identity_id], dtype=np.int64))

    def _read(self) -> list[Template]:
        rows = []
        template_ids = set()
        dimension = None
        with open(self._path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None or tuple(header) != COLUMNS:
                raise ValueError(f"{self._path}: header must be {','.join(COLUMNS)}, got {header}")
            for line, values in enumerate(reader, start=2):
                if len(values) != len(COLUMNS):
                    raise ValueError(f"{self._path}:{line}: expected {len(COLUMNS)} fields, got {len(values)}")
                identity_id, template_id, model_key, created_at, vector = values
                if model_key != self.model_key:
                    raise ValueError(f"{self._path}:{line}: template from {model_key} in a {self.model_key} gallery")
                if template_id in template_ids:
                    raise ValueError(f"{self._path}:{line}: template {template_id} appears twice")
                template_ids.add(template_id)
                parsed = parse_vector(vector)
                if dimension is None:
                    dimension = len(parsed)
                if len(parsed) != dimension:
                    raise ValueError(f"{self._path}:{line}: dimension {len(parsed)}, the rows before {dimension}")
                timestamp = datetime.fromisoformat(created_at)
                if timestamp.tzinfo is None:
                    raise ValueError(f"{self._path}:{line}: created_at must carry its timezone")
                rows.append(Template(vector=parsed, model_name=self._model_name, model_version=self._model_version, normalized=True, template_id=template_id, identity_id=identity_id, created_at=timestamp))
        return rows

    def _write(self, rows: Sequence[Template]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._path.with_name(f"{self._path.name}.{uuid4().hex}.partial")
        with open(partial, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for row in rows:
                created_at = row.created_at.astimezone(timezone.utc).isoformat()
                writer.writerow((row.identity_id, row.template_id, row.model_key, created_at, format_vector(row.vector)))
        partial.replace(self._path)


def split_model_key(model_key: str) -> tuple[str, str]:
    parts = model_key.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"model key {model_key!r} must be name:version")
    return parts[0], parts[1]


def format_vector(vector: Sequence[float]) -> str:
    tokens = []
    for value in vector:
        tokens.append(str(np.float32(value)))
    return " ".join(tokens)


def parse_vector(text: str) -> tuple[float, ...]:
    values = []
    for token in text.split(" "):
        value = float(np.float32(token))
        if not math.isfinite(value):
            raise ValueError(f"vector value {token} is not finite")
        values.append(value)
    return tuple(values)
