"""In-memory matrices of stored embeddings, for a fast similarity scan.

``EmbeddingStore.search`` used to decode every stored vector with
``struct.unpack`` and compute its cosine in pure Python on each call. With
numpy available the vectors of one store and provider are kept as a single
float32 matrix instead, and a search is one matrix product.

The ranking is the pure-Python one, not an approximation of it:

1. A float32 product scores every row. Its error is far below
   ``_COARSE_BAND``, so every row that can reach the top ``limit`` lies
   within that band of the coarse cut-off and becomes a candidate.
2. The candidates are rescored in float64 from the same float32 values the
   pure-Python path decodes.
3. Candidates whose float64 scores lie within ``_TIE_EPSILON`` of another
   candidate (duplicated vectors, mostly) are rescored with the pure-Python
   cosine itself, so their relative order is decided by the very numbers the
   pure-Python path compares. Ties keep the row order of the table scan, as
   a stable sort does there.

Anything this module cannot represent faithfully (rows of differing sizes,
non-finite or extreme values) returns ``None`` and the caller scans in pure
Python, which is also what happens when numpy is not installed.

Matrices are float32, the precision the vectors are stored in: 4 bytes per
component, so 136,713 vectors of 768 dimensions take 420 MB. The cache is a
process-wide LRU bounded by ``CRG_VECTOR_CACHE_MB`` (default 1024; 0 keeps
nothing between calls). An entry is reused only while the database file, its
write-ahead log and the embeddings table still look the way they did when it
was loaded.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_MB = 1024
_BYTES_PER_MB = 1024 * 1024
# The coarse float32 score of a row is off by at most ~dimension * 6e-8 (5e-5
# at 768 dimensions, far less in practice). A row can only be in the true top
# ``limit`` if it is within twice that of the coarse cut-off.
_COARSE_BAND = 1e-3
# Float64 scores from numpy and from the pure-Python sum differ by ~1e-13 at
# most, so two candidates further apart than this are ordered the same way by
# both. Closer ones are settled with the pure-Python cosine.
_TIE_EPSILON = 1e-9
# Outside this range of vector lengths the float32 product can overflow or
# lose its accuracy, so such stores and queries are left to pure Python.
_MIN_NORM = 1e-10
_MAX_NORM = 1e10
_NORM_CHUNK_ROWS = 4096

ExactCosine = Callable[[list[float], list[float]], float]
Fingerprint = tuple[Any, ...]
_CacheKey = tuple[str, str]


def load_numpy() -> Any | None:
    """Return the numpy module, or None when it is disabled or not installed."""
    if os.environ.get("CRG_NUMPY_SEARCH", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        import numpy
    except ImportError:
        return None
    return numpy


@dataclass(frozen=True)
class VectorMatrix:
    """The vectors of one store and provider, in table-scan order."""

    names: tuple[str, ...]
    vectors: Any  # float32, shape (rows, dimension), read-only
    norms: Any  # float64, shape (rows,), read-only
    nbytes: int


def _cache_budget_bytes() -> int:
    raw = os.environ.get("CRG_VECTOR_CACHE_MB", "").strip()
    if not raw:
        return _DEFAULT_CACHE_MB * _BYTES_PER_MB
    try:
        return max(int(raw), 0) * _BYTES_PER_MB
    except ValueError:
        logger.warning(
            "CRG_VECTOR_CACHE_MB=%r is not a whole number; using %d",
            raw, _DEFAULT_CACHE_MB,
        )
        return _DEFAULT_CACHE_MB * _BYTES_PER_MB


class _MatrixCache:
    """Least-recently-used matrices, bounded by their total size in bytes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[_CacheKey, tuple[Fingerprint, VectorMatrix]] = OrderedDict()
        self._bytes = 0

    def get(self, key: _CacheKey, fingerprint: Fingerprint) -> VectorMatrix | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry[0] != fingerprint:
                self._drop(key)
                return None
            self._entries.move_to_end(key)
            return entry[1]

    def put(self, key: _CacheKey, fingerprint: Fingerprint, matrix: VectorMatrix) -> None:
        budget = _cache_budget_bytes()
        with self._lock:
            self._drop(key)
            if matrix.nbytes > budget:
                return
            self._entries[key] = (fingerprint, matrix)
            self._bytes += matrix.nbytes
            while self._bytes > budget and self._entries:
                self._drop(next(iter(self._entries)))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries), "bytes": self._bytes}

    def _drop(self, key: _CacheKey) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry[1].nbytes


_CACHE = _MatrixCache()


def clear_cache() -> None:
    """Forget every cached matrix."""
    _CACHE.clear()


def cache_stats() -> dict[str, int]:
    """Return how many matrices are cached and the bytes they hold."""
    return _CACHE.stats()


def _file_state(path: str, ignore_when_empty: bool = False) -> tuple[int, ...] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    if ignore_when_empty and stat.st_size == 0:
        # SQLite creates an empty write-ahead log each time a database is
        # opened and removes it on the last close, so its inode and mtime say
        # nothing about the content.
        return (0,)
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _fingerprint(conn: sqlite3.Connection, db_path: Path) -> Fingerprint:
    """Describe the store's state; any committed write changes it.

    A commit appends to the write-ahead log or, once checkpointed, rewrites
    the database file, which moves a size or an mtime. The row count and the
    highest rowid cover filesystems whose mtime is too coarse to show two
    writes in quick succession: a re-embedded row is deleted and re-inserted
    under a new rowid.
    """
    row = conn.execute("SELECT COUNT(*), MAX(rowid) FROM embeddings").fetchone()
    return (
        _file_state(str(db_path)),
        _file_state(f"{db_path}-wal", ignore_when_empty=True),
        row[0],
        row[1],
    )


def _build(np: Any, rows: list[Any]) -> VectorMatrix | None:
    names = tuple(row[0] for row in rows)
    if not rows:
        empty = np.zeros((0, 0), dtype=np.float32)
        return VectorMatrix(names, empty, np.zeros(0, dtype=np.float64), 0)

    blobs = [row[1] for row in rows]
    if any(not isinstance(blob, bytes) for blob in blobs):
        return None
    width = len(blobs[0])
    if width == 0 or width % 4 or any(len(blob) != width for blob in blobs):
        return None

    vectors = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), width // 4)
    norms = np.empty(len(blobs), dtype=np.float64)
    for start in range(0, len(blobs), _NORM_CHUNK_ROWS):
        chunk = vectors[start:start + _NORM_CHUNK_ROWS].astype(np.float64)
        norms[start:start + _NORM_CHUNK_ROWS] = np.sqrt(np.einsum("ij,ij->i", chunk, chunk))
    if not bool(np.isfinite(norms).all()):
        return None
    nonzero = norms[norms > 0]
    if nonzero.size and (float(nonzero.min()) < _MIN_NORM or float(nonzero.max()) > _MAX_NORM):
        return None
    norms.setflags(write=False)
    return VectorMatrix(names, vectors, norms, int(vectors.nbytes + norms.nbytes))


def load_matrix(
    conn: sqlite3.Connection, db_path: Path, provider_name: str,
) -> VectorMatrix | None:
    """Return the store's matrix for one provider, from the cache when it is current.

    None means the fast scan does not apply: numpy is missing or disabled, or
    the stored vectors cannot be laid out as one well-behaved matrix.
    """
    np = load_numpy()
    if np is None:
        return None

    try:
        key = (str(db_path.resolve()), provider_name)
    except OSError:
        key = (str(db_path), provider_name)
    before = _fingerprint(conn, db_path)
    cached = _CACHE.get(key, before)
    if cached is not None:
        return cached

    rows = conn.execute(
        "SELECT qualified_name, vector FROM embeddings WHERE provider = ?",
        (provider_name,),
    ).fetchall()
    matrix = _build(np, rows)
    # A write that landed while the rows were being read may or may not be in
    # them. Use the rows for this search, but do not keep them.
    if matrix is not None and _fingerprint(conn, db_path) == before:
        _CACHE.put(key, before, matrix)
    return matrix


def top_matches(
    matrix: VectorMatrix,
    query_vec: list[float],
    limit: int,
    exact_cosine: ExactCosine,
) -> list[tuple[str, float]] | None:
    """Return the ``limit`` most similar rows, ordered as the pure-Python scan orders them.

    ``exact_cosine`` is the pure-Python cosine; it settles near-ties. None
    means the caller must scan in pure Python instead.
    """
    np = load_numpy()
    if np is None or limit <= 0:
        return None
    rows = len(matrix.names)
    if rows == 0:
        return []
    if len(query_vec) != matrix.vectors.shape[1]:
        # The pure-Python cosine of mismatched sizes is 0.0 for every row.
        return [(name, 0.0) for name in matrix.names[:limit]]

    try:
        query = np.asarray(query_vec, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    query_norm = float(np.sqrt(np.dot(query, query)))
    if not math.isfinite(query_norm) or not _MIN_NORM <= query_norm <= _MAX_NORM:
        return None

    lengths = np.where(matrix.norms > 0, matrix.norms, 1.0) * query_norm
    coarse = (matrix.vectors @ query.astype(np.float32)).astype(np.float64) / lengths
    if not bool(np.isfinite(coarse).all()):
        return None

    if limit < rows:
        cutoff = np.partition(coarse, rows - limit)[rows - limit]
        candidates = np.flatnonzero(coarse >= cutoff - _COARSE_BAND)
    else:
        candidates = np.arange(rows)

    fine = (matrix.vectors[candidates].astype(np.float64) @ query) / lengths[candidates]
    scores = [float(score) for score in fine]
    for position in _near_ties(np, fine):
        row = matrix.vectors[candidates[position]].tolist()
        scores[position] = exact_cosine(query_vec, row)

    # ``candidates`` ascends, so this is table-scan order and the stable sort
    # breaks ties the way the pure-Python scan does.
    scored = [(matrix.names[int(index)], scores[i]) for i, index in enumerate(candidates)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def _near_ties(np: Any, scores: Any) -> list[int]:
    """Positions whose score is within ``_TIE_EPSILON`` of another score."""
    if scores.size < 2:
        return []
    order = np.argsort(scores, kind="stable")
    close = np.diff(scores[order]) < _TIE_EPSILON
    tied = np.zeros(scores.size, dtype=bool)
    tied[order[:-1]] |= close
    tied[order[1:]] |= close
    return [int(position) for position in np.flatnonzero(tied)]
