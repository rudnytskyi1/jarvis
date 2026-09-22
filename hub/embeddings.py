"""Text embeddings on the CPU (ТЗ F-414: multilingual-e5-small or bge-m3).

The hub's memory search needs vectors, and the ТЗ names two models that fit a
CPU: ``intfloat/multilingual-e5-small`` and ``BAAI/bge-m3``. Both are
multilingual, so a Russian fact is found by an English question.

Nothing is downloaded behind the operator's back. The model is looked up as a
LOCAL directory (``models/multilingual-e5-small`` by default, or whatever
``ROWAN_EMBEDDING_MODEL`` / ``server.memory.embedding_model`` say); a hub whose
model is not on the machine refuses honestly - :class:`EmbedderUnavailable` -
and the caller falls back to search by words (``hub/memory_search.py``). A
refusal is logged once per model, not once per utterance: the answer does not
change between turns, and a warning storm helps nobody.

Two backends, because a hub may have either: ``sentence-transformers`` when it
is installed (the library these models are published for), otherwise plain
``transformers`` + ``torch`` with mean pooling and L2 normalisation, which is
what sentence-transformers does under the hood for these two models. Both run
on the CPU by design: the 5090 belongs to speech and the language model.
"""
from __future__ import annotations

import importlib.util
import logging
import math
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("jarvis.server.embeddings")

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The models ТЗ 9.4 names, cheapest first: a dorm hub has one CPU budget, and
#: e5-small is 384 dimensions against bge-m3's 1024.
PREFERRED_MODELS = ("multilingual-e5-small", "bge-m3")
#: Where an operator drops the model so the hub never needs the network.
DEFAULT_MODEL = "models/multilingual-e5-small"
ENV_VAR = "ROWAN_EMBEDDING_MODEL"
#: Files that make a directory "a model" and not just "a directory".
MODEL_MARKERS = ("config.json", "modules.json", "model.safetensors", "pytorch_model.bin")
#: One batch of passages at a time - CPU memory, not throughput, is the limit.
MAX_BATCH = 16
#: e5 wants its prompt prefixes; bge does not, and adding them would hurt.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class EmbedderUnavailable(RuntimeError):
    """There is no usable embedding model on this machine."""


class TextEmbedder(Protocol):
    """What the hub needs from an embedder: a size and a vector per text."""

    name: str
    dimension: int

    def encode(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        """One L2-normalised vector per text, in the order given."""


def wanted_model(model: Any = "") -> str:
    """The model this call is about: the argument, the env var, or the default."""
    return str(model or "").strip() or os.environ.get(ENV_VAR, "").strip() or DEFAULT_MODEL


def local_path(model: Any = "") -> Path | None:
    """The directory holding the model, or ``None`` when it is not here.

    An absolute path, a path relative to the repo, a ``models/...`` name and an
    HF id that happens to exist as a directory all resolve the same way; a name
    with nothing behind it returns ``None`` rather than pretending.
    """
    wanted = wanted_model(model)
    if not wanted:
        return None
    candidate = Path(wanted).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.is_dir():
        return None
    if not any((candidate / marker).is_file() for marker in MODEL_MARKERS):
        return None
    return candidate


def is_e5(model: Any = "") -> bool:
    """True for the e5 family, which is trained with ``query:``/``passage:``."""
    return "e5" in wanted_model(model).casefold()


def prompt_prefix(model: Any = "", *, query: bool = False) -> str:
    """The text prefix this model expects, or ``""`` when it wants none."""
    if not is_e5(model):
        return ""
    return QUERY_PREFIX if query else PASSAGE_PREFIX


def _import(name: str) -> Any:
    """Import an optional heavy dependency, or say honestly that it is absent."""
    if importlib.util.find_spec(name) is None:
        raise EmbedderUnavailable(f"{name} is not installed")
    try:
        return __import__(name)
    except Exception as exc:  # noqa: BLE001 - a broken install is a refusal too
        raise EmbedderUnavailable(f"{name} could not be imported ({exc})") from exc


def mean_pool(rows: Sequence[Sequence[float]], mask: Sequence[float]) -> list[float]:
    """Average the token vectors of one text and normalise the result.

    This is the pooling sentence-transformers applies to these encoder models;
    padding positions are excluded by ``mask``. The result is L2-normalised, so
    a dot product of two of them is their cosine similarity.
    """
    if not rows:
        raise ValueError("mean_pool needs at least one token vector")
    width = len(rows[0])
    if width == 0:
        raise ValueError("mean_pool needs vectors with at least one dimension")
    weights = [float(value) for value in mask][:len(rows)]
    total = sum(weights)
    if total <= 0:
        raise ValueError("mean_pool needs at least one unmasked position")
    pooled = [0.0] * width
    for row, weight in zip(rows, weights):
        if weight == 0:
            continue
        if len(row) != width:
            raise ValueError("mean_pool needs vectors of one width")
        for index, value in enumerate(row):
            pooled[index] += float(value) * weight
    pooled = [value / total for value in pooled]
    norm = math.sqrt(sum(value * value for value in pooled))
    if norm <= 0:
        return pooled
    return [value / norm for value in pooled]


class _SentenceTransformerEmbedder:
    """The published library for these models, if it is installed."""

    def __init__(self, target: str, *, name: str) -> None:
        module = _import("sentence_transformers")
        self.name = name
        self._target = target
        try:
            self._model = module.SentenceTransformer(target, device="cpu")
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:  # noqa: BLE001 - loading is the honest failure
            raise EmbedderUnavailable(f"cannot load {name!r} from {target}: {exc}") from exc

    def encode(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        prefix = prompt_prefix(self.name, query=query)
        values = [f"{prefix}{text}" for text in texts]
        if not values:
            return []
        rows = self._model.encode(values, normalize_embeddings=True,
                                  batch_size=MAX_BATCH, convert_to_numpy=True)
        return [[float(value) for value in row] for row in rows]


class _TransformersEmbedder:
    """Plain ``transformers`` + ``torch``: mean pooling over the last layer."""

    def __init__(self, target: str, *, name: str) -> None:
        torch = _import("torch")
        transformers = _import("transformers")
        self.name = name
        self._target = target
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(target)
            self._model = transformers.AutoModel.from_pretrained(target)
            self._model.eval()
            self.dimension = int(self._model.config.hidden_size)
        except Exception as exc:  # noqa: BLE001 - loading is the honest failure
            raise EmbedderUnavailable(f"cannot load {name!r} from {target}: {exc}") from exc
        self._torch = torch

    def encode(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        prefix = prompt_prefix(self.name, query=query)
        values = [f"{prefix}{text}" for text in texts]
        if not values:
            return []
        result: list[list[float]] = []
        for start in range(0, len(values), MAX_BATCH):
            result.extend(self._encode_batch(values[start:start + MAX_BATCH]))
        return result

    def _encode_batch(self, values: Sequence[str]) -> list[list[float]]:
        torch = self._torch
        batch = self._tokenizer(list(values), padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
        mask = batch["attention_mask"].tolist()
        return [mean_pool(rows, weights)
                for rows, weights in zip(hidden.tolist(), mask)]


def load(model: Any = "", *, allow_download: bool = False) -> TextEmbedder:
    """Load the embedder for this model, or raise :class:`EmbedderUnavailable`.

    ``allow_download`` is the only way an HF id reaches the network; without it
    the model has to be a local directory, because a dorm hub that suddenly
    downloads half a gigabyte during an utterance is not a working hub.
    """
    name = wanted_model(model)
    here = local_path(name)
    if here is None and not allow_download:
        raise EmbedderUnavailable(
            f"the embedding model {name!r} is not on this machine: {REPO_ROOT / name} "
            f"holds no model, and downloading is off. Put the model there, set "
            f"{ENV_VAR}, or set server.memory.allow_download to true"
        )
    target = str(here) if here is not None else name
    failures: list[str] = []
    for build in (_SentenceTransformerEmbedder, _TransformersEmbedder):
        try:
            embedder = build(target, name=name)
        except EmbedderUnavailable as exc:
            failures.append(str(exc))
            continue
        log.info("Text embeddings: %s (%d dimensions) from %s", name, embedder.dimension, target)
        return embedder
    raise EmbedderUnavailable(
        f"no backend could load {name!r} from {target}: " + "; ".join(failures))


_cache: dict[str, TextEmbedder | None] = {}
_cache_lock = threading.Lock()


def cached(model: Any = "", *, allow_download: bool = False) -> TextEmbedder | None:
    """The embedder, or ``None`` when it cannot be had - logged once per model.

    The hub asks for the embedder on every turn that wants a vector; the answer
    does not change between turns, so neither does the message.
    """
    name = wanted_model(model)
    key = f"{name}|{int(bool(allow_download))}"
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    try:
        embedder = load(name, allow_download=allow_download)
    except EmbedderUnavailable as exc:
        log.warning("Text embeddings are off, memory is searched by words: %s", exc)
        embedder = None
    except Exception:  # noqa: BLE001 - a broken model is a refusal, not a crash
        log.exception("The embedding model %r could not be loaded", name)
        embedder = None
    with _cache_lock:
        _cache[key] = embedder
    return embedder


def forget() -> None:
    """Drop the cache (tests, and a configuration reload that changes the model)."""
    with _cache_lock:
        _cache.clear()


__all__ = [
    "DEFAULT_MODEL",
    "ENV_VAR",
    "PREFERRED_MODELS",
    "EmbedderUnavailable",
    "TextEmbedder",
    "cached",
    "forget",
    "is_e5",
    "load",
    "local_path",
    "mean_pool",
    "prompt_prefix",
    "wanted_model",
]
