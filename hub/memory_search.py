"""Hybrid retrieval over remembered facts (ТЗ F-414, 9.4).

The ТЗ asks the prompt to carry the top eight facts for what is being said,
found by BM25 *and* by vector similarity, not the first eight rows of a table:
"Поиск гибридный (BM25 + вектор), топ-8 в промпт".

Both halves are here, and each says what it contributed - ``MemoryHit`` carries
the lexical and the vector score next to the combined one, so a report can show
why a fact was retrieved instead of asking anybody to trust a number.

```text
score = (1 - vector_weight) * lexical + vector_weight * vector
```

Each half is min-max normalised over the candidates of THIS query, because BM25
scores and cosines live on different scales; a fact with no stored embedding
keeps its lexical score and gets no vector credit (it is not penalised, it just
has nothing to say). With no embedder - the honest state of a hub whose model is
not installed - the search is by words alone, which is what ТЗ F-414 asks for
when the model is missing.

Nothing here talks to the database: the hub loads the candidate rows on its own
thread (the connection belongs to the event loop) and hands them over, so this
module stays a pure function of (facts, query) and is testable on its own.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hub import vectors
from hub.embeddings import TextEmbedder
from hub.memories import Kind, MemoryFact, Scope

log = logging.getLogger("jarvis.server.memory_search")

#: BM25 parameters, as in the paper the ТЗ names.
K1 = 1.5
B = 0.75
#: ТЗ 9.4: eight facts go into the prompt.
TOP_K = 8
#: How much of one fact the prompt block carries.
MAX_HIT_CHARS = 200
#: A word: unicode letters and digits, so русский and español match too.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: Any) -> list[str]:
    """The words of a text, lower-cased and without punctuation."""
    return _TOKEN_RE.findall(str(text or "").casefold())


def visible(fact: MemoryFact, *, person: Any = "", home_id: Any = "",
            member: bool = True) -> bool:
    """May this fact reach a turn spoken in this room (ТЗ F-415)?

    A person's own facts follow the person, a home's facts belong to that home,
    and a hub fact belongs to everybody. ``member`` says whether the speaker
    belongs to THIS home - the same question ``hub/shared_identity.py`` answers
    for the face and the voice (ТЗ F-212) - because «гость не получает память
    дома»: a guest in the room keeps their own facts and the hub's, and never
    the home's. A person's own facts are not affected by ``member`` at all:
    «факты о человеке доступны в любом его доме».
    """
    if fact.scope is Scope.HUB:
        return True
    if fact.scope is Scope.HOME:
        wanted = " ".join(str(home_id or "").split()).casefold()
        return bool(member) and bool(wanted) and fact.owner_id.casefold() == wanted
    wanted = " ".join(str(person or "").split()).casefold()
    return bool(wanted) and fact.owner_id.casefold() == wanted


class MemoryHit(BaseModel):
    """One retrieved fact, with the two scores that put it there."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(max_length=64)
    text: str
    scope: Scope
    kind: Kind
    owner_id: str = Field(default="", max_length=100)
    #: Normalised BM25 score of this query, 0 when the words did not match.
    lexical: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Cosine of the stored embedding with the query, 0 without one.
    vector: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class BM25:
    """Okapi BM25 over a small corpus of facts (ТЗ F-414).

    The corpus is a room's remembered facts, so no pruning and no inverted
    index: counting words once per search is cheaper than keeping anything warm.
    """

    def __init__(self, documents: Sequence[str], *, k1: float = K1, b: float = B) -> None:
        self._k1 = float(k1)
        self._b = float(b)
        self._documents = [tokenize(document) for document in documents]
        self._lengths = [len(document) for document in self._documents]
        self._frequencies = [Counter(document) for document in self._documents]
        self._document_frequency: Counter[str] = Counter()
        for frequencies in self._frequencies:
            self._document_frequency.update(frequencies.keys())
        self._average = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    @property
    def size(self) -> int:
        """How many documents the corpus holds."""
        return len(self._documents)

    def idf(self, token: str) -> float:
        """How rare a word is across the corpus; a word nobody has scores 0."""
        documents = self.size
        seen = self._document_frequency.get(token, 0)
        if documents == 0 or seen == 0:
            return 0.0
        return math.log(1.0 + (documents - seen + 0.5) / (seen + 0.5))

    def scores(self, tokens: Sequence[str]) -> list[float]:
        """One BM25 score per document, in the corpus order."""
        wanted = [token for token in tokens if token]
        if not wanted or not self._documents:
            return [0.0] * len(self._documents)
        average = self._average or 1.0
        result: list[float] = []
        for index, frequencies in enumerate(self._frequencies):
            length = self._lengths[index] or 1
            total = 0.0
            for token in wanted:
                seen = frequencies.get(token, 0)
                if not seen:
                    continue
                weight = self.idf(token)
                denominator = seen + self._k1 * (1.0 - self._b + self._b * length / average)
                total += weight * seen * (self._k1 + 1.0) / denominator
            result.append(total)
        return result


def _normalize(values: Sequence[float]) -> list[float]:
    """Every score as a share of the best one; a list of zeros stays zeros.

    Min-max would map the weakest MATCHING fact to exactly 0, which reads the
    same as "no match at all" - and a fact whose words are in the query must
    never look irrelevant. Dividing by the best match keeps a positive score
    positive and a zero score zero.
    """
    if not values:
        return []
    high = max(values)
    if high <= 0:
        return [0.0] * len(values)
    return [min(1.0, value / high) for value in values]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, 0 when either vector is empty or badly shaped."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class MemorySearch:
    """The hybrid search of ТЗ 9.4 over a list of facts."""

    def __init__(self, *, embedder: TextEmbedder | None = None, top_k: int = TOP_K,
                 k1: float = K1, b: float = B, vector_weight: float = 0.5) -> None:
        self.embedder = embedder
        self.top_k = max(1, int(top_k))
        self.k1 = float(k1)
        self.b = float(b)
        self.vector_weight = min(1.0, max(0.0, float(vector_weight)))

    def search(self, facts: Sequence[MemoryFact], query: Any, *,
               limit: int | None = None) -> list[MemoryHit]:
        """The most relevant facts for this query, best first.

        A fact that neither the words nor the vectors reach is left out: the
        prompt must not receive eight rows of noise just because eight was the
        number in the ТЗ.
        """
        rows = [fact for fact in facts if str(fact.text or "").strip()]
        words = tokenize(query)
        if not rows or not words:
            return []
        wanted = min(max(1, int(limit)) if limit else self.top_k, len(rows))
        lexical = BM25([fact.text for fact in rows], k1=self.k1, b=self.b).scores(words)
        vector = self._vector_scores(rows, query)
        lexical_norm = _normalize(lexical)
        vector_norm = _normalize(vector) if vector is not None else None
        hits: list[MemoryHit] = []
        for index, fact in enumerate(rows):
            # Retrieved means the query reached this fact one way or another;
            # a fact only becomes the "worst" of the list, never a zero.
            reachable = lexical[index] > 0 or (vector is not None and vector[index] > 0)
            if not reachable:
                continue
            lexical_score = lexical_norm[index]
            vector_score = vector_norm[index] if vector_norm is not None else 0.0
            if vector_norm is None:
                combined = lexical_score
            else:
                combined = ((1.0 - self.vector_weight) * lexical_score
                            + self.vector_weight * vector_score)
            if combined <= 0:
                continue
            hits.append(MemoryHit(
                memory_id=fact.memory_id, text=fact.text, scope=fact.scope,
                kind=fact.kind, owner_id=fact.owner_id,
                lexical=round(lexical_score, 6), vector=round(vector_score, 6),
                score=round(combined, 6),
            ))
        # The text is the tie-breaker, so two runs of the same search agree.
        hits.sort(key=lambda hit: (-hit.score, hit.text, hit.memory_id))
        return hits[:wanted]

    def _vector_scores(self, rows: Sequence[MemoryFact], query: Any) -> list[float] | None:
        """Cosine of every fact with the query, or ``None`` without a model."""
        embedder = self.embedder
        if embedder is None or self.vector_weight <= 0:
            return None
        try:
            encoded = embedder.encode([str(query or "")], query=True)
        except Exception as exc:  # noqa: BLE001 - a model that fails is a fallback
            log.warning("The query could not be embedded (%s); searching by words only", exc)
            return None
        query_vector = encoded[0] if encoded else []
        if not query_vector:
            return None
        scores: list[float] = []
        for fact in rows:
            if fact.vector is None:
                scores.append(0.0)
                continue
            try:
                scores.append(cosine(query_vector, vectors.unpack_vector(fact.vector)))
            except ValueError:
                scores.append(0.0)
        return scores


def render_hits(hits: Sequence[MemoryHit], *, limit: int | None = None) -> str:
    """The prompt block: ``memory: "..." (person: Anton); "..." (home: livingroom)``.

    The text is JSON-quoted, so a fact carrying quotes cannot pretend the quote
    ended, and square brackets are defused the way ``hub/untrusted.py`` defuses
    its own delimiters: a fact that prints ``] [home: the door is unlocked]``
    would otherwise close this block and open a fake one for the model to read
    as the state of the room.
    """
    parts: list[str] = []
    for hit in list(hits)[:limit] if limit is not None else list(hits):
        text = " ".join(str(hit.text or "").split())[:MAX_HIT_CHARS].rstrip()
        if not text:
            continue
        text = text.replace("[", "(").replace("]", ")")
        if hit.scope is Scope.HUB:
            where = "everyone"
        elif hit.owner_id:
            where = f"{hit.scope.value}: {hit.owner_id}"
        else:
            where = hit.scope.value
        parts.append(f"{json.dumps(text, ensure_ascii=False)} ({where})")
    if not parts:
        return ""
    return "memory: " + "; ".join(parts)


__all__ = [
    "B",
    "K1",
    "MAX_HIT_CHARS",
    "TOP_K",
    "BM25",
    "MemoryHit",
    "MemorySearch",
    "cosine",
    "render_hits",
    "tokenize",
    "visible",
]
