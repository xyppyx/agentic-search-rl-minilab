"""Small local BM25 backend for reproducible offline search."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from search_r1_minilab.tools.base import SearchItem, SearchResult, SearchStats


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize English-ish text for a lightweight local baseline."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Document:
    """A normalized document with precomputed token statistics."""

    item: SearchItem
    tokens: list[str]
    term_counts: Counter[str]


@dataclass
class LocalBM25Backend:
    """Simple BM25 search over a JSONL corpus."""

    documents: list[BM25Document]
    name: str = "local_bm25"
    top_k: int = 3
    k1: float = 1.5
    b: float = 0.75
    stats: SearchStats = field(default_factory=SearchStats)
    _doc_freq: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _avg_doc_len: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build corpus statistics."""
        self._avg_doc_len = (
            sum(len(document.tokens) for document in self.documents)
            / max(len(self.documents), 1)
        )
        doc_freq: dict[str, int] = {}
        for document in self.documents:
            for token in set(document.tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self._doc_freq = doc_freq

    @classmethod
    def from_jsonl(
        cls,
        corpus_path: str | Path,
        name: str = "local_bm25",
        top_k: int = 3,
    ) -> "LocalBM25Backend":
        """Load a JSONL corpus with id, title, content, url, and source fields."""
        documents: list[BM25Document] = []
        path = Path(corpus_path)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                item = cls._item_from_record(record, line_number)
                tokens = tokenize(f"{item.title} {item.content}")
                documents.append(
                    BM25Document(item=item, tokens=tokens, term_counts=Counter(tokens))
                )
        return cls(documents=documents, name=name, top_k=top_k)

    @staticmethod
    def _item_from_record(record: dict[str, Any], line_number: int) -> SearchItem:
        """Validate and normalize one JSONL corpus record."""
        missing = [
            field_name
            for field_name in ("id", "title", "content", "url", "source")
            if field_name not in record
        ]
        if missing:
            raise ValueError(
                f"BM25 corpus line {line_number} missing fields: {', '.join(missing)}"
            )
        return SearchItem(
            id=str(record["id"]),
            title=str(record["title"]),
            content=str(record["content"]),
            url=str(record["url"]),
            source=str(record["source"]),
            metadata={
                key: value
                for key, value in record.items()
                if key not in {"id", "title", "content", "url", "source"}
            },
        )

    def search(self, query: str) -> SearchResult:
        """Rank the local corpus with BM25."""
        started = time.perf_counter()
        query_tokens = tokenize(query)
        if not query_tokens:
            result = SearchResult(
                ok=False,
                items=[],
                latency=time.perf_counter() - started,
                backend=self.name,
                error_type="empty_query",
                error="query must contain at least one alphanumeric token",
                metadata={"query": query},
            )
            self.stats.observe(result)
            return result

        scored_items: list[SearchItem] = []
        for document in self.documents:
            score = self._score(query_tokens, document)
            if score <= 0.0:
                continue
            scored_items.append(
                SearchItem(
                    id=document.item.id,
                    title=document.item.title,
                    content=document.item.content,
                    url=document.item.url,
                    source=document.item.source,
                    score=score,
                    metadata=document.item.metadata,
                )
            )
        scored_items.sort(key=lambda item: item.score or 0.0, reverse=True)
        result = SearchResult(
            ok=True,
            items=scored_items[: self.top_k],
            latency=time.perf_counter() - started,
            backend=self.name,
            metadata={"query": query, "query_tokens": query_tokens},
        )
        self.stats.observe(result)
        return result

    def _score(self, query_tokens: list[str], document: BM25Document) -> float:
        """Compute BM25 score for one document."""
        if not document.tokens:
            return 0.0
        score = 0.0
        corpus_size = len(self.documents)
        doc_len = len(document.tokens)
        for token in query_tokens:
            term_frequency = document.term_counts[token]
            if term_frequency == 0:
                continue
            document_frequency = self._doc_freq.get(token, 0)
            idf = math.log(
                1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * doc_len / max(self._avg_doc_len, 1e-9)
            )
            score += idf * (term_frequency * (self.k1 + 1.0)) / denominator
        return score

    def metrics(self) -> dict[str, float]:
        """Return cumulative BM25 backend metrics."""
        return self.stats.metrics(f"tool/{self.name}")
