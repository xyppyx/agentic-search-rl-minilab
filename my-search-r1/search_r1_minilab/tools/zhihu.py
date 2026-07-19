"""Zhihu global search backend adapted to the shared tool interface."""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from search_r1_minilab.tools.base import SearchItem, SearchResult, SearchStats


SEARCH_ENDPOINT = "https://developer.zhihu.com/api/v1/content/global_search"


def parse_zhihu_keys(raw_keys: str | None) -> list[str]:
    """Parse comma/newline separated keys with stable de-duplication."""
    if not raw_keys:
        return []
    return list(dict.fromkeys(item.strip() for item in re.split(r"[,\n]", raw_keys) if item.strip()))


def zhihu_keys_from_env() -> list[str]:
    """Read supported Zhihu key environment variables."""
    return parse_zhihu_keys(
        os.getenv("ZHIHU_SEARCH_KEYS")
        or os.getenv("ZHIHU_SEARCH_KEY")
        or os.getenv("ZHIHU_ACCESS_SECRET")
        or os.getenv("ZHIHU_API_KEY")
    )


@dataclass
class ZhihuSearchBackend:
    """Search Zhihu with key rotation and bounded retry."""

    access_secrets: str | list[str]
    name: str = "zhihu_search"
    timeout: float = 15.0
    max_retries: int = 2
    retry_delay: float = 1.0
    top_k: int = 3
    stats: SearchStats = field(default_factory=SearchStats)
    _next_secret_index: int = field(default=0, init=False, repr=False)
    _rate_limited_secret_indices: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize and validate credentials."""
        if isinstance(self.access_secrets, str):
            secrets = parse_zhihu_keys(self.access_secrets)
        else:
            secrets = list(
                dict.fromkeys(secret.strip() for secret in self.access_secrets if secret.strip())
            )
        if not secrets:
            raise ValueError("at least one Zhihu search key is required")
        self.access_secrets = secrets

    @classmethod
    def from_env(
        cls,
        env_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "ZhihuSearchBackend":
        """Create a backend from environment variables or a local .env file."""
        if env_path:
            load_dotenv(env_path)
        secrets = zhihu_keys_from_env()
        if not secrets:
            raise ValueError(
                "set ZHIHU_SEARCH_KEYS, ZHIHU_SEARCH_KEY, ZHIHU_ACCESS_SECRET, or ZHIHU_API_KEY"
            )
        return cls(access_secrets=secrets, **kwargs)

    def search(self, query: str) -> SearchResult:
        """Search one query with retry and credential failover."""
        started = time.perf_counter()
        credential = self._next_credential()
        if credential is None:
            result = self._error_result(
                started,
                error_type="rate_limited",
                message="all search keys are rate limited",
                status=429,
                query=query,
            )
            self.stats.observe(result)
            return result

        secret_index, access_secret = credential
        attempt = 0
        while True:
            try:
                result = self._request(query, started, access_secret)
                self.stats.observe(result)
                return result
            except urllib.error.HTTPError as error:
                if error.code == 429:
                    self._rate_limited_secret_indices.add(secret_index)
                    credential = self._next_credential()
                    if credential is None:
                        result = self._error_result(
                            started,
                            error_type="rate_limited",
                            message="all search keys are rate limited",
                            status=error.code,
                            query=query,
                        )
                        self.stats.observe(result)
                        return result
                    secret_index, access_secret = credential
                    continue
                if error.code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                result = self._error_result(
                    started,
                    error_type="http_error",
                    message=f"HTTP {error.code}",
                    status=error.code,
                    query=query,
                )
                self.stats.observe(result)
                return result
            except (TimeoutError, socket.timeout):
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                result = self._error_result(
                    started,
                    error_type="timeout",
                    message="request timeout",
                    query=query,
                )
                self.stats.observe(result)
                return result
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    result = self._error_result(
                        started,
                        error_type="timeout",
                        message="request timeout",
                        query=query,
                    )
                    self.stats.observe(result)
                    return result
                result = self._error_result(
                    started,
                    error_type="url_error",
                    message=type(error).__name__,
                    query=query,
                )
                self.stats.observe(result)
                return result
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                result = self._error_result(
                    started,
                    error_type="parse_error",
                    message=type(error).__name__,
                    query=query,
                )
                self.stats.observe(result)
                return result

    def _next_credential(self) -> tuple[int, str] | None:
        """Return the next credential that has not hit rate limits."""
        secrets = self.access_secrets if isinstance(self.access_secrets, list) else []
        for _ in range(len(secrets)):
            index = self._next_secret_index
            self._next_secret_index = (self._next_secret_index + 1) % len(secrets)
            if index not in self._rate_limited_secret_indices:
                return index, secrets[index]
        return None

    def _request(self, query: str, started: float, access_secret: str) -> SearchResult:
        """Perform one HTTP request and normalize the response."""
        params = urllib.parse.urlencode(
            {"Query": query, "Count": self.top_k, "SearchDB": "all"}
        )
        request = urllib.request.Request(
            f"{SEARCH_ENDPOINT}?{params}",
            headers={
                "Authorization": f"Bearer {access_secret}",
                "X-Request-Timestamp": str(int(time.time())),
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            items = [self._parse_item(item) for item in payload["Data"]["Items"]]
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                backend=self.name,
                status=response.status,
                metadata={"query": query},
            )

    def _parse_item(self, item: dict[str, Any]) -> SearchItem:
        """Keep the result fields needed by rollout and reports."""
        source_parts = [str(item.get("ContentType") or "Zhihu")]
        if item.get("AuthorName"):
            source_parts.append(str(item["AuthorName"]))
        return SearchItem(
            title=str(item.get("Title") or "Untitled").strip(),
            content=str(item.get("ContentText") or "").strip()[:1200],
            url=str(item.get("Url") or "").strip(),
            source=" / ".join(source_parts),
        )

    def _error_result(
        self,
        started: float,
        error_type: str,
        message: str,
        query: str,
        status: int | None = None,
    ) -> SearchResult:
        """Convert an exception to a sanitized search result."""
        return SearchResult(
            ok=False,
            items=[],
            latency=time.perf_counter() - started,
            backend=self.name,
            status=status,
            error_type=error_type,
            error=message,
            metadata={"query": query},
        )

    def metrics(self) -> dict[str, float]:
        """Return cumulative Zhihu backend metrics."""
        return self.stats.metrics(f"tool/{self.name}")
