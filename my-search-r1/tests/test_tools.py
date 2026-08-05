"""Tests for the first Search-R1 MiniLab tool backends."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from search_r1_minilab.tools import (
    FailureConfig,
    FailureWrapperBackend,
    LocalBM25Backend,
    MockSearchBackend,
    SearchItem,
    ToolRegistry,
)
from search_r1_minilab.tools.smoke import build_tool_smoke_records
from search_r1_minilab.tools.zhihu import (
    ZhihuSearchBackend,
    parse_zhihu_keys,
    zhihu_keys_from_env,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class MockSearchBackendTest(unittest.TestCase):
    def test_known_query_returns_fixture(self) -> None:
        backend = MockSearchBackend.from_pairs(
            {
                "little prince author": [
                    {
                        "id": "doc-1",
                        "title": "The Little Prince",
                        "content": "The author is Antoine de Saint-Exupery.",
                        "url": "https://example.test/little-prince",
                        "source": "mock",
                    }
                ]
            }
        )

        result = backend.search("  Little   Prince AUTHOR ")

        self.assertTrue(result.ok)
        self.assertEqual(result.backend, "mock_search")
        self.assertEqual(result.items[0].id, "doc-1")
        self.assertEqual(result.metadata["fixture_key"], "little prince author")

    def test_unknown_query_returns_empty_success(self) -> None:
        backend = MockSearchBackend.from_pairs({})

        result = backend.search("unknown")

        self.assertTrue(result.ok)
        self.assertEqual(result.items, [])
        self.assertTrue(result.metadata["empty"])


class LocalBM25BackendTest(unittest.TestCase):
    def test_relevant_document_ranks_first(self) -> None:
        backend = LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl", top_k=3)

        result = backend.search("novella french writer")

        self.assertTrue(result.ok)
        self.assertGreaterEqual(len(result.items), 1)
        self.assertEqual(result.items[0].id, "doc-2")
        self.assertGreater(result.items[0].score or 0.0, 0.0)

    def test_empty_query_is_distinguishable_failure(self) -> None:
        backend = LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl")

        result = backend.search("   ???   ")

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "empty_query")

    def test_no_hit_returns_empty_success(self) -> None:
        backend = LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl")

        result = backend.search("zzzzzzzz unmatchedtoken")

        self.assertTrue(result.ok)
        self.assertEqual(result.items, [])


class FailureWrapperBackendTest(unittest.TestCase):
    def test_seeded_failure_sequence_is_reproducible(self) -> None:
        base = MockSearchBackend.from_pairs(
            {"x": [SearchItem(title="X", content="base result", source="mock")]}
        )
        config = FailureConfig(
            p_timeout=0.25,
            p_rate_limited=0.25,
            p_empty=0.25,
            p_noise=0.25,
            seed=7,
        )
        first = FailureWrapperBackend(base, config, name="unstable_mock")
        second = FailureWrapperBackend(base, config, name="unstable_mock")

        first_types = [first.search("x").error_type for _ in range(20)]
        second_types = [second.search("x").error_type for _ in range(20)]

        self.assertEqual(first_types, second_types)
        self.assertIn("timeout", first_types)
        self.assertIn("empty_result", first_types)
        self.assertIn("noisy_result", first_types)

    def test_metrics_distinguish_failure_types(self) -> None:
        base = MockSearchBackend.from_pairs({})
        wrapper = FailureWrapperBackend(
            base,
            FailureConfig(p_timeout=1.0, seed=3),
            name="always_timeout",
        )

        wrapper.search("x")
        metrics = wrapper.metrics()

        self.assertEqual(metrics["tool/always_timeout/requests"], 1.0)
        self.assertEqual(metrics["tool/always_timeout/timeout_rate"], 1.0)


class RegistryTest(unittest.TestCase):
    def test_registry_dispatches_query_argument(self) -> None:
        registry = ToolRegistry()
        registry.register(
            MockSearchBackend.from_pairs(
                {"agentic rl": [SearchItem(title="Agentic RL", content="Search tools")]}
            )
        )

        result = registry.call("mock_search", {"query": "agentic rl"})

        self.assertTrue(result.ok)
        self.assertEqual(result.items[0].title, "Agentic RL")

    def test_registry_rejects_unknown_or_invalid_calls(self) -> None:
        registry = ToolRegistry()
        registry.register(MockSearchBackend.from_pairs({}))

        unknown = registry.call("missing", {"query": "x"})
        invalid = registry.call("mock_search", {"bad": "x"})

        self.assertEqual(unknown.error_type, "unknown_tool")
        self.assertEqual(invalid.error_type, "invalid_arguments")

    def test_mock_and_bm25_smoke_records_share_schema(self) -> None:
        mock_registry = ToolRegistry()
        mock_registry.register(
            MockSearchBackend.from_pairs(
                {"little prince": [SearchItem(title="Little Prince", content="fixture")]}
            )
        )
        bm25_registry = ToolRegistry()
        bm25_registry.register(LocalBM25Backend.from_jsonl(FIXTURES_DIR / "bm25_corpus.jsonl"))

        mock_record = build_tool_smoke_records(
            mock_registry, "mock_search", ["little prince"]
        )[0]
        bm25_record = build_tool_smoke_records(
            bm25_registry, "local_bm25", ["little prince"]
        )[0]

        self.assertEqual(set(mock_record.keys()), set(bm25_record.keys()))
        self.assertEqual(
            set(mock_record["turns"][1].keys()),
            set(bm25_record["turns"][1].keys()),
        )
        self.assertEqual(mock_record["search_calls"], 1)
        self.assertEqual(bm25_record["search_calls"], 1)


class ZhihuConfigTest(unittest.TestCase):
    def test_parse_zhihu_keys_trims_and_deduplicates(self) -> None:
        keys = parse_zhihu_keys(" key-a, key-b\nkey-a ,, ")

        self.assertEqual(keys, ["key-a", "key-b"])

    def test_zhihu_keys_from_env_includes_secondary_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {"ZHIHU_API_KEY": "primary", "ZHIHU_API_KEY2": "secondary"},
            clear=True,
        ):
            keys = zhihu_keys_from_env()

        self.assertEqual(keys, ["primary", "secondary"])

    def test_rate_limit_payload_rotates_to_secondary_key(self) -> None:
        backend = ZhihuSearchBackend(["primary-key", "secondary-key"])
        calls: list[str | None] = []

        class FakeResponse:
            def __init__(self, payload: dict, status: int = 200) -> None:
                self.payload = payload
                self.status = status

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout: float = 0.0) -> FakeResponse:
            calls.append(request.get_header("Authorization"))
            if len(calls) == 1:
                return FakeResponse(
                    {
                        "Code": 429,
                        "Message": "quota exceeded",
                        "Data": None,
                    }
                )
            return FakeResponse(
                {
                    "Data": {
                        "Items": [
                            {
                                "Title": "Result",
                                "ContentText": "secondary key worked",
                                "Url": "https://example.test",
                                "ContentType": "Zhihu",
                            }
                        ]
                    }
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = backend.search("test query")

        self.assertTrue(result.ok)
        self.assertEqual(result.items[0].title, "Result")
        self.assertEqual(calls, ["Bearer primary-key", "Bearer secondary-key"])

    def test_parse_error_records_sanitized_api_response(self) -> None:
        backend = ZhihuSearchBackend("super-secret-key")

        class FakeResponse:
            status = 200

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"Code": 0, "Data": None}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = backend.search("schema mismatch")

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "parse_error")
        self.assertEqual(result.error, "TypeError")
        self.assertEqual(result.status, 200)
        self.assertEqual(result.metadata["api_status"], 200)
        self.assertIn('"Data": null', result.metadata["api_response_excerpt"])
        self.assertFalse(result.metadata["api_response_truncated"])
        self.assertNotIn("super-secret-key", result.to_dict().__repr__())

    def test_search_errors_do_not_leak_secret(self) -> None:
        backend = ZhihuSearchBackend("super-secret-key")
        result = backend._error_result(
            started=0.0,
            error_type="http_error",
            message="HTTP 500",
            query="test query",
            status=500,
        )

        self.assertEqual(result.error, "HTTP 500")
        self.assertNotIn("super-secret-key", result.to_dict().__repr__())


if __name__ == "__main__":
    unittest.main()
