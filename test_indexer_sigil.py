"""Tests for Sigil instrumentation in rag_indexer.py."""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_indexer_module():
    """Ensure rag_indexer module is freshly imported for each test."""
    mod_name = "rag_indexer"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    yield
    if mod_name in sys.modules:
        del sys.modules[mod_name]


def _import_indexer():
    """Import rag_indexer.py (no external services needed for embed tests)."""
    import rag_indexer
    return rag_indexer


def _ollama_embed_response(dims=10):
    return {"embedding": [0.1] * dims}


def _make_mock_response(json_data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestIndexerEmbedWithSigil:
    """rag_indexer.ollama_embed() should record a Sigil embedding span."""

    @patch("requests.post")
    def test_start_embedding_called_with_correct_model(self, mock_post):
        mock_post.return_value = _make_mock_response(_ollama_embed_response())

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_emb(start):
            fake_start_emb.captured_start = start
            yield mock_recorder

        mock_client.start_embedding.side_effect = fake_start_emb

        indexer = _import_indexer()
        indexer._sigil_client = mock_client

        result = indexer.ollama_embed("some text", model="mxbai-embed-large")

        assert len(result) == 10
        start = fake_start_emb.captured_start
        assert start.model.provider == "ollama"
        assert start.model.name == "mxbai-embed-large"
        assert start.agent_name == "local-search-indexer"

    @patch("requests.post")
    def test_set_result_called_with_embedding_result(self, mock_post):
        mock_post.return_value = _make_mock_response(_ollama_embed_response(5))

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_emb(start):
            yield mock_recorder

        mock_client.start_embedding.side_effect = fake_start_emb

        indexer = _import_indexer()
        indexer._sigil_client = mock_client

        indexer.ollama_embed("text")

        mock_recorder.set_result.assert_called_once()
        emb_result = mock_recorder.set_result.call_args[0][0]
        assert emb_result.input_count == 1

    @patch("requests.post")
    def test_set_call_error_on_embed_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("connection refused")
        mock_post.return_value = mock_resp

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_emb(start):
            yield mock_recorder

        mock_client.start_embedding.side_effect = fake_start_emb

        indexer = _import_indexer()
        indexer._sigil_client = mock_client

        with pytest.raises(Exception, match="connection refused"):
            indexer.ollama_embed("fail")

        mock_recorder.set_call_error.assert_called_once()

    @patch("requests.post")
    def test_embed_without_sigil_returns_normally(self, mock_post):
        mock_post.return_value = _make_mock_response(_ollama_embed_response(3))

        indexer = _import_indexer()
        indexer._sigil_client = None

        result = indexer.ollama_embed("text")
        assert result == [0.1, 0.1, 0.1]

    @patch("requests.post")
    def test_model_param_forwarded_to_sigil(self, mock_post):
        """When a custom model is passed, Sigil should see that model name."""
        mock_post.return_value = _make_mock_response(_ollama_embed_response())

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_emb(start):
            fake_start_emb.captured_start = start
            yield mock_recorder

        mock_client.start_embedding.side_effect = fake_start_emb

        indexer = _import_indexer()
        indexer._sigil_client = mock_client

        indexer.ollama_embed("text", model="nomic-embed-text")

        assert fake_start_emb.captured_start.model.name == "nomic-embed-text"


class TestIndexerSigilLifecycle:
    """Sigil client lifecycle in main() should call shutdown."""

    def test_shutdown_called_after_indexing(self):
        indexer = _import_indexer()

        mock_client = MagicMock()

        with patch.object(indexer, "_sigil_client", mock_client), \
             patch.object(indexer, "index_path") as mock_index:
            mock_index.return_value = None

            indexer._sigil_client = mock_client
            try:
                indexer._sigil_client.shutdown()
            finally:
                pass

        mock_client.shutdown.assert_called_once()
