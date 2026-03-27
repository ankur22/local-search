"""Tests for Sigil instrumentation in app.py."""

import json
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_app_module():
    """Ensure app module is freshly imported with mocked ChromaDB for each test."""
    mod_name = "app"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    yield
    if mod_name in sys.modules:
        del sys.modules[mod_name]


def _import_app():
    """Import app.py with ChromaDB mocked out (no real DB needed)."""
    mock_collection = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.return_value = mock_collection

    with patch("chromadb.PersistentClient", return_value=mock_client_instance):
        import app
    return app


def _ollama_generate_response(text="Hello world", model="llama3.1"):
    return {
        "model": model,
        "response": text,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 42,
        "eval_count": 15,
    }


def _ollama_embed_response(dims=10):
    return {"embedding": [0.1] * dims}


def _make_mock_response(json_data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestOllamaGenerateWithSigil:
    """ollama_generate() should record a Sigil generation span when client is active."""

    @patch("requests.post")
    def test_start_generation_called_with_correct_model(self, mock_post):
        mock_post.return_value = _make_mock_response(_ollama_generate_response())

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_gen(start):
            fake_start_gen.captured_start = start
            yield mock_recorder

        mock_client.start_generation.side_effect = fake_start_gen

        app = _import_app()
        app._sigil_client = mock_client

        app.ollama_generate("What is 2+2?", temperature=0.5)

        start = fake_start_gen.captured_start
        assert start.model.provider == "ollama"
        assert start.model.name == app.CHAT_MODEL
        assert start.agent_name == "local-search"

    @patch("requests.post")
    def test_set_result_called_with_messages(self, mock_post):
        mock_post.return_value = _make_mock_response(
            _ollama_generate_response("The answer is 4")
        )

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_gen(start):
            yield mock_recorder

        mock_client.start_generation.side_effect = fake_start_gen

        app = _import_app()
        app._sigil_client = mock_client

        result = app.ollama_generate("What is 2+2?")

        assert result == "The answer is 4"
        mock_recorder.set_result.assert_called_once()
        call_kwargs = mock_recorder.set_result.call_args
        assert call_kwargs.kwargs.get("input") is not None
        assert call_kwargs.kwargs.get("output") is not None

    @patch("requests.post")
    def test_set_result_includes_token_usage(self, mock_post):
        mock_post.return_value = _make_mock_response(
            _ollama_generate_response("answer")
        )

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_gen(start):
            yield mock_recorder

        mock_client.start_generation.side_effect = fake_start_gen

        app = _import_app()
        app._sigil_client = mock_client

        app.ollama_generate("question")

        call_kwargs = mock_recorder.set_result.call_args
        usage = call_kwargs.kwargs.get("usage")
        assert usage is not None
        assert usage.input_tokens == 42
        assert usage.output_tokens == 15

    @patch("requests.post")
    def test_set_result_includes_stop_reason(self, mock_post):
        mock_post.return_value = _make_mock_response(
            _ollama_generate_response("answer")
        )

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_gen(start):
            yield mock_recorder

        mock_client.start_generation.side_effect = fake_start_gen

        app = _import_app()
        app._sigil_client = mock_client

        app.ollama_generate("question")

        call_kwargs = mock_recorder.set_result.call_args
        assert call_kwargs.kwargs.get("stop_reason") == "stop"

    @patch("requests.post")
    def test_set_call_error_on_http_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
        mock_post.return_value = mock_resp

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_gen(start):
            yield mock_recorder

        mock_client.start_generation.side_effect = fake_start_gen

        app = _import_app()
        app._sigil_client = mock_client

        with pytest.raises(Exception, match="HTTP 500"):
            app.ollama_generate("fail prompt")

        mock_recorder.set_call_error.assert_called_once()

    @patch("requests.post")
    def test_generate_without_sigil_returns_normally(self, mock_post):
        mock_post.return_value = _make_mock_response(
            _ollama_generate_response("plain answer")
        )

        app = _import_app()
        app._sigil_client = None

        result = app.ollama_generate("hello")
        assert result == "plain answer"


class TestOllamaEmbedWithSigil:
    """ollama_embed() should record a Sigil embedding span when client is active."""

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

        app = _import_app()
        app._sigil_client = mock_client

        embedding = app.ollama_embed("some text")

        assert len(embedding) == 10
        start = fake_start_emb.captured_start
        assert start.model.provider == "ollama"
        assert start.model.name == app.EMBED_MODEL
        assert start.agent_name == "local-search"

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

        app = _import_app()
        app._sigil_client = mock_client

        app.ollama_embed("text")

        mock_recorder.set_result.assert_called_once()
        emb_result = mock_recorder.set_result.call_args[0][0]
        assert emb_result.input_count == 1

    @patch("requests.post")
    def test_set_call_error_on_embed_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("timeout")
        mock_post.return_value = mock_resp

        mock_recorder = MagicMock()
        mock_recorder.err.return_value = None
        mock_client = MagicMock()

        @contextmanager
        def fake_start_emb(start):
            yield mock_recorder

        mock_client.start_embedding.side_effect = fake_start_emb

        app = _import_app()
        app._sigil_client = mock_client

        with pytest.raises(Exception, match="timeout"):
            app.ollama_embed("fail")

        mock_recorder.set_call_error.assert_called_once()

    @patch("requests.post")
    def test_embed_without_sigil_returns_normally(self, mock_post):
        mock_post.return_value = _make_mock_response(_ollama_embed_response(3))

        app = _import_app()
        app._sigil_client = None

        result = app.ollama_embed("text")
        assert result == [0.1, 0.1, 0.1]
