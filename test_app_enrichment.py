"""Tests for the B+A enrichment: tool-call recording + preflight/postflight hooks."""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_app_module():
    if "app" in sys.modules:
        del sys.modules["app"]
    yield
    if "app" in sys.modules:
        del sys.modules["app"]


def _import_app():
    mock_collection = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.return_value = mock_collection
    with patch("chromadb.PersistentClient", return_value=mock_client_instance):
        import app
    return app


def _ollama_generate_response(text="answer", model="llama3.1"):
    # OpenAI-compatible /v1/chat/completions shape: ollama_generate now routes
    # through chat_completion over that API for all providers (ollama/openai/anthropic).
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 15},
    }


def _make_mock_response(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TestGenerationEnrichment:
    @patch("requests.post")
    def test_tool_call_recorded_in_output(self, mock_post):
        mock_post.return_value = _make_mock_response(_ollama_generate_response())
        rec = MagicMock()
        rec.err.return_value = None
        client = MagicMock()

        @contextmanager
        def fake_start_gen(start):
            fake_start_gen.start = start
            yield rec

        client.start_generation.side_effect = fake_start_gen

        app = _import_app()
        app._sigil_client = client

        app.ollama_generate(
            "FULL PROMPT WITH CONTEXT",
            record_question="what is v1.3.0?",
            record_system_prompt=app.SYSTEM_PROMPT,
            conversation_id="conv-xyz",
            tools=app.AGENT_TOOLS,
            tool_calls=[{"name": "search_corpus", "id": "tc1", "input_json": b'{"query":"v1.3.0"}'}],
        )

        # GenerationStart carries the system prompt + conversation id separately
        start = fake_start_gen.start
        assert start.system_prompt == app.SYSTEM_PROMPT
        assert start.conversation_id == "conv-xyz"

        # set_result output contains a tool_call part named search_corpus
        kwargs = rec.set_result.call_args.kwargs
        output_msg = kwargs["output"][0]
        tool_parts = [p for p in output_msg.parts if getattr(p, "tool_call", None) is not None]
        assert len(tool_parts) == 1
        assert tool_parts[0].tool_call.name == "search_corpus"

        # the user message records the raw question, not the full prompt
        assert kwargs["input"][0].parts[0].text == "what is v1.3.0?"

    @patch("requests.post")
    def test_backward_compatible_without_enrichment(self, mock_post):
        mock_post.return_value = _make_mock_response(_ollama_generate_response("plain"))
        app = _import_app()
        app._sigil_client = None
        assert app.ollama_generate("hello") == "plain"


class TestHooks:
    def test_run_hook_disabled_returns_none(self):
        app = _import_app()
        app._sigil_client = MagicMock()
        app.HOOKS_ENABLED = False
        assert app._run_hook("preflight", conversation_id="c1", question="hi") is None

    def test_run_hook_sends_conversation_id_tag(self):
        app = _import_app()
        client = MagicMock()
        verdict = MagicMock()
        verdict.is_deny = False
        client.evaluate_hook.return_value = verdict
        app._sigil_client = client
        app.HOOKS_ENABLED = True

        out = app._run_hook("preflight", conversation_id="conv-1", question="hello")
        assert out is verdict
        req = client.evaluate_hook.call_args.args[0]
        assert req.phase == "preflight"
        assert req.context.tags["conversation_id"] == "conv-1"
        assert req.input.system_prompt == app.SYSTEM_PROMPT

    def test_run_hook_postflight_includes_output(self):
        app = _import_app()
        client = MagicMock()
        v = MagicMock(); v.is_deny = True
        client.evaluate_hook.return_value = v
        app._sigil_client = client
        app.HOOKS_ENABLED = True

        app._run_hook("postflight", conversation_id="c", question="q", output_text="the answer")
        req = client.evaluate_hook.call_args.args[0]
        assert req.input.output[0].parts[0].text == "the answer"

    def test_run_hook_fails_open_on_exception(self):
        app = _import_app()
        client = MagicMock()
        client.evaluate_hook.side_effect = RuntimeError("collector down")
        app._sigil_client = client
        app.HOOKS_ENABLED = True
        assert app._run_hook("preflight", conversation_id="c", question="q") is None
