"""What the provider puts on the wire and how it reads Gemini's answers, beyond what the kit checks."""

import httpx
import pytest
from fake_server import FakeGeminiServer
from marvin_integration_sdk.ai import CompletionOptions, ImagePart, Message, ToolCall, ToolDefinition

from marvin_ai_google import GoogleProvider
from marvin_ai_google.provider import gemini_contents

LOOKUP = ToolDefinition(name="lookup_entry", description="Find an entry.", input_schema={"type": "object", "properties": {"q": {"type": "string"}}})


@pytest.fixture
def server():
    return FakeGeminiServer()


@pytest.fixture
def provider(server):
    return GoogleProvider(api_key="test-key", http_client=server.client())


def test_system_messages_become_the_system_instruction(provider, server):
    server.reply_text("ok")
    provider.complete(
        [Message("system", "Be brief."), Message("user", "hi"), Message("assistant", "hello"), Message("user", "again")], "gemini-2.5-flash"
    )
    body = server.last_json()
    assert body["systemInstruction"]["parts"] == [{"text": "Be brief."}]
    assert [(c["role"], c["parts"][0]["text"]) for c in body["contents"]] == [("user", "hi"), ("model", "hello"), ("user", "again")]
    assert server.requests[-1].url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert server.requests[-1].headers["x-goog-api-key"] == "test-key"


def test_a_lone_system_message_is_sent_as_the_prompt():
    assert gemini_contents([Message("system", "Say hi.")]) == (None, [{"role": "user", "parts": [{"text": "Say hi."}]}])


def test_only_the_options_that_are_set_are_sent(provider, server):
    server.reply_text("ok")
    provider.complete([Message("user", "hi")], "gemini-2.5-flash")
    assert not server.last_json().get("generationConfig")
    server.reply_text("ok")
    provider.complete([Message("user", "hi")], "gemini-2.5-flash", CompletionOptions(max_tokens=64, temperature=0.2, top_p=0.9))
    assert server.last_json()["generationConfig"] == {"maxOutputTokens": 64, "temperature": 0.2, "topP": 0.9}


def test_an_image_is_sent_as_inline_data(provider, server):
    server.reply_text("ok")
    provider.complete([Message("user", ["What is this?", ImagePart(data="AAEC", mime_type="image/png")])], "gemini-2.5-flash")
    assert server.last_json()["contents"][0]["parts"] == [{"text": "What is this?"}, {"inlineData": {"mimeType": "image/png", "data": "AAEC"}}]


def test_thinking_tokens_count_as_output_and_thoughts_are_not_the_answer(provider, server):
    server.reply_text("x")
    original = server._handle

    def answer(request):
        response = original(request)
        body = response.json()
        body["candidates"][0]["content"]["parts"].insert(0, {"text": "Let me think.", "thought": True})
        body["candidates"][0]["finishReason"] = "MAX_TOKENS"
        body["usageMetadata"]["thoughtsTokenCount"] = 100
        return httpx.Response(200, json=body)

    provider = GoogleProvider(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(answer)))
    result = provider.complete([Message("user", "hi")], "gemini-2.5-flash")
    assert result.content == "x"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (11, 107, 118)
    assert result.stop_reason == "length"


def test_structured_output_asks_for_json_in_the_schema(provider, server):
    schema = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    server.reply_text('{"tags": []}')
    provider.complete_structured([Message("user", "Tag this.")], "gemini-2.5-flash", schema)
    config = server.last_json()["generationConfig"]
    assert config["responseMimeType"] == "application/json" and config["responseJsonSchema"] == schema


def test_tools_are_declared_and_the_sdk_does_not_run_them(provider, server):
    server.reply_tool_calls([ToolCall(id="call_1", name="lookup_entry", arguments={"q": "towel"})])
    result = provider.complete_with_tools([Message("user", "Find it.")], "gemini-2.5-flash", [LOOKUP], tool_choice="required")
    body = server.last_json()
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "lookup_entry"
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert len(server.requests) == 1
    assert result.stop_reason == "tool_calls" and result.tool_calls[0].arguments == {"q": "towel"}


def test_a_call_without_an_id_gets_one(provider, server):
    server.reply_tool_calls([ToolCall(id="", name="lookup_entry", arguments={})])
    (call,) = provider.complete_with_tools([Message("user", "Find it.")], "gemini-2.5-flash", [LOOKUP]).tool_calls
    assert call.id


def test_tool_results_are_grouped_into_one_turn_and_named_after_their_calls():
    calls = [ToolCall(id="a", name="lookup_entry", arguments={"q": "x"}), ToolCall(id="b", name="other", arguments={})]
    _, contents = gemini_contents(
        [
            Message("user", "go"),
            Message("assistant", "", tool_calls=calls),
            Message("tool", '{"id": 1}', tool_call_id="a"),
            Message("tool", "plain text", tool_call_id="b"),
        ]
    )
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert [p["function_call"]["name"] for p in contents[1]["parts"]] == ["lookup_entry", "other"]
    assert [p["function_response"] for p in contents[2]["parts"]] == [
        {"id": "a", "name": "lookup_entry", "response": {"id": 1}},
        {"id": "b", "name": "other", "response": {"output": "plain text"}},
    ]


def test_a_thought_signature_goes_back_with_its_call(server):
    def answer(request):
        body = server._handle(request).json()
        body["candidates"][0]["content"]["parts"][0]["thoughtSignature"] = "c2lnbmVk"
        return httpx.Response(200, json=body)

    provider = GoogleProvider(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(answer)))
    server.reply_tool_calls([ToolCall(id="call_sig", name="lookup_entry", arguments={})])
    (call,) = provider.complete_with_tools([Message("user", "Find it.")], "gemini-3.5-flash", [LOOKUP]).tool_calls
    server.reply_text("done")
    provider.complete_with_tools(
        [Message("user", "Find it."), Message("assistant", "", tool_calls=[call]), Message("tool", "{}", tool_call_id=call.id)],
        "gemini-3.5-flash",
        [LOOKUP],
    )
    assert server.last_json()["contents"][1]["parts"][0]["thoughtSignature"] == "c2lnbmVk"


def test_a_blocked_prompt_is_an_error(provider, server):
    original = server._handle

    def blocked(request):
        original(request)
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 3}})

    provider = GoogleProvider(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(blocked)))
    with pytest.raises(ValueError, match="SAFETY"):
        provider.complete([Message("user", "hi")], "gemini-2.5-flash")


def test_embeddings_go_in_batches_of_a_hundred(provider, server):
    server.reply_embeddings([[1.0]] * 100)
    server.reply_embeddings([[2.0]] * 5)
    vectors = provider.embed([f"t{i}" for i in range(105)], "gemini-embedding-001")
    assert len(vectors) == 105 and vectors[0] == [1.0] and vectors[-1] == [2.0]
    assert [len(r.content) > 0 for r in server.requests] == [True, True]
    assert len(server.last_json()["requests"]) == 5 and server.last_json()["requests"][0]["model"] == "models/gemini-embedding-001"


def test_list_models_offers_chat_models_only(provider, server):
    server.reply_models(["gemini-2.5-flash", "gemini-2.5-flash-image", "gemini-embedding-001", "gemini-2.5-flash-preview-tts", "gemini-2.5-pro"])
    assert provider.list_models() == ["gemini-2.5-flash", "gemini-2.5-pro"]
