"""Google Gemini through the ``google-genai`` SDK (the Gemini Developer API, authenticated with an API key)."""

from __future__ import annotations

import base64
import json
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from marvin_integration_sdk.ai import (
    API_KEY,
    AIProvider,
    CompletionOptions,
    CompletionResult,
    ImagePart,
    Message,
    ToolCall,
    ToolDefinition,
)

from .prices import GOOGLE_PRICES

# Models the account lists that can't hold a conversation (embeddings, speech, images, video, live audio…).
NON_CHAT_MARKERS = ("embedding", "tts", "image", "imagen", "veo", "lyria", "live", "audio", "transcribe", "aqa", "robotics", "computer-use")
# batchEmbedContents takes at most this many texts per request.
EMBED_BATCH = 100
# Gemini 3 wants the thought signature of a function call echoed back with it. A ToolCall has no field for
# it, so the signatures of the latest calls are kept here, by call id, for the life of the process.
SIGNATURE_MEMORY = 512
TOOL_MODES = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
STOP_REASONS = {"STOP": "stop", "MAX_TOKENS": "length"}

_signatures: OrderedDict[str, bytes] = OrderedDict()


def _remember_signature(call_id: str, signature: bytes) -> None:
    _signatures[call_id] = signature
    while len(_signatures) > SIGNATURE_MEMORY:
        _signatures.popitem(last=False)


def _text_parts(content: str | list) -> list[dict]:
    """Agnostic content as Gemini parts: text, and inline image data."""
    if isinstance(content, str):
        return [{"text": content}] if content else []
    return [
        {"inline_data": {"mime_type": p.mime_type, "data": base64.b64decode(p.data)}} if isinstance(p, ImagePart) else {"text": str(p)}
        for p in content
    ]


def _function_call(call: ToolCall) -> dict:
    part: dict = {"function_call": {"id": call.id, "name": call.name, "args": call.arguments or {}}}
    if call.id in _signatures:
        part["thought_signature"] = _signatures[call.id]
    return part


def _function_response(message: Message, names: Mapping[str, str]) -> dict:
    text = message.content if isinstance(message.content, str) else "".join(str(p) for p in message.content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    call_id = message.tool_call_id or ""
    # Gemini wants an object as the result; a tool that returned anything else is wrapped.
    response = parsed if isinstance(parsed, dict) else {"output": text}
    return {"function_response": {"id": call_id, "name": names.get(call_id, "tool"), "response": response}}


def gemini_contents(messages: list[Message]) -> tuple[str | None, list[dict]]:
    """Messages → ``(system_instruction, contents)``. System messages become the system instruction,
    assistant turns the ``model`` role, and a run of tool results one ``user`` turn of function responses."""
    system: list[str] = []
    contents: list[dict] = []
    names: dict[str, str] = {}
    results: dict | None = None  # the open turn of function responses, while tool messages follow one another
    for m in messages:
        if m.role != "tool":
            results = None
        if m.role == "system":
            system.append(m.content if isinstance(m.content, str) else "".join(str(p) for p in m.content))
        elif m.role == "tool":
            if results is None:
                results = {"role": "user", "parts": []}
                contents.append(results)
            results["parts"].append(_function_response(m, names))
        elif m.role == "assistant":
            names.update({c.id: c.name for c in m.tool_calls or []})
            parts = _text_parts(m.content) + [_function_call(c) for c in m.tool_calls or []]
            if parts:
                contents.append({"role": "model", "parts": parts})
        else:
            parts = _text_parts(m.content)
            if parts:
                contents.append({"role": "user", "parts": parts})
    instruction = "\n\n".join(s for s in system if s) or None
    if instruction and not contents:
        # Gemini refuses a request with no contents; a lone system message is the prompt.
        return None, [{"role": "user", "parts": [{"text": instruction}]}]
    return instruction, contents


def gemini_result(resp, model: str) -> CompletionResult:
    if not resp.candidates:
        feedback = resp.prompt_feedback
        reason = (feedback.block_reason_message or str(feedback.block_reason)) if feedback and feedback.block_reason else "no candidates returned"
        raise ValueError(f"Gemini returned no answer: {reason}")
    candidate = resp.candidates[0]
    text: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in candidate.content.parts if candidate.content and candidate.content.parts else []:
        if part.function_call:
            call = part.function_call
            call_id = call.id or f"call_{uuid.uuid4().hex[:24]}"
            if part.thought_signature:
                _remember_signature(call_id, part.thought_signature)
            tool_calls.append(ToolCall(id=call_id, name=call.name or "", arguments=dict(call.args or {})))
        elif part.text and not part.thought:
            text.append(part.text)
    usage = resp.usage_metadata
    prompt = (usage.prompt_token_count or 0) if usage else 0
    # Gemini caches a repeated prefix on its own (implicit caching) and bills that part at a tenth; it is a part of
    # prompt_token_count.
    cached = (getattr(usage, "cached_content_token_count", None) or 0) if usage else 0
    # Thinking tokens are billed as output but counted apart from the candidate's own.
    completion = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
    finish = candidate.finish_reason.value if candidate.finish_reason else "STOP"
    return CompletionResult(
        content="".join(text),
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        model=resp.model_version or model,
        cached_prompt_tokens=min(cached, prompt),
        raw=resp.model_dump(mode="json", exclude_none=True, exclude={"sdk_http_response"}),
        tool_calls=tool_calls,
        stop_reason="tool_calls" if tool_calls else STOP_REASONS.get(finish, finish.lower()),
    )


class GoogleProvider(AIProvider):
    provider_type = "google"
    logo = "logo.svg"  # the Gemini mark: Marvin shows it beside the provider's name
    display_name = "Google Gemini"
    credentials = (API_KEY,)
    supports_vision = True
    supports_structured_output = True
    supports_embeddings = True
    supports_tool_calls = True
    prices = GOOGLE_PRICES
    default_model = "gemini-3.8-flash"
    suggested_models = tuple(m for m in GOOGLE_PRICES if "embedding" not in m)
    default_embedding_model = "gemini-embedding-001"

    def __init__(self, api_key: str, http_options: Any = None, http_client: Any = None) -> None:
        """``http_options`` (a ``google.genai.types.HttpOptions``) and ``http_client`` (an ``httpx.Client``)
        go to ``genai.Client``; tests point them at a mock transport."""
        self._api_key = api_key
        self._http_options = http_options
        self._http_client = http_client
        self._genai_client = None

    @classmethod
    def from_credentials(cls, values: Mapping[str, Any]) -> GoogleProvider:
        return cls(api_key=values.get("api_key") or "")

    def _client(self):
        if self._genai_client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                raise ImportError("The Google GenAI SDK is not installed (pip install marvin-ai-google brings it).") from None
            options = self._http_options or types.HttpOptions()
            if self._http_client is not None:
                options = options.model_copy(update={"httpx_client": self._http_client})
            self._genai_client = genai.Client(api_key=self._api_key, http_options=options)
        return self._genai_client

    def _generate(self, messages: list[Message], model: str, opts: CompletionOptions, **config) -> CompletionResult:
        from google.genai import types

        system, contents = gemini_contents(messages)
        settings = {
            "system_instruction": system,
            "max_output_tokens": opts.max_tokens,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            **config,
        }
        resp = self._client().models.generate_content(
            model=model, contents=contents, config=types.GenerateContentConfig(**{k: v for k, v in settings.items() if v is not None})
        )
        return gemini_result(resp, model)

    def complete(self, messages: list[Message], model: str, options: CompletionOptions | None = None) -> CompletionResult:
        return self._generate(messages, model, options or CompletionOptions())

    def complete_with_tools(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolDefinition],
        options: CompletionOptions | None = None,
        tool_choice: str = "auto",
    ) -> CompletionResult:
        from google.genai import types

        declarations = [types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.input_schema) for t in tools]
        return self._generate(
            messages,
            model,
            options or CompletionOptions(),
            tools=[types.Tool(function_declarations=declarations)],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=TOOL_MODES.get(tool_choice, "AUTO"))),
            # The agent loop runs the tools itself; the SDK must not call anything on its own.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def complete_structured(self, messages: list[Message], model: str, output_schema: dict, options: CompletionOptions | None = None) -> dict:
        return self.execute_operation(messages, model, output_schema, options)[0]

    def execute_operation(self, messages, model, output_schema, options=None):
        result = self._generate(
            messages, model, options or CompletionOptions(), response_mime_type="application/json", response_json_schema=output_schema
        )
        return json.loads(result.content or "{}"), result

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            resp = self._client().models.embed_content(model=model, contents=texts[start : start + EMBED_BATCH])
            vectors.extend(list(e.values or []) for e in resp.embeddings or [])
        return vectors

    def list_models(self) -> list[str]:
        ids = {m.name.removeprefix("models/") for m in self._client().models.list() if m.name and "generateContent" in (m.supported_actions or [])}
        return sorted(i for i in ids if not any(marker in i for marker in NON_CHAT_MARKERS))

    def test_connection(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
            return True, f"Connected — {len(models)} models available"
        except Exception as e:  # noqa: BLE001 — reported to the admin, never raised
            return False, str(e)
