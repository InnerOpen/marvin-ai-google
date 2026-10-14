"""A fake Gemini API server for the conformance kit: the real ``google-genai`` SDK talks to it through
``httpx.MockTransport``, so requests and answers go through the SDK's own encoding and parsing."""

from __future__ import annotations

import json
from collections import deque

import httpx
from marvin_integration_sdk.ai import ToolCall
from marvin_integration_sdk.ai.fake import FakeTransport

ERROR_STATUS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
}


class FakeGeminiServer(FakeTransport):
    """Queued neutral replies, answered in the wire shape of whichever endpoint the provider calls."""

    def __init__(self) -> None:
        self.replies: deque[dict] = deque()
        self.requests: list[httpx.Request] = []

    # ── the kit's vocabulary ─────────────────────────────────────────────────

    def reply_text(self, text: str, *, prompt_tokens: int = 11, completion_tokens: int = 7, cached: int = 0) -> None:
        """`cached`: the part of the prompt Gemini's implicit cache served (cachedContentTokenCount)."""
        self.replies.append({"text": text, "usage": (prompt_tokens, completion_tokens), "cached": cached})

    def reply_tool_calls(self, calls: list[ToolCall], *, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.replies.append({"tool_calls": list(calls), "usage": (prompt_tokens, completion_tokens)})

    def reply_models(self, ids: list[str]) -> None:
        self.replies.append({"models": list(ids)})

    def reply_embeddings(self, vectors: list[list[float]]) -> None:
        self.replies.append({"embeddings": [list(v) for v in vectors]})

    def reply_error(self, status: int, message: str) -> None:
        self.replies.append({"error": (status, message)})

    def last_request(self) -> str:
        return self.requests[-1].content.decode() if self.requests else ""

    def last_json(self) -> dict:
        return json.loads(self.last_request() or "{}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    # ── the server ───────────────────────────────────────────────────────────

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.popleft() if self.replies else {}
        if "error" in reply:
            status, message = reply["error"]
            return httpx.Response(status, json={"error": {"code": status, "message": message, "status": ERROR_STATUS.get(status, "UNKNOWN")}})
        path = request.url.path
        if path.endswith(":generateContent"):
            model = path.split("/models/")[-1].split(":")[0]
            return httpx.Response(200, json=_generated(reply, model))
        if path.endswith(":batchEmbedContents"):
            return httpx.Response(200, json={"embeddings": [{"values": v} for v in reply.get("embeddings", [])]})
        if path.endswith("/models"):
            models = [{"name": f"models/{i}", "supportedGenerationMethods": ["generateContent", "countTokens"]} for i in reply.get("models", [])]
            return httpx.Response(200, json={"models": models})
        return httpx.Response(404, json={"error": {"code": 404, "message": f"no fake for {path}", "status": "NOT_FOUND"}})


def _generated(reply: dict, model: str) -> dict:
    prompt, completion = reply.get("usage", (0, 0))
    parts: list[dict] = [{"functionCall": {"id": c.id, "name": c.name, "args": c.arguments}} for c in reply.get("tool_calls", [])]
    if "text" in reply:
        parts.append({"text": reply["text"]})
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {
            "promptTokenCount": prompt,
            "candidatesTokenCount": completion,
            "totalTokenCount": prompt + completion,
            "cachedContentTokenCount": reply.get("cached", 0),
        },
        "modelVersion": model,
        "responseId": "resp-1",
    }
