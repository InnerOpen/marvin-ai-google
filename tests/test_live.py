"""One optional smoke test against the real Gemini API. Skipped unless GOOGLE_API_KEY is set; CI has none."""

import os

import pytest
from marvin_integration_sdk.ai import CompletionOptions, Message

from marvin_ai_google import GoogleProvider

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY is not set")
def test_a_real_completion():
    provider = GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"])
    result = provider.complete(
        [Message("user", "Reply with the single word: towel")],
        os.environ.get("GOOGLE_MODEL", GoogleProvider.default_model),
        CompletionOptions(max_tokens=256),
    )
    assert result.content.strip() and result.prompt_tokens > 0
