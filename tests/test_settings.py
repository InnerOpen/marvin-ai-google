"""Credentials, prices and models: what Marvin reads to build, bill and offer the provider."""

import pytest
from marvin_integration_sdk.ai import AIConfigError, read_credentials

from marvin_ai_google import GoogleProvider


def test_it_reads_the_setting_core_reads():
    assert [c.env("google") for c in GoogleProvider.credentials] == ["GOOGLE_API_KEY"]
    built = GoogleProvider.from_credentials(read_credentials(GoogleProvider.credentials, {"api_key": "test-key"}))
    assert built._api_key == "test-key"


def test_a_key_is_required():
    with pytest.raises(AIConfigError, match="api_key"):
        read_credentials(GoogleProvider.credentials, {})


def test_the_defaults_are_priced_and_the_suggestions_are_chat_models():
    assert GoogleProvider.price(GoogleProvider.default_model) and GoogleProvider.price(GoogleProvider.default_embedding_model)
    assert GoogleProvider.default_model in GoogleProvider.suggested_models
    assert GoogleProvider.default_embedding_model not in GoogleProvider.suggested_models


def test_a_snapshot_takes_the_price_of_its_model():
    assert GoogleProvider.price("gemini-2.5-flash-lite-001") == GoogleProvider.price("gemini-2.5-flash-lite")
    assert GoogleProvider.price("gemini-2.5-flash-001") == GoogleProvider.price("gemini-2.5-flash")
    assert GoogleProvider.estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000) == 2.8
