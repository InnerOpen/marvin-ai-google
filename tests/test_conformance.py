"""The SDK's AI provider conformance kit, over a fake Gemini server."""

import pytest
from fake_server import FakeGeminiServer
from marvin_integration_sdk.ai.testing import AIProviderContract

from marvin_ai_google import GoogleProvider, plugin

PLACEHOLDER_KEY = "test-key"


class TestGoogle(AIProviderContract):
    @pytest.fixture
    def transport(self):
        return FakeGeminiServer()

    @pytest.fixture
    def provider(self, transport):
        return GoogleProvider(api_key=PLACEHOLDER_KEY, http_client=transport.client())


def test_the_entry_point_declares_the_plugin():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.load() for ep in entry_points(group="marvin.ai_providers")}
    assert eps["google"] is plugin
