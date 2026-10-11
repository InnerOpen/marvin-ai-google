"""Google Gemini for Marvin, as an AI provider plugin.

One entry point in ``marvin.ai_providers``: ``google``. It replaces Marvin core's built-in provider of the
same slug and reads the same setting (``GOOGLE_API_KEY``) or a workspace's own key.
"""

from marvin_integration_sdk.ai import AIProviderPlugin

from .provider import GoogleProvider

__all__ = ["GoogleProvider", "plugin"]
__version__ = "0.1.0"

plugin = AIProviderPlugin(slug="google", name="Google Gemini", provider=GoogleProvider, description="Google's Gemini models, through the Gemini API.")
