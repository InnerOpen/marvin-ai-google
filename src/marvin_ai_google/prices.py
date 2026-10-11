"""What Google charges for the Gemini API, per model: USD per million tokens, standard tier, text/image/video input.

From LiteLLM's model price list (``gemini/`` entries with ``litellm_provider: gemini``, 2026-10-10), which
tracks ai.google.dev/gemini-api/docs/pricing. Preview, experimental, image, audio, live and TTS variants
are left out. A dated or versioned snapshot (``gemini-2.5-flash-001``) takes the price of the id it
extends; a model that isn't listed has no estimate (Marvin shows "—" and counts the run as unpriced). A new
model's price is a release of this package, not of Marvin.
"""

from marvin_integration_sdk.ai import ModelPrice

GOOGLE_PRICES: dict[str, ModelPrice] = {
    # Gemini 3.x
    "gemini-3.8-flash": ModelPrice(input_per_1m=0.75, output_per_1m=3.75),
    "gemini-3.7-flash": ModelPrice(input_per_1m=0.75, output_per_1m=3.75),
    "gemini-3.6-flash": ModelPrice(input_per_1m=0.75, output_per_1m=3.75),
    "gemini-3.5-flash": ModelPrice(input_per_1m=1.50, output_per_1m=9.00),
    "gemini-3.5-flash-lite": ModelPrice(input_per_1m=0.30, output_per_1m=2.50),
    "gemini-3.1-flash-lite": ModelPrice(input_per_1m=0.25, output_per_1m=1.50),
    # Gemini 2.5
    "gemini-2.5-pro": ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    "gemini-2.5-flash": ModelPrice(input_per_1m=0.30, output_per_1m=2.50),
    "gemini-2.5-flash-lite": ModelPrice(input_per_1m=0.10, output_per_1m=0.40),
    # Aliases that follow the newest model of their tier (priced as that model is today).
    "gemini-pro-latest": ModelPrice(input_per_1m=2.00, output_per_1m=12.00),
    "gemini-flash-latest": ModelPrice(input_per_1m=0.75, output_per_1m=3.75),
    "gemini-flash-lite-latest": ModelPrice(input_per_1m=0.30, output_per_1m=2.50),
    # Embeddings are billed on input only.
    "gemini-embedding-001": ModelPrice(input_per_1m=0.15, output_per_1m=0.0),
    "gemini-embedding-2": ModelPrice(input_per_1m=0.20, output_per_1m=0.0),
}
