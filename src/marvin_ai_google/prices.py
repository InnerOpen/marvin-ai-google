"""What Google charges for the Gemini API, per model: USD per million tokens, standard tier, text/image/video input.

From LiteLLM's model price list (``gemini/`` entries with ``litellm_provider: gemini``, 2026-10-10), which
tracks ai.google.dev/gemini-api/docs/pricing. Preview, experimental, image, audio, live and TTS variants
are left out. A dated or versioned snapshot (``gemini-2.5-flash-001``) takes the price of the id it
extends; a model that isn't listed has no estimate (Marvin shows "—" and counts the run as unpriced). A new
model's price is a release of this package, not of Marvin.
"""

from marvin_integration_sdk.ai import ModelPrice


def _price(input_per_1m: float, output_per_1m: float) -> ModelPrice:
    """A model's price with Gemini's cache rate: a cached (implicitly or explicitly) prompt token is a tenth of the
    input rate (``cache_read_input_token_cost`` in the feed); there is no write premium."""
    return ModelPrice(input_per_1m=input_per_1m, output_per_1m=output_per_1m, cached_input_per_1m=round(input_per_1m * 0.1, 6))


GOOGLE_PRICES: dict[str, ModelPrice] = {
    # Gemini 3.x
    "gemini-3.8-flash": _price(0.75, 3.75),
    "gemini-3.7-flash": _price(0.75, 3.75),
    "gemini-3.6-flash": _price(0.75, 3.75),
    "gemini-3.5-flash": _price(1.50, 9.00),
    "gemini-3.5-flash-lite": _price(0.30, 2.50),
    "gemini-3.1-flash-lite": _price(0.25, 1.50),
    # Gemini 2.5
    "gemini-2.5-pro": _price(1.25, 10.00),
    "gemini-2.5-flash": _price(0.30, 2.50),
    "gemini-2.5-flash-lite": _price(0.10, 0.40),
    # Aliases that follow the newest model of their tier (priced as that model is today).
    "gemini-pro-latest": _price(2.00, 12.00),
    "gemini-flash-latest": _price(0.75, 3.75),
    "gemini-flash-lite-latest": _price(0.30, 2.50),
    # Embeddings are billed on input only.
    "gemini-embedding-001": ModelPrice(input_per_1m=0.15, output_per_1m=0.0),
    "gemini-embedding-2": ModelPrice(input_per_1m=0.20, output_per_1m=0.0),
}
