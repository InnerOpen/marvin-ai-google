# marvin-ai-google

Google Gemini for [Marvin](https://github.com/InnerOpen/marvin), as an AI provider plugin. It talks to the
Gemini API through the unified **`google-genai`** SDK (Marvin core's built-in provider used the deprecated
`google-generativeai`) and supports chat, vision, structured output (Gemini's native JSON-schema mode),
**tool calling** (function declarations), and embeddings.

## Install

Marvin installs site-wide plugins at pod start from the chart's `plugins.packages` list. Only a
platform admin installs plugins; every workspace can then pick the provider in **Settings → AI**.

```yaml
plugins:
  packages:
    - https://github.com/InnerOpen/marvin-integration-sdk/archive/refs/heads/develop.tar.gz
    - https://github.com/InnerOpen/marvin-ai-google/archive/refs/heads/main.tar.gz
```

Requires Marvin with the AI provider contract (`marvin-integration-sdk` 0.8+). Marvin core still has a
built-in `google` provider; once this plugin is installed it **replaces** it (the log says
`AI provider plugin 'google' replaces core's built-in 'google'`) and reads the same setting, so nothing
else changes. **Admin → Plugins** lists `marvin-ai-google` with kind *AI provider*.

> pip puts `google-genai` and its dependencies (`httpx`, `pydantic`, `google-auth`, `requests`,
> `websockets`…) into the plugins volume, and `PYTHONPATH` comes before the image's site-packages, so
> those copies replace the image's own versions at runtime. Check that the versions pip picks match
> Marvin's lock before installing this in production.

## Settings

| Setting | |
|---|---|
| `GOOGLE_API_KEY` | secret, masked; a workspace in *workspace* credential mode uses its own key |

A key whose Google Cloud project has no billing is rationed: the full-size flash models (`gemini-3.8-flash`, `-3.7-`) answer
503 "high demand" and the 2.5 line 404 "no longer available to new users", while the lite tiers (`gemini-3.5-flash-lite`,
`gemini-flash-lite-latest`) answer. Enable billing on the project to use the full models; the key stays the same.

The default chat model is `gemini-3.8-flash` and the default embedding model `gemini-embedding-001`
(3072 dimensions by default). The model list comes from the API (models that can generate content, minus
embedding, image, speech and live variants).

## Behaviour worth knowing

- **Thinking tokens** are counted as completion tokens, since Gemini bills them as output. Leave
  `max_tokens` unset (or generous) for thinking models: their thinking counts against the limit.
- **Tool calling on Gemini 3**: the model's thought signatures are kept in memory (per process, by call
  id) and sent back with the call. A run resumed in another process loses them, and Gemini 3 may refuse
  that conversation. Gemini 2.5 doesn't need them.
- A prompt Gemini blocks (safety) raises an error naming the reason.

## Prices

`prices.py` holds Google's standard-tier prices per model (USD per million tokens), from LiteLLM's price
list. A snapshot id (`gemini-2.5-flash-001`) takes its base id's price; a model that isn't listed has no
estimate, and Marvin shows the run as unpriced rather than free. A new price is a release of this package,
not of Marvin.

## Develop

```sh
uv run --extra dev pytest -q
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
```

The tests never call Google. `tests/fake_server.py` is a fake Gemini server behind `httpx.MockTransport`,
so requests and answers still go through the `google-genai` SDK's own encoding and parsing; the SDK's
conformance kit (`marvin_integration_sdk.ai.testing`) runs over it. `tests/test_live.py` makes one real
call when `GOOGLE_API_KEY` is set.
