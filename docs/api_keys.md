# API keys

The benchmark calls candidate and judge models through LiteLLM. Copy
[`.env.example`](../.env.example) to `.env` and set the keys you need
for the providers your run actually touches.

| Variable | When it's needed |
|---|---|
| `ANTHROPIC_API_KEY` | Candidate or judge from the Claude family |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY`) | Candidate or judge from the Gemini family |
| `OPENAI_API_KEY` | Candidate or judge from the OpenAI family |
| `OPENROUTER_API_KEY` | Any model accessed through OpenRouter |
| `HF_TOKEN` (or `HUGGINGFACE_API_KEY`) | Open-weights candidate via Hugging Face Inference Providers |

You don't need keys for providers you aren't using.

## Hugging Face Inference Providers

Generate a fine-grained token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
with the **Make calls to Inference Providers** permission. HF Pro
accounts get included Inference Provider credits each month;
extra usage is billed at provider rates with no markup.

Setup details and recommended candidates:
[`running_models.md`](running_models.md).
