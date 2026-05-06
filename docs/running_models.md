# Running open-weights candidates via Hugging Face Inference Providers

This page covers running open-weights multimodal candidates through
[HF Inference Providers](https://huggingface.co/docs/inference-providers/index)
(`router.huggingface.co`).

## Authentication

Set `HF_TOKEN` to a fine-grained token with the *Make calls to
Inference Providers* permission. `HUGGINGFACE_API_KEY` works as a
fallback variable name. HF Pro accounts get included Inference
Provider credits each month; additional usage is billed at provider
rates with no markup.

## Model id format

Model ids use `huggingface/<inference_provider>/<hf_org>/<hf_model>`.
The benchmark works with any chat-completion-capable vision-language
model.

## Recommended candidates

| Model | Model id (LiteLLM format) | Notes |
| --- | --- | --- |
| Qwen 2.5 VL 7B | `huggingface/together/Qwen/Qwen2.5-VL-7B-Instruct` | Best balance: leading open VLM, fast and cheap |
| Llama 3.2 Vision 11B | `huggingface/together/meta-llama/Llama-3.2-11B-Vision-Instruct` | Proven, broadly supported across providers |
| Qwen 2.5 VL 72B | `huggingface/fireworks-ai/Qwen/Qwen2.5-VL-72B-Instruct` | Frontier open-weights VLM; higher cost |

Other HF Inference Provider partners that support vision-language
chat completion include Cohere, Featherless AI, Fireworks, Groq,
Hyperbolic, Novita, Nscale, OVHcloud, Together, and Z.ai. The
specific provider segment in the model id determines routing; pick
based on availability and pricing.

## Worked example

```bash
wac-bench \
  --model huggingface/together/Qwen/Qwen2.5-VL-7B-Instruct \
  --judge-family openai \
  --judge-model openrouter/openai/gpt-4o-mini \
  --output-dir runs/qwen-vl-7b
```

`--judge-family` is required for open-weights HF candidates; the
cross-family judge auto-resolution map only covers Claude, Gemini,
and OpenAI today. Pick the family different from the candidate's
training lineage to preserve self-preference bias correction. The
runner will surface a clear error if you pass `--judge-family auto`
against an open-weights HF candidate.
