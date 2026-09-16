# Model Backends

Pluggable model backends for running evaluations:

- **CallableModel** / **PrecomputedModel** — stdlib, no dependencies
- **OpenAIModel** — requires `openai` extra
- **AnthropicModel** — requires `anthropic` extra
- **HFGenModel** — requires `transformers` extra
- **VLLMModel** — requires `vllm` extra
- **LiteLLMModel** — requires `litellm` extra
- **APIModel** — generic OpenAI-compatible HTTP API backend, requires `requests` extra
- **LexsiModel** — Lexsi AI gateway backend, requires `requests` extra
- **GroqModel** — Groq API backend (OpenAI-compatible), requires `requests` extra

Use `AutoModel.resolve(spec, **opts)` to select a backend at runtime, e.g.
`AutoModel.resolve("groq:llama-3.3-70b-versatile")`.
