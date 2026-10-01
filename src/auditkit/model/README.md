# Model Backends

Pluggable model backends for running evaluations:

- **CallableModel** / **PrecomputedModel** — stdlib, no dependencies
- **OpenAIModel** — requires `openai` extra
- **AnthropicModel** — requires `anthropic` extra
- **HFGenModel** — requires `transformers` extra
- **VLLMModel** — requires `vllm` extra
- **LiteLLMModel** — requires `litellm` extra
- **APIModel** — generic OpenAI-compatible HTTP API backend, requires `requests` extra; in chat mode forwards `tools`/`tool_choice`/`parallel_tool_calls` and keeps returned `tool_calls` in `Generated.trace`
- **LexsiModel** — Lexsi AI gateway backend, requires `requests` extra
- **GroqModel** — Groq API backend (OpenAI-compatible), requires `requests` extra
- **AgentEndpointModel** (`agent_endpoint.py`, `agent:` prefix): an externally deployed agent over HTTP, stdlib only; returns the answer plus the agent's transcript, tool-call turns and retrieved contexts as `Generated.trace`

Use `AutoModel.resolve(spec, **opts)` to select a backend at runtime, e.g.
`AutoModel.resolve("groq:llama-3.3-70b-versatile")`.
