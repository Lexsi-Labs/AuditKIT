# Red Teaming

Probes and detectors for adversarial evaluation of LLM safety and robustness.

**Built-in probes** — `prompt_injection`, `jailbreak`, `encoding`, `refusal`

**Built-in detectors** — `keyword`, `refusal`, `injection_success`, `system_prompt_leak`

Run with `RedTeamRunner` which orchestrates probe → model → detector pipelines.
