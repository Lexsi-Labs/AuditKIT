# CLI

AuditKIT provides a command-line interface for quick evaluations.

## Commands

```
auditkit eval      Run an evaluation (default)
auditkit init      Scaffold a new project
auditkit list      List available resources
auditkit compare   Compare multiple models on the same dataset
auditkit redteam   Run red-team probes against a model
auditkit agent     End-to-end agent evaluation: eval / import-agenttune / rescore
```

## `auditkit eval`

Evaluate a model on a dataset.

```bash
# Using a YAML config file
auditkit eval --config auditkit.yaml

# Evaluate a CSV dataset
auditkit eval --csv data.csv --input-col question --target-col answer

# Pipe stdin
echo "hello" | auditkit eval --model hf:gpt2

# Save as markdown report
auditkit eval --csv data.csv --output results.md --format md

# Use a built-in dataset
auditkit eval --dataset mmlu --subject abstract_algebra --model hf:gpt2

# With experiment tracking
auditkit eval --csv data.csv --experiment my_run --output run.json

# Run via the lm-evaluation-harness engine instead of the native spine
auditkit eval --engine lmeval --tasks mmlu,gsm8k --model hf:gpt2
```

### Options

| Flag | Description |
|---|---|
| `--config` | Path to YAML/JSON config file |
| `--model` | Model spec (required — e.g. hf:gpt2, openai:gpt-4o) |
| `--csv` | Path to CSV dataset |
| `--input-col` | CSV input column (default: input) |
| `--target-col` | CSV target column (default: target) |
| `--dataset` | Built-in dataset (mmlu, gsm8k, arc) |
| `--subject` | MMLU subject |
| `--output` / `-o` | Output file path |
| `--format` | json, csv, or md |
| `--experiment` | Experiment name |
| `--tag` | Tag (can repeat) |
| `--temperature` | Generation temperature |
| `--max-tokens` | Max tokens |
| `--seed` | Random seed |
| `--concurrency` | Max concurrent requests |
| `--limit` | Max samples |
| `--adapter` | Input adapter (generation, chat, instruction, fewshot, rag, template) |
| `--system-prompt` | System prompt (for chat adapter) |
| `--split-strategy` | Dataset split strategy |
| `--mlflow-uri` | MLflow tracking URI |
| `--engine` | `native` (default) or `lmeval` — runs the lm-evaluation-harness engine instead |
| `--tasks` | lm-eval task name(s), comma-separated (needs `--engine lmeval`) |

## YAML Config File

Define your evaluation in a YAML file:

```yaml
# auditkit.yaml
model: openai:gpt-4o-mini
temperature: 0.0
max_tokens: 256
seed: 42
concurrency: 8
prompts:
  - "What is the capital of France?"
  - "Explain quantum computing."
output: results.json
tags: [test, v1]
```

```bash
auditkit eval --config auditkit.yaml
```

## `auditkit init`

Scaffold a new project with a default config:

```bash
auditkit init my-project
cd my-project
# Edit auditkit.yaml, then:
auditkit eval --config auditkit.yaml
```

## `auditkit list`

List available metrics, datasets, adapters, annotators, or model backends:

```bash
auditkit list             # List everything
auditkit list metrics     # Available metrics
auditkit list datasets    # Built-in datasets
auditkit list adapters    # Built-in adapters
auditkit list annotators  # Built-in annotators
auditkit list models      # Model backends
```

## `auditkit compare`

Run the same dataset against multiple models and compare performance:

```bash
# Compare two models on a CSV dataset
auditkit compare --models openai:gpt-4o,anthropic:claude-sonnet-5-5 --csv data.csv

# Use a built-in dataset
auditkit compare --models hf:gpt2,hf:distilgpt2 --dataset mmlu

# Pipe stdin
echo "hello world" | auditkit compare --models hf:gpt2,hf:distilgpt2
```

| Flag | Description |
|---|---|
| `--models` | Comma-separated model specs (required) |
| `--csv` | Path to CSV dataset |
| `--input-col` | CSV input column (default: input) |
| `--target-col` | CSV target column (default: target) |
| `--dataset` | Built-in dataset name |
| `--scorers` | Comma-separated scorer/metric names (default: auto) |
| `--baseline` | Model spec (from `--models`) to anchor a base-vs-candidate comparison against |
| `--output` / `-o` | Save results to JSON file |

## `auditkit redteam`

Run the built-in red-team probes against a model and score them with the built-in detectors:

```bash
auditkit redteam --model hf:gpt2
auditkit redteam --model groq:llama-3.3-70b-versatile --probes prompt_injection,jailbreak -o redteam.json
```

| Flag | Description |
|---|---|
| `--model` | Model spec (required) |
| `--probes` | Comma-separated probes: `prompt_injection`, `jailbreak`, `encoding`, `refusal` (default: all) |
| `--detectors` | Comma-separated detectors: `keyword`, `refusal`, `injection_success`, `system_prompt_leak` (default: all) |
| `--output` / `-o` | Save per-probe results to JSON file |

## `auditkit agent`

End-to-end agent evaluation, with three subcommands; see [Agent evaluation](agent_eval.md) for the config format:

```bash
auditkit agent import-agenttune trajectories.jsonl --inspect
auditkit agent eval --config agent-suite.json --dry-run
auditkit agent eval --config agent-suite.json --output agent-run.json
auditkit agent rescore --run agent-run.json --scorers tool_call_f1,task_completion
```

| Subcommand | What it does | Main flags |
|---|---|---|
| `eval` | Run an agent evaluation from a JSON config | `--config` (required), `--output`, `--dry-run`, `--cases`, `--redact-key/-env/-auto` |
| `import-agenttune` | Import or inspect an AgentTune JSONL/report file | `PATH`, `--inspect`, `--output`, `--redact-key/-env/-auto` |
| `rescore` | Rescore saved episodes with new scorers, without calling the agent | `--run`, `--scorers` (required), `--judge`, `--cases` |

The `--redact-*` flags scrub secrets from the saved JSON: `--redact-key KEY` blanks a field, `--redact-env VAR` replaces an env var's value wherever it appears, and `--redact-auto` also detects common secrets (API keys, tokens, private keys).
