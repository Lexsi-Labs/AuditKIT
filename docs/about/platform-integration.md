# Platform Integration

AuditKIT serves two roles: a standalone open-source library you can `pip install` and use immediately, and a platform-integrated evaluation service within the Lexsi ecosystem. This page covers the latter.

---

## Two Sides of AuditKIT

| | Side A: OSS Library | Side B: Platform Integration |
|---|---|---|
| **Distribution** | `pip install auditkit` | Bundled via `evals_service` microservice |
| **Entry point** | `ak.evaluate()` in Python | `POST /runs` via HTTP API |
| **Model backends** | User-configured (OpenAI, HF, vLLM, etc.) | Platform-managed model registry |
| **Storage** | Local files / user-managed | MongoDB with experiment lineage |
| **CI/CD** | `RunComparison.grade()` in a script (no `--gate` CLI flag yet — see [roadmap](../community/roadmap.md)) | Automated regression gates per pipeline |
| **Caching** | Fingerprint-driven local cache | Distributed fingerprint cache across runs |
| **Governance** | Manual | Policy-as-code, audit trails, approval workflows |

---

## evals_service

The `evals_service` is a standalone microservice that vendors the `auditkit` library. It exposes a REST API for triggering evaluations, polling status, and retrieving results, while managing model routing, concurrency, caching, and storage behind the scenes.

### Responsibilities

- Packages AuditKIT as a deployable service
- Routes evaluation requests to the appropriate model backend (GPU cluster, API endpoint, or internal model)
- Manages concurrency, retries, and timeouts for large evaluation runs
- Stores runs, results, and fingerprints in MongoDB
- Computes baseline comparisons and returns pass/warn/fail decisions

---

## Integration Flow

```
Prunkit / Reducto / aligntune  ──►  Model Artifact
                                          │
                                    User triggers eval
                                    (UI or SDK)
                                          │
                                          ▼
                                   evals_service
                                          │
                                    ┌─────┴─────┐
                                    │  auditkit  │
                                    │ evaluate() │
                                    └─────┬─────┘
                                          │
                                    ┌─────┴─────┐
                                    │  Compare   │
                                    │  baseline  │
                                    └─────┬─────┘
                                          │
                                    Pass / Warn / Fail
```

1. **Model production**: A model is produced by Prunkit (fine-tuning), Reducto (quantization/pruning), or aligntune (alignment tuning).
2. **Trigger**: A user or pipeline triggers an evaluation via the Lexsi UI, SDK, or directly through the `evals_service` API.
3. **Execution**: `evals_service` runs the AuditKIT evaluation: loads the dataset, instantiates metrics, dispatches requests to the model, and collects scores.
4. **Comparison**: Results are compared against a stored baseline (if one exists) using `RunDiff` with bootstrap significance testing.
5. **Decision**: A pass/warn/fail verdict is returned based on configured thresholds per metric.

---

## API

### POST /runs

Create and initiate a new evaluation run.

```json
{
  "model_id": "prunkit/qwen-2.5-7b-lora-v3",
  "dataset": "mmlu",
  "metrics": ["exact_match", "f1"],
  "split": "test",
  "seed": 42,
  "thresholds": {
    "exact_match": 0.70,
    "f1": 0.65
  },
  "baseline_run_id": "abc123"
}
```

**Response** (202 Accepted):

```json
{
  "run_id": "run_9f3a2b1c",
  "status": "pending",
  "fingerprint": "a1b2c3d4e5f6...",
  "created_at": "2026-07-01T12:00:00Z"
}
```

### GET /runs/{id}/status

Poll the status of a running evaluation.

```json
{
  "run_id": "run_9f3a2b1c",
  "status": "running",
  "progress": {
    "completed": 42,
    "total": 100,
    "percent": 42.0
  },
  "started_at": "2026-07-01T12:00:05Z"
}
```

Possible statuses: `pending`, `running`, `completed`, `failed`, `cached`.

### GET /runs/{id}/results

Retrieve completed evaluation results.

```json
{
  "run_id": "run_9f3a2b1c",
  "fingerprint": "a1b2c3d4e5f6...",
  "status": "completed",
  "model_id": "prunkit/qwen-2.5-7b-lora-v3",
  "dataset": "mmlu",
  "metrics": [
    {
      "name": "exact_match",
      "value": 0.735,
      "baseline_value": 0.712,
      "delta": 0.023,
      "verdict": "pass"
    },
    {
      "name": "f1",
      "value": 0.689,
      "baseline_value": 0.671,
      "delta": 0.018,
      "verdict": "pass"
    }
  ],
  "overall_verdict": "pass",
  "summary": {
    "total_samples": 100,
    "failed_samples": 0,
    "cached_samples": 0,
    "duration_seconds": 34.2
  }
}
```

---

## Architecture

```
                        ┌──────────┐
                        │  Gateway  │
                        │ (Auth,    │
                        │  Routing) │
                        └────┬─────┘
                             │
                     ┌───────┴────────┐
                     │  evals_service  │
                     │  (auditkit)     │
                     └───────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Model   │  │  Model   │  │  Model   │
        │ (GPU)    │  │ (API)    │  │ (vLLM)   │
        └──────────┘  └──────────┘  └──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                      ┌──────────┐
                      │  MongoDB │
                      │ (Runs,   │
                      │  Results,│
                      │  Baselines)│
                      └──────────┘
```

- **Gateway**: Handles authentication, rate limiting, and request routing to `evals_service`. Single entry point for the Lexsi platform.
- **evals_service**: The core evaluation engine, vendoring the AuditKIT library. Manages the evaluation lifecycle, dispatches model requests, and stores results.
- **Model**: The target under evaluation. Dispatched to GPU nodes (via Modal), external API endpoints (OpenAI, Anthropic), or self-hosted inference servers (vLLM, TGI).
- **MongoDB**: Persistent storage for run metadata, results, baselines, fingerprints, and pipeline configurations.

### GPU Dispatch (Modal)

For large-scale evaluations, the service dispatches model inference to cloud GPU infrastructure via Modal. It partitions evaluation samples across GPU workers, collates results, and stores them back in MongoDB.

---

## CI/CD Integration

The AuditKIT CLI does **not** have a `--gate` flag yet (it's on the
[roadmap](../community/roadmap.md), T5). Today, the same regression-gating
job is done in a short Python script using `RunComparison`:

```python
import auditkit as ak

baseline = ak.evaluate(dataset, model="hf:Qwen/Qwen2.5-7B", scorers=["exact_match"])
candidate = ak.evaluate(dataset, model=candidate_model, scorers=["exact_match"])

cmp = ak.compare(baseline, candidate)
if cmp.grade() != ak.DeltaGrade.PASS:
    raise SystemExit(f"regression gate failed: {cmp.summary()}")
```

### GitHub Actions Example

```yaml
name: Model Evaluation Gate
on: [deployment]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run evaluation gate
        run: python ci_gate.py  # the script above
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: results.json
```

### CI Configuration

The CLI's real YAML config format (no `thresholds:`/`cache:` keys — those
aren't implemented; gating is done in the script above, not the config
file):

```yaml
# auditkit.yaml
model: hf:Qwen/Qwen2.5-7B
temperature: 0.0
prompts:
  - "What is the capital of France?"
output: results.json
```

When paired with the platform API, the pipeline can:

1. POST a new run to `evals_service` via the SDK
2. Poll `GET /runs/{id}/status` until completion
3. Fail the pipeline if `overall_verdict` is `fail`

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY --from=ghcr.io/lexsi/evals-service:latest /app /app

EXPOSE 8000
CMD ["uvicorn", "evals_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: "3.9"
services:
  evals_service:
    image: ghcr.io/lexsi/evals-service:latest
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=mongodb://mongo:27017/evals
      - MODAL_TOKEN_ID=${MODAL_TOKEN_ID}
      - MODAL_TOKEN_SECRET=${MODAL_TOKEN_SECRET}
    depends_on:
      - mongo

  mongo:
    image: mongo:7
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evals-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: evals-service
  template:
    metadata:
      labels:
        app: evals-service
    spec:
      containers:
        - name: evals-service
          image: ghcr.io/lexsi/evals-service:latest
          ports:
            - containerPort: 8000
          env:
            - name: MONGODB_URI
              valueFrom:
                secretKeyRef:
                  name: mongo-uri
                  key: uri
            - name: MODAL_TOKEN_ID
              valueFrom:
                secretKeyRef:
                  name: modal-tokens
                  key: token-id
            - name: MODAL_TOKEN_SECRET
              valueFrom:
                secretKeyRef:
                  name: modal-tokens
                  key: token-secret
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
```

### Modal (Cloud GPU)

For GPU-backed inference, the service dispatches model calls to Modal. It partitions each evaluation batch into chunks, runs them across parallel GPU workers, then merges and stores the results.

```python
# evals_service/dispatch.py (illustrative)
import modal

app = modal.App("evals-worker")

@app.function(gpu="A100", timeout=600)
def evaluate_chunk(model_id: str, samples: list, metrics: list):
    import auditkit as ak
    model = ak.AutoModel.resolve(model_id)
    return [metric.score(sample, model.generate(sample.input)) for sample in samples]
```

---

## Security & Governance

Platform integration adds several layers beyond the OSS library:

- **Authentication**: All API requests are authenticated via the Lexsi platform gateway (OAuth2 / JWT).
- **Audit trails**: Every evaluation run is logged with the user, timestamp, fingerprint, and full RunSpec for compliance review.
- **Policy-as-code**: Thresholds and allowed datasets/metrics can be enforced via policy configuration, preventing unauthorized evaluations.
- **Data isolation**: Tenant data is isolated in MongoDB via database-per-tenant or collection-prefix patterns.
- **Approval workflows**: Sensitive evaluations (e.g., running against production models) can require multi-party approval before execution.

---

*See the [API Reference](../api.md) for the full Python API, or the [CLI docs](../cli.md) for standalone usage.*
