# Configuration

## RunConfig

```python
import auditkit as ak
from auditkit.runspec import RunConfig, SplitConfig

config = RunConfig(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop_sequences=["\n"],
    seed=42,
    timeout=30,
    concurrency=1,  # default; raise for parallel requests across a chunked batch
    limit=100,
    trials=3,
    num_fewshot=5,
    split=SplitConfig(
        strategy="stratified",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=0,
    ),
)

result = ak.evaluate(dataset, model="hf:gpt2", config=config)
```

## Split Strategies

| Strategy | Description |
|---|---|
| `sequential` | First N samples for train, next M for val, rest for test |
| `random` | Shuffle then split |
| `stratified` | Stratified by target label |

## Caching

```python
from auditkit import DiskCache

cache = DiskCache(cache_dir="./.auditkit_cache")
cache.clear()
```

## Logging

```python
from auditkit import configure_logging

configure_logging(level="INFO")
```
