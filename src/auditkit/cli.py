"""auditkit CLI — run evaluations from the command line."""
from __future__ import annotations
import argparse
import sys
import os
from .api import evaluate
from .loaders import load_csv
from .sample import Sample
from .report_format import Report


def _load_yaml_config(path: str) -> dict:
    """Load a YAML config file (pyyaml optional)."""
    with open(path) as f:
        raw = f.read()
    try:
        import yaml as _yaml
        return _yaml.safe_load(raw) or {}
    except ImportError:
        import json as _json
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            raise SystemExit(
                "PyYAML is not installed. Install it with 'pip install pyyaml' "
                "or provide a JSON config file instead."
            )


def _merge_yaml(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    """Overlay YAML keys onto CLI args (CLI flags take precedence when not None)."""
    mapping = {
        "model": "model", "system_prompt": "system_prompt",
        "instruction": "instruction", "template": "template",
        "temperature": "temperature", "top_p": "top_p",
        "max_tokens": "max_tokens", "seed": "seed",
        "limit": "limit", "trials": "trials",
        "num_fewshot": "num_fewshot", "concurrency": "concurrency",
        "experiment": "experiment", "output": "output",
        "format": "format", "mlflow_uri": "mlflow_uri",
        "dataset": "dataset", "subject": "subject",
        "adapter": "adapter",
        "gpu_memory_utilization": "gpu_memory_utilization",
    }
    for yaml_key, ns_key in mapping.items():
        yv = cfg.get(yaml_key)
        if yv is not None and getattr(args, ns_key, None) is None:
            setattr(args, ns_key, yv)

    if "tags" in cfg and not args.tag:
        args.tag = cfg["tags"]
    if "stop" in cfg and not args.stop:
        args.stop = cfg["stop"]
    if "split" in cfg:
        s = cfg["split"]
        if args.split_strategy is None:
            args.split_strategy = s.get("strategy")
            args.train_ratio = s.get("train_ratio", args.train_ratio)
            args.val_ratio = s.get("val_ratio", args.val_ratio)
            args.test_ratio = s.get("test_ratio", args.test_ratio)
            args.split_seed = s.get("seed", args.split_seed)
    return args


def _build_eval_parser(prog: str = "auditkit eval") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Evaluate any model on any task")
    p.add_argument("--config", help="Path to YAML/JSON config file")
    p.add_argument("--model", required=True, help="Model spec (e.g. hf:gpt2, groq:llama-3.3-70b-versatile)")
    p.add_argument("--csv", help="Path to CSV dataset")
    p.add_argument("--input-col", default="input", help="CSV input column")
    p.add_argument("--target-col", default="target", help="CSV target column")
    p.add_argument("--output", "-o", help="Save results to path")
    p.add_argument("--format", choices=["json", "csv", "md"], default="json", help="Output format")
    p.add_argument("--temperature", type=float, help="Generation temperature")
    p.add_argument("--top-p", type=float, help="Nucleus sampling top-p")
    p.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    p.add_argument("--stop", nargs="*", help="Stop sequences")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--limit", type=int, help="Max samples")
    p.add_argument("--trials", type=int, help="Number of trials")
    p.add_argument("--num-fewshot", type=int, help="Few-shot examples count")
    p.add_argument("--concurrency", type=int, help="Max concurrency")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument("--split-strategy", choices=["sequential", "random"], help="Dataset split strategy")
    p.add_argument("--train-ratio", type=float, default=0.0, help="Train split ratio")
    p.add_argument("--val-ratio", type=float, default=0.0, help="Validation split ratio")
    p.add_argument("--test-ratio", type=float, default=1.0, help="Test split ratio")
    p.add_argument("--split-seed", type=int, help="Split random seed")
    p.add_argument("--experiment", help="Experiment name for tracking")
    p.add_argument("--mlflow-uri", help="MLflow tracking URI")
    p.add_argument("--tag", action="append", help="Tags (can repeat)")
    p.add_argument("--dataset", help="Built-in dataset name (mmlu, gsm8k, arc)")
    p.add_argument("--subject", help="MMLU subject (when --dataset=mmlu)")
    p.add_argument("--engine", choices=["native", "lmeval"], default="native",
                   help="Eval engine: 'native' spine (default) or 'lmeval' (lm-eval harness)")
    p.add_argument("--tasks", help="lm-eval task name(s), comma-separated (with --engine lmeval)")
    p.add_argument("--adapter", default="generation", choices=["generation", "chat", "instruction", "fewshot", "rag", "template"], help="Adapter type")
    p.add_argument("--system-prompt", default="You are a helpful assistant.", help="System prompt (for chat adapter)")
    p.add_argument("--instruction", default="Answer the following question:", help="Instruction prefix (for instruction adapter)")
    p.add_argument("--template", default="{input}", help="Template string (for template adapter)")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="vLLM GPU memory fraction (forwarded to AutoModel.resolve)",
    )
    return p


def _run_eval(args: argparse.Namespace, yaml_prompts: list[str] | None = None) -> None:
    """Execute an evaluation from parsed args."""
    cfg = {}
    if args.config:
        cfg = _load_yaml_config(args.config)
        args = _merge_yaml(args, cfg)

    from .runspec import RunConfig, SplitConfig

    split = None
    if args.split_strategy:
        split_kwargs = dict(
            strategy=args.split_strategy,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        # Only override SplitConfig's own seed=0 default when --split-seed
        # was actually passed -- args.split_seed is None whenever it wasn't
        # (argparse has no default= on that flag), and passing seed=None
        # explicitly would silently defeat SplitConfig's own reproducibility
        # guarantee (None draws fresh OS entropy every call, reshuffling
        # differently each run instead of the same way every time).
        if args.split_seed is not None:
            split_kwargs["seed"] = args.split_seed
        split = SplitConfig(**split_kwargs)

    config = RunConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_sequences=args.stop,
        seed=args.seed,
        limit=args.limit,
        trials=args.trials,
        num_fewshot=args.num_fewshot,
        concurrency=args.concurrency if args.concurrency is not None else 1,
        split=split,
    )

    engine = getattr(args, "engine", "native")
    tasks = getattr(args, "tasks", None) or cfg.get("tasks")
    if engine == "lmeval" or tasks:
        from .api import run_lmeval
        if not tasks:
            print("--engine lmeval needs --tasks (e.g. --tasks mmlu,gsm8k).")
            return
        result = run_lmeval(tasks, model=args.model, config=config,
                            experiment_name=args.experiment, tags=args.tag)
        if args.output:
            if args.format == "md":
                with open(args.output, "w") as fh:
                    fh.write(str(Report(result)))
            else:
                result.save(args.output, fmt=args.format)
        else:
            print(result.summary())
        return

    dataset_kwargs = {}
    yaml_prompts = yaml_prompts or cfg.get("prompts")
    if args.dataset:
        dataset = args.dataset
        if args.subject:
            dataset_kwargs["subject"] = args.subject
    elif args.csv:
        samples = load_csv(args.csv, input_col=args.input_col, target_col=args.target_col)
        dataset = samples
    elif yaml_prompts:
        dataset = [Sample(input=p) for p in yaml_prompts]
    else:
        dataset = [Sample(input=line.rstrip()) for line in sys.stdin if line.strip()]

    if not dataset:
        print("No dataset provided. Use --csv, --dataset, --config with prompts:, or pipe input via stdin.")
        return

    if args.adapter == "chat":
        from .adapter import ChatAdapter
        adapter = ChatAdapter(system_prompt=args.system_prompt)
    elif args.adapter == "instruction":
        from .adapter import InstructionAdapter
        adapter = InstructionAdapter(instruction=args.instruction)
    elif args.adapter == "fewshot":
        from .adapter import FewShotAdapter
        adapter = FewShotAdapter(num_shots=args.num_fewshot or 3)
    elif args.adapter == "rag":
        from .adapter import RAGAdapter
        adapter = RAGAdapter()
    elif args.adapter == "template":
        from .adapter import TemplateAdapter
        adapter = TemplateAdapter(template=args.template)
    else:
        adapter = None

    eval_opts = dict(dataset_kwargs)
    gpu_util = getattr(args, "gpu_memory_utilization", None)
    if gpu_util is not None:
        eval_opts["gpu_memory_utilization"] = gpu_util
    result = evaluate(dataset, model=args.model, adapter=adapter, config=config,
                       experiment_name=args.experiment, tags=args.tag, **eval_opts)

    if args.mlflow_uri:
        from .experiment import Experiment
        exp = Experiment(name=args.experiment or "cli_run")
        exp.add(result)
        exp.log_mlflow(experiment_name=args.experiment, tracking_uri=args.mlflow_uri)

    if args.output:
        if args.format == "md":
            with open(args.output, "w") as fh:
                fh.write(str(Report(result)))
        else:
            result.save(args.output, fmt=args.format)
    else:
        print(result.summary())


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a new auditkit project."""
    path = args.path or "."
    config_path = os.path.join(path, "auditkit.yaml")
    if os.path.exists(config_path):
        print(f"File already exists: {config_path}")
        return
    os.makedirs(path, exist_ok=True)
    with open(config_path, "w") as f:
        f.write("""# auditkit evaluation config
model: hf:gpt2
prompts:
  - "What is the capital of France?"
  - "Explain quantum computing in one sentence."
temperature: 0.0
max_tokens: 128
concurrency: 4
output: results.json
""")
    print(f"Created {config_path}")
    print("Edit the file then run: auditkit eval --config auditkit.yaml")


def cmd_list(args: argparse.Namespace) -> None:
    """List available resources."""
    from .registry import ADAPTERS, ANNOTATORS, METRICS, SCENARIOS

    resource = getattr(args, "resource", "all")
    if resource in ("metrics", "all"):
        print("Built-in metrics:")
        print("  " + ", ".join(METRICS.names()))
        print("")
    if resource in ("datasets", "all"):
        print("Built-in datasets:")
        print("  " + ", ".join(SCENARIOS.names()))
        print("")
    if resource in ("adapters", "all"):
        print("Built-in adapters:")
        print("  " + ", ".join(ADAPTERS.names()))
        print("")
    if resource in ("annotators", "all"):
        print("Built-in annotators:")
        print("  " + ", ".join(ANNOTATORS.names()))
        print("")
    if resource in ("models", "all"):
        print("Model backends:")
        print("  precomputed             - Scores samples with actual_output already set")
        print("  openai:<model>          - OpenAI (e.g. openai:gpt-4o)")
        print("  anthropic:<model>       - Anthropic (e.g. anthropic:claude-3-opus)")
        print("  hf:<model>              - HuggingFace Transformers")
        print("  lexsi:<model>           - Lexsi gateway (OpenAI-compatible)")
        print("  vllm:<model>            - vLLM")
        print("  litellm:<model>         - LiteLLM proxy (e.g. litellm:ollama/llama3.1)")
        print("  api:<model>             - Generic OpenAI-compatible endpoint (needs base_url=)")
        print("  groq:<model>            - Groq (e.g. groq:llama-3.3-70b-versatile)")
        print("")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare multiple models on the same dataset."""
    from .model_compare import compare_models
    from .sample import Sample

    models = args.models.split(",")
    dataset = [Sample(input=line.rstrip()) for line in sys.stdin if line.strip()] if not args.csv else None

    if args.csv:
        from .loaders import load_csv
        dataset = load_csv(args.csv, input_col=args.input_col, target_col=args.target_col)
    elif args.dataset:
        dataset = args.dataset

    if not dataset:
        print("Provide a dataset via --csv, --dataset, or stdin")
        return

    scorers = args.scorers.split(",") if getattr(args, "scorers", None) else None
    result = compare_models(models, dataset, scorers=scorers, model_names=models)

    baseline = getattr(args, "baseline", None)
    if baseline:
        if baseline not in result.runs:
            print(f"--baseline {baseline!r} is not one of --models ({', '.join(models)})")
            return
        for name in models:
            if name != baseline:
                print(result.pairwise(baseline, name).summary())
                print()
    else:
        print(result.summary())

    if getattr(args, "output", None):
        import json
        from .api import compare as _leaderboard
        payload = {
            "models": models,
            "baseline": baseline,
            "leaderboard": _leaderboard(list(result.runs.values())),
            "per_metric": [
                {"metric": m.metric, "scores": m.scores, "winner": m.winner}
                for m in result.per_metric()
            ],
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"wrote {args.output}")


def cmd_redteam(args: argparse.Namespace) -> None:
    """Run red team evaluation."""
    from .redteam import RedTeamRunner

    probes = args.probes.split(",") if args.probes else None
    detectors = args.detectors.split(",") if args.detectors else None

    runner = RedTeamRunner(model=args.model)
    result = runner.run(probes=probes, detectors=detectors)
    print(result.summary())
    if args.output:
        import json
        data = [
            {
                "probe": r.probe_name,
                "prompt": r.prompt,
                "output": r.output,
                "passed": r.passed,
                "detector": r.detector_name,
            }
            for r in result.results
        ]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    # Detect subcommand manually so --flags work in flat mode
    args_list = argv if argv is not None else sys.argv[1:]
    first_arg = args_list[0] if args_list else None
    if first_arg and not first_arg.startswith("-") and first_arg in ("init", "list", "eval", "redteam", "compare"):
        command = first_arg
        rest = args_list[1:] if len(args_list) > 1 else []
    else:
        command = None
        rest = args_list

    if command == "init":
        p = argparse.ArgumentParser(prog="auditkit init", description="Scaffold a new project")
        p.add_argument("path", nargs="?", default=".", help="Project directory")
        args = p.parse_args(rest)
        cmd_init(args)
        return

    if command == "list":
        p = argparse.ArgumentParser(prog="auditkit list", description="List available resources")
        p.add_argument("resource", nargs="?", default="all", choices=["all", "metrics", "datasets", "adapters", "annotators", "models"], help="Resource type to list")
        args = p.parse_args(rest)
        cmd_list(args)
        return

    if command == "redteam":
        p = argparse.ArgumentParser(prog="auditkit redteam", description="Run red team evaluation")
        p.add_argument("--model", required=True, help="Model spec (e.g. hf:gpt2, groq:llama-3.3-70b-versatile)")
        p.add_argument("--probes", help="Comma-separated probe names")
        p.add_argument("--detectors", help="Comma-separated detector names")
        p.add_argument("--output", "-o", help="Save results to JSON file")
        args = p.parse_args(rest)
        cmd_redteam(args)
        return

    if command == "compare":
        p = argparse.ArgumentParser(prog="auditkit compare", description="Compare multiple models on the same dataset")
        p.add_argument("--models", required=True, help="Comma-separated model specs")
        p.add_argument("--csv", help="Path to CSV dataset")
        p.add_argument("--input-col", default="input", help="CSV input column")
        p.add_argument("--target-col", default="target", help="CSV target column")
        p.add_argument("--dataset", help="Built-in dataset name")
        p.add_argument("--scorers", help="Comma-separated scorer/metric names (default: auto)")
        p.add_argument("--baseline", help="Model spec (from --models) to anchor a base-vs-candidate comparison against")
        p.add_argument("--output", "-o", help="Save results to JSON file")
        args = p.parse_args(rest)
        cmd_compare(args)
        return

    # Default: eval (with backwards compat for flat argument style)
    parser = _build_eval_parser("auditkit" if command is None else "auditkit eval")
    args = parser.parse_args(rest)
    _run_eval(args)


if __name__ == "__main__":
    main()
