"""auditkit — evaluate any model on any dataset and any task."""

from __future__ import annotations

__version__ = "1.0.0"

from .api import run_lmeval, compare, evaluate, evaluate_many, generate, scorer
from .lmeval_engine import BenchmarkEvaluator, map_model_spec, run_benchmark
from .loaders import load_csv, load_hf, load_croissant
from .model_compare import CompareResult, compare_models
from .experiment import Experiment, ExperimentDB
from .cache import DiskCache
from .logs import configure_logging
from .diff import DeltaGrade, RunDiff
from .comparison import MetricDelta, RunComparison, TaskDelta, grade_delta
from .registry import ADAPTERS, EVALUATORS, METRICS, SCENARIOS
from . import scenarios  # noqa: F401 -- triggers @SCENARIOS.register decorators
from .errors import AuditKitError, ExtraNotInstalled
from .evaluator import Evaluator
from .model import AutoModel
from .scenario import CallableScenario, ListScenario, Scenario
from .metrics.code import (
    Contains, EndsWith, Equals, F1Score, IsJson, Levenshtein, Regex, StartsWith, WordCount,
)
from .metrics.embedding import BM25Similarity, CosineSimilarity, TokenOverlap
from .metrics.generation import (
    BertScore, Bleu, ChrF, Perplexity, RogueL, WordErrorRate,
)
from .metrics.encoder_judge import (
    EncoderJudge, FactualityEncoderJudge, SentimentEncoderJudge,
)
from .metrics.hallucination import FactualConsistency
from .metrics.judge import (
    BiasJudge, ClosedQA, Factuality, GEval, JudgeMetric, LLMJudge, Relevance, RubricItem,
)
from .metrics.pairwise import EloScore, PreferenceAccuracy, WinRate
from .metrics.toxicity import HateSpeechScore, RepresentationSkew, ToxicityScore
from .metrics.guard import GuardJudge
from .metrics.perf import LatencyStats, Throughput
from .metrics.rag import AnswerOverlap, ContextCoverage, ContextOverlap, LexicalGroundedness
from .metrics.security import DefconGrade, KeywordDetector, ThreatCategory
from .adapter import (
    Adapter, ChatAdapter, FewShotAdapter, GenerationAdapter, InstructionAdapter, MCQAdapter,
    RAGAdapter, TemplateAdapter,
)
from .router import route_adapter
from .annotator import Annotator, RegexAnnotator, LLMAnnotator, ThinkingStripAnnotator
from .report import RunResult
from .runspec import RunConfig
from .report_format import Report
from .sample import Sample
from .score import Score, Stat, pass_at_k, pass_hat_k
from .scoring import ScoreGate, WeightedSum
from .types import (
    Capability,
    DataType,
    Direction,
    ScoreKind,
    Source,
    TaskKind,
)

#: Public alias of :class:`RunResult` — the object returned by :func:`evaluate`.
Result = RunResult

__all__ = [
    "evaluate",
    "evaluate_many",
    "run_lmeval",
    "generate",
    "compare",
    "scorer",
    "BenchmarkEvaluator",
    "run_benchmark",
    "map_model_spec",
    "load_csv",
    "load_hf",
    "load_croissant",
    "Experiment",
    "ExperimentDB",
    "Sample",
    "Score",
    "Stat",
    "Result",
    "RunConfig",
    "Adapter",
    "GenerationAdapter",
    "MCQAdapter",
    "ChatAdapter",
    "FewShotAdapter",
    "InstructionAdapter",
    "RAGAdapter",
    "TemplateAdapter",
    "route_adapter",
    "AutoModel",
    "Scenario",
    "ListScenario",
    "CallableScenario",
    "ADAPTERS",
    "METRICS",
    "EVALUATORS",
    "ScoreGate",
    "WeightedSum",
    "Equals",
    "configure_logging",
    "BertScore",
    "Bleu",
    "ChrF",
    "Contains",
    "Perplexity",
    "StartsWith",
    "EndsWith",
    "Evaluator",
    "Regex",
    "Levenshtein",
    "WordCount",
    "IsJson",
    "RogueL",
    "WordErrorRate",
    "F1Score",
    "RunDiff",
    "DeltaGrade",
    "RunComparison",
    "MetricDelta",
    "TaskDelta",
    "grade_delta",
    "LexicalGroundedness",
    "ContextCoverage",
    "ContextOverlap",
    "AnswerOverlap",
    "JudgeMetric",
    "RubricItem",
    "GEval",
    "LLMJudge",
    "EncoderJudge",
    "FactualityEncoderJudge",
    "SentimentEncoderJudge",
    "Factuality",
    "ClosedQA",
    "Relevance",
    "BiasJudge",
    "KeywordDetector",
    "DefconGrade",
    "DiskCache",
    "ThreatCategory",
    "LatencyStats",
    "Throughput",
    "Annotator",
    "RegexAnnotator",
    "LLMAnnotator",
    "ThinkingStripAnnotator",
    "Report",
    "TaskKind",
    "ScoreKind",
    "DataType",
    "Direction",
    "Capability",
    "Source",
    "SCENARIOS",
    "AuditKitError",
    "ExtraNotInstalled",
    "CosineSimilarity",
    "TokenOverlap",
    "BM25Similarity",
    "FactualConsistency",
    "ToxicityScore",
    "RepresentationSkew",
    "HateSpeechScore",
    "GuardJudge",
    "WinRate",
    "EloScore",
    "PreferenceAccuracy",
    "pass_at_k",
    "pass_hat_k",
    "CompareResult",
    "compare_models",
]
