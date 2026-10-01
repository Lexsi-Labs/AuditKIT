"""auditkit — evaluate any model on any dataset and any task."""

from __future__ import annotations

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("auditkit")
except _metadata.PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "1.1.2"

from .api import run_lmeval, compare, evaluate, evaluate_many, generate, scorer
from .lexsi_login import lexsi_login
from .lmeval_engine import BenchmarkEvaluator, map_model_spec, run_benchmark
from .loaders import load_agenttune, load_croissant, load_csv, load_dataset, load_hf, load_jsonl
from .bfcl import bfcl_arg_match, load_bfcl
from .provenance import read_provenance
from .agent_eval import (
    AgentCase, AgentEvent, AgentEpisode, AgentEvalSpec, AgentEvalRunner, AgentEvalResult,
    FinalStateAssertion, ArtifactAssertion, AnswerAssertion, CustomPredicate,
    episodes_from_agenttune, episode_from_openai_messages, episode_from_sample,
)
from .model_compare import CompareResult, compare_models
from .experiment import Experiment, ExperimentDB
from .cache import DiskCache
from .compat import CompatReport, check_compat
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
from .metrics.rag_judge import (ContextPrecision, ContextRecall, Faithfulness,
                                AnswerRelevancy, ResponseGroundedness, Hallucination, ContextRelevance)
from .metrics.retrieval import RetrievalMetrics
from .metrics.rag_stress import (
    CorpusSnapshot, RAGCase, RAGTrace, EvidenceSetRecall, Freshness,
    AclCompliance, CitationSupport, Abstention, make_doc, sha256_digest,
    synthetic_bank_kb,
)
from .metrics.rag_stress_runner import (
    NumericAccuracy, ContextRetention, StressCase, StressCaseResult, StressReport,
    generate_stress_cases, run_stress, DEFAULT_METRICS,
)
from .metrics.rag_stress_ops import (
    HumanReview, judge_calibration, review_queue, operational_stress,
    false_answer_under_partial_retrieval, OPS_KINDS,
)
from .metrics.agent import (
    ParallelToolCalls, RedundantToolCalls, TaskCompletion, ToolCallF1, ToolCallValidity, TrajectoryMatch,
    AgentLoopDetection, ToolPermission, ToolSelectionJudge,
)
from .trace import ToolCall, parse_tool_calls, to_turns
from .metrics.security import DefconGrade, KeywordDetector, ThreatCategory
from .adapter import (
    Adapter, ChatAdapter, FewShotAdapter, GenerationAdapter, InstructionAdapter, MCQAdapter,
    RAGAdapter, TemplateAdapter, ToolCallAdapter,
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
    "lexsi_login",
    "load_csv",
    "load_hf",
    "load_dataset",
    "read_provenance",
    "load_croissant",
    "load_jsonl",
    "load_agenttune",
    "load_bfcl",
    "bfcl_arg_match",
    "AgentCase",
    "AgentEvent",
    "AgentEpisode",
    "AgentEvalSpec",
    "AgentEvalRunner",
    "AgentEvalResult",
    "FinalStateAssertion",
    "ArtifactAssertion",
    "AnswerAssertion",
    "CustomPredicate",
    "episodes_from_agenttune",
    "episode_from_openai_messages",
    "episode_from_sample",
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
    "Faithfulness",
    "AnswerRelevancy",
    "ResponseGroundedness",
    "Hallucination",
    "ContextRelevance",
    "ContextPrecision",
    "ContextRecall",
    "RetrievalMetrics",
    "CorpusSnapshot",
    "RAGCase",
    "RAGTrace",
    "EvidenceSetRecall",
    "Freshness",
    "AclCompliance",
    "CitationSupport",
    "Abstention",
    "make_doc",
    "sha256_digest",
    "synthetic_bank_kb",
    "NumericAccuracy",
    "ContextRetention",
    "StressCase",
    "StressCaseResult",
    "StressReport",
    "generate_stress_cases",
    "run_stress",
    "DEFAULT_METRICS",
    "HumanReview",
    "judge_calibration",
    "review_queue",
    "operational_stress",
    "false_answer_under_partial_retrieval",
    "OPS_KINDS",
    "ToolCallF1",
    "TrajectoryMatch",
    "ParallelToolCalls",
    "ToolCallValidity",
    "RedundantToolCalls",
    "AgentLoopDetection",
    "ToolPermission",
    "ToolSelectionJudge",
    "TaskCompletion",
    "ToolCall",
    "ToolCallAdapter",
    "parse_tool_calls",
    "to_turns",
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
    "check_compat",
    "CompatReport",
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
