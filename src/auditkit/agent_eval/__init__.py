"""End-to-end agent evaluation: the AgentTune bridge (slices A1 + A2).

Public surface::

    from auditkit.agent_eval import (
        AgentCase, AgentEvent, AgentEpisode,
        AgentEvalSpec, AgentEvalRunner, AgentEvalResult,
        FinalStateAssertion, ArtifactAssertion, AnswerAssertion, CustomPredicate,
        episodes_from_agenttune, episode_from_eventlog,
        episode_from_openai_messages, episode_from_sample,
        SourceEvidence, join_evidence, lineage_sidecar,  # Lexsi evidence importer
    )

Everything here is stdlib-only; AgentTune / torch / transformers are never
imported by this package. Slices delivered: A1, A2, A3 (reliability/trials --
:func:`reliability`), A4 (harness-owned loop, ``mode="harness"``), and A5's
achievable sidecar export (:func:`write_sidecar`). ``episode_from_eventlog``
reads an EventLog. Automatic secret detection is :func:`detect_secrets` /
``redact(auto=True)``.
"""

from __future__ import annotations

from .importers import (
    episode_from_eventlog,
    episode_from_openai_messages,
    episode_from_sample,
    episodes_from_agenttune,
    inspect_agenttune,
    supported_scorers_from_coverage,
)
from .outcome import (
    ERROR,
    FAILURE,
    SUCCESS,
    UNKNOWN,
    AnswerAssertion,
    ArtifactAssertion,
    AssertionOracle,
    CustomPredicate,
    FinalStateAssertion,
    Oracle,
    Outcome,
    judge_outcome,
    resolve_outcome,
)
from .outcome import from_spec as outcome_from_spec
from .reliability import aggregate_verdict, reliability
from .report import AgentCaseResult, AgentEvalResult
from .runner import (
    AgentEvalRunner,
    AgentEvalSpec,
    case_from_episode,
    eligibility,
    rescore,
)
from .sidecar import read_sidecar, sidecar_records, write_sidecar
from .types import detect_secrets
from .lexsi import (
    SourceEvidence,
    JoinResult,
    join_evidence,
    lineage_sidecar,
    aligntune_audit_report,
    aligntune_regression_report,
    safetune_audit_event,
    safetune_bench_row,
    curatorkit_data_sample,
    curatorkit_manifest,
    curatorkit_trainer_export_row,
    circuitkit_faithfulness_report,
)
from .types import (
    INFERRED,
    OBSERVED,
    SCHEMA_VERSION,
    UNAVAILABLE,
    AgentCase,
    AgentEpisode,
    AgentEvent,
    redact,
    validate_cases,
)

__all__ = [
    "SCHEMA_VERSION",
    "OBSERVED",
    "INFERRED",
    "UNAVAILABLE",
    "AgentCase",
    "AgentEvent",
    "AgentEpisode",
    "validate_cases",
    "Outcome",
    "Oracle",
    "FinalStateAssertion",
    "ArtifactAssertion",
    "AnswerAssertion",
    "AssertionOracle",
    "CustomPredicate",
    "judge_outcome",
    "resolve_outcome",
    "outcome_from_spec",
    "SUCCESS",
    "FAILURE",
    "UNKNOWN",
    "ERROR",
    "episodes_from_agenttune",
    "episode_from_eventlog",
    "episode_from_openai_messages",
    "episode_from_sample",
    "SourceEvidence",
    "JoinResult",
    "join_evidence",
    "lineage_sidecar",
    "aligntune_audit_report",
    "aligntune_regression_report",
    "safetune_audit_event",
    "safetune_bench_row",
    "curatorkit_data_sample",
    "curatorkit_manifest",
    "curatorkit_trainer_export_row",
    "circuitkit_faithfulness_report",
    "inspect_agenttune",
    "supported_scorers_from_coverage",
    "AgentEvalSpec",
    "AgentEvalRunner",
    "AgentEvalResult",
    "AgentCaseResult",
    "case_from_episode",
    "eligibility",
    "rescore",
    "reliability",
    "aggregate_verdict",
    "sidecar_records",
    "write_sidecar",
    "read_sidecar",
    "redact",
    "detect_secrets",
]
