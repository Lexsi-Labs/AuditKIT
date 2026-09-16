"""Red team evaluation runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..model import AutoModel, Model, Request
from .detector import Detector
from .detectors.builtin import InjectionSuccessDetector, KeywordDetector, RefusalDetector, SystemPromptLeakDetector
from .probe import Probe, ProbeResult
from .probes.builtin import EncodingProbe, JailbreakProbe, PromptInjectionProbe, RefusalProbe


_PROBE_REGISTRY: dict[str, type[Probe]] = {
    "prompt_injection": PromptInjectionProbe,
    "jailbreak": JailbreakProbe,
    "encoding": EncodingProbe,
    "refusal": RefusalProbe,
}

_DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "keyword": KeywordDetector,
    "refusal": RefusalDetector,
    "system_prompt_leak": SystemPromptLeakDetector,
    "injection_success": InjectionSuccessDetector,
}

_PROBE_DETECTOR_MAP: dict[str, list[str]] = {
    "prompt_injection": ["keyword", "injection_success"],
    "jailbreak": ["keyword", "injection_success"],
    "encoding": ["keyword", "injection_success"],
    "refusal": ["refusal"],
}


@dataclass
class RedTeamResult:
    """Results from a red team evaluation run."""

    results: list[ProbeResult]
    model_name: str
    probes_used: list[str]
    detectors_used: list[str]

    @property
    def total_probes(self) -> int:
        return len(self.results)

    @property
    def passed_probes(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_probes(self) -> int:
        return self.total_probes - self.passed_probes

    @property
    def pass_rate(self) -> float:
        if self.total_probes == 0:
            return 1.0
        return self.passed_probes / self.total_probes

    def summary(self) -> str:
        lines = [
            f"Red Team Results — model: {self.model_name}",
            f"  Total probes: {self.total_probes}",
            f"  Passed:       {self.passed_probes}",
            f"  Failed:       {self.failed_probes}",
            f"  Pass rate:    {self.pass_rate:.1%}",
            "",
            "Failed probes:",
        ]
        for r in self.results:
            if not r.passed:
                lines.append(f"  [{r.probe_name}/{r.detector_name}] {r.prompt[:60]}")
        if self.failed_probes == 0:
            lines.append("  (none)")
        return "\n".join(lines)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Push probe rows to a Hub dataset and stamp the AuditKIT card."""
        from ..hf_publish import push_rows_to_hub

        rows = [
            {
                "probe": r.probe_name,
                "detector": r.detector_name,
                "prompt": r.prompt,
                "output": r.output,
                "passed": r.passed,
            }
            for r in self.results
        ]
        return push_rows_to_hub(
            rows,
            repo_id,
            private=private,
            token=token,
            kind="redteam",
            method="redteam",
            model=str(self.model_name),
        )


class RedTeamRunner:
    """Orchestrates red team evaluation of a model using probes and detectors."""

    def __init__(
        self,
        model: Model | Callable[[list[str]], list[str]] | str,
    ) -> None:
        self._model = AutoModel.resolve(model)

    def run(
        self,
        probes: list[str | Probe] | None = None,
        detectors: list[str | Detector] | None = None,
    ) -> RedTeamResult:
        """Run red team evaluation.

        Args:
            probes: List of probe names or Probe instances. Defaults to all built-in probes.
            detectors: List of detector names or Detector instances. Defaults to auto-selected.

        Returns:
            A RedTeamResult with per-probe results.
        """
        resolved_probes = self._resolve_probes(probes)
        resolved_detectors = self._resolve_detectors(detectors, resolved_probes)

        results: list[ProbeResult] = []

        for probe in resolved_probes:
            prompts = probe.generate()
            requests = [Request(prompt=p) for p in prompts]
            results_list = self._model.generate(requests)

            for prompt, result in zip(prompts, results_list):
                output_str = result.text
                probe_results: list[ProbeResult] = []

                for detector in resolved_detectors:
                    det_result = detector.detect(prompt, output_str)
                    pr = ProbeResult(
                        probe_name=probe.name,
                        prompt=prompt,
                        output=output_str,
                        passed=not det_result.detected,
                        detector_name=detector.name,
                        metadata={"confidence": det_result.confidence, "details": det_result.details},
                    )
                    probe_results.append(pr)
                    results.append(pr)

        return RedTeamResult(
            results=results,
            model_name=str(self._model),
            probes_used=[p.name for p in resolved_probes],
            detectors_used=[d.name for d in resolved_detectors],
        )

    def _resolve_probes(self, probes: list[str | Probe] | None) -> list[Probe]:
        if probes is None:
            return [cls() for cls in _PROBE_REGISTRY.values()]
        resolved = []
        for p in probes:
            if isinstance(p, Probe):
                resolved.append(p)
            elif isinstance(p, str):
                cls = _PROBE_REGISTRY.get(p)
                if cls is None:
                    raise ValueError(f"Unknown probe: {p}. Available: {list(_PROBE_REGISTRY.keys())}")
                resolved.append(cls())
            else:
                raise TypeError(f"Expected Probe or str, got {type(p)}")
        return resolved

    def _resolve_detectors(
        self, detectors: list[str | Detector] | None, probes: list[Probe]
    ) -> list[Detector]:
        if detectors is not None:
            resolved = []
            for d in detectors:
                if isinstance(d, Detector):
                    resolved.append(d)
                elif isinstance(d, str):
                    cls = _DETECTOR_REGISTRY.get(d)
                    if cls is None:
                        raise ValueError(f"Unknown detector: {d}. Available: {list(_DETECTOR_REGISTRY.keys())}")
                    resolved.append(cls())
                else:
                    raise TypeError(f"Expected Detector or str, got {type(d)}")
            return resolved

        used: set[str] = set()
        for probe in probes:
            for det_name in _PROBE_DETECTOR_MAP.get(probe.name, []):
                used.add(det_name)
        if not used:
            used = {"keyword"}
        return [_DETECTOR_REGISTRY[name]() for name in used]
