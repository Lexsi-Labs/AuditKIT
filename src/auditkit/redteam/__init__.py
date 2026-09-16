"""Red teaming for LLMs — adversarial probes and detectors.

Usage::

    import auditkit as ak
    from auditkit.redteam import RedTeamRunner, Probe, Detector

    runner = RedTeamRunner(model="groq:llama-3.3-70b-versatile")
    result = runner.run(probes=[Probe("jailbreak_dan")])
    print(result.summary())
"""

from __future__ import annotations

from .probe import Probe, ProbeResult
from .detector import Detector, DetectorResult
from .runner import RedTeamRunner, RedTeamResult

__all__ = [
    "Probe",
    "ProbeResult",
    "Detector",
    "DetectorResult",
    "RedTeamRunner",
    "RedTeamResult",
]
