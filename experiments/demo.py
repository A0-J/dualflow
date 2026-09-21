"""Minimal end-to-end examples of DualFlow.

This script shows *how to call the DualFlow API*, not how well it performs —
for benchmark numbers see bench.py / bench_single_env_v1.py, for figures see
plots.py, for the real-LLM entropy study see entropy_probe.py.

This script is not part of the DualFlow core package — it uses the installed
package like any other caller would (`pip install -e .` from the repository
root, then `python experiments/demo.py`).
"""

from __future__ import annotations

import sys

from dualflow.capability import Budget, Privilege
from dualflow.framework import Config, DelegationTask, DelegationVerifier
from dualflow.semantic import Interpretation


def build_demo_system() -> DelegationVerifier:
    """Construct a DualFlow verifier with default configuration."""
    return DelegationVerifier(Config())


def run_example(system: DelegationVerifier, task: DelegationTask) -> None:
    """Run one delegation task and print the result."""
    verdict = system.run(task)
    print(f"Request:   {task.spec}")
    print(f"Proposed:  {task.candidates[0][0]}")
    print(f"Result:    {verdict.decision}  ({verdict.reason})")
    print()


def main() -> None:
    try:  # Windows 기본 콘솔(cp949 등)의 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    system = build_demo_system()

    examples = [
        # A benign request: what B proposes is exactly what A meant, and
        # it's within A's delegated authority.
        DelegationTask(
            name="benign",
            category="demo",
            spec="Summarize the August sales report for the marketing team.",
            principal_budget=Budget.of(
                Privilege("summarize", "file", "/reports/2026-08/")),
            ceilings=[None],
            candidates=[(Interpretation("summarize", "file", "/reports/2026-08/"), 1.0)],
            truth=Interpretation("summarize", "file", "/reports/2026-08/"),
        ),
        # A semantic mismatch: B confidently proposes "export" when A meant
        # "summarize". Authority allows it (export is in the budget), so
        # only Joint Verification — comparing against A's actual intent —
        # catches it.
        DelegationTask(
            name="semantic_mismatch",
            category="demo",
            spec="Summarize the report for marketing. Don't export the raw file.",
            principal_budget=Budget.of(
                Privilege("summarize", "file", "/reports/2026-08/"),
                Privilege("export", "file", "/reports/2026-08/")),
            ceilings=[None],
            candidates=[(Interpretation("export", "file", "/reports/2026-08/"), 1.0)],
            truth=Interpretation("summarize", "file", "/reports/2026-08/"),
        ),
        # An authority mismatch: B correctly understands the request, but
        # it was never delegated at all. Authority Flow hard-rejects it
        # regardless of how confident B is.
        DelegationTask(
            name="authority_denied",
            category="demo",
            spec="Delete last quarter's reports, they're no longer needed.",
            principal_budget=Budget.of(
                Privilege("summarize", "file", "/reports/")),
            ceilings=[None],
            candidates=[(Interpretation("delete", "file", "/reports/"), 1.0)],
            truth=Interpretation("delete", "file", "/reports/"),
        ),
    ]

    for task in examples:
        run_example(system, task)


if __name__ == "__main__":
    main()
