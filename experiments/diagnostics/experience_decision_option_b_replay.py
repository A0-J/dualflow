"""B7d-v3 Option B replay (NO API CALLS): re-evaluates the ALREADY-FROZEN
real-API candidate distributions from the B7d-v3 main experiment
(docs/experiments/agent_connected_eval.md §24, 200 real API calls, commit
79ce2cc) against the current (Option B, conservative-abstention)
`FrozenCandidateEvidenceHarness` contract (commit 0f1cf7c).

This is NOT a new experiment. It makes zero LLM calls and samples zero new
candidates. It is a deterministic re-evaluation of data that was already
generated and already committed to the historical record under §24 -- the
only thing that changes is which version of `decide()`'s consumption logic
processes it. The original REVISION-1 (automatic-resolution) results in
§24 and the raw JSON they came from are not modified or deleted; this
script's output is a second, clearly labeled column standing alongside
them.

Source data: the 5-run x {ambiguous, explicit_change} x {baseline, v3}
frozen distributions recorded verbatim in the original experiment's
machine-readable output (`belief_by_action`, `entropy`, `baseline_decision`
per row). Reconstructing a `CandidateDistribution` from these fields is
sufficient here because `FrozenCandidateEvidenceHarness.decide(...,
facet="action")` only ever inspects the `.action` attribute of each
candidate `Interpretation` (via `HistoricalEvidenceComparator`/
`rule_engine.field_match`) -- resource/scope/condition are structurally
inert to this facet's decision, so a synthetic per-action-value
`Interpretation` with placeholder resource/scope reproduces byte-identical
stage/final_value/relations output to what the real per-candidate
distribution would have produced.

    python experiments/diagnostics/experience_decision_option_b_replay.py

Historical evidence reused unmodified from the original experiment's
construction path (`experience_decision_experiment.build_historical_
evidence()`, itself built from `experience_transfer.make_experience()` --
both fixed reproducers, imported not copied): confirmed_facets={"action"},
confirmed action="summarize".
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAGNOSTICS_DIR))
from experience_decision_experiment import build_historical_evidence  # noqa: E402
from experience_transfer import load_scenario  # noqa: E402

from dualflow.delegate_agent import CandidateDistribution  # noqa: E402
from dualflow.experience_decision import FrozenCandidateEvidenceHarness  # noqa: E402
from dualflow.semantic import Interpretation  # noqa: E402

# Verbatim from the frozen v3 main-experiment output (scratchpad
# v3_main_results.json, identical scenario/model/samples/repetitions
# identity recorded at the top of that file and in §24). Only the fields
# `decide()` actually consumes (belief_by_action, entropy, baseline_
# decision) are carried over -- token/call counts are provenance for the
# ORIGINAL experiment, not reproduced here since this replay makes no
# calls at all.
FROZEN_AMBIGUOUS_RUNS = [
    {"run": 1, "belief_by_action": {"export": 0.80, "summarize": 0.20}, "entropy": 0.7219280948873623, "baseline_decision": "export"},
    {"run": 2, "belief_by_action": {"summarize": 0.25, "export": 0.75}, "entropy": 0.8112781244591328, "baseline_decision": "export"},
    {"run": 3, "belief_by_action": {"export": 0.90, "summarize": 0.10}, "entropy": 0.4689955935892812, "baseline_decision": "export"},
    {"run": 4, "belief_by_action": {"export": 0.65, "summarize": 0.35}, "entropy": 0.934068055375491, "baseline_decision": "export"},
    {"run": 5, "belief_by_action": {"export": 0.90, "summarize": 0.10}, "entropy": 0.4689955935892812, "baseline_decision": "export"},
]

FROZEN_EXPLICIT_RUNS = [
    {"run": 1, "belief_by_action": {"export": 1.0}, "entropy": 0.0, "baseline_decision": "export"},
    {"run": 2, "belief_by_action": {"export": 1.0}, "entropy": 0.0, "baseline_decision": "export"},
    {"run": 3, "belief_by_action": {"export": 1.0}, "entropy": 0.0, "baseline_decision": "export"},
    {"run": 4, "belief_by_action": {"export": 1.0}, "entropy": 0.0, "baseline_decision": "export"},
    {"run": 5, "belief_by_action": {"export": 1.0}, "entropy": 0.0, "baseline_decision": "export"},
]

# Original (REVISION-1, automatic-resolution) stage/final_decision, exactly
# as recorded in §24's already-committed raw output -- reproduced here only
# so the printed table can show old vs. new side by side without re-reading
# the scratchpad file (which is not part of the repo).
ORIGINAL_AMBIGUOUS_STAGE = {
    1: ("stable_baseline", "export"),
    2: ("ambiguous_resolved_by_evidence", "summarize"),
    3: ("stable_baseline", "export"),
    4: ("ambiguous_resolved_by_evidence", "summarize"),
    5: ("stable_baseline", "export"),
}
ORIGINAL_EXPLICIT_STAGE = {n: ("stable_baseline", "export") for n in range(1, 6)}


def _reconstruct_distribution(belief_by_action: dict[str, float], entropy_value: float,
                              baseline_decision: str, scope: str) -> CandidateDistribution:
    belief: dict[Interpretation, float] = {}
    for action, prob in belief_by_action.items():
        belief[Interpretation(action, "file", scope, frozenset())] = prob
    top = next(i for i in belief if i.action == baseline_decision)
    return CandidateDistribution(belief=belief, entropy=entropy_value, top=top,
                                 top_probability=belief[top], n_unique=len(belief),
                                 n_samples=20, responses=[])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    scenario = load_scenario("external_audit_finance")
    experience = build_historical_evidence(scenario)  # confirmed_facets={"action"}, action="summarize"
    harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)  # unmodified, current contract

    print("=== B7d-v3 Option B replay (deterministic, 0 API calls) ===")
    print("Source: frozen real-API distributions from the B7d-v3 main experiment")
    print("(docs/experiments/agent_connected_eval.md SS24, git 79ce2cc, 200 real calls).")
    print("Re-evaluated against the current harness contract (git 0f1cf7c).\n")

    print(f"{'task':16s} {'run':4s} {'entropy':9s} {'baseline':10s} "
         f"{'ORIGINAL stage':30s} {'ORIGINAL final':10s}   "
         f"{'OPTION-B stage':34s} {'OPTION-B final':10s}")

    ambiguous_new_stages = []
    for row in FROZEN_AMBIGUOUS_RUNS:
        dist = _reconstruct_distribution(row["belief_by_action"], row["entropy"],
                                         row["baseline_decision"], scope="/reports/2026-09/")
        decision = harness.decide(distribution=dist, experience=experience, facet="action")
        ambiguous_new_stages.append(decision.stage)
        orig_stage, orig_final = ORIGINAL_AMBIGUOUS_STAGE[row["run"]]
        print(f"{'ambiguous':16s} {row['run']:<4d} {row['entropy']:<9.4f} "
             f"{row['baseline_decision']:10s} {orig_stage:30s} {orig_final:10s}   "
             f"{decision.stage:34s} {decision.final_value:10s}")
        assert decision.final_value == row["baseline_decision"], (
            "Option B must never change final_value away from baseline -- "
            "regression in the replay itself, not the harness.")

    explicit_new_stages = []
    for row in FROZEN_EXPLICIT_RUNS:
        dist = _reconstruct_distribution(row["belief_by_action"], row["entropy"],
                                         row["baseline_decision"], scope="/reports/2026-09/")
        decision = harness.decide(distribution=dist, experience=experience, facet="action")
        explicit_new_stages.append(decision.stage)
        orig_stage, orig_final = ORIGINAL_EXPLICIT_STAGE[row["run"]]
        print(f"{'explicit_change':16s} {row['run']:<4d} {row['entropy']:<9.4f} "
             f"{row['baseline_decision']:10s} {orig_stage:30s} {orig_final:10s}   "
             f"{decision.stage:34s} {decision.final_value:10s}")
        assert decision.final_value == "export"

    n_amb = len(ambiguous_new_stages)
    n_requires_clarification = sum(1 for s in ambiguous_new_stages
                                   if s == "historical_evidence_requires_clarification")
    n_stable = sum(1 for s in ambiguous_new_stages if s == "stable_baseline")
    n_override_ambiguous = 0  # by construction: Option B never overrides
    n_explicit = len(explicit_new_stages)
    n_stable_explicit = sum(1 for s in explicit_new_stages if s == "stable_baseline")
    n_override_explicit = 0  # by construction

    print("\n=== Option B replay summary (final v3 contract, no API calls) ===")
    print("Ambiguous task:")
    print(f"  history-triggered clarification rate = {n_requires_clarification}/{n_amb}")
    print(f"  automatic historical override rate   = {n_override_ambiguous}/{n_amb}")
    print(f"  stable baseline preservation          = {n_stable}/{n_stable} "
         f"(of the {n_stable} runs the stability gate short-circuited)")
    print("Explicit-change task:")
    print(f"  current decision preservation         = {n_stable_explicit}/{n_explicit}")
    print(f"  automatic historical override         = {n_override_explicit}/{n_explicit}")
    print("\nCompare against the ORIGINAL (REVISION-1, automatic-resolution) contract's")
    print("recorded result on the SAME frozen data (§24): runs_resolved_by_evidence=2/5,")
    print("historical_override_count_F4=0/5 (explicit task sampled stable in all 5 runs,")
    print("so REVISION-1 and Option B agree on the explicit task here -- the divergence")
    print("is entirely in the 2 ambiguous runs that were unstable: REVISION-1 auto-applied")
    print("the historically-supported value; Option B flags them for clarification instead.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
