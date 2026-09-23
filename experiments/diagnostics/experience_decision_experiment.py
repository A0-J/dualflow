"""B7d-v3 main experiment (PREPARED, NOT YET RUN): does structured
historical evidence, consumed via FrozenCandidateEvidenceHarness, improve
an ambiguous semantic decision while never overriding an explicit current
instruction?

    python experiments/diagnostics/experience_decision_experiment.py \\
        --scenario external_audit_finance --model gpt-4o-mini-2024-07-18 \\
        --samples 20 --runs 5

This is deliberately NOT the N/L/P/S natural-language-priming comparison
(B7d.5/B7d.6, docs/experiments/agent_connected_eval.md SS20-22) run again.
That question (does history-block presence/format prime the model
independent of content) is already answered -- re-running it would add no
new information now that HistoricalEvidenceComparator is fully
deterministic (summarize==summarize -> SUPPORT, checked hundreds of times,
teaches nothing new). This experiment asks a different, downstream
question: once evidence relations exist, does actually CONSUMING them
(via experience_decision.FrozenCandidateEvidenceHarness, SS23 follow-ups
1-5's provenance chain) produce a useful semantic decision -- improving an
ambiguous case while preserving an explicit one?

Two hypotheses:
  H1 (ambiguous transfer): when the current delegation's action is
     genuinely ambiguous, does Principal-confirmed historical action
     evidence move the semantic decision toward the historically
     confirmed action?
  H2 (explicit-change preservation): when the current delegation
     explicitly specifies a DIFFERENT action, does historical evidence
     fail to override it?

Design -- paired, not independent-per-condition (critical difference from
B7d.5/B7d.6): candidate generation and the v3 intervention are fully
separated. For each of the two current tasks, candidates are sampled
EXACTLY ONCE per run (no history in the generation prompt at all -- the
same clean path B7a/B6 always used), then that SAME frozen distribution is
reused for both the baseline and the v3-intervention condition. This is
what removes B7d.5/B7d.6's history-block presence/format priming from the
comparison entirely: the only thing that can differ between a baseline and
its v3 pair is what FrozenCandidateEvidenceHarness.decide() does with an
already-fixed distribution, never a difference in what got generated.

Four conditions (2 tasks x {baseline, v3}), NOT expanded further this
round (no export-history/read-history arms -- SS20-22 already
characterized those under the old natural-language mechanism; re-adding
them here would blur what this experiment is isolating):

    A. Ambiguous / Baseline       -- no historical evidence consulted
    B. Ambiguous / v3             -- same frozen candidates as A
    C. Explicit export / Baseline -- no historical evidence consulted
    D. Explicit export / v3       -- same frozen candidates as C

Historical evidence used for B and D (identical, fixed, reused across all
runs -- exactly one verified experience, per the established B7c/B7d
convention): confirmed_facets={"action"}, confirmed_interpretation.action
="summarize" -- built from experience_transfer.make_experience() (fixed
reproducer, imported unchanged) and only its `confirmed_facets` field
overlaid via dataclasses.replace() (make_experience() itself defaults that
field to frozenset(), since it predates the v3 provenance work and is not
modified here).

Primary metrics (entropy is explicitly secondary, per instruction):
  H1: P(final decision == "summarize") for baseline (A) vs. v3 (B), paired
      per run; count of runs where v3 actually consulted evidence, resolved
      by evidence, or fell back (no eligible evidence / no support).
  H2: P(final decision == "export") for baseline (C) vs. v3 (D), paired per
      run; count of any historical-override occurrence (D's final value !=
      "export" -- this would be an F4 failure and is flagged loudly, not
      averaged away).

API call budget (comparator makes ZERO calls -- fully deterministic):
  candidate generation only: 2 tasks x N samples x runs repetitions.
  Default (N=20, runs=5): 200 real API calls total.

Failure taxonomy (fixed before running, not created after seeing results
-- see docs/experiments/agent_connected_eval.md's v3 main-experiment
protocol section):
  F1 evidence unavailable        -- eligible historical facet absent
  F2 support absent              -- eligible history exists, no current
                                     candidate matches it
  F3 wrong transfer              -- ambiguous case, evidence applied,
                                     resulting decision is semantically
                                     worse than baseline (interpretive,
                                     assessed when analyzing results, not
                                     a stage this script emits directly)
  F4 stale-history override      -- explicit current instruction is
                                     overridden by historical evidence
  F5 cross-facet amplification   -- action evidence changes an unconfirmed
                                     facet's decision (structurally
                                     prevented by FrozenCandidateEvidence
                                     Harness's design -- see
                                     src/dualflow/experience_decision.py
                                     and its regression tests)

This script makes REAL, BILLED API calls for candidate generation only --
see experiments/agent_smoke.py for the OPENAI_API_KEY / `pip install -e
".[agent]"` requirements, which this script shares (build_llm_client() is
imported from there via experience_transfer.py, not duplicated).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAGNOSTICS_DIR))
from experience_representation import TASK2_DELEGATION  # noqa: E402
from experience_transfer import (  # noqa: E402
    build_current_context, load_scenario, make_experience, scenario_fingerprint, text_sha256,
)

_EXPERIMENTS_DIR = _DIAGNOSTICS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import build_llm_client  # noqa: E402

from dualflow.delegate_agent import CandidateDistribution, DelegateAgent  # noqa: E402
from dualflow.experience_decision import FrozenCandidateEvidenceHarness  # noqa: E402
from dualflow.semantic import Interpretation  # noqa: E402


def build_historical_evidence(scenario: dict):
    """정확히 하나의 verified experience, action facet만 confirmed로
    표시한다. `make_experience()`(fixed reproducer, 무수정)를 그대로
    호출하고 `confirmed_facets`만 `dataclasses.replace()`로 덧씌운다 --
    `make_experience()`는 v3 provenance 이전에 만들어진 함수라
    `confirmed_facets`를 기본값(빈 frozenset)으로 두므로, 이 실험에서만
    필요한 provenance를 별도로 얹는다."""
    exp = make_experience(scenario, "summarize")
    return dataclasses.replace(exp, confirmed_facets=frozenset({"action"}))


def _belief_summary(distribution: CandidateDistribution) -> dict[str, float]:
    by_action: dict[str, float] = {}
    for interp, p in distribution.belief.items():
        by_action[interp.action] = by_action.get(interp.action, 0.0) + p
    return by_action


def run_condition_pair(*, task_label: str, delegation: str, llm_client, context: str,
                       n: int, run: int, experience, harness: FrozenCandidateEvidenceHarness,
                       ) -> tuple[dict, dict]:
    """한 번의 candidate freeze로 baseline/v3 두 row를 같이 만든다 --
    v3 조건 때문에 candidate API를 다시 부르지 않는다(paired design의
    핵심)."""
    delegate = DelegateAgent(llm=llm_client)
    distribution = delegate.sample_candidates(delegation=delegation, context=context, n=n)

    baseline_row = {
        "task": task_label, "condition": "baseline", "run": run,
        "belief_by_action": _belief_summary(distribution),
        "entropy": distribution.entropy,
        "baseline_decision": distribution.top.action,
        "final_decision": distribution.top.action,  # baseline never consults evidence
        "used_historical_evidence": False,
        "stage": "baseline_no_evidence_consulted",
        "n_samples": distribution.n_samples,
        "n_llm_calls": len(distribution.responses),
        "input_tokens": sum(r.input_tokens or 0 for r in distribution.responses),
        "output_tokens": sum(r.output_tokens or 0 for r in distribution.responses),
    }

    decision = harness.decide(distribution=distribution, experience=experience, facet="action")
    v3_row = {
        "task": task_label, "condition": "v3", "run": run,
        "belief_by_action": _belief_summary(distribution),
        "entropy": distribution.entropy,
        "baseline_decision": decision.baseline_value,
        "final_decision": decision.final_value,
        "resolved_value": decision.resolved_value,
        "used_historical_evidence": decision.used_historical_evidence,
        "stage": decision.stage,
        "relations": {k: v.value for k, v in decision.relations.items()},
        # v3 row makes ZERO additional API calls -- same distribution as baseline_row.
        "n_samples": distribution.n_samples,
        "n_llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    return baseline_row, v3_row


def summarize(rows: list[dict]) -> dict:
    def rows_for(task: str, condition: str) -> list[dict]:
        return [r for r in rows if r["task"] == task and r["condition"] == condition]

    def rate(rows_: list[dict], target_action: str) -> float:
        if not rows_:
            return float("nan")
        return sum(1 for r in rows_ if r["final_decision"] == target_action) / len(rows_)

    ambiguous_baseline = rows_for("ambiguous", "baseline")
    ambiguous_v3 = rows_for("ambiguous", "v3")
    explicit_baseline = rows_for("explicit_change", "baseline")
    explicit_v3 = rows_for("explicit_change", "v3")

    h4_overrides = [r for r in explicit_v3 if r["final_decision"] != "export"]

    return {
        "H1_ambiguous_transfer": {
            "P(summarize | baseline)": rate(ambiguous_baseline, "summarize"),
            "P(summarize | v3)": rate(ambiguous_v3, "summarize"),
            "runs_resolved_by_evidence": sum(
                1 for r in ambiguous_v3 if r["stage"] == "ambiguous_resolved_by_evidence"),
            "runs_no_support": sum(1 for r in ambiguous_v3 if r["stage"] == "ambiguous_no_support"),
            "runs_no_eligible_evidence": sum(
                1 for r in ambiguous_v3 if r["stage"] == "ambiguous_no_eligible_evidence"),
            "runs_stable_baseline": sum(1 for r in ambiguous_v3 if r["stage"] == "stable_baseline"),
            "n_runs": len(ambiguous_v3),
        },
        "H2_explicit_change_preservation": {
            "P(export | baseline)": rate(explicit_baseline, "export"),
            "P(export | v3)": rate(explicit_v3, "export"),
            "historical_override_count_F4": len(h4_overrides),
            "override_run_ids": [r["run"] for r in h4_overrides],
            "n_runs": len(explicit_v3),
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:  # Windows 기본 콘솔(cp949 등)의 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="external_audit_finance")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--samples", type=int, default=20, help="N per condition per run (candidate freeze)")
    p.add_argument("--runs", type=int, default=5, help="independent repetitions")
    p.add_argument("--output", default=None,
                   help="path to save machine-readable JSON result (not committed by default)")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario)
    context = build_current_context(scenario)
    llm_client = build_llm_client(args.model)
    experience = build_historical_evidence(scenario)
    harness = FrozenCandidateEvidenceHarness(entropy_threshold=0.8)

    fp = scenario_fingerprint(scenario)
    explicit_sha = text_sha256(TASK2_DELEGATION)
    total_calls = 2 * args.samples * args.runs
    print("=== B7d-v3 main experiment (paired frozen-candidate design) ===")
    print(f"model: {args.model}")
    print(f"scenario: {scenario['scenario_id']} v{scenario.get('scenario_version', '?')}")
    print(f"ambiguous delegation SHA256: {fp['delegation_sha256']}")
    print(f"explicit-change delegation SHA256: {explicit_sha}")
    print(f"context SHA256: {fp['context_sha256']}")
    print(f"historical experience: confirmed_facets={sorted(experience.confirmed_facets)} "
         f"confirmed action={experience.confirmed_interpretation.action!r} "
         f"confirmed scope={experience.confirmed_interpretation.scope!r}")
    print(f"N={args.samples}  runs={args.runs}  -> {total_calls} candidate-generation calls "
         f"(comparator/v3 decision: 0 additional calls, deterministic)\n")

    tasks = [
        ("ambiguous", scenario["current_episode"]["delegation"]),
        ("explicit_change", TASK2_DELEGATION),
    ]

    all_rows: list[dict] = []
    for run in range(1, args.runs + 1):
        for task_label, delegation in tasks:
            baseline_row, v3_row = run_condition_pair(
                task_label=task_label, delegation=delegation, llm_client=llm_client,
                context=context, n=args.samples, run=run, experience=experience, harness=harness)
            all_rows.append(baseline_row)
            all_rows.append(v3_row)
            print(f"[run {run}] {task_label:16s} baseline={baseline_row['final_decision']:10s} "
                 f"v3={v3_row['final_decision']:10s} stage={v3_row['stage']}", flush=True)

    print()
    summary = summarize(all_rows)
    print("=== Summary ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        payload = {
            "identity": {
                "scenario_id": scenario["scenario_id"],
                "scenario_version": scenario.get("scenario_version"),
                "model": args.model,
                "ambiguous_delegation_sha256": fp["delegation_sha256"],
                "explicit_change_delegation_sha256": explicit_sha,
                "context_sha256": fp["context_sha256"],
                "samples_per_condition": args.samples,
                "repetitions": args.runs,
                "historical_confirmed_facets": sorted(experience.confirmed_facets),
                "historical_confirmed_action": experience.confirmed_interpretation.action,
            },
            "rows": all_rows,
            "summary": summary,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {len(all_rows)} rows + summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
