"""B7d.4 real-API validation: does v2's historical CONTENT matter, or just
its format?

    python experiments/diagnostics/experience_semantic_ablation.py \\
        --scenario external_audit_finance --samples 20 --runs 10

A NEW script, not an extension of experience_transfer.py (the fixed
B7d.1/B7d.2 reproducer) or experience_representation.py (the fixed B7d.3
reproducer) -- neither of those files is modified by this script. Each
diagnostic in this directory answers one fixed question and stays a stable
reproducer once recorded; this one answers the question the canonical
B7d.3 rerun left open (docs/experiments/agent_connected_eval.md SS16):

B7d.3's canonical rerun found that v2 (render_experience_block_v2) shows
only a weak, run-inconsistent P(summarize) lift on the ambiguous task
versus a true (non-deterministic) no-experience baseline. That result
alone cannot say WHY: it could be that v2's format is doing genuine
content-sensitive semantic transfer that is just weak in magnitude, or it
could be that v2, like v1 before it (B7d.1/B7d.2), is dominated by
format/priming effects that are largely independent of what the historical
experience actually confirmed.

This script isolates that question with a content-only ablation. Same
canonical ambiguous September delegation, same v2 renderer, same retrieval
path (ExperienceAwareDelegate) -- the ONLY thing that varies between
conditions B and C is the confirmed action inside the stored experience:

    A. no experience
    B. v2 rendering of a "summarize"-confirmed historical experience
    C. v2 rendering of an "export"-confirmed historical experience

B and C are built from the same shared answer template
(experience_transfer.make_experience(), the wording-symmetry fix from
B7d.2/agent_connected_eval.md SS10) so they stay structurally identical --
same sentence shape, same length, same header -- differing only in the one
confirmed-action word. If v2 is doing real semantic-content transfer,
B should push toward summarize and C should push toward export,
i.e. both of:

    delta_sum    = P(summarize | B) - P(summarize | C)   should be > 0
    delta_export = P(export    | C) - P(export    | B)   should be > 0

If B and C instead behave similarly to each other (regardless of which
action was "confirmed"), that would mean v2's effect -- like v1's -- is not
selectively carrying the historical experience's semantic content, and the
B7d.3 canonical lift cannot yet be attributed to genuine semantic transfer.

Reuses load_scenario()/build_current_context()/make_experience()/
text_sha256()/scenario_fingerprint() from experience_transfer.py (sibling
import), and render_experience_block_v2 from
src/dualflow/experience_aware_delegate.py -- nothing in either module is
modified here.

Entropy is NOT the primary success criterion here -- P(summarize)/P(export)
conditioned on which history was shown are. Entropy is still recorded per
row (secondary), consistent with the established B7d methodology
(docs/experiments/agent_connected_eval.md SS11).

This script makes REAL, BILLED API calls -- see experiments/agent_smoke.py
for the OPENAI_API_KEY / `pip install -e ".[agent]"` requirements, which
this script shares (build_llm_client() is imported from there via
experience_transfer.py, not duplicated).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAGNOSTICS_DIR))
from experience_transfer import (  # noqa: E402
    build_current_context, load_scenario, make_experience, scenario_fingerprint, text_sha256,
)

_EXPERIMENTS_DIR = _DIAGNOSTICS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import build_llm_client  # noqa: E402

from dualflow.agent_experience import AgentExperienceStore  # noqa: E402
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent  # noqa: E402
from dualflow.experience_aware_delegate import ExperienceAwareDelegate, render_experience_block_v2  # noqa: E402
from dualflow.semantic import Interpretation  # noqa: E402

CONDITIONS = ("no_experience", "v2_summarize_history", "v2_export_history")


def extract_row(condition: str, run: int, experience_count: int,
                dist: CandidateDistribution, scenario: dict) -> dict:
    cur = scenario["current_episode"]
    p_by_action = {
        action: dist.belief.get(Interpretation(action, cur["resource"], cur["scope"], frozenset()), 0.0)
        for action in scenario["action_vocabulary"]
    }
    return {
        "condition": condition, "run": run, "experience_count": experience_count,
        "belief": {f"{i.action}:{i.resource}@{i.scope}": p for i, p in dist.belief.items()},
        "entropy": dist.entropy, "top1": dist.top.action, "p_by_action": p_by_action,
        "scopes_seen": sorted({i.scope for i in dist.belief}),
        "n_samples": dist.n_samples, "n_llm_calls": len(dist.responses),
        "input_tokens": sum(r.input_tokens or 0 for r in dist.responses),
        "output_tokens": sum(r.output_tokens or 0 for r in dist.responses),
    }


def run_condition(condition: str, scenario: dict, llm_client, context: str, n: int, run: int) -> dict:
    delegation = scenario["current_episode"]["delegation"]
    delegate = DelegateAgent(llm=llm_client)
    store = AgentExperienceStore()

    if condition == "no_experience":
        wrapped = ExperienceAwareDelegate(delegate, store)  # empty store -> no block either way
    elif condition == "v2_summarize_history":
        store.add(make_experience(scenario, "summarize"))
        wrapped = ExperienceAwareDelegate(delegate, store, render_experience_block=render_experience_block_v2)
    elif condition == "v2_export_history":
        store.add(make_experience(scenario, "export"))
        wrapped = ExperienceAwareDelegate(delegate, store, render_experience_block=render_experience_block_v2)
    else:
        raise ValueError(f"unknown condition: {condition!r}")

    result = wrapped.sample_candidates(
        principal_id=scenario["principal_id"], task_category=scenario["task_category"],
        delegation=delegation, context=context, n=n)
    return extract_row(condition, run, result.experience_count, result.distribution, scenario)


def summarize_ablation(all_rows: list[dict]) -> dict:
    """Aggregate P(summarize)/P(export) per condition (mean over runs), and
    the two delta metrics the B7d.4 interpretation rule is based on."""
    def mean_p(condition: str, action: str) -> float:
        vals = [row["p_by_action"][action] for row in all_rows if row["condition"] == condition]
        return sum(vals) / len(vals) if vals else float("nan")

    p_sum_given_sum_hist = mean_p("v2_summarize_history", "summarize")
    p_sum_given_exp_hist = mean_p("v2_export_history", "summarize")
    p_exp_given_sum_hist = mean_p("v2_summarize_history", "export")
    p_exp_given_exp_hist = mean_p("v2_export_history", "export")

    return {
        "P(summarize | summarize-history)": p_sum_given_sum_hist,
        "P(summarize | export-history)": p_sum_given_exp_hist,
        "P(export | summarize-history)": p_exp_given_sum_hist,
        "P(export | export-history)": p_exp_given_exp_hist,
        "delta_sum": p_sum_given_sum_hist - p_sum_given_exp_hist,
        "delta_export": p_exp_given_exp_hist - p_exp_given_sum_hist,
        "P(summarize | no_experience)": mean_p("no_experience", "summarize"),
        "P(export | no_experience)": mean_p("no_experience", "export"),
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
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--samples", type=int, default=20, help="N per condition per run")
    p.add_argument("--runs", type=int, default=10, help="independent repetitions")
    p.add_argument("--output", default=None,
                   help="optional path to save raw per-run rows as JSON (not committed by default)")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario)
    context = build_current_context(scenario)
    llm_client = build_llm_client(args.model)

    fp = scenario_fingerprint(scenario)
    total_calls = len(CONDITIONS) * args.samples * args.runs
    print(f"Scenario: {scenario['scenario_id']} v{scenario.get('scenario_version', '?')}")
    print(f"Delegation SHA256: {fp['delegation_sha256']}")
    print(f"Context SHA256: {fp['context_sha256']}")
    print(f"Conditions: {list(CONDITIONS)}  N={args.samples}  runs={args.runs}  "
         f"-> {total_calls} sampling calls\n")

    all_rows = []
    for run in range(1, args.runs + 1):
        for condition in CONDITIONS:
            row = run_condition(condition, scenario, llm_client, context, args.samples, run)
            all_rows.append(row)
            p_str = " ".join(f"P({a})={row['p_by_action'][a]:.2f}"
                             for a in scenario["action_vocabulary"])
            print(f"[run {run}] {condition:22s} H={row['entropy']:.3f} "
                 f"top1={row['top1']:10s} {p_str} scopes={row['scopes_seen']} "
                 f"calls={row['n_llm_calls']} tok_in={row['input_tokens']} "
                 f"tok_out={row['output_tokens']}", flush=True)

    print()
    agg = summarize_ablation(all_rows)
    for key, value in agg.items():
        print(f"{key}: {value:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"rows": all_rows, "aggregate": agg}, f, indent=2)
        print(f"\nSaved {len(all_rows)} rows + aggregate to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
