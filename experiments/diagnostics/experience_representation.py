"""B7d.3 real-API validation: v1 vs. v2 experience representation.

    python experiments/diagnostics/experience_representation.py \\
        --scenario external_audit_finance --samples 20 --runs 5

A separate script from experience_transfer.py on purpose --
experience_transfer.py is now the fixed reproducer for the B7d.1/B7d.2
semantic-transfer diagnostics recorded in
docs/experiments/agent_connected_eval.md SS7/SS10 and should keep
reproducing exactly those results unchanged; this script answers a
different question introduced by B7d.3 (SS14): does the redesigned
experience representation (render_experience_block_v2, added in
src/dualflow/experience_aware_delegate.py) transfer Principal-confirmed
semantics on an AMBIGUOUS current task, while still deferring to an
EXPLICIT current instruction that contradicts the historical experience?

Reuses load_scenario()/build_current_context()/make_experience() from
experience_transfer.py (sibling import) rather than duplicating the
scenario/environment constants -- experience_transfer.py itself is not
modified by this script.

Two current tasks, both against the SAME historical "summarize" verified
experience (built via make_experience(scenario, "summarize")) and the SAME
current environment (RESOURCE=file, SCOPE=/reports/2026-09/):

  Task 1 -- ambiguous: the scenario's own current_episode delegation
    ("Please prepare the September 2026 financial report for the
    external audit.") -- genuinely ambiguous between summarize/export
    (see agent_connected_eval.md SS3). Primary metric: P(summarize).
    Desirable: v2 > no-experience.

  Task 2 -- explicit change: "This time, export the September 2026
    financial report to the external auditor." -- explicitly names
    export, contradicting the historical summarize experience. Primary
    metric: P(export). Desirable: export stays dominant even with v2's
    experience present -- the redesigned representation must not
    override an explicit current instruction.

Three conditions per task: A (no experience), B (v1 rendering), C (v2
rendering). 2 tasks x 3 conditions x N samples x runs repetitions --
default N=20, runs=5 -> 600 real API calls.

Per the B7d.3 evaluation rule (agent_connected_eval.md SS14): entropy is
NOT the primary success signal here. The primary metrics are
P(summarize) for Task 1 and P(export) for Task 2, reported separately;
entropy is reported alongside but secondary.

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
from experience_transfer import build_current_context, load_scenario, make_experience  # noqa: E402

_EXPERIMENTS_DIR = _DIAGNOSTICS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import build_llm_client  # noqa: E402

from dualflow.agent_experience import AgentExperienceStore  # noqa: E402
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent  # noqa: E402
from dualflow.experience_aware_delegate import (  # noqa: E402
    ExperienceAwareDelegate, render_experience_block_v1, render_experience_block_v2,
)
from dualflow.semantic import Interpretation  # noqa: E402

TASK2_DELEGATION = ("This time, export the September 2026 financial report "
                    "to the external auditor.")

RENDERERS = {"v1": render_experience_block_v1, "v2": render_experience_block_v2}


def extract_row(task: str, condition: str, run: int, experience_count: int,
                dist: CandidateDistribution, scenario: dict) -> dict:
    cur = scenario["current_episode"]
    p_by_action = {
        action: dist.belief.get(Interpretation(action, cur["resource"], cur["scope"], frozenset()), 0.0)
        for action in scenario["action_vocabulary"]
    }
    return {
        "task": task, "condition": condition, "run": run,
        "experience_count": experience_count,
        "belief": {f"{i.action}:{i.resource}@{i.scope}": p for i, p in dist.belief.items()},
        "entropy": dist.entropy, "top1": dist.top.action, "p_by_action": p_by_action,
        "scopes_seen": sorted({i.scope for i in dist.belief}),
        "n_samples": dist.n_samples, "n_llm_calls": len(dist.responses),
        "input_tokens": sum(r.input_tokens or 0 for r in dist.responses),
        "output_tokens": sum(r.output_tokens or 0 for r in dist.responses),
    }


def run_task_condition(task_label: str, delegation: str, condition: str, scenario: dict,
                       llm_client, context: str, n: int, run: int) -> dict:
    delegate = DelegateAgent(llm=llm_client)
    store = AgentExperienceStore()

    if condition == "no_experience":
        wrapped = ExperienceAwareDelegate(delegate, store)  # empty store -> no block either way
    else:
        store.add(make_experience(scenario, "summarize"))
        wrapped = ExperienceAwareDelegate(delegate, store,
                                          render_experience_block=RENDERERS[condition])

    result = wrapped.sample_candidates(
        principal_id=scenario["principal_id"], task_category=scenario["task_category"],
        delegation=delegation, context=context, n=n)
    return extract_row(task_label, condition, run, result.experience_count,
                       result.distribution, scenario)


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
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", default=None,
                   help="optional path to save raw per-run rows as JSON (not committed by default)")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario)
    context = build_current_context(scenario)
    llm_client = build_llm_client(args.model)

    tasks = [
        ("ambiguous", scenario["current_episode"]["delegation"], "summarize"),
        ("explicit_change", TASK2_DELEGATION, "export"),
    ]
    conditions = ["no_experience", "v1", "v2"]

    total_calls = len(tasks) * len(conditions) * args.samples * args.runs
    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Tasks: {[t[0] for t in tasks]}  Conditions: {conditions}  "
         f"N={args.samples}  runs={args.runs}  -> {total_calls} sampling calls\n")

    all_rows = []
    for run in range(1, args.runs + 1):
        for task_label, delegation, primary_action in tasks:
            for cond in conditions:
                row = run_task_condition(task_label, delegation, cond, scenario,
                                         llm_client, context, args.samples, run)
                row["primary_action"] = primary_action
                row["p_primary"] = row["p_by_action"][primary_action]
                all_rows.append(row)
                p_str = " ".join(f"P({a})={row['p_by_action'][a]:.2f}"
                                 for a in scenario["action_vocabulary"])
                print(f"[run {run}] {task_label:16s} {cond:12s} H={row['entropy']:.3f} "
                     f"top1={row['top1']:10s} {p_str} scopes={row['scopes_seen']} "
                     f"calls={row['n_llm_calls']} tok_in={row['input_tokens']} "
                     f"tok_out={row['output_tokens']}", flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nSaved {len(all_rows)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
