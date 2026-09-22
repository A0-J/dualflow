"""B7d.5 real-API validation: does v2's block PRESENCE/FORMAT alone prime
the model's existing prior, independent of historical content?

    python experiments/diagnostics/experience_neutral_control.py \\
        --scenario external_audit_finance --samples 20 --runs 10

A NEW script. Does not modify experience_transfer.py (fixed B7d.1/B7d.2
reproducer), experience_representation.py (fixed B7d.3 reproducer), or
experience_semantic_ablation.py (fixed B7d.4 reproducer) -- none of those
three files is touched here. Also does NOT modify
src/dualflow/experience_aware_delegate.py's render_experience_block_v2 --
the neutral control renderer lives only in this script (see
render_experience_block_v2_neutral below), so the production/research
implementation stays frozen regardless of this experiment's outcome.

Why this experiment: B7d.4 (docs/experiments/agent_connected_eval.md §18)
found delta_sum = delta_export = +0.23, but decomposed that turned out to
be entirely one-sided -- export-history collapsed deterministically to
P(export)=1.00 in 10/10 runs (matching the model's pre-existing export
lean), while summarize-history showed no improvement over the
no-experience baseline at all. That result cannot distinguish between two
different explanations for the export-history collapse:

  (a) v2's block PRESENCE/FORMAT alone reinforces whatever the model's
      existing prior already favors, regardless of what the historical
      content actually says (the same "priming" mechanism already found
      under v1's bare-label format in B7d.1/B7d.2, docs/experiments/
      agent_connected_eval.md SS7-SS10) -- content would be close to
      irrelevant.
  (b) v2's block presence is roughly neutral, and it is specifically the
      export-congruent CONTENT that adds an extra push in the direction
      the model already leans -- content matters, but only when it agrees
      with the prior; counter-prior content (summarize) still doesn't
      transfer.

This script adds a fourth condition, D (neutral-history): a v2-SHAPED
block that preserves the same header, same "Example N" structure, and the
same three content lines as render_experience_block_v2, but replaces the
two content-bearing lines (Principal clarification / Confirmed
interpretation) with text that names no action at all. If D behaves like
C (collapses toward export), that supports (a) -- block presence/format
is still the dominant effect under v2, same as it was under v1. If D stays
close to the A (no-experience) baseline while C still collapses, that
supports (b) -- asymmetric, prior-congruent content sensitivity, still not
evidence v2 transfers counter-prior (summarize) content.

Four conditions, same canonical ambiguous September task as B7d.3/B7d.4:

    A. no_experience
    B. v2_summarize_history   (render_experience_block_v2, real content)
    C. v2_export_history      (render_experience_block_v2, real content)
    D. v2_neutral_history     (render_experience_block_v2_neutral, no content)

B and C are unchanged from experience_semantic_ablation.py's conditions
(same make_experience() calls, same renderer) so this run is directly
comparable to B7d.4's. N=20, repetitions=10, 4 conditions -> 800 calls.

Primary comparisons (entropy remains secondary, per SS11):

    format_effect            = P(export | neutral-history) - P(export | no-experience)
    summarize_content_effect = P(summarize | summarize-history) - P(summarize | neutral-history)
    export_content_effect    = P(export | export-history) - P(export | neutral-history)

Interpretation rule, fixed before running:
  - If neutral-history ALSO collapses toward export (format_effect large),
    conclude v2's block presence/format itself still strongly primes the
    existing export prior -- the same mechanism found under v1.
  - If neutral-history stays near baseline (format_effect small) but
    export-history still collapses, conclude v2 shows asymmetric,
    prior-congruent content sensitivity -- content matters only when it
    agrees with the prior; counter-prior (summarize) transfer still fails.
  - Only if summarize-history selectively raises P(summarize) AND
    export-history selectively raises P(export), both relative to neutral,
    would that be evidence of bidirectional semantic-content sensitivity.

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

from dualflow.agent_experience import AgentExperience, AgentExperienceStore  # noqa: E402
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent  # noqa: E402
from dualflow.experience_aware_delegate import (  # noqa: E402
    _EXPERIENCE_HEADER_V2, ExperienceAwareDelegate, render_experience_block_v2,
)
from dualflow.semantic import Interpretation  # noqa: E402

CONDITIONS = ("no_experience", "v2_summarize_history", "v2_export_history", "v2_neutral_history")


def render_experience_block_v2_neutral(experiences: list[AgentExperience]) -> str:
    """B7d.5's structure-only control for v2. Local to this script --
    src/dualflow/experience_aware_delegate.py is not modified. Reuses
    render_experience_block_v2's exact header (_EXPERIENCE_HEADER_V2,
    imported, not retyped -- retyping it would risk exactly the kind of
    byte-for-byte drift docs/REPRODUCIBILITY.md exists to prevent) and the
    same "Example N" / three-line shape, but replaces the two
    content-bearing lines with text that names no action at all -- kept
    close in register/length to the real lines so this control isn't its
    own new wording confound."""
    if not experiences:
        return ""
    examples = []
    for i, exp in enumerate(experiences, start=1):
        examples.append(
            f"Example {i}\n"
            f"Previous ambiguous delegation: {exp.delegation}\n"
            f"Principal clarification: The intended operation for this prior "
            f"task was clarified with the Principal.\n"
            f"Confirmed interpretation: a meaning was confirmed for this "
            f"prior interaction; no specific action preference is indicated here.")
    return _EXPERIENCE_HEADER_V2 + "\n\n" + "\n\n".join(examples)


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
    elif condition == "v2_neutral_history":
        # action passed to make_experience() is arbitrary here -- the
        # neutral renderer ignores confirmed_interpretation/principal_answer
        # entirely and substitutes fixed neutral text instead.
        store.add(make_experience(scenario, "summarize"))
        wrapped = ExperienceAwareDelegate(delegate, store, render_experience_block=render_experience_block_v2_neutral)
    else:
        raise ValueError(f"unknown condition: {condition!r}")

    result = wrapped.sample_candidates(
        principal_id=scenario["principal_id"], task_category=scenario["task_category"],
        delegation=delegation, context=context, n=n)
    return extract_row(condition, run, result.experience_count, result.distribution, scenario)


def summarize_ablation(all_rows: list[dict]) -> dict:
    def mean_p(condition: str, action: str) -> float:
        vals = [row["p_by_action"][action] for row in all_rows if row["condition"] == condition]
        return sum(vals) / len(vals) if vals else float("nan")

    p_exp_no_exp = mean_p("no_experience", "export")
    p_exp_neutral = mean_p("v2_neutral_history", "export")
    p_sum_neutral = mean_p("v2_neutral_history", "summarize")
    p_sum_sumhist = mean_p("v2_summarize_history", "summarize")
    p_exp_exphist = mean_p("v2_export_history", "export")

    return {
        "P(export | no_experience)": p_exp_no_exp,
        "P(summarize | neutral-history)": p_sum_neutral,
        "P(export | neutral-history)": p_exp_neutral,
        "P(summarize | summarize-history)": p_sum_sumhist,
        "P(export | export-history)": p_exp_exphist,
        "format_effect = P(export|neutral) - P(export|no-exp)": p_exp_neutral - p_exp_no_exp,
        "summarize_content_effect = P(summarize|sumhist) - P(summarize|neutral)":
            p_sum_sumhist - p_sum_neutral,
        "export_content_effect = P(export|exphist) - P(export|neutral)": p_exp_exphist - p_exp_neutral,
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
