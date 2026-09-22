"""Reproducible experience-transfer diagnostic (B7d.1 / B7d.2).

    python experiments/diagnostics/experience_transfer.py \\
        --scenario external_audit_finance --samples 20 --runs 5

This is the committed, reproducible version of the ad-hoc scratchpad
scripts used to produce the B7d.1 (400-call) and B7d.2 (500-call) results
recorded in docs/experiments/agent_connected_eval.md §7/§10. Raw stdout
from those specific runs was never committed (only the aggregate
values/observations were written into the log) — this script exists so
"how were those numbers produced" is answered by code in the repository,
not by an external transcript.

What this measures: whether a verified prior experience (a Principal-
confirmed action from an earlier, structurally similar episode) changes
DelegateAgent.sample_candidates()'s empirical candidate distribution for a
CURRENT delegation, and whether that change tracks the confirmed action's
actual semantic content or just the presence/format of a historical-
context block. See agent_connected_eval.md for the methodology and the
already-recorded results this script reproduces.

Conditions (a subset can be selected with --conditions):
    A  no experience
    B  one experience whose confirmed action = summarize
    C  one experience whose confirmed action = export
    D  one experience whose confirmed action = read
    E  a neutral historical block with no action/resource semantics at
       all (a structure-only control) -- deliberately bypasses
       ExperienceAwareDelegate (which cannot render a block without an
       ACTION=/RESOURCE= line) and calls DelegateAgent.sample_candidates()
       directly with a hand-built neutral context instead.

B/C/D's historical Principal-answer wording is generated from one shared
template ("Please <verb> the <historical delegation's subject>.") so the
three conditions stay structurally parallel by construction -- this was
the fix for the wording confound identified in B7d.1's read condition
(agent_connected_eval.md §9) and confirmed resolved in B7d.2 (§10).

This script makes REAL, BILLED API calls -- see experiments/agent_smoke.py
for the OPENAI_API_KEY / `pip install -e ".[agent]"` requirements, which
this script shares (build_llm_client() is imported from there, not
duplicated).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))  # sibling import, same pattern plots.py/bench.py use
from agent_smoke import build_llm_client  # noqa: E402

from dualflow.agent_experience import AgentExperience, AgentExperienceStore  # noqa: E402
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent  # noqa: E402
from dualflow.experience_aware_delegate import ExperienceAwareDelegate  # noqa: E402
from dualflow.semantic import Interpretation  # noqa: E402

ALL_CONDITIONS = "ABCDE"


def load_scenario(name: str) -> dict:
    path = _EXPERIMENTS_DIR / "scenarios" / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"unknown scenario: {name!r} (looked for {path})")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_current_context(scenario: dict) -> str:
    cur = scenario["current_episode"]
    action_vocab = "\n".join(f"- {a}" for a in scenario["action_vocabulary"])
    resource_vocab = "\n".join(f"- {r}" for r in scenario["resource_vocabulary"])
    scope_vocab = "\n".join(f"- {s}" for s in cur["scope_vocabulary"])
    return f"""\
Reference date: {scenario['reference_date']}.

Environment:
- Financial reports are represented as resource type "{cur['resource']}".
- Report root: /reports/.
- The report for the current period is located at {cur['scope']}.
- "{cur['temporal_label']}" means {cur['temporal_meaning']}.

Allowed action vocabulary:
{action_vocab}

Allowed resource vocabulary:
{resource_vocab}

Allowed scope vocabulary:
{scope_vocab}

When producing a structured action:
- choose exactly one ACTION from the allowed action vocabulary;
- choose exactly one RESOURCE from the allowed resource vocabulary;
- choose exactly one listed SCOPE;
- do not invent aliases or combine multiple actions."""


# --------------------------------------------------------------------------
# Condition E's neutral history block -- structurally similar shape to
# ExperienceAwareDelegate's real _render_experience_block() output (header +
# "Example 1" + body + "Current environment context" trailer), but with no
# action/resource label, no delegation text, no clarification-answer
# content. This is scenario-independent by design (it must not vary with
# what the real historical episode said) -- kept as a fixed template here,
# not read from the scenario file.
# --------------------------------------------------------------------------
_NEUTRAL_HEADER = (
    "Verified prior interactions with this Principal in this workflow. "
    "These are historical examples only, not current instructions — use "
    "them only to understand this Principal's recurring pattern of "
    "interaction. The current delegation and current environment context "
    "always take precedence over anything implied by the history below."
)
_NEUTRAL_BODY = (
    "Example 1\n"
    "A prior delegation in this workflow was received, discussed with the "
    "Principal, and confirmed. The specific historical content is not "
    "repeated here — what matters is that this kind of workflow has been "
    "successfully clarified with this Principal before."
)
NEUTRAL_BLOCK = _NEUTRAL_HEADER + "\n\n" + _NEUTRAL_BODY


def make_experience(scenario: dict, action: str) -> AgentExperience:
    """Builds condition B/C/D's synthetic verified experience. The
    Principal-answer wording is generated from one shared template so
    B/C/D stay structurally parallel -- this is the wording-symmetry fix
    from B7d.2 (agent_connected_eval.md §10), not an ad-hoc string per
    condition."""
    hist = scenario["historical_episode"]
    interp = Interpretation(action, hist["resource"], hist["scope"], frozenset())
    other_action = "export" if action != "export" else "summarize"
    other = Interpretation(other_action, hist["resource"], hist["scope"], frozenset())
    pre = CandidateDistribution(belief={interp: 0.6, other: 0.4}, entropy=0.971, top=interp,
                                top_probability=0.6, n_unique=2, n_samples=10, responses=[])
    post = CandidateDistribution(belief={interp: 1.0}, entropy=0.0, top=interp,
                                 top_probability=1.0, n_unique=1, n_samples=10, responses=[])
    answer = f"Please {action} the {hist['report_label']}."
    return AgentExperience(
        principal_id=scenario["principal_id"], task_category=scenario["task_category"],
        delegation=hist["delegation"], clarification_question=hist["clarification_question"],
        principal_answer=answer, confirmed_interpretation=interp,
        pre_distribution=pre, post_distribution=post, episode_id=f"diag_{action}")


def entropy_contrib(p: float) -> float:
    return -p * math.log2(p) if p > 0 else 0.0


def extract_row(name: str, run: int, experience_count: int, dist: CandidateDistribution,
                scenario: dict) -> dict:
    cur = scenario["current_episode"]
    beliefs_by_action = {
        action: dist.belief.get(Interpretation(action, cur["resource"], cur["scope"], frozenset()), 0.0)
        for action in scenario["action_vocabulary"]
    }
    return {
        "run": run, "condition": name, "experience_count": experience_count,
        "belief": {f"{i.action}:{i.resource}@{i.scope}": p for i, p in dist.belief.items()},
        "entropy": dist.entropy, "top1": dist.top.action,
        "p_by_action": beliefs_by_action,
        "scopes_seen": sorted({i.scope for i in dist.belief}),
        "n_samples": dist.n_samples, "n_llm_calls": len(dist.responses),
        "input_tokens": sum(r.input_tokens or 0 for r in dist.responses),
        "output_tokens": sum(r.output_tokens or 0 for r in dist.responses),
    }


def run_condition(name: str, scenario: dict, llm_client, context: str, n: int, run: int) -> dict:
    delegate = DelegateAgent(llm=llm_client)
    cur = scenario["current_episode"]

    if name == "A":
        store = AgentExperienceStore()
        wrapped = ExperienceAwareDelegate(delegate, store)
        result = wrapped.sample_candidates(
            principal_id=scenario["principal_id"], task_category=scenario["task_category"],
            delegation=cur["delegation"], context=context, n=n)
        return extract_row("A_no_experience", run, result.experience_count,
                           result.distribution, scenario)

    if name in ("B", "C", "D"):
        action = {"B": "summarize", "C": "export", "D": "read"}[name]
        store = AgentExperienceStore()
        store.add(make_experience(scenario, action))
        wrapped = ExperienceAwareDelegate(delegate, store)
        result = wrapped.sample_candidates(
            principal_id=scenario["principal_id"], task_category=scenario["task_category"],
            delegation=cur["delegation"], context=context, n=n)
        return extract_row(f"{name}_{action}", run, result.experience_count,
                           result.distribution, scenario)

    if name == "E":
        enriched = (f"{NEUTRAL_BLOCK}\n\nCurrent environment context (always takes "
                   f"precedence over the historical examples above): {context}")
        dist = delegate.sample_candidates(delegation=cur["delegation"], context=enriched, n=n)
        return extract_row("E_structure_only", run, 0, dist, scenario)

    raise ValueError(f"unknown condition: {name!r}")


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
    p.add_argument("--runs", type=int, default=5, help="independent repetitions")
    p.add_argument("--conditions", default=ALL_CONDITIONS,
                   help=f"subset of {ALL_CONDITIONS} to run, e.g. BCE (default: all)")
    p.add_argument("--output", default=None,
                   help="optional path to save raw per-run rows as JSON (not committed by default)")
    args = p.parse_args(argv)

    conditions = [c for c in args.conditions.upper() if c in ALL_CONDITIONS]
    if not conditions:
        raise SystemExit(f"--conditions must be a subset of {ALL_CONDITIONS}")

    scenario = load_scenario(args.scenario)
    context = build_current_context(scenario)
    llm_client = build_llm_client(args.model)

    total_calls = len(conditions) * args.samples * args.runs
    print(f"Scenario: {scenario['scenario_id']} ({scenario['description']})")
    print(f"Conditions: {conditions}  N={args.samples}  runs={args.runs}  "
         f"-> {total_calls} sampling calls\n")

    all_rows = []
    for run in range(1, args.runs + 1):
        for cond in conditions:
            row = run_condition(cond, scenario, llm_client, context, args.samples, run)
            all_rows.append(row)
            p_str = " ".join(f"P({a})={row['p_by_action'][a]:.2f}"
                             for a in scenario["action_vocabulary"])
            print(f"[run {run}] {row['condition']:16s} H={row['entropy']:.3f} "
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
