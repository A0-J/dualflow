"""B7d.6 real-API validation: is the replicated counter-prior summarize
effect (B7d.5/B7d.5R) caused by structured semantic experience, or merely
by the literal action token "summarize"?

    python experiments/diagnostics/experience_lexical_semantic_control.py \\
        --scenario external_audit_finance --model gpt-4o-mini --samples 20 --runs 10

A NEW script. Does not modify experience_transfer.py (fixed B7d.1/B7d.2
reproducer), experience_representation.py (fixed B7d.3 reproducer),
experience_semantic_ablation.py (fixed B7d.4 reproducer), or
experience_neutral_control.py (fixed B7d.5/B7d.5R reproducer) -- none of
those four files is touched here. src/dualflow/experience_aware_delegate.py
(render_experience_block_v2) is also NOT modified.

Why this experiment: B7d.5/B7d.5R (docs/experiments/agent_connected_eval.md
SS19/SS20) found a replicated, run-consistent effect -- v2's
summarize-history condition raises P(summarize) relative to a structurally
matched neutral-history control in 10/10 runs, twice, in two independent
800-call batches (summarize_content_effect +0.270 then +0.220). That effect
was explicitly NOT called "semantic transfer" -- it could be driven by (a)
the literal token "summarize" appearing in the historical block acting as
a lexical/action-token prime, independent of any structured meaning, or (b)
the structured historical relation (ambiguous delegation -> clarification
-> Principal-confirmed meaning) carrying real semantic content, independent
of whether the literal word "summarize" appears at all. B7d.5/B7d.5R cannot
separate these because the existing summarize-history condition has both
at once. This experiment isolates them with a 2x2-style four-condition
design:

    N  neutral-history           -- no summarize semantics, no literal token
       (byte-identical to B7d.5/B7d.5R's v2_neutral_history condition --
       imported from experience_neutral_control.py, not reimplemented)
    L  lexical-only control      -- literal token "summarize" present, but
       explicitly NOT presented as the Principal's confirmed historical
       action (a disconnected, incidental mention)
    P  semantic-paraphrase       -- the full structured historical relation
       (ambiguous delegation -> clarification -> confirmed meaning) is
       preserved, expressed as "produce a concise account of the report's
       contents" -- no "summarize"/"summary"/"summarization" root anywhere
    S  existing structured summarize-history -- byte-identical to
       B7d.5/B7d.5R's v2_summarize_history condition (structured relation +
       literal token both present) -- reused via render_experience_block_v2,
       not reimplemented

No no_experience/raw-baseline condition this round -- N (neutral) is
already the more appropriate control (per B7d.5's finding that raw
no-experience carries its own uncontrolled variance), and the causal
question here is entirely about differences WITHIN the v2-shaped block,
not about whether a block exists at all.

N=20, repetitions=10, 4 conditions -> 800 calls.

Primary metrics (entropy not computed as a success criterion here at all --
this experiment is purely about P(summarize) across conditions):

    lexical_effect               = P(summarize | L) - P(summarize | N)
    semantic_without_token_effect = P(summarize | P) - P(summarize | N)
    full_structured_effect        = P(summarize | S) - P(summarize | N)

Interpretation rule, fixed before running:
  - L increases, P does not  -> evidence favors literal/action-token priming.
  - P increases, L does not  -> strong evidence the effect does not depend
    on the literal "summarize" token -- consistent with structured
    semantic-content sensitivity.
  - both L and P increase    -> both lexical priming and semantic content
    may contribute.
  - neither L nor P increases but S does -> possible interaction between
    the literal token and the structured historical relation (neither part
    alone is sufficient).

No outcome here is automatically labeled "semantic transfer."

Before any real API call, verify_blocks() renders and prints all four
experience blocks and asserts (via plain substring checks, not by grepping
comments/docstrings) the required token properties:
  - N contains no "summar"-root token at all.
  - L contains the literal token "summarize", but not on the "Confirmed
    interpretation" line (i.e. not presented as the confirmed historical
    action).
  - P contains no "summar"-root token at all (covers summarize/summary/
    summarization).
  - S is byte-identical to render_experience_block_v2's own output for a
    summarize-confirmed experience (the exact function/experience already
    used by B7d.5/B7d.5R's condition B) -- reused, not retyped.

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
from experience_neutral_control import render_experience_block_v2_neutral  # noqa: E402
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

CONDITIONS = ("neutral", "lexical_only", "semantic_paraphrase", "structured_summarize")


def render_lexical_only_block(experiences: list[AgentExperience]) -> str:
    """L -- the literal token "summarize" is present, but only as an
    incidental, disconnected mention -- explicitly not the Principal's
    confirmed historical action. Same header/shape as the neutral control;
    only the "Principal clarification" line differs from
    render_experience_block_v2_neutral's, and only by inserting the token
    in a sentence that denies it was the confirmed action."""
    if not experiences:
        return ""
    examples = []
    for i, exp in enumerate(experiences, start=1):
        examples.append(
            f"Example {i}\n"
            f"Previous ambiguous delegation: {exp.delegation}\n"
            f"Principal clarification: The Principal's reply mentioned the "
            f"word \"summarize\" only in an unrelated remark, without "
            f"indicating it as the intended action for this task.\n"
            f"Confirmed interpretation: a meaning was confirmed for this "
            f"prior interaction; no specific action preference is indicated here.")
    return _EXPERIENCE_HEADER_V2 + "\n\n" + "\n\n".join(examples)


def render_semantic_paraphrase_block(experiences: list[AgentExperience]) -> str:
    """P -- preserves the full structured relation (ambiguous delegation ->
    clarification -> confirmed meaning), same as render_experience_block_v2,
    but the confirmed meaning is expressed as "produce a concise account of
    the report's contents" instead of the literal token "summarize" (and
    without any "summary"/"summarization" root either). Deliberately does
    NOT add an explicit anti-export statement -- that would introduce a new
    directional cue not present in the real S condition."""
    if not experiences:
        return ""
    examples = []
    for i, exp in enumerate(experiences, start=1):
        examples.append(
            f"Example {i}\n"
            f"Previous ambiguous delegation: {exp.delegation}\n"
            f"Principal clarification: Please produce a concise account of "
            f"the report's contents for the prior task.\n"
            f"Confirmed interpretation: producing a concise account of the "
            f"report's contents")
    return _EXPERIENCE_HEADER_V2 + "\n\n" + "\n\n".join(examples)


def verify_blocks(scenario: dict) -> dict[str, str]:
    """Renders all four blocks and asserts the required token properties
    BEFORE any real API call is made. Raises AssertionError (loudly, before
    any billed call happens) if any property fails."""
    exp = make_experience(scenario, "summarize")  # container only; custom
    # renderers below ignore its principal_answer/confirmed_interpretation
    # content except where explicitly noted (S).

    blocks = {
        "N (neutral)": render_experience_block_v2_neutral([exp]),
        "L (lexical-only)": render_lexical_only_block([exp]),
        "P (semantic-paraphrase)": render_semantic_paraphrase_block([exp]),
        "S (structured summarize-history)": render_experience_block_v2([exp]),
    }

    print("=== B7d.6 experience blocks (pre-run verification) ===\n")
    for label, text in blocks.items():
        print(f"--- {label} ---")
        print(text)
        print()

    def has_summar_root(text: str) -> bool:
        return "summar" in text.lower()  # covers summarize/summary/summarization

    n_text = blocks["N (neutral)"]
    l_text = blocks["L (lexical-only)"]
    p_text = blocks["P (semantic-paraphrase)"]
    s_text = blocks["S (structured summarize-history)"]

    assert not has_summar_root(n_text), "N must contain no summar-root token"
    assert "summarize" in l_text.lower(), "L must contain the literal token 'summarize'"
    l_confirmed_line = next(ln for ln in l_text.splitlines() if ln.startswith("Confirmed interpretation:"))
    assert "summarize" not in l_confirmed_line.lower(), (
        "L's 'Confirmed interpretation' line must NOT contain 'summarize' -- "
        "the token must not be presented as the confirmed historical action")
    assert not has_summar_root(p_text), "P must contain no summar-root token at all"

    expected_s = render_experience_block_v2([make_experience(scenario, "summarize")])
    assert s_text == expected_s, "S must be byte-identical to render_experience_block_v2's own output"

    print("All pre-run token/structure checks passed:")
    print("  - N: no summar-root token")
    print("  - L: contains 'summarize', but not on the Confirmed-interpretation line")
    print("  - P: no summar-root token (structured relation preserved via paraphrase)")
    print("  - S: byte-identical to render_experience_block_v2's real output")
    print()

    return blocks


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


_RENDERERS = {
    "neutral": render_experience_block_v2_neutral,
    "lexical_only": render_lexical_only_block,
    "semantic_paraphrase": render_semantic_paraphrase_block,
    "structured_summarize": render_experience_block_v2,
}


def run_condition(condition: str, scenario: dict, llm_client, context: str, n: int, run: int) -> dict:
    delegation = scenario["current_episode"]["delegation"]
    delegate = DelegateAgent(llm=llm_client)
    store = AgentExperienceStore()
    store.add(make_experience(scenario, "summarize"))  # container; renderer picks what to use
    wrapped = ExperienceAwareDelegate(delegate, store, render_experience_block=_RENDERERS[condition])

    result = wrapped.sample_candidates(
        principal_id=scenario["principal_id"], task_category=scenario["task_category"],
        delegation=delegation, context=context, n=n)
    return extract_row(condition, run, result.experience_count, result.distribution, scenario)


def summarize_ablation(all_rows: list[dict]) -> dict:
    def mean_p(condition: str, action: str) -> float:
        vals = [row["p_by_action"][action] for row in all_rows if row["condition"] == condition]
        return sum(vals) / len(vals) if vals else float("nan")

    p_sum_n = mean_p("neutral", "summarize")
    p_sum_l = mean_p("lexical_only", "summarize")
    p_sum_p = mean_p("semantic_paraphrase", "summarize")
    p_sum_s = mean_p("structured_summarize", "summarize")

    return {
        "P(summarize | N neutral)": p_sum_n,
        "P(summarize | L lexical-only)": p_sum_l,
        "P(summarize | P semantic-paraphrase)": p_sum_p,
        "P(summarize | S structured-summarize)": p_sum_s,
        "lexical_effect = P(sum|L) - P(sum|N)": p_sum_l - p_sum_n,
        "semantic_without_token_effect = P(sum|P) - P(sum|N)": p_sum_p - p_sum_n,
        "full_structured_effect = P(sum|S) - P(sum|N)": p_sum_s - p_sum_n,
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

    # Structural verification FIRST -- before building the (possibly real)
    # LLM client and before any API call is made.
    verify_blocks(scenario)

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
