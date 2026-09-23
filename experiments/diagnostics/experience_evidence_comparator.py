"""V3 feasibility diagnostic (isolated, NOT wired into any runtime): does
moving historical experience from Delegate candidate generation to a
post-hoc evidence comparison remove the slot-token dependence found in
B7d.6 (docs/experiments/agent_connected_eval.md SS22)?

    python experiments/diagnostics/experience_evidence_comparator.py \\
        --scenario external_audit_finance --model gpt-4o-mini \\
        --freeze-samples 20 --samples 10

Does NOT modify AgentDelegationRuntime.run(), Authority Flow, existing
SemanticVerifierAgent.verify_agent_proposal() semantics, or any of the
B7d.1-B7d.6 fixed reproducers (experience_transfer.py/
experience_representation.py/experience_semantic_ablation.py/
experience_neutral_control.py/experience_lexical_semantic_control.py) --
none of those files is touched by this script. Exercises the new,
opt-in src/dualflow/experience_evidence.py module only.

Design (v3 architecture review, agreed before this was written):

  1. DelegateAgent.sample_candidates() is called EXACTLY ONCE, with no
     historical experience in its prompt (identical to the clean B7a
     path) -- this freezes the current-episode candidate set. Candidate
     generation is never repeated per condition; only the historical
     evidence text varies across conditions, so any difference across
     conditions can only come from the evidence, not from generation
     variance. This directly answers the concern that re-generating
     candidates per condition would reintroduce a confound B7d.6 was
     designed to remove.

  2. For each DISTINCT candidate interpretation observed in that single
     generation, and for each of four evidence conditions --

         N  neutral-history            (no summarize semantics, no token)
         L  lexical-only               (literal "summarize" token, not
                                         connected to the confirmed action)
         P  semantic-paraphrase        (confirmed relation preserved, no
                                         summar-root token)
         S  structured summarize-history (confirmed relation + literal
                                         "summarize" token -- the existing
                                         v2 condition, unchanged)

     -- reusing the EXACT renderers already used in B7d.5/B7d.6
     (render_experience_block_v2_neutral, render_lexical_only_block,
     render_semantic_paraphrase_block, render_experience_block_v2 -- none
     reimplemented here), HistoricalEvidenceComparator.judge() is called
     repeatedly to estimate an empirical relation-rate (SUPPORT/CONFLICT/
     IRRELEVANT/UNCERTAIN) for that (candidate, condition) pair -- the same
     way DelegateAgent.sample_candidates() estimates a candidate
     distribution via repeated independent calls.

  3. Exactly ONE verified historical experience is used throughout (via
     make_experience(scenario, "summarize"), same as B7d.5/B7d.6) -- no
     multi-experience aggregation or ordering is introduced here.

  4. The comparator's output is NOT fed back into candidate probabilities
     or any execution decision in this script -- this measures the
     comparator's behavior in isolation only. There is no fusion logic
     here at all.

Primary comparison: semantic-paraphrase (P) vs. structured-literal (S),
per candidate. Desired v3-hypothesis evidence: P and S both support the
same semantically-corresponding candidate at similar rates, neutral (N)
does not systematically move support toward or away from either candidate,
and lexical-only (L) does not reproduce the full S effect. If S >> P
again, that would mean slot-token dependence moved from Delegate
generation to verifier comparison -- interpret that as evidence against
the v3 hypothesis, not as something to immediately prompt-tune away.

This script makes REAL, BILLED API calls -- see experiments/agent_smoke.py
for the OPENAI_API_KEY / `pip install -e ".[agent]"` requirements, which
this script shares (build_llm_client() is imported from there via
experience_transfer.py, not duplicated).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAGNOSTICS_DIR))
from experience_lexical_semantic_control import (  # noqa: E402
    render_lexical_only_block, render_semantic_paraphrase_block,
)
from experience_neutral_control import render_experience_block_v2_neutral  # noqa: E402
from experience_transfer import (  # noqa: E402
    build_current_context, load_scenario, make_experience, scenario_fingerprint, text_sha256,
)

_EXPERIMENTS_DIR = _DIAGNOSTICS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import build_llm_client  # noqa: E402

from dualflow.delegate_agent import DelegateAgent  # noqa: E402
from dualflow.experience_aware_delegate import render_experience_block_v2  # noqa: E402
from dualflow.experience_evidence import HistoricalEvidenceComparator  # noqa: E402
from dualflow.semantic import Interpretation  # noqa: E402

CONDITIONS = ("neutral", "lexical_only", "semantic_paraphrase", "structured_summarize")

_RENDERERS = {
    "neutral": render_experience_block_v2_neutral,
    "lexical_only": render_lexical_only_block,
    "semantic_paraphrase": render_semantic_paraphrase_block,
    "structured_summarize": render_experience_block_v2,
}


def freeze_candidates(scenario: dict, llm_client, context: str, n: int) -> list[Interpretation]:
    """DelegateAgent.sample_candidates()를 history 없이 정확히 1회
    호출해서 나온 distinct Interpretation 전부를 반환한다 -- 이후 모든
    (candidate, condition) 판단은 이 고정된 목록에 대해서만 이뤄진다.
    이 함수가 끝난 뒤에는 다시 sample_candidates()를 호출하지 않는다."""
    delegate = DelegateAgent(llm=llm_client)
    delegation = scenario["current_episode"]["delegation"]
    distribution = delegate.sample_candidates(delegation=delegation, context=context, n=n)
    return list(distribution.belief.keys())


def run_cell(comparator: HistoricalEvidenceComparator, *, delegation: str,
            candidate: Interpretation, evidence_text: str, n: int) -> dict:
    """한 (candidate, condition) 쌍에 대해 judge()를 n번 반복 호출해서
    relation별 empirical rate를 추정한다. 파싱 실패는 세되 분포에서는
    제외한다(DelegateAgent.sample_candidates()와 동일한 fail-closed 관례)."""
    relations: list = []
    responses = []
    parse_failures = 0
    for _ in range(n):
        try:
            judgment = comparator.judge(
                delegation=delegation, candidate=candidate, evidence_text=evidence_text)
        except ValueError:
            parse_failures += 1
            continue
        relations.append(judgment.relation.value)
        responses.append(judgment.response)

    counts = Counter(relations)
    total = len(relations)
    rates = {r: (counts.get(r, 0) / total if total else float("nan"))
             for r in ("SUPPORT", "CONFLICT", "IRRELEVANT", "UNCERTAIN")}

    return {
        "candidate": f"{candidate.action}:{candidate.resource}@{candidate.scope}",
        "n_requested": n, "n_judged": total, "parse_failures": parse_failures,
        "rates": rates,
        "input_tokens": sum(r.input_tokens or 0 for r in responses),
        "output_tokens": sum(r.output_tokens or 0 for r in responses),
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
    p.add_argument("--freeze-samples", type=int, default=20,
                   help="N for the single, history-free candidate-generation call")
    p.add_argument("--samples", type=int, default=10,
                   help="repeated comparator judgments per (candidate, condition) cell")
    p.add_argument("--output", default=None,
                   help="optional path to save raw per-cell rows as JSON (not committed by default)")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario)
    context = build_current_context(scenario)
    llm_client = build_llm_client(args.model)

    fp = scenario_fingerprint(scenario)
    print(f"Scenario: {scenario['scenario_id']} v{scenario.get('scenario_version', '?')}")
    print(f"Delegation SHA256: {fp['delegation_sha256']}")
    print(f"Context SHA256: {fp['context_sha256']}")
    print("NOTE: this diagnostic does not feed comparator output into any "
         "execution decision -- it measures HistoricalEvidenceComparator "
         "behavior in isolation only.\n")

    print(f"Freezing candidates (1 call to sample_candidates(), n={args.freeze_samples})...")
    candidates = freeze_candidates(scenario, llm_client, context, args.freeze_samples)
    print(f"Frozen candidate set ({len(candidates)}): "
         f"{[f'{c.action}:{c.resource}@{c.scope}' for c in candidates]}\n")

    exp = make_experience(scenario, "summarize")  # exactly one verified experience, throughout
    delegation = scenario["current_episode"]["delegation"]
    total_calls = args.freeze_samples + len(candidates) * len(CONDITIONS) * args.samples
    print(f"Conditions: {list(CONDITIONS)}  candidates={len(candidates)}  "
         f"samples/cell={args.samples}  -> {total_calls} total sampling calls\n")

    comparator = HistoricalEvidenceComparator(llm=llm_client)
    all_rows = []
    for condition in CONDITIONS:
        evidence_text = _RENDERERS[condition]([exp])
        for candidate in candidates:
            row = run_cell(comparator, delegation=delegation, candidate=candidate,
                           evidence_text=evidence_text, n=args.samples)
            row["condition"] = condition
            all_rows.append(row)
            print(f"[{condition:22s}] {row['candidate']:35s} "
                 f"n_judged={row['n_judged']}/{row['n_requested']} "
                 f"parse_failures={row['parse_failures']} "
                 f"SUPPORT={row['rates']['SUPPORT']:.2f} "
                 f"CONFLICT={row['rates']['CONFLICT']:.2f} "
                 f"IRRELEVANT={row['rates']['IRRELEVANT']:.2f} "
                 f"UNCERTAIN={row['rates']['UNCERTAIN']:.2f} "
                 f"tok_in={row['input_tokens']} tok_out={row['output_tokens']}", flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"rows": all_rows}, f, indent=2)
        print(f"\nSaved {len(all_rows)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
