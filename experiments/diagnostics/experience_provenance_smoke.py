"""V3 CONTRACT SMOKE TEST -- NOT the next N/L/P/S-style experiment.

    python experiments/diagnostics/experience_provenance_smoke.py \\
        --model gpt-4o-mini-2024-07-18

Purpose (per docs/experiments/agent_connected_eval.md §23 follow-up 5's
closing note): confirm the full v3 provenance chain --

    IG target selection -> singleton clarifying question
    -> post-clarification resolution -> confirmed_facets
    -> HistoricalEvidenceComparator

-- actually holds together end to end against REAL API-generated data, at
minimal cost, BEFORE committing to a large-scale experiment. This is a
smoke test, not a statistics-gathering run: one historical clarification
round, one frozen candidate-generation call, then a handful of zero-API
deterministic comparator calls.

Does not modify any fixed reproducer, the v3 comparator/provenance code,
prompts, IG selection, entropy thresholds, or scenario files. Failures are
recorded and categorized (A-E, see module bottom), not silently patched.

Two real-API steps are kept structurally separate, on purpose (candidate
generation must never see historical evidence):

  1. Historical verified experience, via the REAL clarification pipeline
     (ClarifyingDelegate.resolve() -> build_verified_experience()) on the
     canonical AUGUST scenario constants already validated in
     experiments/agent_smoke.py (EXAMPLE_AMBIGUOUS_DELEGATION/
     EXAMPLE_CONTEXT/EXAMPLE_GOAL -- reused verbatim, not reinvented).
     Real API calls: pre-sampling (n), one clarifying question, one
     Principal answer, post-sampling (n).
  2. A frozen CURRENT candidate set, via DelegateAgent.sample_candidates()
     with NO history, on the canonical SEPTEMBER scenario
     (experiments/scenarios/external_audit_finance.json, the same
     fingerprinted scenario every B7d.3-B7d.6/v3 diagnostic has used).
     Real API calls: N independent completions. The historical episode
     (step 1) is August; the current episode (step 2) is September --
     genuinely different real scopes, which is what lets this smoke test
     exercise cross-facet isolation (action matches, scope differs) on
     real API-generated data rather than synthetic Interpretations.

HistoricalEvidenceComparator.judge() is then called directly against the
frozen candidates using the real verified experience from step 1 --
ZERO additional API calls, fully deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAGNOSTICS_DIR))
from experience_transfer import build_current_context, load_scenario, scenario_fingerprint  # noqa: E402

_EXPERIMENTS_DIR = _DIAGNOSTICS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import (  # noqa: E402
    EXAMPLE_AMBIGUOUS_DELEGATION, EXAMPLE_CONTEXT, EXAMPLE_GOAL, build_llm_client,
)

from dualflow.agent_experience import build_verified_experience  # noqa: E402
from dualflow.clarification import ClarifyingDelegate  # noqa: E402
from dualflow.delegate_agent import DelegateAgent  # noqa: E402
from dualflow.experience_evidence import EvidenceRelation, HistoricalEvidenceComparator  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402


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
    p.add_argument("--n", type=int, default=10, help="samples per DelegateAgent call "
                   "(matches B7b's own established convention, not B7d's N=20)")
    p.add_argument("--freeze-samples", type=int, default=20,
                   help="N for the separate, history-free current-candidate freeze call")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario)
    fp = scenario_fingerprint(scenario)
    llm_client = build_llm_client(args.model)

    print("=== v3 contract smoke test (NOT a full experiment) ===")
    print(f"model: {args.model}")
    print(f"scenario: {scenario['scenario_id']} v{scenario.get('scenario_version', '?')}")
    print(f"current (September) delegation SHA256: {fp['delegation_sha256']}")
    print(f"current (September) context SHA256: {fp['context_sha256']}")
    print("historical (August) delegation/context: agent_smoke.py's "
         "EXAMPLE_AMBIGUOUS_DELEGATION/EXAMPLE_CONTEXT (reused verbatim)")
    print()

    calls = {"clarification_pre": 0, "clarification_question": 0,
            "clarification_answer": 0, "clarification_post": 0, "freeze": 0}
    tokens_in = {"clarification": 0, "freeze": 0}
    tokens_out = {"clarification": 0, "freeze": 0}

    # ------------------------------------------------------------------
    # Step 1: REAL clarification pipeline -> verified historical experience
    # ------------------------------------------------------------------
    print("--- Step 1: historical clarification round (real API) ---")
    principal = PrincipalAgent(llm=llm_client)
    delegate_for_clarification = DelegateAgent(llm=llm_client)
    clarifier = ClarifyingDelegate(principal=principal, delegate=delegate_for_clarification,
                                   n=args.n, entropy_threshold=0.8)

    result = clarifier.resolve(goal=EXAMPLE_GOAL, context=EXAMPLE_CONTEXT,
                              delegation=EXAMPLE_AMBIGUOUS_DELEGATION)

    calls["clarification_pre"] = result.pre_distribution.n_samples
    tokens_in["clarification"] += sum(r.input_tokens or 0 for r in result.pre_distribution.responses)
    tokens_out["clarification"] += sum(r.output_tokens or 0 for r in result.pre_distribution.responses)

    print(f"clarified: {result.clarified}")
    print(f"pre_distribution: entropy={result.pre_distribution.entropy:.3f} "
         f"belief={ {f'{i.action}': round(p, 2) for i, p in result.pre_distribution.belief.items()} }")

    failure_stage = None
    if not result.clarified:
        failure_stage = "A"  # target selection never even ran a real clarification round
        print("FAILURE STAGE A: pre-distribution entropy did not exceed threshold -- "
             "no clarification was triggered at all. Cannot proceed to steps requiring "
             "a real verified experience.")
    else:
        calls["clarification_question"] = 1
        calls["clarification_answer"] = 1
        calls["clarification_post"] = result.post_distribution.n_samples
        tokens_in["clarification"] += (result.question.response.input_tokens or 0)
        tokens_out["clarification"] += (result.question.response.output_tokens or 0)
        tokens_in["clarification"] += (result.answer.response.input_tokens or 0)
        tokens_out["clarification"] += (result.answer.response.output_tokens or 0)
        tokens_in["clarification"] += sum(r.input_tokens or 0 for r in result.post_distribution.responses)
        tokens_out["clarification"] += sum(r.output_tokens or 0 for r in result.post_distribution.responses)

        print(f"varied_facets: {sorted(result.varied_facets)}")
        print(f"target_facets: {sorted(result.question.target_facets)}")
        print(f"clarifying question: {result.question.question!r}")
        print(f"Principal answer: {result.answer.answer!r}")
        post_action_values = sorted({i.action for i in result.post_distribution.belief})
        print(f"post_distribution: entropy={result.post_distribution.entropy:.3f} "
             f"action values={post_action_values} "
             f"belief={ {f'{i.action}': round(p, 2) for i, p in result.post_distribution.belief.items()} }")
        print(f"final_interpretation.action: {result.final_interpretation.action!r}")

        if result.question.target_facets != frozenset({"action"}):
            failure_stage = "A"
            print(f"FAILURE STAGE A: target_facets was {sorted(result.question.target_facets)}, "
                 "expected exactly {'action'}.")
        elif len(post_action_values) != 1:
            failure_stage = "C"
            print("FAILURE STAGE C: post-clarification resampling did not converge to a "
                 "single action value -- clarification did not actually resolve the "
                 "target facet.")

    experience = None
    if failure_stage is None:
        experience = build_verified_experience(
            result, principal_id=scenario["principal_id"], task_category=scenario["task_category"],
            delegation=EXAMPLE_AMBIGUOUS_DELEGATION)
        if experience is None:
            failure_stage = "D"
            print("FAILURE STAGE D: build_verified_experience() returned None -- "
                 "clarified/question/answer/post_distribution preconditions or the "
                 "entropy gate were not satisfied.")
        else:
            print(f"confirmed_facets: {sorted(experience.confirmed_facets)}")
            print(f"confirmed_interpretation: action={experience.confirmed_interpretation.action!r} "
                 f"scope={experience.confirmed_interpretation.scope!r}")
            if experience.confirmed_facets != frozenset({"action"}):
                failure_stage = "D"
                print(f"FAILURE STAGE D: confirmed_facets was "
                     f"{sorted(experience.confirmed_facets)}, expected exactly {{'action'}}.")

    print()

    # ------------------------------------------------------------------
    # Step 2: frozen CURRENT candidates -- no historical evidence involved
    # ------------------------------------------------------------------
    print("--- Step 2: frozen current (September) candidates (real API, no history) ---")
    context = build_current_context(scenario)
    delegate_for_freeze = DelegateAgent(llm=llm_client)
    distribution = delegate_for_freeze.sample_candidates(
        delegation=scenario["current_episode"]["delegation"], context=context, n=args.freeze_samples)

    calls["freeze"] = distribution.n_samples
    tokens_in["freeze"] = sum(r.input_tokens or 0 for r in distribution.responses)
    tokens_out["freeze"] = sum(r.output_tokens or 0 for r in distribution.responses)

    candidates = list(distribution.belief.keys())
    print(f"frozen candidates ({len(candidates)}): "
         f"{[f'{c.action}:{c.resource}@{c.scope}' for c in candidates]}")

    candidate_summarize = next((c for c in candidates if c.action == "summarize"), None)
    candidate_export = next((c for c in candidates if c.action == "export"), None)
    if candidate_summarize is None or candidate_export is None:
        if failure_stage is None:
            failure_stage = "B"
        print("FAILURE STAGE B (candidate generation): the frozen set did not contain both "
             "a 'summarize' and an 'export' candidate -- cannot run the comparator check "
             "as specified.")
    print()

    # ------------------------------------------------------------------
    # Step 3: deterministic comparator -- ZERO additional API calls
    # ------------------------------------------------------------------
    print("--- Step 3: HistoricalEvidenceComparator (0 API calls, deterministic) ---")
    if experience is not None and candidate_summarize is not None and candidate_export is not None:
        comparator = HistoricalEvidenceComparator()

        r_action_summarize = comparator.judge(candidate=candidate_summarize, historical=experience,
                                              facet="action")
        r_action_export = comparator.judge(candidate=candidate_export, historical=experience,
                                           facet="action")
        r_scope_summarize = comparator.judge(candidate=candidate_summarize, historical=experience,
                                             facet="scope")

        print(f"candidate(summarize) action vs. historical(summarize) action -> "
             f"{r_action_summarize.relation.value} (expected SUPPORT)")
        print(f"candidate(export) action vs. historical(summarize) action -> "
             f"{r_action_export.relation.value} (expected CONFLICT)")
        print(f"candidate(summarize) scope vs. historical(August) scope, facet=scope "
             f"(not a confirmed facet) -> {r_scope_summarize.relation.value} (expected IRRELEVANT)")

        if r_action_summarize.relation != EvidenceRelation.SUPPORT:
            failure_stage = failure_stage or "E"
            print("FAILURE STAGE E: candidate(summarize) vs. historical(summarize) on "
                 "facet=action was not SUPPORT.")
        if r_action_export.relation != EvidenceRelation.CONFLICT:
            failure_stage = failure_stage or "E"
            print("FAILURE STAGE E: candidate(export) vs. historical(summarize) on "
                 "facet=action was not CONFLICT.")
        if r_scope_summarize.relation != EvidenceRelation.IRRELEVANT:
            failure_stage = failure_stage or "E"
            print("FAILURE STAGE E: querying an unconfirmed facet (scope) was not IRRELEVANT.")
    else:
        print("Skipped -- an earlier stage already failed (see above).")

    print()
    print("=== Summary ===")
    print(f"calls: clarification_pre={calls['clarification_pre']} "
         f"clarification_question={calls['clarification_question']} "
         f"clarification_answer={calls['clarification_answer']} "
         f"clarification_post={calls['clarification_post']} "
         f"freeze={calls['freeze']} comparator=0 (deterministic, no API)")
    total_calls = sum(calls.values())
    print(f"total real API calls: {total_calls}")
    print(f"tokens (clarification pipeline): input={tokens_in['clarification']} "
         f"output={tokens_out['clarification']}")
    print(f"tokens (candidate freeze): input={tokens_in['freeze']} output={tokens_out['freeze']}")
    print(f"tokens (total): input={tokens_in['clarification'] + tokens_in['freeze']} "
         f"output={tokens_out['clarification'] + tokens_out['freeze']}")
    print(f"failure stage: {failure_stage if failure_stage else 'none -- full chain held'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
