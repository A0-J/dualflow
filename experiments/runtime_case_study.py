"""Runtime Case Study (docs/experiments/agent_connected_eval.md §29) --
Option A: small, representative real-API traces through the actual
`AgentDelegationRuntime`, not another large-scale measurement.

    python experiments/runtime_case_study.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt"

Runs exactly 2 real episodes end-to-end through the production runtime
(in-process, all 4 agents real GPT-4o-mini -- no separate-process HTTP
transport needed here; that already validated deployability in Phase
2A/2B, not the point of this case study):

  Case 1 (source-side ambiguity, recovered):
    `calib_scope_and_action_ambiguous` (already-established real
    ambiguous task, Phase 2C-P5's V2), `use_source_verification=True`
    -- shows source_pre_entropy > threshold -> clarification -> safe
    execution, through the real, wired-in runtime path.

  Case 2 (Authority violation despite semantic agreement):
    `condition_violation` (already-established task), baseline runtime
    settings (no source/receiver clarification) -- shows Semantic Flow
    agreeing while Authority Flow independently blocks on the missing
    'reviewed' condition.

Case 3 (confident shared misinterpretation) is NOT run here -- it is
presented in the paper as a standalone `GroundedIntentVerifier`
boundary case using the already-collected Phase 3D GPT-4.1 data
(`phase3d_gpt41_full.jsonl`), since that mechanism is not wired into
`AgentDelegationRuntime` (a deliberate scoping decision -- see the
docs for why Option B, full runtime integration, was rejected as
out-of-scope for this case study).

API key handling identical to every other real-API step in this
project: read once from `--api-key-file`, removed in `finally`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_bench_calibration_tasks import CALIBRATION_TASKS  # noqa: E402
from agent_bench_tasks import SHARED_BUDGET, TASKS  # noqa: E402
from agent_smoke import build_llm_client  # noqa: E402
from runtime_e2e_real import DEFAULT_MODEL, _read_api_key  # noqa: E402

from dualflow.agent_runtime import AgentDelegationRuntime  # noqa: E402
from dualflow.authority_feedback import AuthorityVerifierAgent  # noqa: E402
from dualflow.delegate_agent import DelegateAgent  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402
from dualflow.semantic import SemanticVerifierAgent  # noqa: E402


def _find(tasks, name):
    return next(t for t in tasks if t.name == name)


def _interp_dict(interp):
    return {"action": interp.action, "resource": interp.resource, "scope": interp.scope,
           "condition": sorted(interp.condition)}


def run_case_1(model: str) -> dict:
    """Source-side ambiguity, recovered via real clarification."""
    task = _find(CALIBRATION_TASKS, "calib_scope_and_action_ambiguous")
    llm = build_llm_client(model)
    runtime = AgentDelegationRuntime(
        principal=PrincipalAgent(llm=llm), delegate=DelegateAgent(llm=llm),
        semantic_verifier=SemanticVerifierAgent(llm_client=llm),
        authority_verifier=AuthorityVerifierAgent(llm_client=llm),
        use_source_verification=True, source_n=20, entropy_threshold=0.8)

    result = runtime.run(goal=task.goal, context=task.context, budget=SHARED_BUDGET)
    scr = result.source_clarification_result
    return {
        "case": "1: source-side ambiguity, recovered",
        "task": task.name, "goal": task.goal,
        "source_pre_entropy": scr.pre_distribution.entropy if scr else None,
        "source_clarified": scr.clarified if scr else None,
        "clarified_facet": scr.target_facet if scr else None,
        "source_post_entropy": (scr.post_distribution.entropy
                                if scr and scr.post_distribution else None),
        "confirmed_delegation": scr.final_delegation if scr else None,
        "final_interpretation": _interp_dict(result.final_interpretation),
        "principal_match": result.principal_match,
        "semantic_confirmed": result.semantic_verdict.confirmed,
        "authority_allowed": result.authority_verdict.allowed,
        "decision": result.decision, "reason": result.reason,
    }


def run_case_2(model: str) -> dict:
    """Semantic agreement, Authority independently blocks."""
    task = _find(TASKS, "condition_violation")
    llm = build_llm_client(model)
    runtime = AgentDelegationRuntime(
        principal=PrincipalAgent(llm=llm), delegate=DelegateAgent(llm=llm),
        semantic_verifier=SemanticVerifierAgent(llm_client=llm),
        authority_verifier=AuthorityVerifierAgent(llm_client=llm))  # baseline, no flags

    result = runtime.run(goal=task.goal, context=task.context, budget=SHARED_BUDGET)
    return {
        "case": "2: authority violation despite semantic agreement",
        "task": task.name, "goal": task.goal,
        "delegation": result.delegation.delegation,
        "proposal": _interp_dict(result.proposal.interpretation),
        "final_interpretation": _interp_dict(result.final_interpretation),
        "principal_match": result.principal_match,
        "semantic_confirmed": result.semantic_verdict.confirmed,
        "semantic_route": result.semantic_verdict.route,
        "authority_allowed": result.authority_verdict.allowed,
        "authority_reason": result.authority_verdict.reason,
        "authority_negotiated": result.authority_verdict.negotiated,
        "decision": result.decision, "reason": result.reason,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    print("=== Runtime Case Study -- 2 real episodes through AgentDelegationRuntime ===")
    print(f"model={args.model}  computed budget: ~40-90 calls total (2 episodes, "
         f"Case 1 up to ~1 source-side clarification round, Case 2 minimal)")

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key
    try:
        rows = []
        for name, fn in [("case1", run_case_1), ("case2", run_case_2)]:
            print(f"\n--- {name} ---")
            row = fn(args.model)
            rows.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nSaved to {args.output}")
        return 0
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
