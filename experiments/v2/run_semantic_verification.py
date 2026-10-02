"""E2 -- shared-sample semantic verification runner (Experiment Plan v2,
docs/experiments/phase_e_experiment_plan_v2.md). Per episode, exactly one
real `DelegateAgent.propose()` call plus one shared Principal sample bank
(`sample_principal_intents()`, n=20) -- Arms A/B/C are all recomputed from
that ONE bank (no per-arm resampling, so no Phase-3C-style independent-
sample confound to discover and fix later; it's absent from the design).

    # cheapest possible first step -- 1 Delegate call per base scenario
    # (24 calls total), 0 Principal sampling, no entropy/arm computation --
    # purely checks the real API/logging path end to end:
    python experiments/v2/run_semantic_verification.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt" \\
        --scenarios experiments/v2/scenarios/semantic_ambiguity.jsonl \\
        --output results/v2_smoke.jsonl --smoke

    # full run (per the plan: only after the smoke run's action
    # distribution is inspected and the design is frozen/committed):
    python experiments/v2/run_semantic_verification.py \\
        --api-key-file "C:\\Users\\user\\Downloads\\files\\openai_api_key.txt" \\
        --scenarios experiments/v2/scenarios/semantic_ambiguity.jsonl \\
        --output results/v2_e2_full.jsonl --repeats 3 --n 20

Telemetry (per instruction -- real, measured, not assumed): every stored
response carries `requested_model`/`served_model`/`input_tokens`/
`output_tokens`/`cached_input_tokens`/`latency_ms`/`prompt_hash` (all from
`LLMResponse`, src/dualflow/llm.py) plus a `sample_bank_id` stamped by
this runner -- the same id on every one of an episode's n bank responses,
so a later audit can directly verify Arms B and C really did consume the
identical bank (not just assert it).

API key handling identical to every other real-API step in this project:
read once from --api-key-file, removed in a finally block.
"""

from __future__ import annotations

import argparse, json, os, uuid
from pathlib import Path

import sys
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR.parent / "src"))
from dualflow.delegate_agent import DelegateAgent
from dualflow.llm import OpenAILLMClient
from dualflow.principal_agent import PrincipalAgent

MODEL = "gpt-4o-mini-2024-07-18"


def _read_api_key(path: str) -> str:
    key = Path(path).read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"Empty API key file: {path}")
    return key


def _response_dict(r, sample_bank_id: str) -> dict:
    return {
        "text": r.text, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
        "cached_input_tokens": r.cached_input_tokens, "latency_ms": r.latency_ms,
        "served_model": r.model, "requested_model": r.requested_model,
        "temperature": r.temperature, "top_p": r.top_p, "prompt_hash": r.prompt_hash,
        "sample_bank_id": sample_bank_id,
    }


def load_scenarios(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_smoke(scenarios: list[dict], delegate: DelegateAgent) -> list[dict]:
    """1 propose() call per base scenario -- exactly 24 calls across the
    full manifest. Checks the real API/logging path; does NOT by itself
    validate the summarize/export ambiguity (inspect the action
    distribution printed below for that)."""
    rows = []
    for i, sc in enumerate(scenarios, 1):
        bank_id = f"smoke_{sc['name']}_{uuid.uuid4().hex[:8]}"
        proposal = delegate.propose(delegation=sc["goal"], context=sc["context"])
        correct = proposal.interpretation.action == sc["ideal_action"]
        print(f"[{i}/{len(scenarios)}] {sc['name']}: ideal={sc['ideal_action']} "
             f"proposed={proposal.interpretation.action} {'OK' if correct else 'MISREAD'}",
             flush=True)
        rows.append({"name": sc["name"], "domain": sc["domain"], "intent": sc["intent"],
                     "split": sc["split"], "ideal_action": sc["ideal_action"],
                     "proposed_action": proposal.interpretation.action, "correct": correct,
                     "response": _response_dict(proposal.response, bank_id)})
    return rows


def run_episode(scenario: dict, principal: PrincipalAgent, delegate: DelegateAgent,
                n: int, repeat: int) -> dict:
    """1 Delegate.propose() + n Principal.restate_intent() calls, all real.

    Calls `restate_intent()` directly in its own loop -- NOT via
    `sample_principal_intents()` -- because that helper only returns an
    aggregated `CandidateDistribution` (belief/top/entropy), not an
    ordered per-sample (interpretation, response) pairing. This is
    exactly the pattern `phase3d_shared_sample_replication.py`'s
    `collect_one_episode()` already established for the identical reason
    -- reused here, not reinvented."""
    bank_id = f"{scenario['name']}_rep{repeat}_{uuid.uuid4().hex[:8]}"
    proposal = delegate.propose(delegation=scenario["goal"], context=scenario["context"])

    bank = []
    for _ in range(n):
        intent = principal.restate_intent(goal=scenario["goal"], context=scenario["context"])
        interp = intent.intended_action
        bank.append({"action": interp.action, "resource": interp.resource, "scope": interp.scope,
                     "condition": sorted(interp.condition),
                     "response": _response_dict(intent.response, bank_id)})

    return {
        "scenario": scenario["name"], "domain": scenario["domain"], "intent": scenario["intent"],
        "split": scenario["split"], "repeat": repeat,
        "ideal_action": scenario["ideal_action"],
        "delegate_proposal": {"action": proposal.interpretation.action,
                              "resource": proposal.interpretation.resource,
                              "scope": proposal.interpretation.scope,
                              "condition": sorted(proposal.interpretation.condition)},
        "delegate_response": _response_dict(proposal.response, bank_id),
        "sample_bank_id": bank_id,
        "principal_bank": bank,   # n raw (interp, response) pairs, in call order --
                                  # Arms A/B/C are all recomputed from this offline
                                  # (experiments/v2/analyze.py), 0 further API calls,
                                  # same discipline as phase3d_analysis.py.
    }


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--scenarios", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--entropy-threshold", type=float, default=0.8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--smoke", action="store_true",
                  help="1 propose() call per scenario only (24 calls total) -- "
                       "verifies real API/logging, does not run the full bank")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--budget-only", action="store_true")
    args = p.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    if args.smoke:
        calls = len(scenarios)
    else:
        calls = len(scenarios) * args.repeats * (1 + args.n)
    print(f"scenarios={len(scenarios)} smoke={args.smoke} repeats={args.repeats if not args.smoke else '-'} "
         f"n={args.n if not args.smoke else '-'} expected_new_calls={calls}")
    if args.budget_only:
        return 0

    api_key = _read_api_key(args.api_key_file)
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        import openai
        llm = OpenAILLMClient(openai.OpenAI(api_key=api_key), args.model,
                              temperature=args.temperature, top_p=args.top_p)
        del api_key
        delegate = DelegateAgent(llm=llm)
        principal = PrincipalAgent(llm=llm)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        if args.smoke:
            rows = run_smoke(scenarios, delegate)
            with out.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            from collections import Counter
            print("\n=== Smoke action distribution (inspect before freezing E2) ===")
            for intent in ("summarize", "export"):
                dist = Counter(r["proposed_action"] for r in rows if r["intent"] == intent)
                print(f"  ideal={intent}: proposed={dict(dist)}")
            print(f"saved={out}")
            return 0

        # Incremental write (flushed per episode, not buffered to the end) --
        # same hardening as phase3d_shared_sample_replication.py: integrity
        # checkable by line count alone, a crash partway through doesn't
        # lose already-completed episodes.
        total_episodes = len(scenarios) * args.repeats
        done, n_errors = 0, 0
        with out.open("w", encoding="utf-8") as f:
            for scenario in scenarios:
                for repeat in range(1, args.repeats + 1):
                    done += 1
                    try:
                        row = run_episode(scenario, principal, delegate, args.n, repeat)
                    except Exception as exc:  # noqa: BLE001 -- log and keep going
                        n_errors += 1
                        print(f"[{done}/{total_episodes}] {scenario['name']} rep{repeat}: "
                             f"ERROR: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    print(f"[{done}/{total_episodes}] {scenario['name']} rep{repeat}: "
                         f"ideal={scenario['ideal_action']} "
                         f"delegate={row['delegate_proposal']['action']} "
                         f"bank_id={row['sample_bank_id']}", flush=True)
        print(f"\nDone: {done - n_errors}/{total_episodes} episodes saved, "
             f"{n_errors} error(s), to {out}")
        return 1 if n_errors else 0
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
