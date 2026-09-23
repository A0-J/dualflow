"""B7e Phase 1 -- minimal sequential experience-chain experiment.

    python experiments/diagnostics/experience_chain_experiment.py \\
        --model gpt-4o-mini-2024-07-18 --samples 20 --chains 3

NOT YET RUN WITH REAL API CALLS as of this commit -- this module only
builds the chain (scenario + driver) and validates it deterministically
with a fake LLM client (see tests/test_experience_chain.py). Per
instruction, no API calls are made until the design below is reviewed.

RQ-B7e (docs/experiments/agent_connected_eval.md SS25, replacing the
pre-v3 "does history reduce uncertainty" framing): across sequential
episodes, can verified historical semantic evidence remain
provenance-bounded and non-authoritative while still identifying when
clarification is warranted, and while only Principal-verified outcomes
(never merely-stable model output) enter the store that the next episode
reads?

Design (per instruction, Phase 1 minimum, do not expand without review):
  - chain length = 4 episodes (E1 August seed / E2 September ambiguous /
    E3 October explicit-change / E4 November sequential-adaptation
    observation point -- see experiments/scenarios/
    external_audit_finance_chain.json for the full rationale each
    episode's role encodes)
  - runs = 3 independent chains (--chains, default 3)
  - experience selection = latest-only, reusing the exact recency
    convention ExperienceAwareDelegate already established
    (`all_experiences[-self.max_experiences:]` with max=1 here):
    `experiences[-1] if experiences else None`. No aggregation, no
    voting, no weighting, no similarity retrieval, no per-facet
    search-back -- explicitly out of scope for Phase 1.
  - v3 consumption = FrozenCandidateEvidenceHarness (frozen at commit
    0f1cf7c, Option B conservative abstention), completely unmodified.
    Historical evidence is looked at purely to compute the SUPPORT/
    CONFLICT relation audit trail against each episode's OWN pre-
    clarification candidate distribution -- never injected into any
    generation prompt (ExperienceAwareDelegate's v1/v2 natural-language
    injection is deliberately NOT used anywhere in this chain, per the
    v3 architectural boundary B7d.6 established).
  - clarification-triggering = the existing entropy threshold inside
    `ClarifyingDelegate.resolve()`, unchanged, same threshold value
    (0.8) as the harness -- these two gates are structurally the SAME
    condition (same distribution, same threshold), so "did the harness
    see a stable distribution" and "did resolve() decide not to
    clarify" always agree.
  - storage = `build_verified_experience()`'s existing gate, unchanged.
    A stable model output that never went through clarification
    (`result.clarified is False`) is NEVER added to the store -- this is
    the chain-level form of the project's standing invariant "a stable
    model output is not the same thing as a verified historical fact."
    An experience is stored only if `build_verified_experience()`
    returns non-None AND its `confirmed_facets` is non-empty (defends
    against the theoretical edge case where clarification happened but
    the target facet still didn't converge -- see build_verified_
    experience()'s own docstring; unreachable under the default
    max_verified_entropy=0.0 this chain uses, kept explicit anyway).

Failure taxonomy extension (F1-F5 unchanged, two new sequence-specific
codes only, per instruction -- "새 taxonomy를 많이 만들 필요 없다"):
  F6 unverified-store contamination -- semantic output that never went
     through Principal clarification/verification enters the store.
     Guarded structurally (see `stored` computation above) and asserted
     explicitly in `run_chain()` below (`assert result.clarified`).
  F7 sequence-order violation -- an episode consults a "historical"
     experience that was not actually verified-and-stored before this
     episode ran (i.e., not really in the past at execution time).
     Guarded explicitly in `run_chain()` below via `seen_episode_ids`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAGNOSTICS_DIR))
from experience_transfer import load_scenario as _load_base_scenario  # noqa: E402

_EXPERIMENTS_DIR = _DIAGNOSTICS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
from agent_smoke import (  # noqa: E402
    EXAMPLE_AMBIGUOUS_DELEGATION, EXAMPLE_CONTEXT, EXAMPLE_GOAL, build_llm_client,
)

from dualflow.agent_experience import AgentExperienceStore, build_verified_experience  # noqa: E402
from dualflow.clarification import ClarifyingDelegate  # noqa: E402
from dualflow.delegate_agent import DelegateAgent  # noqa: E402
from dualflow.experience_decision import FrozenCandidateEvidenceHarness  # noqa: E402
from dualflow.principal_agent import PrincipalAgent  # noqa: E402

_SCENARIOS_DIR = _EXPERIMENTS_DIR / "scenarios"
DEFAULT_ENTROPY_THRESHOLD = 0.8  # same value ClarifyingDelegate/FrozenCandidateEvidenceHarness both default to


def load_chain_scenario(name: str = "external_audit_finance_chain") -> dict:
    path = _SCENARIOS_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def episode_texts(chain: dict, base_scenario_name: str = "external_audit_finance") -> list[dict]:
    """4 episode의 실제 goal/context/delegation 문자열을 만든다. E1은
    agent_smoke.py, E2의 context/delegation은 external_audit_finance.json
    의 current_episode를 그대로(재타이핑 없이) 재사용한다 -- B7d.3의
    wording-drift 교훈(tests/test_scenario_reproducibility.py) 그대로.
    E3/E4는 chain scenario 파일 자신의 새 canonical text를 그대로 쓴다."""
    base = _load_base_scenario(base_scenario_name)
    texts = []
    for ep in chain["episodes"]:
        eid = ep["episode_id"]
        if eid == "E1_august_seed":
            texts.append({"episode_id": eid, "goal": EXAMPLE_GOAL, "context": EXAMPLE_CONTEXT,
                         "delegation": EXAMPLE_AMBIGUOUS_DELEGATION})
        elif eid == "E2_september_ambiguous":
            texts.append({"episode_id": eid, "goal": ep["goal"],
                         "context": base["current_episode"]["context"],
                         "delegation": base["current_episode"]["delegation"]})
        else:
            texts.append({"episode_id": eid, "goal": ep["goal"], "context": ep["context"],
                         "delegation": ep["delegation"]})
    return texts


@dataclass
class EpisodeLog:
    """B7e primary metrics(SS25 §6 A-E), 한 episode 실행 결과 전부."""

    episode_id: str

    # A. chain integrity
    store_size_before: int
    selected_history_episode_id: str | None
    selected_history_action: str | None
    selected_history_confirmed_facets: list[str] | None
    stored: bool
    store_size_after: int

    # B. advisory activation / D. non-authoritative safety
    decision_stage: str | None            # None <=> no history was available to consult at all
    decision_relations: dict[str, str]
    decision_baseline_value: str | None
    decision_final_value: str | None
    automatic_override: bool              # MUST always be False under Option B -- asserted, not just recorded

    # C. clarification outcome
    clarified: bool
    target_facets: list[str]
    pre_entropy: float
    post_entropy: float | None
    baseline_action: str
    confirmed_action: str | None
    confirmed_facets: list[str]

    # secondary
    n_pre_samples: int = 0
    n_post_samples: int = 0
    n_question_calls: int = 0
    n_answer_calls: int = 0
    pre_tokens_in: int = 0
    pre_tokens_out: int = 0
    post_tokens_in: int = 0
    post_tokens_out: int = 0

    def total_calls(self) -> int:
        return self.n_pre_samples + self.n_post_samples + self.n_question_calls + self.n_answer_calls


def run_chain(*, principal_id: str, task_category: str, episodes: list[dict], llm_client,
             n: int, entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
             ) -> tuple[AgentExperienceStore, list[EpisodeLog]]:
    """4개 episode를 순서대로 한 번 실행한다 -- store는 이 호출 안에서만
    산다(체인 간에 공유되지 않는다, 매 chain은 독립적인 store로
    시작한다). 매 episode 시작 시점에 store를 읽어 latest-only experience
    를 고르고(F7 가드), FrozenCandidateEvidenceHarness로 advisory relation
    만 계산한 뒤(자동 적용 없음, Option B), 기존 ClarifyingDelegate.
    resolve()를 그대로 호출해 실제 clarification을 (필요하면) 진행하고,
    build_verified_experience()가 승인한 경우에만 store에 추가한다(F6
    가드)."""
    store = AgentExperienceStore()
    principal = PrincipalAgent(llm=llm_client)
    delegate = DelegateAgent(llm=llm_client)
    clarifier = ClarifyingDelegate(principal=principal, delegate=delegate, n=n,
                                   entropy_threshold=entropy_threshold)
    harness = FrozenCandidateEvidenceHarness(entropy_threshold=entropy_threshold)

    seen_episode_ids: set[str] = set()  # F7 guard: only episodes already processed may be "history"
    logs: list[EpisodeLog] = []

    for ep in episodes:
        before = store.get(principal_id, task_category)
        store_size_before = len(before)
        selected = before[-1] if before else None

        if selected is not None:
            # F7 guard -- the selected experience must have been produced by
            # an episode this loop has already finished, never a future one.
            assert selected.episode_id in seen_episode_ids, (
                f"sequence-order violation (F7): episode {ep['episode_id']!r} selected "
                f"history from {selected.episode_id!r}, which has not been processed yet.")

        result = clarifier.resolve(goal=ep["goal"], context=ep["context"], delegation=ep["delegation"])

        if selected is not None:
            decision = harness.decide(distribution=result.pre_distribution, experience=selected,
                                      facet="action")
            decision_stage = decision.stage
            decision_relations = {k: v.value for k, v in decision.relations.items()}
            decision_baseline_value = decision.baseline_value
            decision_final_value = decision.final_value
            # D. non-authoritative safety -- MUST be False by construction (Option B
            # never overrides); asserted here, not just recorded, so a future
            # regression in experience_decision.py would fail this chain loudly.
            automatic_override = decision.final_value != decision.baseline_value
            assert not automatic_override, (
                f"automatic historical override detected at {ep['episode_id']!r} -- "
                "this must be structurally impossible under the frozen Option B contract "
                "(commit 0f1cf7c).")
        else:
            decision_stage = None
            decision_relations = {}
            decision_baseline_value = None
            decision_final_value = None
            automatic_override = False

        experience = build_verified_experience(
            result, principal_id=principal_id, task_category=task_category,
            delegation=ep["delegation"], episode_id=ep["episode_id"])
        # F6 guard -- only a genuinely clarified-and-confirmed episode may be stored.
        stored = experience is not None and len(experience.confirmed_facets) > 0
        if stored:
            assert result.clarified, (
                f"unverified-store contamination (F6): {ep['episode_id']!r} would have "
                "been stored without ever going through Principal clarification.")
            store.add(experience)

        seen_episode_ids.add(ep["episode_id"])
        after = store.get(principal_id, task_category)

        logs.append(EpisodeLog(
            episode_id=ep["episode_id"],
            store_size_before=store_size_before,
            selected_history_episode_id=selected.episode_id if selected else None,
            selected_history_action=selected.confirmed_interpretation.action if selected else None,
            selected_history_confirmed_facets=sorted(selected.confirmed_facets) if selected else None,
            stored=stored,
            store_size_after=len(after),
            decision_stage=decision_stage,
            decision_relations=decision_relations,
            decision_baseline_value=decision_baseline_value,
            decision_final_value=decision_final_value,
            automatic_override=automatic_override,
            clarified=result.clarified,
            target_facets=sorted(result.question.target_facets) if result.question else [],
            pre_entropy=result.pre_distribution.entropy,
            post_entropy=result.post_distribution.entropy if result.post_distribution else None,
            baseline_action=result.pre_distribution.top.action,
            confirmed_action=experience.confirmed_interpretation.action if experience else None,
            confirmed_facets=sorted(experience.confirmed_facets) if experience else [],
            n_pre_samples=result.pre_distribution.n_samples,
            n_post_samples=result.post_distribution.n_samples if result.post_distribution else 0,
            n_question_calls=1 if result.clarified else 0,
            n_answer_calls=1 if result.clarified else 0,
            pre_tokens_in=sum(r.input_tokens or 0 for r in result.pre_distribution.responses),
            pre_tokens_out=sum(r.output_tokens or 0 for r in result.pre_distribution.responses),
            post_tokens_in=(sum(r.input_tokens or 0 for r in result.post_distribution.responses)
                            if result.post_distribution else 0),
            post_tokens_out=(sum(r.output_tokens or 0 for r in result.post_distribution.responses)
                             if result.post_distribution else 0),
        ))

    return store, logs


def compute_call_budget(*, n_episodes: int, n_chains: int, n_samples: int) -> dict:
    """API 실행 전 min/max budget -- clarification 발생 여부에 따라 달라
    지므로 고정 숫자로 위장하지 않는다. Fixed: 모든 episode가 반드시 최소
    한 번 pre-sampling을 한다(resolve()가 항상 먼저 하는 일). Variable:
    ambigu한 episode마다 question(1) + answer(1) + post-sampling(n_samples)
    이 추가된다."""
    total_episodes = n_episodes * n_chains
    fixed = total_episodes * n_samples
    per_ambiguous_episode = 1 + 1 + n_samples
    return {
        "total_episodes": total_episodes,
        "fixed_candidate_calls": fixed,
        "per_ambiguous_episode_extra_calls": per_ambiguous_episode,
        "minimum_total_calls": fixed,  # 0 episodes ambiguous
        "maximum_total_calls": fixed + total_episodes * per_ambiguous_episode,  # all ambiguous
    }


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="external_audit_finance_chain")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--samples", type=int, default=20, help="N per pre/post sampling call, "
                   "matching the B7d/v3 convention (not B7b's own default of 10)")
    p.add_argument("--chains", type=int, default=3, help="independent chain repetitions")
    p.add_argument("--budget-only", action="store_true",
                   help="print the min/max call budget and exit -- no API client built, "
                        "no API calls made")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    chain = load_chain_scenario(args.scenario)
    budget = compute_call_budget(n_episodes=len(chain["episodes"]), n_chains=args.chains,
                                 n_samples=args.samples)

    print("=== B7e Phase 1 -- sequential experience-chain experiment ===")
    print(f"scenario: {chain['scenario_id']} v{chain['scenario_version']}")
    print(f"principal_id={chain['principal_id']!r} task_category={chain['task_category']!r}")
    print(f"chain length: {len(chain['episodes'])} episodes x {args.chains} chains "
         f"= {budget['total_episodes']} episode executions")
    print(f"N per sampling call: {args.samples}")
    print()
    print("=== API call budget (candidate generation only; comparator/harness: 0 calls) ===")
    print(f"fixed candidate calls (every episode always pre-samples): {budget['fixed_candidate_calls']}")
    print(f"per-ambiguous-episode extra (1 question + 1 answer + {args.samples} post-samples): "
         f"{budget['per_ambiguous_episode_extra_calls']}")
    print(f"minimum total calls (0/{budget['total_episodes']} episodes ambiguous): "
         f"{budget['minimum_total_calls']}")
    print(f"maximum total calls ({budget['total_episodes']}/{budget['total_episodes']} "
         f"episodes ambiguous): {budget['maximum_total_calls']}")

    if args.budget_only:
        return 0

    llm_client = build_llm_client(args.model)
    episodes = episode_texts(chain)

    all_logs: list[list[EpisodeLog]] = []
    for chain_idx in range(1, args.chains + 1):
        print(f"\n--- chain {chain_idx}/{args.chains} ---")
        store, logs = run_chain(principal_id=chain["principal_id"], task_category=chain["task_category"],
                                episodes=episodes, llm_client=llm_client, n=args.samples)
        all_logs.append(logs)
        for log in logs:
            print(f"[{log.episode_id}] pre_H={log.pre_entropy:.3f} baseline={log.baseline_action:10s} "
                 f"clarified={log.clarified!s:5s} confirmed={log.confirmed_action} "
                 f"stored={log.stored!s:5s} store_before={log.store_size_before} "
                 f"store_after={log.store_size_after} selected_history="
                 f"{log.selected_history_episode_id}:{log.selected_history_action} "
                 f"decision_stage={log.decision_stage} override={log.automatic_override}")

    if args.output:
        payload = {
            "identity": {"scenario_id": chain["scenario_id"], "model": args.model,
                        "samples_per_call": args.samples, "chains": args.chains},
            "budget": budget,
            "chains": [[log.__dict__ for log in logs] for logs in all_logs],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nSaved {sum(len(l) for l in all_logs)} episode logs to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
