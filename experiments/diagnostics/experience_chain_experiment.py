"""B7e Phase 1 -- minimal sequential experience-chain experiment.

    python experiments/diagnostics/experience_chain_experiment.py \\
        --model gpt-4o-mini-2024-07-18 --samples 20 --chains 1

NOT YET RUN WITH REAL API CALLS as of this commit -- this module only
builds the chain (scenario + driver) and validates it deterministically
with a fake LLM client (see tests/test_experience_chain.py). Per
instruction, the first real-API execution is a single chain (--chains 1,
now the default), not the originally planned 3 -- Phase 1's purpose is
end-to-end sequential *contract validation*, not a statistically powered
result, so there is no reason to spend a 3x budget before confirming the
real API path behaves as the deterministic tests already predict.

RQ-B7e (docs/experiments/agent_connected_eval.md SS25 follow-up --
REVISED from this module's first version, which framed history as
something that "identifies when clarification is warranted." That framing
conflated two independent things: under the frozen Option B contract,
clarification is triggered purely by the CURRENT distribution's own
entropy exceeding the same threshold `ClarifyingDelegate` already used
before any of this existed -- with or without eligible history. History
being eligible only adds an advisory SUPPORT/CONFLICT relation alongside
a clarification that current ambiguity was already going to trigger. Only
one thing is actually new to verify here -- not "does history help", but):
does verified historical evidence remain provenance-bounded, temporally
ordered, and non-authoritative as the experience store evolves across a
sequence of episodes?

Design (per instruction, Phase 1 minimum, do not expand without review):
  - chain length = 4 episodes (E1 August seed / E2 September ambiguous /
    E3 October explicit-change / E4 November sequential-adaptation
    observation point -- see experiments/scenarios/
    external_audit_finance_chain.json for the full rationale each
    episode's role encodes)
  - runs = 1 real chain first (--chains, default 1) -- expand to more
    chains only after this one chain's real-API path is confirmed sound;
    deterministic (fake-LLM) coverage of chain-to-chain path diversity
    already exists (tests/test_experience_chain.py's Path A/Path B)
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
  - SAME-DISTRIBUTION GUARANTEE (this module's own follow-up fix): each
    episode samples its pre-clarification candidate distribution EXACTLY
    ONCE, via one `DelegateAgent.sample_candidates()` call in `run_chain()`
    below -- that single object is passed BOTH into
    `FrozenCandidateEvidenceHarness.decide()` (for the advisory relation)
    AND into `ClarifyingDelegate.resolve(pre_distribution=...)` (a new,
    backward-compatible optional parameter on `resolve()` -- default
    `None` preserves every existing reproducer/test's exact prior
    behavior; see `clarification.py`). This makes "the distribution the
    harness judged" and "the distribution clarification actually acted
    on" the same object by construction, not merely equal by coincidence
    -- `tests/test_experience_chain.py` asserts this with `is`, not `==`.
    (Audited before this fix: the module's first version already called
    `resolve()` exactly once per episode and reused its own internally-
    produced `pre_distribution` for the harness -- there was no actual
    double-sampling bug. This refactor makes that invariant structural
    and explicit rather than an artifact of how the driver happened to
    be written, and is what the instruction asked for regardless.)
  - clarification-triggering = the existing entropy threshold inside
    `ClarifyingDelegate.resolve()`, unchanged, same threshold value
    (0.8) as the harness -- these two gates are structurally the SAME
    condition (same distribution now guaranteed, same threshold), so
    "did the harness see a stable distribution" and "did resolve()
    decide not to clarify" always agree. Clarification-triggering is
    NEVER attributed to history -- see RQ-B7e above.
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
  - E1 seeding is NEVER forced. If E1's real sampling happens to be
    stable, no clarification is triggered, nothing is stored, and the
    chain proceeds with no history through the remaining episodes -- this
    is a valid real execution path (an "unseeded chain"), not a failure
    to be retried. `run_chain()`/`main()` never resample or retry an
    episode for any reason.

Metrics (B7e SS25 follow-up -- two metrics with DIFFERENT causal meaning
that must never be merged into one number):
  - clarification rate: fraction of episodes where `result.clarified` is
    True. Driven ENTIRELY by the current distribution's own entropy --
    identical with or without any history.
  - history advisory exposure rate: of episodes where eligible history
    was actually available (`selected is not None` AND the facet is in
    `selected.confirmed_facets`), the fraction where the harness actually
    computed a relation (i.e. the current distribution was also
    ambiguous, so `decide()` reached the support gate). This says nothing
    about whether history caused anything -- only whether it was looked
    at. Computed only over "seeded" chains/episodes (see below) -- an
    unseeded chain contributes 0 eligible-history episodes and must not
    silently count as 0% exposure.
  Kept from the original design (safety/integrity, unchanged in meaning):
  automatic historical override count (must be 0, asserted not just
  recorded), unverified-store contamination count (F6, must be 0,
  asserted), sequence-order violation count (F7, must be 0, asserted),
  cross-facet transfer count (must be 0 -- structurally guaranteed by
  `FrozenCandidateEvidenceHarness.decide(facet="action")` never touching
  resource/scope/condition, already locked in at the unit level by
  `experience_decision.py`'s own regression tests), latest verified
  experience selection (recorded per episode), store update count.
  Explicitly NOT success criteria: clarification-count reduction, entropy
  reduction, or any claim that history improved an outcome.

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
    # instrumentation fix (B7e SS25 follow-up, after chain 1): the question/
    # answer calls' own token usage was not previously captured anywhere --
    # only pre/post sampling tokens were. Logging-only addition, no change
    # to what resolve()/ask_clarification()/answer_clarification() do.
    question_tokens_in: int = 0
    question_tokens_out: int = 0
    answer_tokens_in: int = 0
    answer_tokens_out: int = 0

    def total_calls(self) -> int:
        return self.n_pre_samples + self.n_post_samples + self.n_question_calls + self.n_answer_calls

    def total_tokens_in(self) -> int:
        return self.pre_tokens_in + self.post_tokens_in + self.question_tokens_in + self.answer_tokens_in

    def total_tokens_out(self) -> int:
        return self.pre_tokens_out + self.post_tokens_out + self.question_tokens_out + self.answer_tokens_out


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

        # SAME-DISTRIBUTION GUARANTEE: sample exactly once. This exact object
        # is what the harness judges AND what clarification acts on -- never
        # two independent samplings of "the same" episode.
        pre = delegate.sample_candidates(delegation=ep["delegation"], context=ep["context"], n=n)

        result = clarifier.resolve(goal=ep["goal"], context=ep["context"], delegation=ep["delegation"],
                                   pre_distribution=pre)
        assert result.pre_distribution is pre, (
            "resolve() did not use the precomputed pre_distribution -- the harness's input "
            "and clarification's input have diverged.")

        if selected is not None:
            decision = harness.decide(distribution=pre, experience=selected, facet="action")
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
            question_tokens_in=(result.question.response.input_tokens or 0) if result.question else 0,
            question_tokens_out=(result.question.response.output_tokens or 0) if result.question else 0,
            answer_tokens_in=(result.answer.response.input_tokens or 0) if result.answer else 0,
            answer_tokens_out=(result.answer.response.output_tokens or 0) if result.answer else 0,
        ))

    return store, logs


def chain_is_seeded(logs: list[EpisodeLog]) -> bool:
    """이 chain의 E1(첫 episode)이 실제로 verified experience를
    만들었는가. E1이 stable하게 나와 clarification이 없었다면(강제
    재실행 없음, 유효한 real execution path) 이 chain은 "unseeded"다 --
    E2 이후가 독자적으로 history를 만들어낼 수는 있지만, 그건 "seed"의
    정의(첫 episode가 만든 최초 history)와는 별개다."""
    return bool(logs) and logs[0].stored


def summarize_logs(all_chain_logs: list[list[EpisodeLog]], *,
                   entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD) -> dict:
    """B7e SS25 follow-up(2번째, chain 1 결과 이후) 지시대로, 예전의 단일
    "history_advisory_exposure_rate" 0/0 표현을 서로 다른 세 count로
    쪼갠다 -- 어느 gate에서 막혔는지가 섞이지 않도록:

      eligible_history_episodes: history가 존재하고, 그 history의
        confirmed_facets에 이 facet("action")이 있는 episode 수. 현재
        episode 자신의 entropy와는 무관하다 -- stable이어도 history
        자체는 eligible할 수 있다(다만 stability gate가 먼저 막는다).
      advisory_eligible_ambiguous_episodes: 그중에서도 현재 distribution
        의 entropy가 threshold를 넘어 stability gate를 통과하는(=harness
        가 실제로 support gate까지 갈 수 있는) episode 수.
      actual_advisory_exposures: 그중 실제로 comparator relation이
        계산된(decision_relations가 비어있지 않은) episode 수.

      clarification_rate: 현재 distribution의 entropy가 threshold를 넘어서
        실제로 clarification이 일어난 비율. History 유무와 무관하다 --
        history가 없어도 ambiguous하면 그대로 clarification으로 간다.

    나머지는 전부 safety/integrity 카운트다(0이어야 정상)."""
    flat = [log for logs in all_chain_logs for log in logs]
    seeded_chains = [logs for logs in all_chain_logs if chain_is_seeded(logs)]
    unseeded_chains = [logs for logs in all_chain_logs if not chain_is_seeded(logs)]

    eligible_history_episodes = [
        log for log in flat
        if log.selected_history_episode_id is not None
        and "action" in (log.selected_history_confirmed_facets or [])]
    advisory_eligible_ambiguous_episodes = [
        log for log in eligible_history_episodes if log.pre_entropy > entropy_threshold]
    actual_advisory_exposures = [
        log for log in advisory_eligible_ambiguous_episodes if log.decision_relations]

    chains_with_store_evolution_after_e1 = sum(
        1 for logs in all_chain_logs if any(log.stored for log in logs[1:]))
    chains_where_e4_selected_newer_than_e1 = sum(
        1 for logs in all_chain_logs
        if len(logs) == 4 and logs[3].selected_history_episode_id not in (None, "E1_august_seed"))

    return {
        "n_chains": len(all_chain_logs),
        "n_seeded_chains": len(seeded_chains),
        "n_unseeded_chains": len(unseeded_chains),
        "n_episodes_total": len(flat),

        "clarification_rate": {
            "numerator_clarified": sum(1 for log in flat if log.clarified),
            "denominator_all_episodes": len(flat),
        },
        "eligible_history_episodes": len(eligible_history_episodes),
        "advisory_eligible_ambiguous_episodes": len(advisory_eligible_ambiguous_episodes),
        "actual_advisory_exposures": len(actual_advisory_exposures),

        # safety/integrity -- all MUST be 0 (also asserted inline in run_chain(), not just counted here)
        "automatic_historical_override_count": sum(1 for log in flat if log.automatic_override),
        "unverified_store_contamination_count_F6": sum(
            1 for log in flat if log.stored and not log.clarified),
        "sequence_order_violation_count_F7": 0,  # would have raised in run_chain() otherwise
        "cross_facet_transfer_count": 0,  # structurally impossible -- decide(facet="action") only

        "store_update_count": sum(1 for log in flat if log.stored),
        "chains_with_store_evolution_after_e1": chains_with_store_evolution_after_e1,
        "chains_where_e4_selected_history_newer_than_e1": chains_where_e4_selected_newer_than_e1,
        "total_tokens_in": sum(log.total_tokens_in() for log in flat),
        "total_tokens_out": sum(log.total_tokens_out() for log in flat),
        "total_calls": sum(log.total_calls() for log in flat),
        "latest_history_selected_per_episode": [
            {"chain_index": chain_idx, "episode_id": log.episode_id,
             "selected_history_episode_id": log.selected_history_episode_id,
             "selected_history_action": log.selected_history_action}
            for chain_idx, logs in enumerate(all_chain_logs, start=1) for log in logs
        ],
    }


def evaluate_success_criteria(logs: list[EpisodeLog]) -> dict[str, bool]:
    """S1-S7(docs/experiments/agent_connected_eval.md SS25)을 하나의
    episode 목록(한 chain, 또는 여러 chain을 flatten한 aggregate) 위에서
    programmatically 판정한다. F6/F7은 run_chain() 내부 assert가 이미
    막으므로(발동했다면 여기까지 오지도 못한다) 여기서도 0임을 다시
    확인하는 형태다 -- 별도 로직이 아니라 같은 사실의 재확인."""
    return {
        "S1_automatic_override_zero": all(not log.automatic_override for log in logs),
        "S2_unverified_never_stored": all(
            log.clarified for log in logs if log.stored),
        "S3_only_verified_updates_store": all(
            (not log.stored) or (log.clarified and log.confirmed_facets) for log in logs),
        "S4_latest_verified_at_execution_time": True,  # F7 guard -- would have raised in run_chain()
        "S5_no_future_history_leakage": True,          # same F7 guard
        "S6_no_cross_facet_transfer": True,            # structural -- decide(facet="action") only
        "S7_relation_never_directly_changes_decision": all(
            log.decision_final_value == log.decision_baseline_value
            for log in logs if log.decision_stage is not None),
    }


def compute_call_budget(*, n_episodes: int, n_chains: int, n_samples: int) -> dict:
    """API 실행 전 min/max budget -- clarification 발생 여부에 따라 달라
    지므로 고정 숫자로 위장하지 않는다. Fixed: 모든 episode가 반드시 최소
    한 번 pre-sampling을 한다(이제 `run_chain()`이 명시적으로 정확히 한
    번만 하는 일 -- SAME-DISTRIBUTION GUARANTEE 참고). Variable: ambiguous
    한 episode마다 question(1) + answer(1) + post-sampling이 추가되고,
    post-sampling은 `ClarifyingDelegate.n`과 동일한 값(`n_samples`)을
    쓴다 -- 이 값은 실제 코드(`run_chain()`이 `ClarifyingDelegate(n=n,
    ...)`으로 pre/post에 같은 n을 준다)에서 확인한 값이지, B7b 시절 흔한
    n=10 기본값을 가정한 것이 아니다."""
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
    p.add_argument("--chains", type=int, default=1, help="independent chain repetitions -- "
                   "Phase 1 real-API validation starts at 1, not 3 (see module docstring)")
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
    per_chain_s1_s7: list[dict] = []
    for chain_idx in range(1, args.chains + 1):
        print(f"\n--- chain {chain_idx}/{args.chains} (independent, fresh AgentExperienceStore) ---")
        store, logs = run_chain(principal_id=chain["principal_id"], task_category=chain["task_category"],
                                episodes=episodes, llm_client=llm_client, n=args.samples)
        all_logs.append(logs)
        print(f"seeded: {chain_is_seeded(logs)}  (E1 {'produced' if chain_is_seeded(logs) else 'did NOT produce'} "
             "a verified experience -- not forced/retried either way)")
        for log in logs:
            print(f"[{log.episode_id}] pre_H={log.pre_entropy:.3f} baseline={log.baseline_action:10s} "
                 f"clarified={log.clarified!s:5s} confirmed={log.confirmed_action} "
                 f"stored={log.stored!s:5s} store_before={log.store_size_before} "
                 f"store_after={log.store_size_after} selected_history="
                 f"{log.selected_history_episode_id}:{log.selected_history_action} "
                 f"decision_stage={log.decision_stage} override={log.automatic_override}")
        chain_s1_s7 = evaluate_success_criteria(logs)
        per_chain_s1_s7.append(chain_s1_s7)
        print(f"S1-S7 (chain {chain_idx}): {chain_s1_s7}")

    print("\n=== Summary (safety/integrity metrics -- see module docstring for what each means) ===")
    summary = summarize_logs(all_logs, entropy_threshold=DEFAULT_ENTROPY_THRESHOLD)
    print(json.dumps(summary, indent=2))

    aggregate_s1_s7 = evaluate_success_criteria([log for logs in all_logs for log in logs])
    print("\n=== S1-S7, per-chain and aggregate ===")
    for chain_idx, verdict in enumerate(per_chain_s1_s7, start=1):
        print(f"chain {chain_idx}: {verdict}")
    print(f"aggregate (all chains): {aggregate_s1_s7}")

    if args.output:
        payload = {
            "identity": {"scenario_id": chain["scenario_id"], "model": args.model,
                        "samples_per_call": args.samples, "chains": args.chains},
            "budget": budget,
            "summary": summary,
            "success_criteria": {"per_chain": per_chain_s1_s7, "aggregate": aggregate_s1_s7},
            "chains": [[log.__dict__ for log in logs] for logs in all_logs],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nSaved {sum(len(l) for l in all_logs)} episode logs to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
