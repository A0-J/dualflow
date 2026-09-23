"""B7e Phase 1 -- deterministic (0 API calls) validation of
experience_chain_experiment.run_chain() with a fake LLM client. Locks in
every invariant the B7e design (docs/experiments/agent_connected_eval.md
SS25) requires before any real API call is made:

  - store insertion order / latest-only selection
  - a stable episode (never clarified) is never stored
  - a clarified AND verified episode IS stored
  - E3 stored vs. not-stored changes what E4 selects as its latest history
    (the whole reason this chain is 4 episodes, not 3)
  - automatic historical override is always False (Option B, structural)
  - F6 (unverified-store contamination) / F7 (sequence-order violation)
    guards inside run_chain() never fire on a well-formed chain

Two hand-constructed fake-response paths exercise both branches of E3:

  Path A -- E3 samples STABLE (all export, entropy=0): resolve() never
    clarifies, so nothing is stored for E3. E4 still sees E1/E2's
    "summarize" experience.
  Path B -- E3 samples AMBIGUOUS (entropy=1.0): resolve() clarifies,
    Principal confirms "export", and a new verified experience IS stored.
    E4 now sees E3's "export" experience instead -- a different
    selected_history_action than Path A, from the identical E4 inputs.

n=4 per sampling call (not the real N=20 this experiment will eventually
use) -- this file only tests control flow, not real API behavior, so a
small n keeps the hand-built response queues legible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "diagnostics"
sys.path.insert(0, str(_DIAGNOSTICS_DIR))

from experience_chain_experiment import chain_is_seeded, run_chain, summarize_logs  # noqa: E402

from dualflow.llm import LLMResponse  # noqa: E402

PRINCIPAL_ID = "finance_lead_A"
TASK_CATEGORY = "external_audit_report"

EPISODES = [
    {"episode_id": "E1_august_seed", "goal": "Create an internal summary of the August report. Do not export.",
     "context": "August context.", "delegation": "Please prepare the August 2026 financial report."},
    {"episode_id": "E2_september_ambiguous", "goal": "Create an internal summary of the September report. Do not export.",
     "context": "September context.", "delegation": "Please prepare the September 2026 financial report."},
    {"episode_id": "E3_october_explicit_export", "goal": "Export the October report as a file.",
     "context": "October context.", "delegation": "This time, export the October 2026 financial report."},
    {"episode_id": "E4_november_observation", "goal": "Create an internal summary of the November report. Do not export.",
     "context": "November context.", "delegation": "Please prepare the November 2026 financial report."},
]


class _FakeLLMClient:
    """tests/test_clarification.py의 것과 동일한 패턴 -- 준비된 응답을
    호출 순서대로 돌려주는 순수 fake."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        if not self._responses:
            raise AssertionError("fake LLM client ran out of queued responses -- "
                                 "the hand-built response sequence doesn't match "
                                 "run_chain()'s actual call order.")
        return self._responses.pop(0)


def _structured(action: str, scope: str) -> LLMResponse:
    return LLMResponse(text=f"ACTION: {action}\nRESOURCE: file\nSCOPE: {scope}\nCONDITION: none")


def _ambiguous_pre(scope: str) -> list[LLMResponse]:
    return [_structured("export", scope), _structured("export", scope),
           _structured("summarize", scope), _structured("summarize", scope)]  # H=1.0


def _stable(action: str, scope: str, count: int = 4) -> list[LLMResponse]:
    return [_structured(action, scope) for _ in range(count)]  # H=0.0


_QUESTION = LLMResponse(text="Summarize or export?")

_SCOPES = {"E1_august_seed": "/reports/2026-08/", "E2_september_ambiguous": "/reports/2026-09/",
          "E3_october_explicit_export": "/reports/2026-10/", "E4_november_observation": "/reports/2026-11/"}


def _build_path_a() -> tuple[_FakeLLMClient, _FakeLLMClient]:
    """E3 samples stable (export, H=0) -- never clarified, never stored."""
    delegate_responses = (
        _ambiguous_pre(_SCOPES["E1_august_seed"]) + [_QUESTION] + _stable("summarize", _SCOPES["E1_august_seed"])
        + _ambiguous_pre(_SCOPES["E2_september_ambiguous"]) + [_QUESTION] + _stable("summarize", _SCOPES["E2_september_ambiguous"])
        + _stable("export", _SCOPES["E3_october_explicit_export"])  # E3: stable, no question/post
        + _ambiguous_pre(_SCOPES["E4_november_observation"]) + [_QUESTION] + _stable("summarize", _SCOPES["E4_november_observation"])
    )
    principal_responses = [
        LLMResponse(text="summarize"),  # E1
        LLMResponse(text="summarize"),  # E2
        # E3: no clarification, no answer call
        LLMResponse(text="summarize"),  # E4
    ]
    return _FakeLLMClient(delegate_responses), _FakeLLMClient(principal_responses)


def _build_path_c_unseeded() -> tuple[_FakeLLMClient, _FakeLLMClient]:
    """Every episode samples stable -- E1 never seeds the store, and it
    stays empty through E4. A valid real execution path (not forced/
    retried), used to confirm chain_is_seeded()==False and that
    summarize_logs() does not count this chain's episodes as "0%%
    exposure" in the history-advisory-exposure-rate denominator."""
    delegate_responses = (
        _stable("export", _SCOPES["E1_august_seed"])
        + _stable("export", _SCOPES["E2_september_ambiguous"])
        + _stable("export", _SCOPES["E3_october_explicit_export"])
        + _stable("export", _SCOPES["E4_november_observation"])
    )
    return _FakeLLMClient(delegate_responses), _FakeLLMClient([])


def _build_path_b() -> tuple[_FakeLLMClient, _FakeLLMClient]:
    """E3 samples ambiguous (H=1.0) -- clarified, Principal confirms
    "export", stored as a new verified experience."""
    delegate_responses = (
        _ambiguous_pre(_SCOPES["E1_august_seed"]) + [_QUESTION] + _stable("summarize", _SCOPES["E1_august_seed"])
        + _ambiguous_pre(_SCOPES["E2_september_ambiguous"]) + [_QUESTION] + _stable("summarize", _SCOPES["E2_september_ambiguous"])
        + _ambiguous_pre(_SCOPES["E3_october_explicit_export"]) + [_QUESTION] + _stable("export", _SCOPES["E3_october_explicit_export"])
        + _ambiguous_pre(_SCOPES["E4_november_observation"]) + [_QUESTION] + _stable("summarize", _SCOPES["E4_november_observation"])
    )
    principal_responses = [
        LLMResponse(text="summarize"),  # E1
        LLMResponse(text="summarize"),  # E2
        LLMResponse(text="export"),     # E3 -- genuine Principal-confirmed change
        LLMResponse(text="summarize"),  # E4
    ]
    return _FakeLLMClient(delegate_responses), _FakeLLMClient(principal_responses)


class _CombinedClient:
    """`run_chain()`은 하나의 `llm_client`만 받아 `PrincipalAgent`/
    `DelegateAgent` 둘 다에 넘긴다(real-API 모드와 동일한 호출 패턴 --
    `experience_provenance_smoke.py`도 그렇게 한다). fake 모드에서는
    Principal 쪽 호출과 Delegate 쪽 호출을 구분해서 서로 다른 큐에서
    꺼내야 하므로, instructions 문자열로 어느 쪽 호출인지 구분해 라우팅
    하는 아주 얇은 wrapper가 필요하다."""

    def __init__(self, delegate_client: _FakeLLMClient, principal_client: _FakeLLMClient):
        self._delegate = delegate_client
        self._principal = principal_client

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        if "Delegate's question" in input_text:  # PrincipalAgent.answer_clarification()
            return self._principal.generate(instructions=instructions, input_text=input_text)
        return self._delegate.generate(instructions=instructions, input_text=input_text)


def _run(path_builder) -> tuple:
    delegate_client, principal_client = path_builder()
    combined = _CombinedClient(delegate_client, principal_client)
    store, logs = run_chain(principal_id=PRINCIPAL_ID, task_category=TASK_CATEGORY,
                            episodes=EPISODES, llm_client=combined, n=4)
    return store, logs


class TestChainIntegrity:
    def test_store_insertion_order(self):
        store, logs = _run(_build_path_a)
        records = store.get(PRINCIPAL_ID, TASK_CATEGORY)
        assert [e.episode_id for e in records] == [
            "E1_august_seed", "E2_september_ambiguous", "E4_november_observation"]

    def test_e1_seed_is_clarified_and_stored(self):
        _, logs = _run(_build_path_a)
        e1 = logs[0]
        assert e1.selected_history_episode_id is None  # store was empty
        assert e1.decision_stage is None                # no history to consult at all
        assert e1.clarified is True
        assert e1.confirmed_action == "summarize"
        assert e1.stored is True
        assert e1.store_size_before == 0
        assert e1.store_size_after == 1


class TestStableEpisodeNeverStored:
    def test_e3_stable_in_path_a_is_not_stored(self):
        _, logs = _run(_build_path_a)
        e3 = logs[2]
        assert e3.clarified is False
        assert e3.stored is False
        assert e3.confirmed_action is None
        assert e3.store_size_before == e3.store_size_after == 2


class TestClarifiedVerifiedEpisodeIsStored:
    def test_e3_ambiguous_in_path_b_is_stored(self):
        _, logs = _run(_build_path_b)
        e3 = logs[2]
        assert e3.clarified is True
        assert e3.confirmed_action == "export"
        assert e3.stored is True
        assert e3.store_size_before == 2
        assert e3.store_size_after == 3


class TestSequentialAdaptation:
    """B7e를 4 episode로 하는 핵심 이유: 동일한 E4 입력인데, E3가
    저장됐는지 여부에 따라 E4가 실제로 다른 latest history를 본다."""

    def test_e4_sees_old_summarize_history_when_e3_was_not_verified(self):
        _, logs = _run(_build_path_a)
        e4 = logs[3]
        assert e4.selected_history_episode_id == "E2_september_ambiguous"
        assert e4.selected_history_action == "summarize"

    def test_e4_sees_new_export_history_when_e3_was_verified(self):
        _, logs = _run(_build_path_b)
        e4 = logs[3]
        assert e4.selected_history_episode_id == "E3_october_explicit_export"
        assert e4.selected_history_action == "export"

    def test_e4_outcome_itself_stays_governed_by_principal_not_history_in_either_path(self):
        """history가 advisory일 뿐이라는 핵심 주장의 가장 직접적인 증거:
        Path B에서 E4가 보는 history는 export인데, 그래도 Principal이
        실제로 확인해준 값(summarize)이 이깁니다 -- history가 outcome을
        강제하지 않습니다."""
        _, logs_a = _run(_build_path_a)
        _, logs_b = _run(_build_path_b)
        assert logs_a[3].confirmed_action == "summarize"
        assert logs_b[3].confirmed_action == "summarize"  # unchanged despite conflicting history
        assert logs_a[3].selected_history_action != logs_b[3].selected_history_action  # but history itself differed


class TestNonAuthoritativeSafety:
    def test_automatic_override_is_always_false_in_both_paths(self):
        for path_builder in (_build_path_a, _build_path_b):
            _, logs = _run(path_builder)
            for log in logs:
                assert log.automatic_override is False
                if log.decision_stage is not None:
                    assert log.decision_final_value == log.decision_baseline_value

    def test_e2_and_e4_advisory_relations_are_recorded_when_history_is_eligible(self):
        """relations는 자동 적용은 안 하지만, 계산되고 기록은 돼야 한다."""
        _, logs = _run(_build_path_a)
        e2 = logs[1]
        assert e2.decision_stage == "historical_evidence_requires_clarification"
        assert e2.decision_relations == {"export": "CONFLICT", "summarize": "SUPPORT"}

    def test_e4_relations_flip_direction_between_paths(self):
        """Path A에서 E4의 history는 summarize이므로 relation은
        export=CONFLICT/summarize=SUPPORT; Path B에서는 history가
        export이므로 정반대로 뒤집힌다 -- 두 경우 모두 override는
        일어나지 않는다(위 test에서 이미 확인)."""
        _, logs_a = _run(_build_path_a)
        _, logs_b = _run(_build_path_b)
        assert logs_a[3].decision_relations == {"export": "CONFLICT", "summarize": "SUPPORT"}
        assert logs_b[3].decision_relations == {"export": "SUPPORT", "summarize": "CONFLICT"}


class TestSequenceGuardsDoNotFalselyTrigger:
    """F6/F7 가드(run_chain() 내부 assert)는 정상 chain에서는 절대
    발동하지 않아야 한다 -- 발동하면 이 테스트가 AssertionError로
    실패한다(가드 자체가 켜져 있는지 확인하는 회귀 테스트)."""

    def test_well_formed_chain_never_trips_f6_or_f7_guards(self):
        for path_builder in (_build_path_a, _build_path_b):
            _run(path_builder)  # would raise AssertionError from run_chain() if either guard fired


class TestFakeClientExhaustion:
    def test_response_queues_are_exactly_consumed(self):
        """큐가 남거나 모자라면 이 hand-built response 구성 자체가 실제
        호출 순서와 안 맞는다는 뜻이다 -- 조용히 넘어가지 않는다."""
        delegate_client, principal_client = _build_path_a()
        combined = _CombinedClient(delegate_client, principal_client)
        run_chain(principal_id=PRINCIPAL_ID, task_category=TASK_CATEGORY,
                 episodes=EPISODES, llm_client=combined, n=4)
        assert delegate_client._responses == []
        assert principal_client._responses == []


class TestSameDistributionGuarantee:
    """B7e SS25 follow-up의 1번 요구사항: harness가 판단에 쓴 distribution
    과 clarification이 실제로 쓰는 pre_distribution이 정확히 같은
    object여야 한다 -- run_chain()이 `delegate.sample_candidates()`를
    episode당 정확히 한 번만 부르고, 그 결과를 `harness.decide()`와
    `clarifier.resolve(pre_distribution=...)` 둘 다에 넘긴다는 것을
    call-count로 검증한다(같은 큐를 두 번 소비했다면 카운트가 어긋난다)."""

    def test_delegate_call_count_matches_exactly_once_pre_sampling_per_episode(self):
        """Path A: E1/E2/E4는 ambiguous(4 pre + 1 question + 4 post = 9),
        E3는 stable(4 pre만). 만약 pre-sampling이 어딘가에서 중복
        호출됐다면 이 합계(9+9+4+9=31)를 넘어서다가 fake client가
        "큐 소진" AssertionError를 던진다 -- 즉 이 테스트가 그냥
        통과한다는 사실 자체가 이중 sampling이 없다는 증거다."""
        delegate_client, principal_client = _build_path_a()
        combined = _CombinedClient(delegate_client, principal_client)
        run_chain(principal_id=PRINCIPAL_ID, task_category=TASK_CATEGORY,
                 episodes=EPISODES, llm_client=combined, n=4)
        assert len(delegate_client.calls) == 31  # 9+9+4+9, not 9+9+4+9 + any extra pre-sampling


class TestUnseededChain:
    def test_e1_stable_never_forces_a_seed(self):
        _, logs = _run(_build_path_c_unseeded)
        e1 = logs[0]
        assert e1.clarified is False
        assert e1.stored is False
        assert chain_is_seeded(logs) is False

    def test_unseeded_chain_has_no_history_available_at_any_later_episode(self):
        _, logs = _run(_build_path_c_unseeded)
        for log in logs[1:]:
            assert log.selected_history_episode_id is None
            assert log.decision_stage is None  # nothing to consult -- not "no support", just absent


class TestSummarizeLogsMetricSeparation:
    """clarification_rate과 history_advisory_exposure_rate는 서로 다른
    causal 질문에 답한다 -- 하나로 합쳐지면 안 된다."""

    def test_clarification_rate_is_independent_of_history_availability(self):
        """Path A/Path B 둘 다 E1/E2/E4가 ambiguous이므로 clarified=True
        (E1은 history가 아예 없었는데도 clarified됐다) -- clarification은
        history 유무와 무관하게 현재 entropy만으로 결정된다."""
        _, logs = _run(_build_path_a)
        summary = summarize_logs([logs])
        # E1, E2, E4 clarified (ambiguous); E3 stable, not clarified.
        assert summary["clarification_rate"]["numerator_clarified"] == 3
        assert summary["clarification_rate"]["denominator_all_episodes"] == 4

    def test_history_advisory_exposure_excludes_unseeded_chain_from_denominator(self):
        """history가 한 번도 생긴 적 없는 chain은 exposure-rate 분모에
        아예 안 들어가야 한다 -- "0%% 노출"로 잘못 세면 안 된다."""
        _, logs_unseeded = _run(_build_path_c_unseeded)
        summary = summarize_logs([logs_unseeded])
        assert summary["history_advisory_exposure_rate"]["denominator_ambiguous_with_history_available"] == 0
        assert summary["n_unseeded_chains"] == 1
        assert summary["n_seeded_chains"] == 0

    def test_history_advisory_exposure_counts_only_ambiguous_episodes_with_available_history(self):
        """Path A: E2와 E4가 ambiguous + history 존재 + eligible ->
        exposure 분자/분모 모두 2. E1은 ambiguous였지만 history가 없었으므로
        분모에서 제외된다. E3는 history는 있었지만 stable이라 애초에
        ambiguous 집합에 안 들어간다."""
        _, logs = _run(_build_path_a)
        summary = summarize_logs([logs])
        assert summary["history_advisory_exposure_rate"] == {
            "numerator_eligible_and_consulted": 2,
            "denominator_ambiguous_with_history_available": 2,
            "note": summary["history_advisory_exposure_rate"]["note"],
        }

    def test_safety_integrity_counts_are_all_zero_on_well_formed_chains(self):
        for path_builder in (_build_path_a, _build_path_b, _build_path_c_unseeded):
            _, logs = _run(path_builder)
            summary = summarize_logs([logs])
            assert summary["automatic_historical_override_count"] == 0
            assert summary["unverified_store_contamination_count_F6"] == 0
            assert summary["sequence_order_violation_count_F7"] == 0
            assert summary["cross_facet_transfer_count"] == 0

    def test_store_update_count_matches_number_of_stored_episodes(self):
        _, logs_a = _run(_build_path_a)
        _, logs_b = _run(_build_path_b)
        assert summarize_logs([logs_a])["store_update_count"] == 3  # E1, E2, E4
        assert summarize_logs([logs_b])["store_update_count"] == 4  # E1, E2, E3, E4
