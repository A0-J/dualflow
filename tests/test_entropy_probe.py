"""entropy_probe 하네스의 배관(파싱→MLE→entropy→objective count) 검증.

mock sampler로만 테스트한다 — 이건 "entropy가 실측됐다"를 증명하는 게 아니라
"하네스가 API 키만 생기면 정상 작동할 준비가 됐다"를 증명하는 테스트다.
"""

from __future__ import annotations

import math

from dualflow.entropy_probe import (
    FILE_INVENTORY, SCENARIO_REDESIGN_CASES, SPEC_OBJECTIVE_REFERENTS,
    judge_scenario_validation, make_deterministic_mock,
    objective_referent_count, run_probe, run_probe_suite,
    run_scenario_validation,
)
from dualflow.semantic import Interpretation as I


class TestObjectiveReferentCount:
    """느낌이 아니라 얼린 인벤토리로 센다."""

    def test_specific_folder_counts_its_own_files(self):
        assert objective_referent_count("/reports/2026-08/") == 1
        assert objective_referent_count("/finance/") == 2

    def test_broader_scope_sums_subfolders(self):
        assert objective_referent_count("/reports/") == 2  # 2026-08 + 2026-09
        assert objective_referent_count("*") == sum(len(v) for v in FILE_INVENTORY.values())

    def test_unknown_scope_counts_zero(self):
        assert objective_referent_count("/nonexistent/") == 0


class TestProbePlumbing:
    """mock으로 하네스 자체가 올바르게 배선됐는지 — 실제 LLM 없이도 지금 돈다."""

    def test_single_dominant_candidate_gives_near_zero_entropy(self):
        target = I("read", "file", "/reports/2026-08/", label="8월 리포트 읽기")
        mock = make_deterministic_mock({target: 1.0})
        r = run_probe("지난달 매출 리포트 읽고 요약해줘", mock, n=20,
                       hypothesis_label="명확(가설)")
        assert r.measured_entropy < 1e-6
        assert r.n_parsed == 20
        assert r.objective_referent_count == 1

    def test_split_candidates_give_positive_entropy(self):
        a = I("export", "file", "/reports/", label="반출")
        b = I("read", "file", "/reports/", label="열람")
        mock = make_deterministic_mock({a: 0.5, b: 0.5})
        r = run_probe("필요한 데이터 좀 확인해서 처리해줘", mock, n=20,
                       hypothesis_label="모호(가설)")
        assert math.isclose(r.measured_entropy, 1.0, abs_tol=0.05)

    def test_unparseable_responses_are_not_silently_dropped_from_denominator(self):
        target = I("read", "file", "/reports/2026-08/", label="8월 리포트 읽기")
        mock = make_deterministic_mock({target: 1.0})
        r = run_probe("...", mock, n=20)
        assert r.n_requested == 20  # 분모는 그대로 유지 — 파싱 실패도 측정치다

    def test_suite_runs_multiple_cases_and_keeps_hypothesis_labels(self):
        target = I("read", "file", "/reports/2026-08/", label="8월 리포트 읽기")
        mock = make_deterministic_mock({target: 1.0})
        cases = [("지난달 매출 리포트 읽고 요약해줘", "명확(가설)"),
                 ("필요한 데이터 좀 확인해서 처리해줘", "모호(가설)")]
        results = run_probe_suite(cases, mock, n=10)
        assert [r.hypothesis_label for r in results] == ["명확(가설)", "모호(가설)"]
        assert all(r.n_requested == 10 for r in results)


class TestObjectiveReferentsAreDecoupledFromModelOutput:
    """회귀 방지: 최초 구현은 objective_referent_count(_dominant_scope(parsed))
    로 모델이 가장 많이 고른 답에서 거꾸로 참조 개수를 셌다 — "entropy가
    objective referent count와 상관있는가"를 주장할 수 없게 만드는 순환이었다.
    이제 referent count는 spec 문장에 미리 등록된 값만 쓰고, 모델이 뭘
    답했는지는 전혀 참조하지 않는다는 걸 고정한다."""

    def test_referent_count_ignores_model_answer_entirely(self):
        spec = "지난달 매출 리포트 읽고 요약해줘"  # 등록값 1
        # 모델이 무엇을 답하든(여기서는 완전히 다른 넓은 scope) referent count는
        # spec 하나로만 결정되고 변하지 않는다.
        wide = I("read", "file", "*", label="전체")
        mock = make_deterministic_mock({wide: 1.0})
        r = run_probe(spec, mock, n=10)
        assert r.objective_referent_count == SPEC_OBJECTIVE_REFERENTS[spec] == 1

    def test_context_dependent_specs_are_explicitly_undefined(self):
        """"이번에도"/"이 요약"처럼 이전 턴을 전제하는 spec은 단일턴 probe로는
        참조 개수가 정의되지 않는다 — 억지로 숫자를 매기지 않고 None."""
        for spec, ref in SPEC_OBJECTIVE_REFERENTS.items():
            if "이번에도" in spec or spec.startswith("이 요약"):
                assert ref is None, spec

    def test_explicit_override_takes_precedence_over_registry(self):
        target = I("read", "file", "/reports/2026-08/", label="8월")
        mock = make_deterministic_mock({target: 1.0})
        r = run_probe("지난달 매출 리포트 읽고 요약해줘", mock, n=5,
                       objective_referents=999)
        assert r.objective_referent_count == 999

    def test_unregistered_spec_defaults_to_none_not_a_guess(self):
        target = I("read", "file", "/reports/2026-08/", label="8월")
        mock = make_deterministic_mock({target: 1.0})
        r = run_probe("등록되지 않은 새 spec", mock, n=5)
        assert r.objective_referent_count is None


class TestScenarioValidationPhase1:
    """M1/M2/M3(misread/scope-exceeded/condition 재설계)의 PASS/FAIL 판정
    로직 — mock으로 배관만 검증한다. 실제 통과 여부는 real LLM으로만
    확인 가능(여기선 안 함)."""

    def test_three_redesigned_cases_registered(self):
        assert len(SCENARIO_REDESIGN_CASES) == 3
        labels = [c[1] for c in SCENARIO_REDESIGN_CASES]
        assert any("misread" in l for l in labels)
        assert any("scope-exceeded" in l for l in labels)
        assert any("condition" in l for l in labels)

    def test_judge_passes_when_dominant_meets_threshold_and_matches_truth(self):
        truth = I("summarize", "file", "/reports/2026-08/")
        mock = make_deterministic_mock({truth: 0.95, I("export", "file", "/reports/2026-08/"): 0.05})
        r = run_probe("M1 spec", mock, n=20, objective_referents=1)
        assert judge_scenario_validation(r, truth) is True

    def test_judge_fails_when_dominant_does_not_match_truth(self):
        """지난번 실측처럼 — dominant가 다른 interpretation으로 수렴하면
        확률이 높아도(90%+) FAIL이어야 한다. '확신에 찼다'가 '맞다'를
        보장하지 않는다는 걸 이 판정 로직 자체가 지켜야 한다."""
        truth = I("summarize", "file", "/reports/2026-08/")
        wrong = I("export", "file", "/reports/2026-08/")
        mock = make_deterministic_mock({wrong: 1.0})
        r = run_probe("M1 spec", mock, n=20)
        assert judge_scenario_validation(r, truth) is False

    def test_judge_fails_when_distribution_is_too_spread_even_if_truth_is_dominant(self):
        truth = I("summarize", "file", "/reports/2026-08/")
        mock = make_deterministic_mock({
            truth: 0.5,
            I("export", "file", "/reports/2026-08/"): 0.3,
            I("read", "file", "/reports/2026-08/"): 0.2,
        })
        r = run_probe("M1 spec", mock, n=20)
        assert judge_scenario_validation(r, truth, threshold=0.90) is False

    def test_run_scenario_validation_reports_all_three_without_hiding_failures(self):
        # 전부 틀린 답으로 수렴하는 mock — 3개 다 FAIL이어야 하고, 숨기지 않고
        # 3개 결과가 그대로 나와야 한다.
        wrong = I("read", "file", "*")
        mock = make_deterministic_mock({wrong: 1.0})
        rows = run_scenario_validation(mock, n=10)
        assert len(rows) == 3
        assert all(passed is False for _, _, passed in rows)


class TestAnthropicSamplerIsLazy:
    """anthropic 패키지가 없어도 모듈 import/함수 생성 자체는 실패하면 안 된다
    — 실제 호출 시점에만 ImportError 가 나야 API 없는 환경에서도 이 파일 전체가
    깨지지 않는다."""

    def test_factory_does_not_require_anthropic_installed(self):
        from dualflow.entropy_probe import make_anthropic_sampler
        sampler = make_anthropic_sampler()
        assert callable(sampler)
