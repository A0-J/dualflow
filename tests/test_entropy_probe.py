"""entropy_probe 하네스의 배관(파싱→MLE→entropy→objective count) 검증.

mock sampler로만 테스트한다 — 이건 "entropy가 실측됐다"를 증명하는 게 아니라
"하네스가 API 키만 생기면 정상 작동할 준비가 됐다"를 증명하는 테스트다.
"""

from __future__ import annotations

import math

from dualflow.entropy_probe import (
    FILE_INVENTORY, make_deterministic_mock, objective_referent_count,
    run_probe, run_probe_suite,
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


class TestAnthropicSamplerIsLazy:
    """anthropic 패키지가 없어도 모듈 import/함수 생성 자체는 실패하면 안 된다
    — 실제 호출 시점에만 ImportError 가 나야 API 없는 환경에서도 이 파일 전체가
    깨지지 않는다."""

    def test_factory_does_not_require_anthropic_installed(self):
        from dualflow.entropy_probe import make_anthropic_sampler
        sampler = make_anthropic_sampler()
        assert callable(sampler)
