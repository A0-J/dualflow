"""
엔트로피 실측 하네스 — 아직 미착수인 "1번(entropy 측정 방법 검증)" 작업의
실행 준비물이다. API 키가 없어서 지금은 돌릴 수 없지만, 키만 꽂으면 바로
실행되는 상태로 완성해 둔다.

파이프라인(docs 대화에서 확정한 6단계 formal spec을 그대로 코드화):

    입력(spec)
        │
        ▼
    ① 후보 생성 — CandidateSampler가 spec을 받아 N개의 독립 completion을 낸다
        │           (LLM 자유생성 → 닫힌 스키마로 파싱, 실패한 샘플은 버림)
        ▼
    ② 확률 — MLE: p_i = n_i / N  (다항분포 최대우도추정)
        ▼
    ③ 엔트로피 — semantic.entropy() 그대로 재사용 (H = -Σ p_i log2 p_i)
        ▼
    ④ objective ground truth와 대조 — "이 표현에 몇 개의 후보가 원칙적으로
       부합하는가"를 사람의 느낌이 아니라 **미리 얼린 파일 인벤토리**로 센다
       (순환 방지 — 결과를 보기 전에 인벤토리를 고정한다는 걸 커밋 이력으로 보증)
        ▼
    ⑤ 보고 — H와 objective count의 상관관계, 가설(v1 모듈의 "명확/모호" 라벨)과
       실측의 일치/불일치를 숨기지 않고 그대로 기록

이 모듈은 **메인 pilot(bench_single_env_v1)과 완전히 분리**돼 있다 — Kuhn/
Farquhar(semantic entropy 선행연구)와 같은 패턴: "entropy가 실제 의미적
불확실성과 상관있는가"라는 경험적 주장을, "Semantic+Authority+Joint 구조가
안전한가"라는 메커니즘 증명과 절대 섞지 않는다(docs 대화, 분리 결정 근거).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from .semantic import Interpretation, entropy as shannon_entropy, normalize

# --------------------------------------------------------------------------
# 닫힌 vocabulary — bench_single_env_v1.py와 동일한 환경을 공유한다. 실측
# 실험이 pilot과 "같은 세계"를 봐야 나중에 두 결과를 나란히 놓을 수 있다.
# --------------------------------------------------------------------------
SCOPE_VOCAB = ("/reports/2026-08/", "/reports/2026-09/", "/reports/",
               "/finance/", "/hr/", "*")
ACTION_VOCAB = ("read", "summarize", "write", "export", "send", "delete")
RESOURCE_VOCAB = ("file", "email")

# --------------------------------------------------------------------------
# 파일 인벤토리 — entropy 실측(1번)보다 먼저 얼린다. 이 모듈을 커밋하는 시점
# 자체가 "LLM 응답을 보기 전에 정답이 고정됐다"는 pre-registration 증거다.
# 값을 바꾸려면 반드시 새 커밋으로 남길 것 — 조용히 수정하지 않는다.
# --------------------------------------------------------------------------
FILE_INVENTORY: dict[str, tuple[str, ...]] = {
    "/reports/2026-08/": ("sales_summary_final.pdf",),
    "/reports/2026-09/": ("sales_draft_v3.pdf",),          # 아직 review 안 끝남
    "/finance/": ("q3_ledger.xlsx", "budget_2026.xlsx"),
    "/hr/": ("performance_review_2026H1.pdf",),
}


def objective_referent_count(scope: str) -> int:
    """이 scope 표현이 인벤토리에서 몇 개의 파일과 부합하는가 — "느낌"이 아니라
    셀 수 있는 사실. scope가 폴더(접두어)면 그 밑 전체 파일 수를 센다."""
    if scope in FILE_INVENTORY:
        return len(FILE_INVENTORY[scope])
    return sum(len(files) for path, files in FILE_INVENTORY.items()
               if path.startswith(scope) or scope == "*")


# --------------------------------------------------------------------------
# ① 후보 생성 — pluggable. 실제 LLM 백엔드는 이 프로토콜만 만족하면 된다.
# --------------------------------------------------------------------------
class CandidateSampler(Protocol):
    def __call__(self, spec: str, n: int) -> list[Interpretation | None]:
        """spec에 대해 n개의 독립 completion을 받아 Interpretation으로 파싱한
        결과를 돌려준다. 파싱 실패(스키마 밖 응답)는 None으로 표시 — 버리지
        않고 개수를 남겨서 "얼마나 스키마를 못 지켰는지"도 드러낸다."""
        ...


@dataclass
class ProbeResult:
    spec: str
    n_requested: int
    n_parsed: int
    belief: dict[Interpretation, float]
    measured_entropy: float
    objective_referent_count: int
    hypothesis_label: str  # v1 모듈에서 가져온 "설계자 가설" — 미검증 딱지 유지

    def summary_line(self) -> str:
        return (f"H={self.measured_entropy:.3f}bits  "
                f"objective_referents={self.objective_referent_count}  "
                f"parsed={self.n_parsed}/{self.n_requested}  "
                f"가설={self.hypothesis_label}  spec={self.spec!r}")


def run_probe(spec: str, sampler: CandidateSampler, n: int = 30,
              hypothesis_label: str = "미검증") -> ProbeResult:
    """②③ MLE 확률 + entropy 계산. 스키마 밖 응답(None)은 분모(n)에는 남기고
    분자 집계에서는 뺀다 — "후보가 안 나왔다"는 것도 측정의 일부다."""
    raw = sampler(spec, n)
    parsed = [c for c in raw if c is not None]
    counts = Counter(parsed)
    belief = normalize({interp: float(cnt) for interp, cnt in counts.items()})
    h = shannon_entropy(belief) if belief else float("nan")
    return ProbeResult(spec, n, len(parsed), belief, h,
                        objective_referent_count(_dominant_scope(parsed)),
                        hypothesis_label)


def _dominant_scope(candidates: list[Interpretation]) -> str:
    if not candidates:
        return "*"
    return Counter(c.scope for c in candidates).most_common(1)[0][0]


def run_probe_suite(cases: Iterable[tuple[str, str]], sampler: CandidateSampler,
                     n: int = 30) -> list[ProbeResult]:
    """(spec, hypothesis_label) 목록을 한 번에 돌린다. 결과를 그대로 출력하되
    가설과 실측이 어긋나는 항목을 숨기지 않는다 — 정렬도 H 기준으로만 한다."""
    return [run_probe(spec, sampler, n, label) for spec, label in cases]


# --------------------------------------------------------------------------
# 테스트/스모크용 — API 없이 하네스 배관 자체를 검증하기 위한 결정론적 mock.
# 실측에는 절대 쓰지 않는다(그러면 다시 순환논리가 된다) — pytest에서만 사용.
# --------------------------------------------------------------------------
def make_deterministic_mock(distribution: dict[Interpretation, float]) -> CandidateSampler:
    """고정된 분포를 그대로 흉내내는 mock — 하네스 배관(파싱→MLE→entropy 계산
    →보고)이 올바른지 검증하는 용도. 이 mock의 출력을 "실측"이라고 부르면 안 된다.
    """
    items = list(distribution.items())
    total = sum(w for _, w in items)

    def _sample(spec: str, n: int) -> list[Interpretation | None]:
        out: list[Interpretation | None] = []
        acc = 0.0
        # 결정론적 비례 배분(진짜 확률 샘플링이 아니라 배관 검증용).
        for interp, w in items:
            k = round(n * w / total)
            out.extend([interp] * k)
        while len(out) < n:
            out.append(None)
        return out[:n]

    return _sample


# --------------------------------------------------------------------------
# 실제 LLM 백엔드 — API 키가 생기면 이 함수를 CandidateSampler로 넘기면 된다.
# anthropic 패키지가 없으면 import 시점이 아니라 **호출 시점**에만 실패하도록
# lazy import — 이 모듈 자체는 API 키/SDK 없이도 정상적으로 import/테스트된다.
# --------------------------------------------------------------------------
def make_anthropic_sampler(model: str = "claude-sonnet-5", temperature: float = 1.0,
                            system_prompt: str = "",
                            parse_fn: Callable[[str], Interpretation | None] | None = None
                            ) -> CandidateSampler:
    """사용 전 `pip install anthropic` + `ANTHROPIC_API_KEY` 필요.

    parse_fn: 모델의 원문 응답(JSON 문자열 기대)을 Interpretation으로 바꾸는
    함수. 기본 파서는 `{"action":..,"resource":..,"scope":..,"condition":[..]}`
    형태를 가정한다 — B에게 이 형식으로만 답하라고 system_prompt에 강제해야
    한다(이게 "닫힌 스키마" 요구사항의 실제 구현 지점).
    """
    def _default_parse(text: str) -> Interpretation | None:
        import json
        try:
            d = json.loads(text)
            if d.get("action") not in ACTION_VOCAB:
                return None
            if d.get("scope") not in SCOPE_VOCAB:
                return None
            return Interpretation(d["action"], d.get("resource", "file"),
                                   d["scope"], frozenset(d.get("condition", ())))
        except Exception:
            return None

    parser = parse_fn or _default_parse

    def _sample(spec: str, n: int) -> list[Interpretation | None]:
        import anthropic  # lazy — 호출 시점에만 필요
        client = anthropic.Anthropic()
        out: list[Interpretation | None] = []
        for _ in range(n):
            resp = client.messages.create(
                model=model, max_tokens=200, temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": spec}],
            )
            out.append(parser(resp.content[0].text))
        return out

    return _sample
