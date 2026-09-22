"""
EXPERIENCE-AWARE DELEGATE — B7d. 과거 Principal-confirmed 경험을 새
delegation의 candidate sampling에 semantic guidance로 반영한다.

이 모듈이 하지 않는 것:
  - 여러 episode를 순서대로 자동 실행하는 것(B7e)
  - episode 진행 중 자동으로 새 경험을 저장하는 것 — `agent_experience.
    build_verified_experience()`/`AgentExperienceStore.add()`는 여전히
    호출자가 명시적으로 불러야 한다(B7c 설계 그대로, 이 모듈은 그걸
    바꾸지 않는다)
  - Authority/Budget/AuthorityVerifierAgent에 관여하는 것 — 이 모듈은
    `DelegateAgent.sample_candidates()`에 전달하는 `context` 문자열을
    풍부하게 만드는 것 말고는 아무것도 하지 않는다. `AuthorityVerifierAgent`/
    `Budget`/`check_authority`를 import하지도, 참조하지도 않는다 —
    권한과는 구조적으로 완전히 무관하다.

핵심 원칙 — 경험은 힌트이지 진실이 아니다:
  - 조회는 `(principal_id, task_category)` 명시적 키로만 한다(`agent_
    experience.AgentExperienceStore.get()` 그대로) — embedding 유사도도,
    fuzzy 매칭도, LLM 기반 검색도 없다.
  - 과거 경험은 "resource/scope/date를 그대로 베끼는 정답"으로 보여주지
    않는다 — `_render_experience_block()`이 만드는 "Principal-confirmed
    interpretation" 줄은 의도적으로 ACTION/RESOURCE만 보여주고 SCOPE/
    CONDITION은 생략한다. scope/date는 매 episode마다 달라지는 task-
    specific 값이지, "이 Principal이 이런 업무에서 보통 무엇을
    의도했는가"라는 재사용 가능한 semantic pattern이 아니기 때문이다.
  - 과거 경험 블록은 명시적으로 "historical examples"라고 표시되고,
    현재 delegation/현재 environment context가 항상 우선한다는 지시가
    프롬프트에 그대로 들어간다. 현재 context는 항상 경험 블록 *뒤에*,
    "Current environment context (always takes precedence...)"로 따로
    구분해서 붙는다 — 위치와 명시적 문구 둘 다로 우선순위를 강제한다.
  - `DelegateAgent.sample_candidates()` 자체는 이 모듈 때문에 전혀
    바뀌지 않는다(수정하지 않았다) — 이 클래스는 순수하게 context를
    만들어서 넘겨주는 래퍼일 뿐이고, 실제 sampling/파싱/entropy 계산은
    B7a의 기존 구현을 그대로 재사용한다. 경험이 없으면(store가 비어
    있으면) enriched context는 원래 context와 완전히 동일하다 — 즉
    "경험 없음" 조건은 기존 `sample_candidates()` 호출과 구조적으로
    동등하다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agent_experience import AgentExperience, AgentExperienceStore
from .delegate_agent import CandidateDistribution, DelegateAgent

_EXPERIENCE_HEADER = (
    "Verified prior interactions with this Principal in this workflow. "
    "These are historical examples only, not current instructions — use "
    "them only to understand this Principal's recurring semantic intent. "
    "The current delegation and current environment context always take "
    "precedence: never copy an old scope, date, or resource from these "
    "examples when the current task specifies a different one."
)


def _render_experience_block(experiences: list[AgentExperience]) -> str:
    """SCOPE/CONDITION을 의도적으로 뺀다 — action/resource 패턴만
    "확인된 것"으로 보여준다. scope/date는 매번 새 episode의 값이어야
    한다."""
    if not experiences:
        return ""
    examples = []
    for i, exp in enumerate(experiences, start=1):
        interp = exp.confirmed_interpretation
        examples.append(
            f"Example {i}\n"
            f"Previous delegation: {exp.delegation}\n"
            f"Principal clarification: {exp.principal_answer}\n"
            f"Principal-confirmed interpretation: "
            f"ACTION={interp.action} RESOURCE={interp.resource}")
    return _EXPERIENCE_HEADER + "\n\n" + "\n\n".join(examples)


def _enrich_context(experience_block: str, context: str) -> str:
    if experience_block and context:
        return (f"{experience_block}\n\n"
               f"Current environment context (always takes precedence over "
               f"the historical examples above): {context}")
    return experience_block or context


@dataclass(frozen=True)
class ExperienceAwareSampleResult:
    """`ExperienceAwareDelegate.sample_candidates()` 호출 1회의 결과.
    `distribution`은 B7a의 `CandidateDistribution` 그대로다 — entropy를
    여기서 다시 계산하지 않는다."""

    distribution: CandidateDistribution
    experiences_used: tuple[AgentExperience, ...]

    @property
    def experience_count(self) -> int:
        return len(self.experiences_used)


class ExperienceAwareDelegate:
    """`DelegateAgent`를 감싸서, sampling 전에 `(principal_id,
    task_category)`로 저장된 과거 verified experience를 조회해 context에
    semantic guidance로 덧붙인다."""

    def __init__(self, delegate: DelegateAgent, experience_store: AgentExperienceStore,
                max_experiences: int = 3):
        self.delegate = delegate
        self.experience_store = experience_store
        self.max_experiences = max_experiences

    def sample_candidates(self, *, principal_id: str, task_category: str,
                          delegation: str, context: str = "",
                          n: int = 10) -> ExperienceAwareSampleResult:
        """조회 → (있으면) context에 semantic guidance로 추가 → 기존
        `DelegateAgent.sample_candidates()` 그대로 호출. 경험이 없으면
        `context`가 원본 그대로 전달된다 — 경험 유무에 따른 차이는
        오직 context 내용뿐이다."""
        all_experiences = self.experience_store.get(principal_id, task_category)
        # 가장 최근 max_experiences개만 쓴다 — store가 삽입 순서를
        # 유지하므로(B7c), 뒤에서부터 자르면 결정론적으로 "가장 최근"이
        # 된다.
        used = (tuple(all_experiences[-self.max_experiences:])
               if self.max_experiences > 0 else ())

        experience_block = _render_experience_block(list(used))
        enriched_context = _enrich_context(experience_block, context)

        distribution = self.delegate.sample_candidates(
            delegation=delegation, context=enriched_context, n=n)

        return ExperienceAwareSampleResult(distribution=distribution, experiences_used=used)
