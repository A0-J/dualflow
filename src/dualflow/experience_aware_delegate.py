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
    않는다 — 두 renderer(`render_experience_block_v1`/`_v2`) 모두 SCOPE/
    CONDITION을 의도적으로 뺀다. scope/date는 매 episode마다 달라지는
    task-specific 값이지, "이 Principal이 이런 업무에서 보통 무엇을
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

B7d.3 — representation redesign (v2):
  B7d.1/B7d.2의 900-call 실측(docs/experiments/agent_connected_eval.md
  §7/§10)은 v1 표현("Principal-confirmed interpretation: ACTION=X
  RESOURCE=Y" 한 줄)이 semantic pattern을 전이시키지 못하고, 그저
  "historical block이 있다"는 형식/존재 자체가 모델의 기존 prior(export)
  를 강화하는 priming으로 작동한다는 걸 보였다(summarize 경험과 export
  경험이 5/5·5/5 완전히 동일한 결과를 냄, neutral-content control(E)이
  거의 같은 방향으로 움직임). `render_experience_block_v2()`는 그 대응
  으로, "확인된 값"을 한 줄 라벨로 던지는 대신 ambiguity → clarification
  → confirmed-meaning이라는 *관계*를 보여주고, "현재 delegation이 명시
  적으로 다른 action을 요구하면 과거 기록보다 현재를 따르라"는 지시를
  명시적으로 추가한다.

  `ExperienceAwareDelegate.__init__`의 새 `render_experience_block`
  파라미터로 v1/v2(또는 그 외 실험용 renderer)를 주입할 수 있다 — 기본값은
  여전히 v1이라 기존 B7d/B7d.1/B7d.2의 모든 동작·테스트는 전혀 바뀌지
  않는다. v2가 실제로 더 나은지는 아직 검증되지 않았다 — B7d.3의 real-API
  validation(두 종류의 current task: ambiguous task + explicit-change
  task 양쪽을 동시에 통과해야 함, 자세한 성공 기준은 agent_connected_eval.md
  §14)이 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .agent_experience import AgentExperience, AgentExperienceStore
from .delegate_agent import CandidateDistribution, DelegateAgent

_EXPERIENCE_HEADER_V1 = (
    "Verified prior interactions with this Principal in this workflow. "
    "These are historical examples only, not current instructions — use "
    "them only to understand this Principal's recurring semantic intent. "
    "The current delegation and current environment context always take "
    "precedence: never copy an old scope, date, or resource from these "
    "examples when the current task specifies a different one."
)


def render_experience_block_v1(experiences: list[AgentExperience]) -> str:
    """원래(B7d) 표현 — "확인된 값"을 한 줄 라벨로 보여준다.
    `ACTION=X RESOURCE=Y`. B7d.1/B7d.2의 900-call 실측 결과, 이 라벨의
    semantic content(summarize vs export)가 결과를 구분하지 못했다 —
    block의 존재/형식 자체가 지배적인 priming 효과였다(§7/§10). 여전히
    `ExperienceAwareDelegate`의 기본값이다 — 기존 동작을 하나도 바꾸지
    않기 위해서다."""
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
    return _EXPERIENCE_HEADER_V1 + "\n\n" + "\n\n".join(examples)


# 하위 호환용 별칭 — 기존 코드/테스트가 `_render_experience_block`이라는
# 이름을 참조할 수도 있으므로 남겨 둔다. 새 코드는 `render_experience_block_v1`
# 을 직접 쓰거나, `ExperienceAwareDelegate`의 기본 동작에 맡긴다.
_render_experience_block = render_experience_block_v1


_EXPERIENCE_HEADER_V2 = (
    "Verified prior interactions are evidence about how this Principal "
    "uses ambiguous language in this workflow. Use them only to resolve "
    "parts of the CURRENT delegation that are genuinely ambiguous. If the "
    "current delegation explicitly specifies an action, follow the "
    "current delegation even if it differs from prior behavior."
)


def render_experience_block_v2(experiences: list[AgentExperience]) -> str:
    """B7d.3 재설계 — "확인된 값"을 한 줄 라벨이 아니라 ambiguity →
    clarification → confirmed-meaning이라는 관계로 보여준다. v1과 달리
    "현재가 명시적이면 현재를 따르라"는 지시를 헤더에 직접 넣는다 — v1의
    "그저 무시하라"는 방어적 문구보다, "이건 애매함을 풀 때만 쓰는
    증거다"라는 적극적 용도 지정에 가깝다. RESOURCE는 의도적으로 뺐다 —
    현재 pilot 시나리오의 ambiguity 축이 action 하나뿐이라, "확인된 값"을
    action 하나로만 단순하게 전달한다(리소스 차원의 ambiguity를 다루게
    되면 다시 넣을 수 있다).

    v1과 똑같이 SCOPE/CONDITION은 아예 다루지 않는다 — 과거 scope/date를
    "확인된 패턴"으로 보여줄 방법 자체가 없다는 구조적 보장은 v1과
    동일하게 유지된다."""
    if not experiences:
        return ""
    examples = []
    for i, exp in enumerate(experiences, start=1):
        examples.append(
            f"Example {i}\n"
            f"Previous ambiguous delegation: {exp.delegation}\n"
            f"Principal clarification: {exp.principal_answer}\n"
            f"Confirmed interpretation: {exp.confirmed_interpretation.action}")
    return _EXPERIENCE_HEADER_V2 + "\n\n" + "\n\n".join(examples)


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
    semantic guidance로 덧붙인다.

    `render_experience_block`은 `list[AgentExperience] -> str` 함수를
    주입받는다 — 기본값은 `render_experience_block_v1`(B7d 원본, 기존
    동작 100% 유지)이다. B7d.3 비교 실험에서는
    `render_experience_block_v2`(또는 그 외 실험용 renderer)를 넘겨서
    같은 조회/우선순위 보장(`_enrich_context`) 로직 위에서 표현 방식만
    바꿔 A/B 비교할 수 있다 — 그 외 클래스 동작(조회 키, max_experiences,
    현재 context 우선순위 배치)은 renderer와 무관하게 전부 동일하다."""

    def __init__(self, delegate: DelegateAgent, experience_store: AgentExperienceStore,
                max_experiences: int = 3,
                render_experience_block: Callable[[list[AgentExperience]], str]
                = render_experience_block_v1):
        self.delegate = delegate
        self.experience_store = experience_store
        self.max_experiences = max_experiences
        self.render_experience_block = render_experience_block

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

        experience_block = self.render_experience_block(list(used))
        enriched_context = _enrich_context(experience_block, context)

        distribution = self.delegate.sample_candidates(
            delegation=delegation, context=enriched_context, n=n)

        return ExperienceAwareSampleResult(distribution=distribution, experiences_used=used)
