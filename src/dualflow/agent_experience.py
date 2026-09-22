"""
AGENT EXPERIENCE STORE — B7c. Principal이 실제로 확인해준 clarification
결과만 저장하는, 순수 in-memory 기록 장치.

이 모듈이 하지 않는 것 — 전부 다른 단계의 일이다:
  - 저장된 경험을 `DelegateAgent`의 sampling에 반영하는 것, entropy를
    낮추거나 clarification을 생략하는 것 (B7d) — 이 모듈은 저장/조회만
    한다. `DelegateAgent`/`PrincipalAgent`/`ClarifyingDelegate` 중
    어느 것도 이 모듈을 참조하지 않는다(반대 방향 의존성만 있다 —
    이 모듈이 `clarification.ClarificationResult`를 읽는다).
  - authority/권한에 관여하는 것 — 이 store는 semantic interaction
    history만 다룬다. `AuthorityVerifierAgent`/`Budget`/
    `check_authority()`는 이 모듈의 존재를 전혀 모르고 영향받지도
    않는다 — experience가 권한을 만들어내는 일은 구조적으로 없다.

이 모듈에는 LLM 호출이 전혀 없다 — 순수 결정론적 저장/조회다. embedding
유사도도, fuzzy matching도 없다 — 조회 키는 `(principal_id,
task_category)` 명시적 쌍이다("이 Principal이, 이런 반복 업무에서, 전에
어떻게 clarify했는가"). `task_category`는 runtime-visible workflow
metadata(예: "external_audit_report")이지 evaluation label이 아니다 —
`task.truth`/ground_truth와는 완전히 다른 것이다.

무엇이 "verified experience"로 저장될 자격이 있는가 — `build_verified_
experience()`가 강제하는 최소 조건 세 가지:
  1. clarification이 실제로 트리거됐다(`ClarificationResult.clarified
     = True`) — entropy가 낮아서 곧장 확정된 결과는 저장하지 않는다.
     그건 "확인된 경험"이 아니라 그냥 "B 혼자만의 추측"이다.
  2. `PrincipalAgent.answer_clarification()`이 실제로 답을 만들었다.
  3. clarification 이후의 `CandidateDistribution`(post)이 존재하고,
     그 post-entropy가 configurable한 `max_verified_entropy` 기준을
     만족한다(기본값 0.0 — B7c pilot 기준, 최종 연구 threshold로
     확정하지 않는다).

post-entropy가 기준을 넘으면(clarification 후에도 여전히 애매하면)
저장하지 않는다 — "A는 summarize를 원한다"처럼 약하게 확인된 결과를
강하게 기억해버리면, 이후(B7d) 그 불확실한 결론이 마치 확정된 사실처럼
재사용될 위험이 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .delegate_agent import CandidateDistribution
from .semantic import Interpretation

# TYPE_CHECKING 없이 직접 import해도 순환 참조가 없다 — clarification.py는
# 이 모듈을 전혀 모른다(참조 방향이 반대다).
from .clarification import ClarificationResult


@dataclass(frozen=True)
class AgentExperience:
    """Principal이 실제로 확인해준 clarification 결과 하나.

    `pre_entropy`/`post_entropy`는 `pre_distribution`/`post_distribution`
    이 이미 갖고 있는 값의 별칭일 뿐이다 — 새로 저장하지 않는다(중복 금지)."""

    principal_id: str
    task_category: str

    delegation: str
    clarification_question: str
    principal_answer: str

    confirmed_interpretation: Interpretation

    pre_distribution: CandidateDistribution
    post_distribution: CandidateDistribution

    episode_id: str | None = None

    @property
    def pre_entropy(self) -> float:
        return self.pre_distribution.entropy

    @property
    def post_entropy(self) -> float:
        return self.post_distribution.entropy


def build_verified_experience(result: ClarificationResult, *, principal_id: str,
                              task_category: str, delegation: str,
                              max_verified_entropy: float = 0.0,
                              episode_id: str | None = None) -> AgentExperience | None:
    """`ClarifyingDelegate.resolve()`의 결과를 검증된 경험으로 바꾼다 —
    자격이 안 되면 `None`을 돌려준다. 이 함수 자체는 아무것도 저장하지
    않는다 — 호출자가 반환값이 `None`이 아닐 때만 명시적으로
    `store.add(...)`해야 한다(B7c는 의도적으로 자동 저장을 하지 않는다,
    `ClarifyingDelegate` 자체는 전혀 건드리지 않았다).

    `delegation`을 별도 인자로 받는 이유: `ClarificationResult`는
    `resolve()`에 넘겼던 원본 delegation 텍스트를 자체적으로 갖고 있지
    않다 — 호출자(= `resolve(delegation=...)`를 부른 쪽)가 이미 알고
    있으므로 그대로 넘겨주면 된다. `clarification.py`는 이 함수 때문에
    수정되지 않았다."""
    if not result.clarified:
        return None
    if result.question is None or result.answer is None or result.post_distribution is None:
        return None
    if result.post_distribution.entropy > max_verified_entropy:
        return None

    return AgentExperience(
        principal_id=principal_id,
        task_category=task_category,
        delegation=delegation,
        clarification_question=result.question.question,
        principal_answer=result.answer.answer,
        confirmed_interpretation=result.final_interpretation,
        pre_distribution=result.pre_distribution,
        post_distribution=result.post_distribution,
        episode_id=episode_id,
    )


class AgentExperienceStore:
    """`(principal_id, task_category)` 키로 `AgentExperience`를 쌓는
    결정론적 in-memory store. LLM 호출도, embedding도, fuzzy matching도
    없다 — 순수 dict[tuple, list] 조작이다."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], list[AgentExperience]] = {}

    def add(self, experience: AgentExperience) -> None:
        key = (experience.principal_id, experience.task_category)
        self._records.setdefault(key, []).append(experience)

    def get(self, principal_id: str, task_category: str) -> list[AgentExperience]:
        """삽입 순서 그대로(가장 오래된 것부터) 반환한다. 매번 새
        리스트를 복사해서 돌려주므로, 호출자가 반환값을 변형해도 store
        내부 상태는 안전하다."""
        return list(self._records.get((principal_id, task_category), ()))

    def count(self, principal_id: str, task_category: str) -> int:
        return len(self._records.get((principal_id, task_category), ()))

    def clear(self, principal_id: str | None = None,
             task_category: str | None = None) -> None:
        """인자 없이 부르면 전체를 비운다. `principal_id`/`task_category`
        를 (하나 또는 둘 다) 주면 그 조건에 맞는 키만 지운다."""
        if principal_id is None and task_category is None:
            self._records.clear()
            return
        to_delete = [
            key for key in self._records
            if (principal_id is None or key[0] == principal_id)
            and (task_category is None or key[1] == task_category)
        ]
        for key in to_delete:
            del self._records[key]
