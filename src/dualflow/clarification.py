"""
CLARIFICATION LOOP — B7b. Principal-Delegate 단일 라운드 clarification.

`DelegateAgent.sample_candidates()`(B7a)가 measure한 entropy가 threshold를
넘으면, `ClarifyingDelegate`가 `DelegateAgent.ask_clarification()`으로
질문을 만들고 `PrincipalAgent.answer_clarification()`으로 답을 받아, 그
답을 반영해 다시 sampling한다. entropy가 낮으면 곧장 top candidate로
확정한다.

이 모듈이 다루지 않는 것 — 전부 다른 단계의 일이다:
  - Semantic/Authority verification, 최종 EXECUTE/REJECT fusion
    (`agent_runtime.AgentDelegationRuntime`, B6)
  - verified experience 저장/조회, task_category 기반 검색 (B7c) — 이
    모듈은 episode 사이에 아무것도 기억하지 않는다. 매 `resolve()` 호출은
    완전히 독립적이다.

최대 1라운드만 진행한다 — clarification 후에도 entropy가 여전히 높으면
그 상태를 그대로 정직하게 반환한다("post_distribution이 여전히 갈려
있다"). 자동으로 다시 묻지 않는다(무한/반복 루프 없음) — 이건 의도적
설계다: B7b의 목표는 "질문 한 번이 실제로 uncertainty를 줄이는가"를
측정하는 것이지, uncertainty를 반드시 0으로 만드는 알고리즘을 만드는
게 아니다.

호출 순서(매 단계가 독립적인 model call — `sample_candidates()`의 N회
포함):
    1. DelegateAgent.sample_candidates(delegation, context, n) -> pre
    2. pre.entropy <= entropy_threshold ?
       예 -> 끝(clarified=False, final=pre.top)
       아니오 -> 계속
    3. DelegateAgent.ask_clarification(delegation, pre, context) -> question
       (pre.belief만 본다 — task.truth/PrincipalIntent/verdict는 없다)
    4. PrincipalAgent.answer_clarification(goal, context, question.question)
       -> answer
       (goal/context/question 텍스트만 본다 — B의 확률 분포, verdict는
       없다)
    5. DelegateAgent.sample_candidates(delegation, context + 답변, n) -> post
    6. 끝(clarified=True, final=post.top)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .delegate_agent import CandidateDistribution, ClarificationQuestion, DelegateAgent
from .principal_agent import PrincipalAgent, PrincipalClarification
from .semantic import Belief, Interpretation, select_question

_ALL_FACETS = ("action", "resource", "scope", "condition")


def _facets_that_varied(belief: Belief) -> frozenset[str]:
    """이 pre-clarification candidate 분포에서 값이 하나 이상으로 갈렸던
    facet 전부. **진단/기록용일 뿐, evidence eligibility의 근거가 아니다**
    — `ClarificationResult.varied_facets`에만 쓰인다. "이 facet이
    ambiguous했다"와 "이 facet이 실제로 질문의 대상이었다/Principal이
    확인했다"는 서로 다른 사실이라는 게 v3 §23 follow-up 4의 핵심
    구분이다: 이 함수는 전자만 답한다. 후자(무엇을 물을지)는
    `_select_target_facet()`이 information gain으로 딱 하나만 고른다."""
    varied: set[str] = set()
    for facet in _ALL_FACETS:
        values = {getattr(interp, facet) for interp in belief}
        if len(values) > 1:
            varied.add(facet)
    return frozenset(varied)


def _select_target_facet(belief: Belief) -> str | None:
    """이번 clarification round가 물을 facet을 정확히 하나만 고른다 —
    새 heuristic이 아니라 기존 legacy Phase A 로직
    (`semantic.select_question()`/`information_gain()`)을 그대로
    재사용한다: information gain(=그 facet을 알면 줄어드는 기대 잔여
    entropy)이 가장 큰 facet 하나. `asked`는 빈 `Counter()`로 준다 — B7b는
    아직 단일 라운드라 반복 페널티(`lam`)가 적용될 이력이 없다.

    반환값이 `None`이면(수학적으로, `pre.entropy > entropy_threshold >
    0`인 한 이 분기에서는 일어나지 않는다 — entropy(p) > 0이면 최소 한
    facet은 반드시 양의 information gain을 갖는다) 어떤 facet도 이번
    round의 target으로 삼지 않는다 — 호출자는 이 경우 `target_facets`를
    빈 `frozenset()`으로 둬야 한다(아무것도 confirmed되지 않음, 여전히
    안전한 쪽)."""
    question, _scored = select_question(belief, Counter())
    return question.dimension if question is not None else None


@dataclass(frozen=True)
class ClarificationResult:
    """`ClarifyingDelegate.resolve()` 호출 1회의 결과.

    `pre_distribution`/`post_distribution`이 이미 `.entropy`를 담고 있으므로
    entropy를 여기서 따로 다시 계산/저장하지 않는다 — 필요하면
    `result.pre_distribution.entropy`/`result.post_distribution.entropy`를
    본다.

    `varied_facets` — 진단/기록용(agent_connected_eval.md §23 follow-up 4).
    pre-clarification 분포에서 값이 갈렸던 facet 전부(`_facets_that_
    varied()`) — evidence eligibility와는 무관하다. 실제로 confirmed
    evidence 자격을 얻는 건 `question.target_facets`뿐이다(항상 정확히
    0개 또는 1개 — `_select_target_facet()`이 IG 최댓값 facet 하나만
    고른다). `varied_facets ⊇ question.target_facets`가 항상 성립한다."""

    clarified: bool

    pre_distribution: CandidateDistribution
    post_distribution: CandidateDistribution | None   # clarified=True일 때만

    question: ClarificationQuestion | None             # clarified=True일 때만
    answer: PrincipalClarification | None              # clarified=True일 때만

    final_interpretation: Interpretation

    varied_facets: frozenset[str] = frozenset()


class ClarifyingDelegate:
    """`PrincipalAgent`와 `DelegateAgent`를 하나로 묶어 단일 라운드
    clarification 루프를 실행한다. Semantic/Authority verification이나
    최종 fusion은 다루지 않는다 — 그건 `agent_runtime.
    AgentDelegationRuntime`(B6)의 일이고, 이 클래스는 그 앞단, "B가
    확신 없으면 A에게 한 번 더 묻는다"는 것 자체를 검증한다."""

    def __init__(self, *, principal: PrincipalAgent, delegate: DelegateAgent,
                n: int = 10, entropy_threshold: float = 0.8):
        self.principal = principal
        self.delegate = delegate
        self.n = n
        self.entropy_threshold = entropy_threshold

    def resolve(self, *, goal: str, context: str, delegation: str) -> ClarificationResult:
        """delegation을 n번 sampling하고, entropy가 threshold를 넘을 때만
        Principal에게 한 번 되물어 다시 sampling한다. `goal`은
        `answer_clarification()`에만 쓰인다(Principal이 자기 지식으로
        답하려면 필요) — Delegate 쪽 호출(sampling/질문 생성)은 여전히
        `delegation`/`context`만 본다, `goal`을 직접 보지 않는다.

        v3 §23 follow-up 4: 이번 round는 정확히 하나의 facet만 target으로
        삼는다(`_select_target_facet()`, information gain 최댓값 —
        여러 facet을 한 번에 물어서 "일부만 답변에서 다뤄졌는지 모르는"
        상태를 만들지 않는다). 여러 facet을 실제로 confirm해야 한다면
        별도의 clarification round가 필요하다는 뜻이고, 이 클래스는
        여전히 의도적으로 단일 라운드만 한다(B7b 설계 그대로)."""
        pre = self.delegate.sample_candidates(
            delegation=delegation, context=context, n=self.n)

        if pre.entropy <= self.entropy_threshold:
            return ClarificationResult(False, pre, None, None, None, pre.top)

        varied_facets = _facets_that_varied(pre.belief)
        target_facet = _select_target_facet(pre.belief)
        target_facets = frozenset({target_facet}) if target_facet is not None else frozenset()

        question = self.delegate.ask_clarification(
            delegation=delegation, distribution=pre, context=context,
            target_facets=target_facets)
        answer = self.principal.answer_clarification(
            goal=goal, context=context, question=question.question)

        enriched_context = (f"{context}\nPrincipal clarification: {answer.answer}"
                            if context else f"Principal clarification: {answer.answer}")
        post = self.delegate.sample_candidates(
            delegation=delegation, context=enriched_context, n=self.n)

        return ClarificationResult(True, pre, post, question, answer, post.top,
                                   varied_facets=varied_facets)
