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

from dataclasses import dataclass

from .delegate_agent import CandidateDistribution, ClarificationQuestion, DelegateAgent
from .principal_agent import PrincipalAgent, PrincipalClarification
from .semantic import Interpretation


@dataclass(frozen=True)
class ClarificationResult:
    """`ClarifyingDelegate.resolve()` 호출 1회의 결과.

    `pre_distribution`/`post_distribution`이 이미 `.entropy`를 담고 있으므로
    entropy를 여기서 따로 다시 계산/저장하지 않는다 — 필요하면
    `result.pre_distribution.entropy`/`result.post_distribution.entropy`를
    본다."""

    clarified: bool

    pre_distribution: CandidateDistribution
    post_distribution: CandidateDistribution | None   # clarified=True일 때만

    question: ClarificationQuestion | None             # clarified=True일 때만
    answer: PrincipalClarification | None              # clarified=True일 때만

    final_interpretation: Interpretation


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
        `delegation`/`context`만 본다, `goal`을 직접 보지 않는다."""
        pre = self.delegate.sample_candidates(
            delegation=delegation, context=context, n=self.n)

        if pre.entropy <= self.entropy_threshold:
            return ClarificationResult(False, pre, None, None, None, pre.top)

        question = self.delegate.ask_clarification(
            delegation=delegation, distribution=pre, context=context)
        answer = self.principal.answer_clarification(
            goal=goal, context=context, question=question.question)

        enriched_context = (f"{context}\nPrincipal clarification: {answer.answer}"
                            if context else f"Principal clarification: {answer.answer}")
        post = self.delegate.sample_candidates(
            delegation=delegation, context=enriched_context, n=self.n)

        return ClarificationResult(True, pre, post, question, answer, post.top)
