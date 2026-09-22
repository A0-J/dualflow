"""
AGENT RUNTIME — Phase B(`research/agent-connected-eval`)의 oracle-free
실행 경로.

기존 controlled `framework.DelegationVerifier.run()`과는 완전히 별개다 —
그 경로는 `task.truth`에서 유도된 `task.intent_fields`를 매칭 기준으로
쓴다(의도적으로 disclosed된 evaluation boundary, docs/EXPERIMENTS.md
§1.3, docs/ARCHITECTURE.md §6). 이 모듈은 그 오라클을 런타임에서 전혀
쓰지 않는다 — 대신 `PrincipalAgent.restate_intent()`가 만드는, B의
proposal과 무관하게 독립적으로 재구성한 의도를 최종 판정 기준으로 쓴다.

왜 `SemanticVerdict.interpretation`을 기준으로 쓰면 안 되는가 —
silent_misread: B가 자신 있게 잘못 이해하고(예: export), Semantic
Verifier도 같은 잘못을 반복하면(export == export, confirmed=True), 그
자기 자신과의 비교는 tautology가 된다. 실제로 controlled 경로에서 이걸
시험해서 확인했다 — `Full = 0% unsafe`였던 것이 11.1%로 깨졌다
(docs/EXPERIMENTS.md §1.3/§6.3). 그래서 이 runtime은 Semantic Verifier의
판단이 아니라, B의 proposal을 전혀 보지 않고 원래 goal/context에서
독립적으로 다시 만든 `PrincipalAgent.restate_intent()`의 결과를 최종
기준으로 쓴다.

호출 순서(매 단계가 독립적인 model call이고, 서로의 결과를 미리 보지
않는다 — 아래 주석에서 각 단계가 "무엇을 안 보는지"를 명시한다):

    1. PrincipalAgent.delegate(goal, context) -> delegation
    2. PrincipalAgent.restate_intent(goal, context) -> principal_intent
       (delegation과 독립 — 이 시점에 Agent B의 proposal은 아직 코드
       상에도 존재하지 않는다. own_delegation도 일부러 넘기지 않는다 —
       원래 goal/context에서 곧바로 재구성한다.)
    3. DelegateAgent.propose(delegation, context) -> proposal
    4. SemanticVerifierAgent.verify_agent_proposal(delegation, proposal,
       context) -> semantic_verdict
       (principal_intent도, AuthorityVerdict도 보지 않는다 — 시그니처
       자체에 그런 파라미터가 없다.)
    5. AuthorityVerifierAgent.verify_agent_proposal(proposal, budget,
       context) -> authority_verdict
       (principal_intent도, SemanticVerdict도 보지 않는다.)
    6. 결정론적 fusion — 여기서 처음으로 세 결과(semantic_verdict,
       authority_verdict, principal_intent)를 합친다. 새 LLM 호출은
       없다 — "fusion judge" 같은 것은 만들지 않는다.

`task.truth`/ground_truth/evaluation label은 이 모듈 어디에도 존재하지
않는다 — `run()`의 시그니처 자체에 그런 파라미터가 없다. B7의
`agent_benchmark.py`가 이 runtime의 결과를 실행 *이후에* 채점할 때만
별도로 ground truth를 쓴다.

Principal 독립 재구성과 최종(협상 반영) action의 비교는 순수 구조적
동등성(`Interpretation.__eq__`)으로 한다 — `rule_engine.match_intent()`는
쓰지 않는다. 이유는 `verify_agent_proposal()`에서 §rule_engine 참고
아래 주석에.
"""

from __future__ import annotations

from dataclasses import dataclass

from .authority_feedback import AuthorityVerdict, AuthorityVerifierAgent
from .capability import Budget
from .delegate_agent import DelegateAgent, DelegateProposal
from .framework import EXECUTE, REJECT
from .principal_agent import PrincipalAgent, PrincipalDelegation, PrincipalIntent
from .semantic import Interpretation, SemanticVerdict, SemanticVerifierAgent


@dataclass(frozen=True)
class AgentRuntimeResult:
    """`AgentDelegationRuntime.run()` 한 번의 결과.

    `task.truth`/ground_truth/benchmark label은 의도적으로 이 안에 없다 —
    B7이 실행 *이후에* 별도로 채점할 때만 붙인다. 여기 담긴 것은 runtime이
    실제로 만들어낸 것뿐이다."""

    decision: str
    reason: str

    delegation: PrincipalDelegation
    principal_intent: PrincipalIntent
    proposal: DelegateProposal

    semantic_verdict: SemanticVerdict
    authority_verdict: AuthorityVerdict

    final_interpretation: Interpretation
    principal_match: bool


class AgentDelegationRuntime:
    """Phase B의 oracle-free end-to-end 실행 진입점. 네 개의 독립 Agent를
    정해진 순서로 호출하고, 그 결과를 결정론적으로 융합한다.

    controlled `framework.DelegationVerifier`와 이 클래스는 완전히
    분리돼 있다 — 서로를 참조하지 않는다(`EXECUTE`/`REJECT` 문자열
    상수만 재사용한다, 아래 import 참고). 어느 한쪽이 바뀌어도 다른 쪽
    동작에는 영향이 없다 — controlled benchmark(`experiments/
    benchmark.py`)는 이 클래스의 존재를 몰라도 되고, 실제로 모른다.
    """

    def __init__(self, *, principal: PrincipalAgent, delegate: DelegateAgent,
                 semantic_verifier: SemanticVerifierAgent,
                 authority_verifier: AuthorityVerifierAgent):
        self.principal = principal
        self.delegate = delegate
        self.semantic_verifier = semantic_verifier
        self.authority_verifier = authority_verifier

    def run(self, *, goal: str, context: str, budget: Budget) -> AgentRuntimeResult:
        """ground_truth/task.truth를 받지 않는다 — 파라미터가 goal/context/
        budget 셋뿐이다. 이게 이 runtime이 배포 가능하다고 주장하는 근거의
        구조적 절반이다(나머지 절반은 아래 _fuse()가 SemanticVerdict.
        interpretation이 아니라 principal_intent를 기준으로 쓴다는 것)."""
        # 1) Agent A가 delegation을 만든다.
        delegation = self.principal.delegate(goal=goal, context=context)

        # 2) Agent A가 원래 goal/context에서만 독립적으로 의도를
        #    재구성한다 — 이 시점에 proposal 변수는 아직 존재하지 않는다
        #    (아래 3번에서야 만들어진다), 그러니 코드 구조상으로도
        #    Delegate의 proposal이 여기 섞여 들어갈 방법이 없다.
        principal_intent = self.principal.restate_intent(goal=goal, context=context)

        # 3) Agent B는 delegation 텍스트만 보고 proposal을 만든다 —
        #    원래 goal/context 원문도, principal_intent도 보지 않는다
        #    (DelegateAgent.propose()의 시그니처 자체에 그런 파라미터가
        #    없다).
        proposal = self.delegate.propose(delegation=delegation.delegation, context=context)

        # 4) Semantic — delegation + B의 proposal만 본다. principal_intent
        #    도 AuthorityVerdict도, 이 시점엔 존재하지도 않는 정보다(5번은
        #    아직 실행 전).
        semantic_verdict = self.semantic_verifier.verify_agent_proposal(
            delegation=delegation.delegation, proposal=proposal.interpretation,
            context=context)

        # 5) Authority — B의 proposal + budget만 본다. principal_intent도
        #    SemanticVerdict도 보지 않는다.
        authority_verdict = self.authority_verifier.verify_agent_proposal(
            proposal=proposal.interpretation, budget=budget, context=context)

        # 6) 결정론적 fusion — 여기서 처음으로 세 결과를 합친다.
        return self._fuse(delegation, principal_intent, proposal,
                          semantic_verdict, authority_verdict)

    def _fuse(self, delegation: PrincipalDelegation, principal_intent: PrincipalIntent,
             proposal: DelegateProposal, semantic_verdict: SemanticVerdict,
             authority_verdict: AuthorityVerdict) -> AgentRuntimeResult:
        # AuthorityVerifierAgent가 협상으로 proposal을 좁혔을 수 있다 —
        # 실행 후보는 B의 원래 proposal이 아니라 authority_verdict.
        # interpretation이다(B5와 같은 관례: 협상 결과가 최종본).
        final_interpretation = authority_verdict.interpretation

        # Principal의 독립 재구성과 최종(협상 반영) action이 구조적으로
        # 같은지 비교한다 — task.truth도, SemanticVerdict.interpretation도
        # 아니라 이 값이 최종 기준이다(모듈 docstring의 silent_misread
        # 설명 참고).
        #
        # rule_engine.match_intent()는 쓰지 않는다 — 직접 확인했다.
        # match_intent(interp, intent: Fields, sysvars: dict, ...)의 두
        # 번째 인자는 raw Interpretation이 아니라 이미 classify()를 거친
        # Fields이고, classify() 자체가 sysvars(sensitive_scopes/
        # broad_scopes/required_conditions/approval_required)에 근본적으로
        # 의존한다. sysvars는 controlled benchmark(DelegationTask)의
        # 개념이고, Phase B agent runtime 어디에도 그런 정책 분류 정보가
        # 없다. 빈/기본 sysvars={}를 억지로 넣으면 문법적으로는 동작하지만
        # sensitivity는 항상 "low", condition_met은 항상 True로 붕괴해
        # SOP 경로 비교(Sim_path)가 사실상 무의미해진다 — 이건 "기존 규칙
        # 재사용"이 아니라 관찰되지 않는 새 fuzzy 판정 기준을 조용히
        # 만들어내는 것과 같다(B7 task별 sysvars를 실제로 설계하기 전까지는
        # 하지 않기로 한 일). 그래서 대신, B4/B5가 각자의 confirmed 판정에
        # 이미 쓰고 있는 것과 같은 순수 구조적 동등성
        # (`Interpretation.__eq__` — action/resource/scope/condition 비교,
        # label은 제외)을 쓴다. report vs file 같은 ontology drift나 날짜
        # 추측은 여기서 흡수하지 않는다 — B7이 각 role에 공유
        # environment context(reference date, resource vocabulary 등)를
        # 일관되게 제공해서 애초에 어긋나지 않게 만드는 게 맞는 해법이다.
        principal_match = (principal_intent.intended_action == final_interpretation)

        if not semantic_verdict.confirmed:
            decision, reason = REJECT, f"의미 확정 실패 — {semantic_verdict.route}"
        elif not authority_verdict.allowed:
            decision, reason = REJECT, f"권한 위반 — {authority_verdict.reason}"
        elif not principal_match:
            decision, reason = REJECT, (
                "Principal의 독립 재구성과 최종 action이 불일치 — "
                f"{principal_intent.intended_action} vs {final_interpretation}")
        else:
            decision, reason = EXECUTE, "의미·권한 확인 및 Principal 독립 확인 모두 통과"

        return AgentRuntimeResult(
            decision, reason, delegation, principal_intent, proposal,
            semantic_verdict, authority_verdict, final_interpretation, principal_match)
