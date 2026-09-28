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

RUNTIME INTEGRATION PHASE 1 (2026-09-28, docs/experiments/
agent_connected_eval.md §26) — B7 시리즈(entropy 기반 clarification,
Principal-verified experience storage, v3 Option B advisory-only evidence
harness, B7e Phase 1에서 real API로 검증 완료, commit `8b701d4`)를 이
runtime에 처음으로 연결한다. 전부 opt-in, additive — 새 생성자 파라미터
(`use_clarification`/`n`/`entropy_threshold`/`experience_store`/
`max_verified_entropy`)를 하나도 안 넘기면 이 클래스는 오늘까지와
byte-identical하게 동작한다(`tests/test_agent_runtime.py`의 기존 테스트
전부가 이 경로만 쓰고, 전혀 수정되지 않았다).

`use_clarification=True`일 때만 3번 단계(`DelegateAgent.propose()` 단일
호출)가 `DelegateAgent.sample_candidates()` + `ClarifyingDelegate.
resolve()`로 바뀐다. 이때 `experience_store`/`principal_id`/
`task_category`가 전부 주어지면 `FrozenCandidateEvidenceHarness.decide()`
가 advisory relation을 계산하지만(§`experience_decision`), 이 값은
`AgentRuntimeResult`에 감사용으로만 실릴 뿐 `clarifier.resolve()`에도,
semantic/authority verification에도, `_fuse()`의 결정 로직에도 전달되지
않는다 — Option B("historical evidence is advisory, not authoritative")를
runtime 레벨에서도 구조적으로 지킨다. Verified experience 저장도
자동이 아니다 — `build_verified_experience()`가 만든 값을
`AgentRuntimeResult.candidate_experience`로만 돌려주고, `run()` 자신은
`experience_store.add(...)`를 절대 호출하지 않는다(B7c의 "이 함수는
저장하지 않는다, 호출자가 명시적으로 add()해야 한다" 원칙을 그대로
유지).

RUNTIME INTEGRATION — SOURCE-SIDE SEMANTIC GATE (2026-09-28, docs/
experiments/agent_connected_eval.md §29, Phase 2C-P5 Finding P2C-F2)
— `source_semantic_gate.SourceClarifyingPrincipal`(P5에서 200+ real API
호출로 standalone 검증 완료, commit `b1e2f86`/`615f238`)을 이 runtime의
1번 단계에 opt-in으로 연결한다. 새 생성자 파라미터
`use_source_verification`/`source_n`을 하나도 안 넘기면 이 클래스는
지금까지와 byte-identical하게 동작한다.

`use_source_verification=True`일 때만 1번 단계(`principal.delegate()`
단일 호출)가 `SourceClarifyingPrincipal.resolve()`로 바뀐다 — 독립적으로
`source_n`번 delegate()를 뽑아 facet-level entropy를 재고(§29 gate,
threshold는 기존 `entropy_threshold`를 그대로 재사용 — 새 threshold를
만들지 않는다), 불안정하면 정확히 그 disagreeing facet 하나만 결정론적
질문으로 확인한 뒤 그 확정값을 constraint로 재생성한다. 이 결과의
`final_principal_delegation`(실제로 생성된, 합성되지 않은
`PrincipalDelegation`)이 2번 단계 이후 기존 파이프라인 전체(principal_
intent 재구성, Delegate propose/clarification, Semantic/Authority
Verification, `_fuse()`)에 그대로 흘러간다 — 그 이후 로직은 하나도
수정되지 않았다. `AgentRuntimeResult.source_clarification_result`는
감사/기록용으로만 실린다(`source_pre_entropy`/`source_clarified`/
`clarified_facet`/`source_post_entropy`는 이 값에서 파생된다).
"""

from __future__ import annotations

from dataclasses import dataclass

from .agent_experience import AgentExperience, AgentExperienceStore, build_verified_experience
from .authority_feedback import AuthorityVerdict, AuthorityVerifierAgent
from .capability import Budget
from .clarification import ClarificationResult, ClarifyingDelegate
from .delegate_agent import DelegateAgent, DelegateProposal
from .experience_decision import FacetDecision, FrozenCandidateEvidenceHarness
from .framework import EXECUTE, REJECT
from .principal_agent import PrincipalAgent, PrincipalDelegation, PrincipalIntent
from .semantic import Interpretation, SemanticVerdict, SemanticVerifierAgent, parse_structured_action
from .source_semantic_gate import SourceClarificationResult, SourceClarifyingPrincipal


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

    # Runtime Integration Phase 1 -- 전부 `use_clarification=False`(기본값)
    # 에서는 항상 None. 세 필드 모두 감사/기록용이며 `decision`/
    # `final_interpretation`에 영향을 준 적이 없다(줄 수 있는 코드 경로
    # 자체가 없다 -- agent_runtime.py 모듈 docstring 참고).
    clarification_result: ClarificationResult | None = None
    experience_decision: FacetDecision | None = None
    candidate_experience: AgentExperience | None = None

    # Source-side Semantic Flow (§29 Phase 2C-P5, Finding P2C-F2) --
    # `use_source_verification=False`(기본값)에서는 항상 None. 감사용일
    # 뿐이며 `decision`/`final_interpretation`에 영향을 준 적이 없다 --
    # 영향을 주는 건 `final_principal_delegation`이 1번 단계의 delegation
    # 자체를 대체한다는 것뿐이고, 그 이후 로직(2번 단계~_fuse())은 이
    # 필드를 전혀 참조하지 않는다.
    source_clarification_result: SourceClarificationResult | None = None


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
                 authority_verifier: AuthorityVerifierAgent,
                 use_clarification: bool = False,
                 n: int = 10,
                 entropy_threshold: float = 0.8,
                 experience_store: AgentExperienceStore | None = None,
                 max_verified_entropy: float = 0.0,
                 use_source_verification: bool = False,
                 source_n: int = 20):
        self.principal = principal
        self.delegate = delegate
        self.semantic_verifier = semantic_verifier
        self.authority_verifier = authority_verifier

        # Runtime Integration Phase 1 -- 전부 opt-in. use_clarification이
        # False(기본값)면 아래 두 객체는 만들어지기만 하고(LLM 호출/IO 없음)
        # run()에서 전혀 쓰이지 않는다 -- 기존 동작은 완전히 그대로다.
        self.use_clarification = use_clarification
        self.n = n
        self.entropy_threshold = entropy_threshold
        self.experience_store = experience_store
        self.max_verified_entropy = max_verified_entropy
        self._clarifying_delegate = ClarifyingDelegate(
            principal=principal, delegate=delegate, n=n, entropy_threshold=entropy_threshold)
        self._experience_harness = FrozenCandidateEvidenceHarness(entropy_threshold=entropy_threshold)

        # Source-side Semantic Flow -- 마찬가지로 opt-in. use_source_
        # verification이 False(기본값)면 아래 객체는 만들어지기만 하고
        # run()에서 전혀 쓰이지 않는다. threshold는 receiver-side와 같은
        # entropy_threshold를 그대로 재사용한다 -- 새 threshold를 만들지
        # 않는다(§29 freeze).
        self.use_source_verification = use_source_verification
        self.source_n = source_n
        self._source_clarifying_principal = SourceClarifyingPrincipal(
            principal=principal, delegate=delegate, n=source_n,
            entropy_threshold=entropy_threshold)

    def run(self, *, goal: str, context: str, budget: Budget,
           principal_id: str | None = None, task_category: str | None = None,
           episode_id: str | None = None) -> AgentRuntimeResult:
        """ground_truth/task.truth를 받지 않는다 — 파라미터가 goal/context/
        budget과(Runtime Integration Phase 1의) principal_id/task_category/
        episode_id뿐이다. 이게 이 runtime이 배포 가능하다고 주장하는
        근거의 구조적 절반이다(나머지 절반은 아래 _fuse()가 SemanticVerdict.
        interpretation이 아니라 principal_intent를 기준으로 쓴다는 것).

        `principal_id`/`task_category`는 `self.use_clarification=True`이고
        `self.experience_store`가 주어졌을 때만 의미가 있다 — experience
        조회/저장 후보 계산에만 쓰이고, 그 외에는 아무 영향도 없다(둘 다
        기본값 None으로 둬도 기존 동작과 완전히 동일)."""
        # 1) Agent A가 delegation을 만든다. use_source_verification=False
        #    (기본값)면 기존 그대로 단일 delegate() 호출. True면 source-
        #    side Semantic Flow 경계(§29 Phase 2C-P5, Finding P2C-F2)가
        #    먼저 발동한다 -- 독립적으로 source_n번 delegate()를 뽑아
        #    facet-level entropy를 재고, 불안정하면 그 disagreeing facet
        #    하나만 확인 후 재생성한다. 이후 코드(2번 단계부터)는 결과로
        #    나온 delegation 하나만 보고, use_source_verification이
        #    켜졌는지조차 모른다 -- 완전히 동일한 PrincipalDelegation
        #    모양이기 때문이다.
        source_clarification_result: SourceClarificationResult | None = None
        if not self.use_source_verification:
            delegation = self.principal.delegate(goal=goal, context=context)
        else:
            source_clarification_result = self._source_clarifying_principal.resolve(
                goal=goal, context=context)
            delegation = source_clarification_result.final_principal_delegation

        # 2) Agent A가 원래 goal/context에서만 독립적으로 의도를
        #    재구성한다 — 이 시점에 proposal 변수는 아직 존재하지 않는다
        #    (아래 3번에서야 만들어진다), 그러니 코드 구조상으로도
        #    Delegate의 proposal이 여기 섞여 들어갈 방법이 없다.
        principal_intent = self.principal.restate_intent(goal=goal, context=context)

        # 3) Agent B가 proposal을 만든다. use_clarification=False(기본값)면
        #    기존 그대로 propose() 단일 호출 — delegation 텍스트만 보고
        #    원래 goal/context 원문도, principal_intent도 보지 않는다.
        #    use_clarification=True면 sample_candidates()+entropy+(필요시)
        #    실제 clarification round로 바뀐다 — 여전히 principal_intent는
        #    보지 않는다(ClarifyingDelegate.resolve()의 goal 사용은
        #    Principal이 스스로 답하기 위한 것일 뿐, Delegate 쪽 호출에는
        #    전달되지 않는다 — clarification.py 자체의 기존 보장).
        if not self.use_clarification:
            proposal = self.delegate.propose(delegation=delegation.delegation, context=context)
            clarification_result: ClarificationResult | None = None
            experience_decision: FacetDecision | None = None
            candidate_experience: AgentExperience | None = None
        else:
            pre = self.delegate.sample_candidates(
                delegation=delegation.delegation, context=context, n=self.n)

            # Advisory 전용 -- Option B(v3, commit 0f1cf7c, B7e Phase 1에서
            # 실측 검증). 이 relation은 아래 clarifier.resolve()에도,
            # semantic/authority verification에도, _fuse()의 결정 로직에도
            # 전달되지 않는다 -- AgentRuntimeResult.experience_decision으로
            # 감사용으로만 실린다.
            latest_experience: AgentExperience | None = None
            if (self.experience_store is not None
                    and principal_id is not None and task_category is not None):
                history = self.experience_store.get(principal_id, task_category)
                latest_experience = history[-1] if history else None

            experience_decision = None
            if latest_experience is not None:
                experience_decision = self._experience_harness.decide(
                    distribution=pre, experience=latest_experience, facet="action")

            clarification_result = self._clarifying_delegate.resolve(
                goal=goal, context=context, delegation=delegation.delegation,
                pre_distribution=pre)
            proposal = _proposal_from_clarification(clarification_result)

            # 저장은 여전히 명시적이다(B7c 원칙) -- run()은 여기서
            # experience_store.add(...)를 절대 호출하지 않는다. 호출자가
            # candidate_experience를 보고 직접 store.add()해야 한다.
            candidate_experience = None
            if principal_id is not None and task_category is not None:
                candidate_experience = build_verified_experience(
                    clarification_result, principal_id=principal_id, task_category=task_category,
                    delegation=delegation.delegation, max_verified_entropy=self.max_verified_entropy,
                    episode_id=episode_id)

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
                          semantic_verdict, authority_verdict,
                          clarification_result=clarification_result,
                          experience_decision=experience_decision,
                          candidate_experience=candidate_experience,
                          source_clarification_result=source_clarification_result)

    def _fuse(self, delegation: PrincipalDelegation, principal_intent: PrincipalIntent,
             proposal: DelegateProposal, semantic_verdict: SemanticVerdict,
             authority_verdict: AuthorityVerdict, *,
             clarification_result: ClarificationResult | None = None,
             experience_decision: FacetDecision | None = None,
             candidate_experience: AgentExperience | None = None,
             source_clarification_result: SourceClarificationResult | None = None
             ) -> AgentRuntimeResult:
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
            semantic_verdict, authority_verdict, final_interpretation, principal_match,
            clarification_result=clarification_result,
            experience_decision=experience_decision,
            candidate_experience=candidate_experience,
            source_clarification_result=source_clarification_result)


def _proposal_from_clarification(result: ClarificationResult) -> DelegateProposal:
    """clarification 경로의 final_interpretation을 propose()가 돌려주는
    것과 같은 `DelegateProposal` 모양으로 감싼다 — semantic/authority
    verification과 `_fuse()`가 여전히 `proposal.interpretation`만 보면
    되도록. raw_text/response는 그 final_interpretation을 실제로 만들어낸
    `sample_candidates()` 호출 하나를 그대로 재사용한다(새로 합성하지
    않는다) — `distribution.top`/`final_interpretation`은 항상 같은
    `distribution.responses`를 다수결로 집계한 값이므로, 그중 실제로
    `final`을 파싱해낸 응답이 반드시 존재한다. `responses[0]` fallback은
    `responses=[]`로 손수 만든 distribution(예: 테스트 헬퍼)에 대해서만
    방어적으로 존재하며, 이 runtime 자신의 호출 경로에서는 도달하지
    않는다."""
    distribution = result.post_distribution if result.clarified else result.pre_distribution
    final = result.final_interpretation
    matched = next((r for r in distribution.responses
                    if parse_structured_action(r.text) == final),
                   distribution.responses[0])
    return DelegateProposal(interpretation=final, raw_text=matched.text, response=matched)
