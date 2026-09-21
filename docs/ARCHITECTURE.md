# Architecture

> 이 문서는 DualFlow 구현의 각 메커니즘을 소스 코드 수준에서 설명한다 — 클래스/함수 단위로 실제 동작을 추적하며, 왜 그렇게 설계했는지는 [DESIGN_NOTES.md](DESIGN_NOTES.md), 각 메커니즘이 어떤 실험으로 검증됐는지는 [EXPERIMENTS.md](EXPERIMENTS.md)를 참고할 것.

---

## 1. System Overview

DualFlow는 Agent A(principal)가 Agent B(delegate)에게 작업을 위임하는 한 홉(또는 다홉) 상황을 검증한다. 전체 조립은 `framework.py`의 모듈 docstring이 그대로 코드 구조다([framework.py:1-18](../src/dualflow/framework.py#L1-L18)):

```text
Agent A --위임--> Agent B
  │
  ├─ AUTHORITY FLOW  허용 범위 A 검사 (규칙 기반, 하드 제약)
  │
  └─ SEMANTIC FLOW   Experience Score
                      ├ 충분 → 자율 판단
                      └ 부족 → Entropy H
                                ├ H ≤ θ → 확정
                                └ H > θ → Clarification (최대 k회)
                                            └ K회 소진 → LLM 의미 판단

JOINT VERIFICATION   E = Authority ∩ Semantic  +  매칭 검증 → Execute / Reject
```

두 흐름(Authority Flow, Semantic Flow)은 서로 독립적으로 계산되며, Joint Verification 단계에서만 합쳐진다 — 어느 한쪽의 확신이 다른 쪽의 판정을 대신하지 않는다([DESIGN_NOTES.md](DESIGN_NOTES.md) §1).

README의 프레이밍을 그대로 따라, 시스템은 두 층으로 나뉜다.

```text
Core Safety Mechanism ── 사전 경험이 전무해도 안전
  1. Semantic Flow            entropy, information gain, clarification
  2. Authority Flow           capability / resource / scope / condition 검증
  3. Authority Feedback Loop  APPROVE / CORRECT / RESTRICT / REJECT, bounded rounds
  4. Joint Verification       확정된 의도 · 확정된 권한에 대한 최종 검사

Optimization Layer ── principal 호출 비용만 줄임, 안전성 자체는 바꾸지 않음
  Verified Authority Store → Adaptive Feedback Gate → 필요할 때만 principal 호출
```

`Config.use_verified_experience=False`로 Optimization Layer를 완전히 꺼도 Core만으로 동일한 안전성 보장이 유지된다([framework.py:65](../src/dualflow/framework.py#L65), README "Architecture" 절). 본 문서의 §3–§6은 Core, §7·§9·§10은 Optimization Layer, §8은 두 층을 관통하는 구조적 불변식, §11은 두 층이 실제로 한 번의 `DelegationVerifier.run()` 호출 안에서 어떻게 엮이는지를 다룬다.

---

## 2. Delegation Model

위임 대상 권한은 4-튜플 `(action, resource, scope, condition)`으로 표현되는 `Privilege`다([capability.py:56-88](../src/dualflow/capability.py#L56-L88)). `condition`이 많을수록 더 약한(좁은) 권한이며, `scope`는 경로 접두사(`/reports/`)나 도메인 접미사(`*.corp.com`) 두 형태를 지원하는 `scope_leq()`로 순서관계를 판단한다([capability.py:28-40](../src/dualflow/capability.py#L28-L40)).

`Budget`은 downward-closed privilege 집합을 극대 원소(generators)만으로 표현한 정규형이다([capability.py:104-135](../src/dualflow/capability.py#L104-L135)). 빈 예산 `TOP = Budget.of()`가 fail-closed 기본값이다([capability.py:138](../src/dualflow/capability.py#L138)) — 아무것도 위임하지 않으면 아무것도 못 한다.

위임 체인은 `Agent(name, budget)`들의 연쇄이고, 한 홉의 위임은 `delegate(principal, spec_ceiling)`가 `principal.meet(spec_ceiling)`으로 계산한다 — 명세가 아무리 넓은 권한을 요구해도 결과는 `principal`을 넘지 못한다([capability.py:150-158](../src/dualflow/capability.py#L150-L158)). `delegation_chain(principal, ceilings)`는 다홉 위임의 각 단계별 유효 예산을 리스트로 반환한다([capability.py:161-166](../src/dualflow/capability.py#L161-L166)).

평가 단위는 `DelegationTask`다 — 위임 명세(`spec`), 시작 예산(`principal_budget`)과 홉별 상한(`ceilings`), 해석 후보 집합(`candidates`), 평가용 실제 의도(`truth`), 시스템 변수(`sysvars`), 협상 불가 차원(`refuses`) 등을 담는다([framework.py:128-164](../src/dualflow/framework.py#L128-L164)). `effective_budget()`은 `delegation_chain(...)[-1]`이고, `ideal_decision()`은 파이프라인과 무관하게 "`truth`가 유효 예산 안에 있고(§4) SOP상 EXECUTE 대상인가(§6)"만으로 정의되는 상한선 판정이다([framework.py:152-164](../src/dualflow/framework.py#L152-L164)) — 어떤 설정도 이보다 더 잘할 수는 없다는 평가 기준선이다.

---

## 3. Semantic Flow

**"B가 A의 요청을 무엇으로 이해했는가"**를 다룬다. 해석 후보는 `Interpretation(action, resource, scope, condition)`이며 `.privilege()`로 `Privilege`로 변환된다([semantic.py:36-58](../src/dualflow/semantic.py#L36-L58)). `Belief`는 `dict[Interpretation, float]`이고, 불확실성은 Shannon entropy `entropy(p) = -Σ p·log2(p)`로 측정한다([semantic.py:71-73](../src/dualflow/semantic.py#L71-L73)).

**초기 belief.** `build_belief(candidates, experience, weight)`는 후보의 사전 그럴듯함(plausibility)과 `ExperienceStore`에서 가져온 과거 카운트를 결합해 정규화한다([semantic.py:128-136](../src/dualflow/semantic.py#L128-L136)).

**Experience Score 게이트.** `ExperienceStore.score(key)`는 `share * n / (n + smoothing)` — 가장 많이 나온 해석의 점유율(`share`)과 표본 수(`n`)를 함께 반영하는 0~1 값이다([semantic.py:108-115](../src/dualflow/semantic.py#L108-L115)). `framework._fast()`는 `score ≥ cfg.sigma`(기본 0.8)면 엔트로피 계산 자체를 건너뛰고 `ExperienceStore.best(key)`로 자율 판단한다([framework.py:191-199](../src/dualflow/framework.py#L191-L199)).

**Entropy 게이트 + Clarification.** 점수가 부족하면 `entropy(belief) ≤ cfg.theta`(기본 0.5 bits)인지 검사한다. 넘으면 최대 `cfg.k`회(기본 2) 역질의를 반복한다([framework.py:210-229](../src/dualflow/framework.py#L210-L229)). 질문 선택은 정보이득 기준이다 — `information_gain(p, q) = H(p) − E_r[H(p|r)]`는 상호정보량이므로 항상 0 이상이다([semantic.py:180-182](../src/dualflow/semantic.py#L180-L182)). `select_question()`은 `IG − λ·(이미 물어본 횟수)`가 최대인 질문을 고르고, 이득이 0 이하면 더 묻지 않는다([semantic.py:185-196](../src/dualflow/semantic.py#L185-L196)). 답변은 `apply_answer()`가 모순되는 후보를 제거하고 재정규화한다([semantic.py:199-206](../src/dualflow/semantic.py#L199-L206)).

**LLM fallback.** `H ≤ θ`로 수렴하면 LLM 호출 없이 확정한다. 그래도 안 되면(`k`회 소진) `LLMJudge.judge()`를 1회 호출한다(§6-2 참고). 즉 LLM은 "우선 시도"가 아니라 마지막 자원이다.

**Fast / Slow / AND / Adaptive.** `Config.mode`가 라우팅 전략을 고른다.
- `fast` — 위 게이트 파이프라인 그대로(`_fast`, [framework.py:182-247](../src/dualflow/framework.py#L182-L247)).
- `slow` — B가 정리한 해석 전체를 A에게 제시하고 `Principal.review()`로 승인/교정/불확정 중 하나를 받는다(`_slow`, [framework.py:279-306](../src/dualflow/framework.py#L279-L306); `Principal.review`, [semantic.py:258-293](../src/dualflow/semantic.py#L258-L293)).
- `and` — Fast로 좁힌 해석을 다시 Slow로 확인받는다. `strict=True`이므로 A의 교정(`correct`)조차 "불일치"로 처리해 확정하지 않는다 — 오탐을 0으로 유지하는 대신 미탐을 감내하는 보수적 결합([framework.py:371-382](../src/dualflow/framework.py#L371-L382)).
- `adaptive` — 평소엔 Fast만 돌리고, Fast가 미확정이거나 확정 결과가 누적 경험(`ExperienceStore`)과 정면으로 모순될 때만 Slow로 에스컬레이션한다(`_adaptive`, [framework.py:309-347](../src/dualflow/framework.py#L309-L347)). 이것은 **Semantic Flow 내부의** adaptive 전략이며, §7·§9·§10의 Optimization Layer(Authority 쪽 Adaptive Feedback)와는 다른 메커니즘이다([DESIGN_NOTES.md](DESIGN_NOTES.md) §12).

Principal 시뮬레이션은 `semantic.Principal`이 맡는다 — `truth`를 알고 있는 오라클이지만, 물어본 차원에만 답하고(`answer`), `carelessness`(부주의한 승인)와 `overcaution`(과잉신중한 반려) 두 축으로 완벽하지 않은 검토자를 흉내낸다([semantic.py:225-293](../src/dualflow/semantic.py#L225-L293)).

---

## 4. Authority Flow

**"B가 무엇을 하도록 허용됐는가"**를 다룬다. `Privilege.__le__`(`p ⪯ q`)는 action/resource가 같고, scope가 `scope_leq`로 포함되며, condition이 역포함(⊇)일 때 참이다 — condition이 많을수록 약한 권한이라는 정의와 일치한다([capability.py:71-75](../src/dualflow/capability.py#L71-L75)). `Budget.meet(other)`는 두 예산의 교집합을 generator별 `Privilege.meet` 쌍의 합집합으로 계산한다 — ChainCaps Eq.(2)의 meet rule([capability.py:119-127](../src/dualflow/capability.py#L119-L127)).

핵심 함수는 `check_authority(effective, requested)`다([capability.py:190-230](../src/dualflow/capability.py#L190-L230)). 요청이 유효 예산 안에 있으면 즉시 통과. 아니면 실패를 세 종류로 분류한다:

| `failure_kind` | 조건 | 처리 |
|---|---|---|
| `no_grant` | action/resource 자체가 예산에 없음, 또는 예산 자체가 소진(`is_empty`) | 하드 리젝트, 협상 불가 |
| `condition_missing` | scope는 맞는데 필수 condition이 빠짐 — B가 스스로 채울 수 없는 시스템 요구사항 | 하드 리젝트, 협상 불가 |
| `scope_exceeded` | action/resource/condition은 맞는데 범위만 넘음 | **협상 가능**(§5) — `suggested` 필드에 실제 허용 상한을 채워 반환 |

`scope_exceeded`일 때 `suggested`는 요청 scope와, condition을 만족하는 generator들의 scope의 meet(교집합) 중 최대값이다([capability.py:219-228](../src/dualflow/capability.py#L219-L228)). 이 계산은 위임 예산 자체에서만 나오며 `task.truth`를 전혀 보지 않는다 — 오라클이 아니다([capability.py:190-197](../src/dualflow/capability.py#L190-L197)).

`framework.DelegationVerifier.run()`은 시작 시 `delegation_chain(task.principal_budget, task.ceilings)`로 전체 홉의 유효 예산을 계산하고([framework.py:390-393](../src/dualflow/framework.py#L390-L393)), Semantic Flow가 확정한 해석에 대해 `check_authority(effective, interp.privilege())`를 호출한다(§11).

---

## 5. Authority Feedback

Authority Flow가 `scope_exceeded`로 막았을 때, `run_feedback()`이 A와의 bounded negotiation을 수행한다([authority_feedback.py:141-229](../src/dualflow/authority_feedback.py#L141-L229)). 결정 종류는 `FeedbackDecision`의 네 값이다([authority_feedback.py:48-52](../src/dualflow/authority_feedback.py#L48-L52)):

- `APPROVE` — 제시된 상한이 이미(또는 그대로) 맞음
- `CORRECT` — A가 정확한 값으로 고쳐줌(`suggested`보다 더 좁을 수 있음)
- `RESTRICT` — 제안된 상한까지만 축소하면 충분
- `REJECT` — A가 원하는 것이 이 범위 밖이거나 확인을 거부

루프 구조([authority_feedback.py:173-228](../src/dualflow/authority_feedback.py#L173-L228)):

```text
매 반복:
  1. check_authority(effective, current.privilege()) 로 재검증
     → allowed 면 확정, ConfirmedAuthority 반환
  2. failure_kind != "scope_exceeded" 면 A 에게 묻지 않고 즉시 REJECT
  3. (adaptive 모드, §10) VerifiedAuthorityStore 로 자동 해결 시도 — 협상당 최대 1회
  4. asked >= max_rounds 면 미해결 REJECT
  5. principal.review_authority(current, suggested) 로 실제 문의
     → REJECT/None 이면 종료, 아니면 current = fb.confirmed 후 1번으로
```

`principal.review_authority()`는 `AuthorityPrincipal` Protocol만 요구한다 — 이 모듈은 concrete `Principal`의 내부(`task.truth`)에 접근하지 않고, 그 응답(`AuthorityFeedback`)만 본다([authority_feedback.py:61-65](../src/dualflow/authority_feedback.py#L61-L65)). 이 경계가 [DESIGN_NOTES.md](DESIGN_NOTES.md) §3이 명시하는 "runtime은 `task.truth`를 읽지 않는다"는 원칙의 구체적 구현이다.

매 확정 직전에는 항상 top의 `check_authority()`로 다시 검증한다 — A의 응답도, 재사용한 경험(§10)도 그 자체로는 최종 권한이 아니다([authority_feedback.py:158-161](../src/dualflow/authority_feedback.py#L158-L161)). 이것이 careless한 `APPROVE`가 그대로 권한 확장으로 이어지지 않는 이유다(§8).

`NegotiationResult.rounds`는 A에게 **실제로** 물어본 횟수만 센다 — auto-restrict(§10)로 풀리면 0, 애초에 협상 불가로 즉시 거부돼도 0이다([authority_feedback.py:84](../src/dualflow/authority_feedback.py#L84), [authority_feedback.py:163-164](../src/dualflow/authority_feedback.py#L163-L164)).

---

## 6. Joint Verification

`rule_engine.py`가 담당하며, SAGE-Bench의 SOP 그래프 구조를 위임 검증에 재적용한다([rule_engine.py:1-15](../src/dualflow/rule_engine.py#L1-L15)). `classify(interp, sysvars)`가 해석을 거친(coarse) 분류 필드 `Fields(action_type, sensitivity, scope_breadth, condition_met, ...)`로 변환하고([rule_engine.py:56-78](../src/dualflow/rule_engine.py#L56-L78)), `RuleEngine.trace()`가 결정론적 SOP를 따라간다([rule_engine.py:92-107](../src/dualflow/rule_engine.py#L92-L107)):

```text
stage1 Classification
stage2 ActionType   read→stage3 | modify→stage4 | transfer→stage5
stage3 Sensitivity  low→EXECUTE | high→stage6
stage4 ScopeBreadth narrow→stage6 | broad→ESCALATE
stage5 ConditionMet true→stage6 | false→REJECT
stage6 ApprovalRequired(sysvar) false→EXECUTE | true→ESCALATE
```

두 겹의 비교로 매칭을 판정한다([rule_engine.py:157-193](../src/dualflow/rule_engine.py#L157-L193)):

1. **경로 유사도** `sim_path(p, p*) = |p ∩ p*| / |p*|`(SAGE-Bench Eq.6) — B의 해석 경로와 A의 원본 의도(`intent`)로 계산한 정답 경로의 비교([rule_engine.py:114-118](../src/dualflow/rule_engine.py#L114-L118)).
2. **필드 매칭** `V_action ∧ V_resource ∧ V_scope ∧ V_condition` — 원본 값의 exact match([rule_engine.py:121-134](../src/dualflow/rule_engine.py#L121-L134)).

`match_intent()`는 `matched = sim ≥ tau and action == ref_action and fields_ok`로 두 조건을 결합한다([rule_engine.py:157-193](../src/dualflow/rule_engine.py#L157-L193)). `fields_ok`는 `require_fields`(=`Config.use_field_match`) 인자로 제어되며 **기본값은 `False`**다 — exact-field 매칭은 `task.truth`에서 유도된 reference와 직접 비교하는 oracle/ablation 기능이라 live gate에는 쓰지 않는다([DESIGN_NOTES.md](DESIGN_NOTES.md) §4·§8, [framework.py:429-430](../src/dualflow/framework.py#L429-L430)). 기본 live gate는 `Sim_path ≥ τ` **AND** `Terminal(p) = Terminal(p*)`(EXECUTE/ESCALATE/REJECT) 두 조건이다.

`sim_path`만으로는 잡지 못하는 구멍(같은 SOP 버킷 안의 자원 치환)을 필드 매칭이 메우도록 설계됐지만, 그 필드 매칭 자체가 기본 비활성이라는 점이 §3 Experiment 1의 misread(M1) 사례가 Authority Feedback이 아니라 Joint Verification의 `action` 필드 불일치(경로 유사도 기반이 아니라 `action == ref_action`)로 잡히는 이유다(`docs/EXPERIMENTS.md` §6.3).

---

## 7. Optimization Layer

Core(§3–§6)만으로 안전성이 성립한다는 것을 전제로, Optimization Layer는 **반복되는 principal 호출 비용**만을 줄인다. 구성 요소는 두 가지다 — `authority_feedback.VerifiedAuthorityStore`(§9)와 `run_feedback()` 내부의 adaptive 재사용 분기(§10). 스위치는 `Config.use_verified_experience`(기본 `True`)이며, `verified_experience_n_min=3`·`verified_experience_sigma=0.8`이 재사용 기준을 정한다([framework.py:65-67](../src/dualflow/framework.py#L65-L67)).

비용 모델은 `Verdict.cost(cfg)`가 계산한다 — 역질의(`cost_question`=1.0), Slow 리뷰(`cost_review`=3.0), LLM 호출(`cost_llm`=10.0), Authority Feedback 1라운드(`cost_authority_feedback`=3.0, review와 동급 — 둘 다 A 호출)의 가중합이다([framework.py:120-124](../src/dualflow/framework.py#L120-L124), [framework.py:57-60](../src/dualflow/framework.py#L57-L60)). Optimization Layer가 줄이는 것은 정확히 이 비용의 `n_authority_feedback` 항이다.

이 층을 꺼도(`use_verified_experience=False`) `run_feedback()`은 §5의 루프에서 3단계(자동 재사용 시도)를 건너뛰고 바로 4단계(`principal.review_authority` 호출)로 가므로, Core의 안전 보장과 판정 결과 집합은 바뀌지 않고 A에게 매번 묻게 될 뿐이다([authority_feedback.py:186-203](../src/dualflow/authority_feedback.py#L186-L203) 분기 자체가 `verified is not None`을 전제).

---

## 8. Non-amplification Property

[DESIGN_NOTES.md](DESIGN_NOTES.md) §7이 명시하듯 DualFlow에는 별도의 `non_amplification_check()` 함수가 없다 — 같은 **intersection + revalidation** 패턴이 세 지점에서 반복 적용되어 구조적으로 성립하는 불변식이다.

**(1) 위임 체인.** 각 홉은 `Budget.meet()`으로 합성된다(`__le__`, [capability.py:129-130](../src/dualflow/capability.py#L129-L130); `meet`, [capability.py:119-127](../src/dualflow/capability.py#L119-L127)):

$$
C_{\text{next}} = C_{\text{current}} \cap C_{\text{ceiling}}
\qquad\Rightarrow\qquad
C_n \subseteq C_{n-1} \subseteq \cdots \subseteq C_A
$$

`delegation_chain()`이 이 단조 감소 수열 전체를 반환한다([capability.py:161-166](../src/dualflow/capability.py#L161-L166)).

**(2) Authority Feedback.** Principal이 어떤 수정안을 반환하더라도(`APPROVE`·`CORRECT`·`RESTRICT` 무관), 매 라운드 `check_authority(effective, current.privilege())`로 현재 예산에 대해 재검증한다([authority_feedback.py:174-179](../src/dualflow/authority_feedback.py#L174-L179)) — careless한 `APPROVE`가 예산 밖이면 non-amplification 검사가 이후 단계에서 반드시 다시 걸러낸다([semantic.py:301-303](../src/dualflow/semantic.py#L301-L303) 주석).

**(3) Adaptive reuse(§10).** 검증된 이력도 그대로 신뢰하지 않고 현재 예산과 다시 meet한다([authority_feedback.py:193](../src/dualflow/authority_feedback.py#L193)):

$$
C_{\text{adaptive}} = C_{\text{experience}} \cap C_{\text{current budget}}
\qquad\Rightarrow\qquad
C_{\text{adaptive}} \subseteq C_{\text{current budget}} \subseteq C_A
$$

세 지점 모두 "합성(위임)은 권한을 보존하거나 줄일 뿐, 늘릴 수 없다"는 동일한 불변식의 인스턴스다([capability.py:13](../src/dualflow/capability.py#L13)). 따라서 정확한 표현은 "non-amplification을 검사한다"가 아니라 "non-amplification이 구조적으로 보장된다"이다.

---

## 9. Verified Authority Store

**Optimization Layer 구성 요소.** `authority_feedback.VerifiedAuthorityStore`는 Principal이 `review_authority()`로 실제 확인해주고, 재검증까지 통과해 최종 EXECUTE로 이어진 scope만 저장한다 — 일반 `semantic.ExperienceStore`(§3)와 달리 auto-restrict로 재사용된 결과는 기록하지 않는다(캐시가 스스로를 강화하는 순환을 막기 위함)([authority_feedback.py:93-121](../src/dualflow/authority_feedback.py#L93-L121)).

`record(key, interp)`는 새로 확인된 값이 지금까지의 이력과 다르면 낡은 이력을 버리고 새로 시작한다(drift 대응, [authority_feedback.py:106-121](../src/dualflow/authority_feedback.py#L106-L121)). `agreement_ratio(key)`는 가장 많이 확인된 scope가 전체 확인 이력에서 차지하는 비율이다([authority_feedback.py:126-131](../src/dualflow/authority_feedback.py#L126-L131)).

기록은 `framework.DelegationVerifier.run()`이 EXECUTE 확정 후에만 수행하며, 조건은 세 가지를 모두 만족해야 한다 — 실제 negotiation이 있었고(`authority_negotiated`), A에게 실제로 물어봤고(`n_authority_feedback > 0`), auto-restrict로 풀린 게 아니어야 한다(`not authority_auto_restricted`)([framework.py:455-458](../src/dualflow/framework.py#L455-L458)). 즉 이 저장소에 들어가는 것은 "실행됐다"가 아니라 "A가 직접 확인해줬고 그 결과가 실제로 안전하게 실행까지 이어졌다"는 근거를 가진 값뿐이다.

---

## 10. Adaptive Feedback

**Optimization Layer 구성 요소.** `run_feedback()` 내부에서 매 `scope_exceeded` 발생 시, 아직 이번 협상에서 자동 재사용을 시도하지 않았다면(`tried_auto`) 다음을 확인한다([authority_feedback.py:186-203](../src/dualflow/authority_feedback.py#L186-L203)):

```text
n_confirmed(key) >= verified_n_min(=3)  and  agreement_ratio(key) >= verified_sigma(=0.8)
  → cand = VerifiedAuthorityStore.best(key)
  → cand.action/resource 가 auth.suggested 와 같으면
      intersected = cand.privilege().meet(auth.suggested)     # §8 (3)
      intersected is not None → A 에게 묻지 않고 그 값으로 재시도
      intersected is None(=드리프트) → 실제 Feedback 으로 진행
```

자동 재사용은 협상당 **최대 1회**만 시도한다(`tried_auto` 플래그, [authority_feedback.py:170,186-187](../src/dualflow/authority_feedback.py#L170)). 재사용된 값도 다음 반복에서 top의 `check_authority()`로 다시 검증되므로(§8), 경험은 "무엇을 시도해볼지"를 줄여줄 뿐 "허용되는지"를 대신 판단하지 않는다([authority_feedback.py:158-161](../src/dualflow/authority_feedback.py#L158-L161)).

이력이 부족하거나 불일치하거나 현재 budget과 안 맞으면(intersection이 `None`) 그대로 §5의 실제 `principal.review_authority()` 문의로 넘어간다 — gate 자체가 단순한 두 수치 비교(`n_confirmed`, `agreement_ratio`)라는 점이 [DESIGN_NOTES.md](DESIGN_NOTES.md) §10이 강조하는 "복잡한 learned risk score 대신 해석 가능한 단순 규칙"이다.

---

## 11. End-to-end Decision Flow

`framework.DelegationVerifier.run(task)`가 orchestration 진입점이다([framework.py:385-464](../src/dualflow/framework.py#L385-L464)). 단계별로:

**1) Authority Flow — 유효 예산.** `delegation_chain(task.principal_budget, task.ceilings)`로 모든 홉의 예산을 계산하고 마지막 값을 `effective`로 둔다(§8의 단조 감소 수열이 여기서 만들어진다).

**2) Semantic Flow — 해석 확정.** `_semantic(task, log)`가 `cfg.mode`(fast/slow/and/adaptive/sage)에 따라 분기해 `SemanticOutcome`(interpretation, confirmed, route, h_initial, h_final, n_questions, n_llm, n_reviews)과 `principal` 인스턴스를 반환한다(§3). `semantic_ok = sem.confirmed`.

**3) Joint Verification — Authority 체크.** `check_authority(effective, interp.privilege())`를 호출한다(§4).

**4) Authority Feedback Loop.** `not auth.allowed and auth.failure_kind == "scope_exceeded" and cfg.use_authority_feedback`일 때만 `run_feedback()`을 실행한다(§5, §10). 협상이 풀리면(`neg.resolved`) `interp`를 협상 결과로 교체하고 `check_authority()`로 **재검사**한다 — `n_authority_feedback`, `authority_negotiated`, `authority_auto_restricted`가 이 단계에서 채워진다([framework.py:409-427](../src/dualflow/framework.py#L409-L427)).

**5) Joint Verification — 매칭.** `match_intent(interp, task.intent_fields, task.sysvars, engine, cfg.tau, require_fields=cfg.use_field_match)`로 `MatchResult`를 계산한다(§6).

**6) 판정 결합.** 세 게이트를 각각의 ablation 스위치와 함께 평가한다([framework.py:435-446](../src/dualflow/framework.py#L435-L446)):

```text
authority_ok = auth.allowed or not cfg.use_authority
matching_ok  = (m.matched and m.executable) or not cfg.use_matching
semantic_gate = semantic_ok or not cfg.use_semantic

not authority_ok    → REJECT "권한 위반"
not semantic_gate   → REJECT "의미 확정 실패"
not matching_ok     → REJECT "의도 불일치"
그 외                → EXECUTE "권한·의미 양쪽 통과"
```

**7) Optimization Layer 기록.** `ExperienceStore.record(task.key, interp, decision == EXECUTE)`는 항상 호출되지만 내부적으로 `accepted=False`면 아무것도 쌓지 않는다(§3). EXECUTE이고 §9의 세 조건을 만족하면 `VerifiedAuthorityStore.record()`도 호출된다(§9).

**8) Verdict 반환.** 최종적으로 다음 필드를 가진 `Verdict`를 만든다([framework.py:96-114](../src/dualflow/framework.py#L96-L114)):

```text
decision, reason, route, interpretation,
h_initial, h_final, n_questions, n_llm, n_reviews,
authority_ok, semantic_ok, match, effective_budget,
n_authority_feedback, authority_negotiated, authority_auto_restricted,
log
```

`docs/EXPERIMENTS.md` §6.3(Mechanism Attribution)은 바로 이 `Verdict`의 `authority_ok`/`semantic_ok`/`match.matched`/`n_authority_feedback` 필드를 직접 읽어, 세 가지 공격을 각각 어떤 메커니즘(Joint Verification / Authority Feedback Loop / Authority 하드 리젝트)이 담당했는지 재구성한다 — 이 문서의 §4–§6·§9–§10이 그 각 메커니즘의 구현이다.

---

관련 문서: [EXPERIMENTS.md](EXPERIMENTS.md) · [BASELINES.md](BASELINES.md) · [DESIGN_NOTES.md](DESIGN_NOTES.md) · [README.md](../README.md)
