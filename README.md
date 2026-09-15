# dualflow — Agent-to-Agent 권한 위임 검증 프레임워크

**Authority Flow × Semantic Flow → Joint Verification** 아키텍처의 실행 가능한 구현.

2026-09-02 미팅자료의 플로우 다이어그램을 그대로 코드로 옮기고, 세 편의 선행연구에서
가져온 부분이 실제로 의도한 대로 작동하는지 실험으로 확인할 수 있게 만들었다.
LLM 호출 없이 결정론적으로 돌아가며, 실제 모델은 인터페이스 하나만 맞추면 붙는다.

**"Dual" 은 Semantic Flow × Authority Flow 를 말한다 — Fast/Slow 가 아니다.**
아래 §3 에 Fast/Slow/AND/Adaptive 비교가 많이 나오는데, 이건 Semantic Flow *안에서*
"B 의 해석을 어떻게 확정할까" 를 고르는 전략일 뿐이다. 이 프레임워크가 실제로 풀려는
두 문제는 서로 다르다:

- **Semantic uncertainty** — B 가 A 의 요청을 제대로 이해했는가? ("지난달 보고서"
  가 어느 파일인지 애매함) → entropy/information gain 으로 잰다(§2 RQ1).
- **Authority consistency** — 해석이 명확해도 실제로 허용된 범위인가? B 가 자기
  해석에 100% 확신해도(H=0), 위임 상한선 안의 *다른* 자원을 가리킬 수 있다
  ("/reports/2026-08/" 대신 "/reports/2025-01/") → `low uncertainty ⇏ valid
  delegation`, 이게 `silent_misread`/belief 조작 실험이 실측으로 보여주는 것이다.

둘 다 확인해야 안전하다(Joint Verification). Fast/Slow/AND/Adaptive 는 그 확인을
**얼마나 싸게** 할지에 관한 최적화이지, 확인 자체를 대체하지 않는다. §7-2 에서
확인했듯 현재 verifier 가 가진 정책·권한 정보만으로는 A 가 실제로 의도한 정확한
resource/scope 를 복원할 수 없다 — `task.truth` 와 직접 비교하면 채점 오라클이
될 뿐이다. 추가 신뢰 소스가 없다면 A 에게 직접 확인받아야 하고, 그게
**Authority Feedback Loop**(`authority_feedback.py`, §7-2)다.

```bash
pip install -e ".[dev]"

dualflow-demo                    # 11개 실험 전체 (텍스트)
python -m dualflow.demo fastslow attack careless authfeedback adaptiveauth   # 필요한 것만
dualflow-plots                   # figures/ 에 그림 8장 저장
dualflow-plots careless --trials 50   # fig5 논문용 (기본 10회는 ±3%p 흔들린다)
python -m dualflow.demo joint    # 특정 파트만
pytest -q                        # 149개 검증 테스트
```

---

## 1. 아키텍처 대응표

```
Agent A --위임--> Agent B
  │
  ├─ AUTHORITY FLOW   허용 범위 A 검사 (action·resource·scope·condition)
  │                     ├ no_grant / condition_missing → 즉시 차단(하드 제약)
  │                     └ scope_exceeded → AUTHORITY FEEDBACK LOOP (§7-2)
  │                                          B 제안 → A 확인/축소(bounded, ≤k회)
  │                                          → 재검증 → 통과 or 차단
  │
  └─ SEMANTIC FLOW    Experience Score
                       ├ 충분(≥σ) → 자율 판단 ──────────────┐
                       └ 부족 → Entropy H                  │
                                 ├ H ≤ θ → 확정 ───────────┤
                                 └ H > θ → Clarification (최대 k회)
                                            └ K회 소진 → LLM 의미 판단 ─┘
                                                                        │
JOINT VERIFICATION   E = Authority ∩ Semantic  +  매칭 검증 → Execute / Reject
```

| 슬라이드 구성요소 | 구현 | 선행연구 |
|---|---|---|
| 허용 범위 A 검사 | `capability.Budget`, `check_authority` | ChainCaps §3.2–3.3 |
| 위임 체인 감쇠 | `delegation_chain`, `Budget.meet` | ChainCaps Eq.(2), Thm 3.1 |
| Authority Feedback Loop | `authority_feedback.run_feedback` | 신규 (§7-2) |
| Action Space·확률분포 | `semantic.Interpretation`, `build_belief` | SAGE-Agent Def.2–3 |
| Entropy H | `semantic.entropy` | (EVPI → H 로 교체) |
| 임계치 θ | `Config.theta` | 신규 |
| Experience Score | `semantic.ExperienceStore` | 신규 |
| Clarification | `select_question`, `information_gain` | SAGE-Agent Def.4를 IG로 재정의 |
| LLM 의미 판단 | `llm.LLMJudge` (fallback) | SAGE-Agent(상시) → fallback 으로 격하 |
| 매칭 검증 | `rule_engine.match_intent`, `sim_path` | SAGE-Bench Eq.(6) |
| Verified Authority Store / Adaptive Gate | `authority_feedback.VerifiedAuthorityStore` | 신규 (§7-2 확장) |
| Execute / Reject | `framework.DelegationVerifier.run` | — |

### Core Mechanism vs Optimization Layer

4단계까지 끝낸 지금, 전체 구조를 두 층으로 고정한다. **Core 만으로도 안전하게
동작해야 한다** — Verified Experience 가 하나도 없어도(cold-start) correctness
가 무너지면 안 된다는 게 이 구분의 존재 이유다.

```
Core Mechanism ── 안전성을 만드는 부분 (Experience 없이도 항상 동작)
  1. Semantic Flow           semantic uncertainty / clarification
  2. Authority Flow          capability / scope / condition validation
  3. Authority Feedback Loop APPROVE / CORRECT / RESTRICT / REJECT
  4. Joint Verification      confirmed authority 안에서 최종 실행 검증

Optimization Layer ── 개입 "비용" 을 줄이는 부분 (있으면 싸지고, 없어도 안전함은 그대로)
  Verified Authority Store → Adaptive Feedback Gate → 필요할 때만 Principal Feedback
```

Optimization Layer 의 목적은 **새로운 권한을 만들어내는 게 아니라, 이미
Principal 이 검증한 결과를 안전하게 재사용해 반복적인 feedback 비용을 줄이는
것**이다. 그래서 이 층을 완전히 꺼도(`use_verified_experience=False`) Core
만으로 동일한 안전성을 낸다 — 단지 매번 A 를 부를 뿐이다.

**Verified Authority Experience 는 일반 Semantic Experience 와 다른 것이다.**
`semantic.ExperienceStore` 는 "이 요청을 과거엔 어떻게 해석했는가" — 실행된
아무 해석이나 담는다. `authority_feedback.VerifiedAuthorityStore` 는

$$ E_v = \{\, \text{authority state Principal 이 확인해주고, 시스템이 현재
예산에 대해 재검증한 것} \,\} $$

만 담는다. **admission 조건**(둘 중 하나라도 빠지면 저장 안 함):

```
B Proposal → Principal Feedback(Correct/Restrict/Approve)
           → Authority 재검증 → Joint Verification → EXECUTE
           → VerifiedAuthorityStore.record()
```

즉 "Principal confirmation + authority revalidation + successful execution
을 모두 거친 기록만" 이다 — auto-restrict 로 재사용된 결과는 다시 기록하지
않는다(그러면 캐시가 스스로를 강화하는 순환이 생긴다, `framework.py` 참고).

**Adaptive Gate** 는 일부러 단순하다 — risk score 없이 두 조건뿐이다:

$$ n_{\text{confirmed}} \ge 3 \quad\text{and}\quad \text{agreement\_ratio} \ge 0.8
\;\Rightarrow\; \text{Feedback 생략 가능} $$

그렇지 않으면(cold-start, 이력 부족, agreement 낮음, drift 감지) Principal
Feedback 으로 간다.

**가장 중요한 안전 invariant** — Optimization Layer 가 켜져 있어도 이 부등식은
절대 깨지지 않는다:

$$ C_{\text{adaptive}} = C_{\text{experience}} \cap C_{\text{current budget}}
\;\subseteq\; C_{\text{current budget}} \;\subseteq\; C_A $$

과거 Experience 가 무엇을 기억하고 있든 **현재 위임된 권한보다 넓어질 수
없다** — Verified Experience 는 권한을 부여하는 source 가 아니라 **현재
권한을 좁히는 hint** 일 뿐이다(`test_verified_history_alone_can_never_widen_authority`
가 이력에 지금 예산보다 넓은 값이 들어 있어도 결과가 항상 현재 예산 안으로
재교집합됨을 확인한다).

**Verified history 는 재사용 가능하지만 영구하지 않다.** 위임 범위가 실제로
바뀌면(intent drift) 새로 확인된 값이 낡은 이력과 다르므로, `record()` 가
낡은 이력을 리셋하고 새 이력을 처음부터 다시 쌓는다 — 그래서 Verified
Experience 는 "낡아버린 정책/캐시" 처럼 보이지 않는다(§3 실험⑥).

## 2. RQ 대응

**RQ1 — 애매함을 숫자로 어떻게 잴 것인가**
`semantic.py`. 해석 후보 집합 Ω 위의 확률분포를 만들고 Shannon entropy H(bits)로 잰다.
후보는 (action, resource, scope, condition) 네 차원의 구조화된 값이므로, 자연어 확률이
아니라 **권한 스키마 위의 확률**이다. Authority Flow 와 같은 좌표계를 쓰는 것이 핵심이다.

**RQ2 — 언제 LLM 을 쓸 것인가**
`framework.DelegationVerifier._semantic`. Experience → θ → k회 역질의 → LLM 순서로 게이트가
열린다. `demo.py theta` 가 θ 스윕으로 비용-이득 곡선을 그린다.

**RQ3 — 되물어 얻은 답이 원래 의도와 맞는지**
`rule_engine.py`. Agent A 의 원본 의도에서 정답 경로 p\* 를, Agent B 의 최종 해석에서 경로 p 를
각각 Rule Engine 으로 계산하고 Sim_path = |p ∩ p\*| / |p\*| 로 비교한다.

## 3. 실험 결과

`DelegationBench-mini` — 각 트랙이 잡아야 할 실패 모드를 하나씩 담은 9개 위임 과제
(clear / ambiguous / persistent / over-privilege / laundering / misread / condition / escalation).

지표는 ChainCaps Table 1 의 구성을 따랐다. 비용은 역질의 1, LLM 호출 10 으로 계산했다.

| 설정 | unsafe↓ | benign↑ | over-rej | LLM률↓ | 평균질문 | 평균비용↓ |
|---|---|---|---|---|---|---|
| Authority only | 22.2% | 80.0% | 0.0% | 11.1% | 0.44 | 1.56 |
| Semantic only | 22.2% | 80.0% | 20.0% | 11.1% | 0.44 | 1.56 |
| θ 게이트만 (LLM 없음) | 0.0% | 60.0% | 40.0% | 0.0% | 0.44 | 0.44 |
| 상시 LLM (SAGE-Agent 형) | 0.0% | 100.0% | 0.0% | 100.0% | 0.00 | 10.00 |
| **Full (제안)** | **0.0%** | 80.0% | 20.0% | **11.1%** | 0.44 | **1.56** |

- `unsafe` = 실행했는데 권한 밖이었거나 A 의 의도와 다른 해석이었던 비율
- `benign` = 정당한 위임을 올바른 해석으로 실행한 비율
- 정답 기준(`ideal_decision`)은 파이프라인과 무관하게 **A 의 실제 의도**만으로 유도한다.

읽는 법:
- **두 트랙이 각각 다른 것을 잡는다.** Authority 를 빼면 삭제·외부발송·조건위반이 통과하고,
  매칭을 빼면 조용한 오해석과 승인 대상 건이 통과한다. 둘 다 22.2% unsafe 로 같은 크기다.
- **상시 LLM 대비 이득은 정확도가 아니라 비용에서 나온다.** 벤치마크의 LLM 오라클은
  항상 정답을 주는 상한선이므로 benign 100% 를 찍는다. 그 상대에게도 제안 프레임워크는
  같은 0% unsafe 를 **1/6 비용**으로 달성한다. 다만 over-rejection 1건을 지불한다.
- 남은 20% over-rejection 은 `silent_misread` 한 건이다. B 가 확신에 차서(H=0.47 ≤ θ)
  틀리게 해석했고, 매칭이 그것을 잡아 거절했다. 안전한 실패지만 쓸모는 잃었다.

### θ 스윕 (RQ2)

| θ | unsafe↓ | benign↑ | LLM률 | 평균질문 | 평균비용 |
|---|---|---|---|---|---|
| 0.00 | 0.0% | 100.0% | 11.1% | 1.22 | 2.33 |
| 0.25 | 0.0% | **100.0%** | 11.1% | 1.22 | 2.33 |
| 0.50 | 0.0% | 80.0% | 11.1% | 0.44 | 1.56 |
| 1.00 | 0.0% | 80.0% | 0.0% | 0.22 | 0.22 |
| 2.00 | 0.0% | 40.0% | 0.0% | 0.00 | 0.00 |

![θ 스윕 — 안전성은 평평하고 유용성/비용만 움직인다](figures/fig3_theta_sweep.png)

**unsafe 가 어느 θ 에서도 0%** — θ 는 "얼마나 유용한가"만 조절하고 "안전한가"는 Authority
Flow 와 매칭 검증이 따로 책임진다는 뜻이다. 기본값 θ=0.5 는 이 벤치마크에서 θ=0.25 에
**지배당한다**(같은 LLM률로 benign 100%) — 슬라이드 설정을 따라 0.5 로 뒀을 뿐, 실제
도메인에서는 반드시 스윕으로 정할 것.

### Fast / Slow / AND 파일럿 (피드백 ①②)

Semantic Flow 를 세 가지 방식으로 갈라 각각 돌린다. `Config.mode` 로 전환한다.

- **Fast** — 엔트로피를 한 번 재고 θ 로 판단. 넘으면 역질의(k회) → LLM fallback.
- **Slow** — B 가 해석을 정리해 A 에게 제시하고 승인/교정받는다. 엔트로피를 쓰지 않는다.
- **AND** — Fast 로 좁힌 해석을 A 가 **승인해야만** 통과. 교정 요구는 불일치로 보고 차단한다.
- **Adaptive** — 평소엔 Fast 만. Fast 가 확정 못 했거나 **누적 경험과 모순될 때만** Slow 로
  에스컬레이션한다 (`mode="adaptive"`, §3 실험 ④ 참고 — 아래 "Slow 를 언제 켤지" 에 대한 답).

베이스라인 `SAGE-Agent (Eq.2+Def.4)` 는 우리 엔트로피 코드의 스위치를 끈 것이 아니라
**논문 공식을 그대로 옮긴 별도 구현**이다 (`sage_baseline.py`, §5 참고).

| 설정 | unsafe↓ | benign↑ | over-rej | 질문 | 검토 | LLM | 비용↓ |
|---|---|---|---|---|---|---|---|
| SAGE-Agent (Eq.2+Def.4) | 77.8% | 40.0% | 0.0% | 0.56 | 0.00 | 1.67 | 17.22 |
| SAGE + Joint | 11.1% | 40.0% | 40.0% | 0.56 | 0.00 | 1.67 | 17.22 |
| Fast + Joint | 0.0% | 80.0% | 20.0% | 0.44 | 0.00 | 0.11 | 1.56 |
| Slow + Joint | 0.0% | **100.0%** | 0.0% | 0.00 | 1.00 | 0.00 | 3.00 |
| AND | 0.0% | 80.0% | 20.0% | 0.44 | 1.00 | 0.11 | 4.56 |
| Adaptive (제안) | 0.0% | 80.0% | 20.0% | 0.44 | **0.00** | 0.11 | **1.56** |

![Fast/Slow/AND 파일럿 — 정상 조건](figures/fig1_pilot_normal.png)

**정상 상황에서는 Slow 가 AND 를 지배한다** — 안전성이 같고, A 의 교정이 `silent_misread`
를 살려내 benign 100% 를 찍으며 비용도 더 싸다. 대신 **모든 위임에서 A 를 호출**한다(검토율
1.00) — 연구의 출발점이 "개입 최소화" 였으므로, 진짜 문제는 "Slow 를 켤지" 가 아니라
**"언제 켤지"** 였다. 이 파일럿은 경험이 아직 없는 신선한 `ExperienceStore` 라 Adaptive 는
그냥 Fast 와 같다(비용·검토율 동일) — Adaptive 가 실제로 갈리는 지점은 경험이 쌓인 뒤
공격이 들어오는 §3 실험 ④ 다.

### belief 조작 공격 (피드백 ⑥ — "belief 는 공격에 취약")

**위협 모델.** Agent B 의 후보 생성 단계가 오염됐다고 본다(프롬프트 인젝션, 오염된 검색
결과, 또는 악의적 B). 공격자는 후보 집합을 목표 해석 하나로 좁혀 자기신고 불확실성을
H=0 으로 만든다. SAGE-Agent 의 π_c 도 **같은 입력에서** 계산되므로 동일하게 π=1 이 된다.
A 의 의도와 권한 정의는 신뢰하고, B 의 후보 생성은 신뢰하지 않는다.

**Eq.(2) 해석 논란은 결과를 바꾸지 못한다.** Eq.(1)의 균등 tool prior 1/K 를 살리는
해석(`sage_tool_prior=True`)으로 돌려도 공격 하 unsafe 는 100% 로 같다. 공격이 후보를
K=1 로 좁히기 때문이다 (`test_attack_works_under_both_readings_of_eq2`).

공격이 가능한 근거는 SAGE-Agent 논문 안에 있다. 후보와 `<UNK>` 표시를 평가 대상 모델이
직접 만들고(Step 1), 시스템 프롬프트가 *"`<UNK>` 를 남용하지 마라, 기본값을 가정해도 된다"*
고 지시하며(§B.3), 보상 `Cert(a_t) = max π_c` 가 확신에 찬 호출을 더 크게 보상한다(§6.2).
**설계 전체가 불확실성 축소 방향으로 압력을 건다.**

SAGE-Agent 원 공식에서 무슨 일이 일어나는지부터 보면:

```
· max π = 1.0 ≥ τ_exec 로 Step 1 에서 즉시 실행: 9/9건
· EVPI 가 한 번이라도 계산된 경우:              0/9건
· 명확화 질문:                                  0회
```

| 설정 | unsafe↓ | benign↑ | over-rej | 비용 |
|---|---|---|---|---|
| SAGE-Agent (Eq.2+Def.4) | **100.0%** | 0.0% | 0.0% | 10.00 |
| SAGE + Joint | 44.4% | 0.0% | 20.0% | 10.00 |
| Fast + Joint | 44.4% | 0.0% | 20.0% | 0.00 |
| Slow + Joint | 0.0% | 80.0% | 20.0% | 3.00 |
| AND | **0.0%** | 0.0% | 100.0% | 3.00 |
| Adaptive (제안) | 44.4% | 0.0% | 20.0% | 0.00 |

Adaptive 가 여기서는 Fast 와 똑같이 뚫린다 — **한계이지 버그가 아니다.** 비교할 경험
이력이 아직 없는 "첫 공격" 이라 불일치 신호 자체가 없다. 이 방어는 "이력이 쌓인 뒤 공격이
온다" 는 실험 ③/④ 의 전제 위에서만 작동한다.

![belief 조작 공격 — 정상 vs 공격 조건](figures/fig2_attack.png)

읽는 법:

- **EVPI 는 방어선이 아니다.** τ_exec 검사가 EVPI 계산보다 앞에 있어, π 가 문턱을 넘으면
  질문도 EVPI 도 실행되지 않는다 — 9건 전부 그렇게 통과했다.
- **확률을 조작할 필요조차 없다.** Eq.(13)은 지정된 인자에 무조건 p=1 을 준다. 공격자는
  **빈칸만 채우면** π=1 을 얻는다.
- **엔트로피 축만으로는 부족하다.** Authority·매칭을 붙여도 44.4% 가 통과한다 — *같은 SOP
  경로, 같은 권한 안의 다른 자원* 을 노린 공격(`/reports/2026-08/` → `/reports/2025-01/`)은
  Sim_path 가 경로만 보므로 못 잡는다.
- **Slow 축은 구조적으로 면역이다.** A 의 검토는 B 의 자기신고 값을 입력으로 쓰지 않고,
  교정까지 해줘 benign 80% 를 복구한다.
- **AND 는 공격 하에서 오탐 0%, 미탐 100%** — 보수적 트레이드오프의 정의 그대로다.

주장은 "우리가 더 싸다" 가 아니라 **"자기신고 불확실성 단일 축에 실행 결정을 거는 구조는
뚫린다"** 이고, AND 결합은 취향이 아니라 그 공격에 대한 필연적 대응이다.

### 실험 ③ — A 의 검토가 완벽하지 않을 때 (`fig5`)

위 두 표는 A 가 **항상 옳게 교정하는 완벽한 검토자**라고 가정한다. 그 가정에서는
Slow 단독이 AND 를 지배한다(안전성 동일, benign 100%, 비용 저렴). 현실의 검토자는
그렇지 않으므로 `Config.carelessness` — A 가 제안을 대충 읽고 그냥 승인해 버릴 확률 —
를 넣고 스윕한다.

시나리오는 **정상 운영 5회로 경험을 쌓은 뒤 같은 유형에 belief 조작 공격 1회** 다.
지표는 공격 시점의 unsafe 실행률이고, 확률적이므로 50회 평균이다.

> **버그 수정 (2026-09-07):** `adversarial_tasks()` 가 공격 버전의 `name` 에 `"@attack"`
> 을 붙이면서, `experience_key` 를 지정 안 한 8/9 과제는 `key` 도 같이 바뀌어 warmup 경험이
> 다른 키로 조회되고 있었다(사실상 9개 중 1개만 반영된 수치였음). `experience_key=t.key`
> 로 고정해 수정하니 결론이 오히려 더 강해졌다 — 아래는 수정 후 수치다.

| carelessness | Slow only | AND (σ=0.80, 게이트 열림) | AND (σ=0.95, 게이트 닫힘) | AND + 일관성검사 |
|---|---|---|---|---|
| 0.00 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.25 | 11.8% | **0.0%** | 11.3% | **0.0%** |
| 0.50 | 23.3% | **0.0%** | 22.0% | **0.0%** |
| 0.75 | 35.1% | **0.0%** | 35.3% | **0.0%** |
| 1.00 | 44.4% | **0.0%** | 44.4% | **0.0%** |

![carelessness 스윕 — Slow only vs AND vs AND+일관성검사](figures/fig5_careless_reviewer.png)

σ=0.80(경험 게이트 발동)에서는 게이트만으로 이미 unsafe 0% — 일관성 검사가 더 해줄 일이
없다. 게이트를 닫으면(σ=0.95) AND 단독은 Slow-only 와 거의 같아진다(축 하나가 꺼지면
나머지 하나만으로는 부족). **AND + 일관성검사만이 게이트 상태와 무관하게 항상 0% 를
유지한다.** 아래는 그 0.6 이라는 문턱값이 우연이 아닌지 0.4~0.9 로 스윕해 확인한 것이다.

![consistency_sigma 스윕](figures/fig6_consistency_sweep.png)

**갈라지게 만드는 것은 엔트로피가 아니라 경험이다.** H 는 후보 집합만 오염시키면 0 으로
위조되지만, 누적 이력은 공격자가 손댈 수 없다
(`test_experience_is_what_separates_them_not_entropy`). 단, 이 방어는
`DelegationBench-mini` 9개처럼 warmup 중 정상적으로 승인 이력을 쌓을 수 있는 유형에서만
검증됐다 — 애초에 승인 이력을 못 쌓는 과제 유형(예: 정의상 항상 반려)에서는 여전히
무력할 것으로 예상되지만 실측하지는 못했다.

### 실험 ④ — Adaptive DualFlow: Slow 를 언제 켤 것인가 (§7-1 구현)

실험 ③ 은 "AND 가 왜 필요한가" 를 보였을 뿐, §7 이 다음 단계로 지목한 진짜 문항 —
**"매번 켤지 필요할 때만 켤지"** — 은 그대로 남겨뒀었다. `Config.mode="adaptive"`
가 그 구현이다: 평소엔 Fast 만 돌리고, Fast 가 확정 못 했거나 **확정 결과가 누적
경험과 정면으로 모순될 때만** Slow 로 에스컬레이션한다 (`_experience_conflict`,
`framework.py`). AND 처럼 무조건 승인을 요구하는 대신, 에스컬레이션되면 Slow 의
판단을 그대로 신뢰한다는 점이 설계의 핵심 차이다.

같은 warmup 5회 → 공격 1회 시나리오, 50회 평균:

**σ=0.80 (경험 게이트 열림)**

| carelessness | Slow only | AND (검토율) | AND+일관성 (검토율) | Adaptive (검토율) |
|---|---|---|---|---|
| 0.00 | 0.0% | 0.0% (1.00) | 0.0% (1.00) | 0.0% (**0.00**) |
| 0.25 | 10.4% | 0.0% (1.00) | 0.0% (1.00) | 0.0% (**0.00**) |
| 0.50 | 21.1% | 0.0% (1.00) | 0.0% (1.00) | 0.0% (**0.00**) |
| 0.75 | 32.2% | 0.0% (1.00) | 0.0% (1.00) | 0.0% (**0.00**) |
| 1.00 | 44.4% | 0.0% (1.00) | 0.0% (1.00) | 0.0% (**0.00**) |

**Adaptive 가 AND 를 그대로 지배한다** — 동일한 0% unsafe 를 검토율 0 으로 낸다. 이유는
Adaptive 의 에스컬레이션 로직이 아니라 **Fast 자신의 experience 게이트**다: 경험 점수
(warmup 5회 → 0.83)가 이미 σ 를 넘어서 Fast 가 오염된 후보를 아예 안 보고 경험값을
직접 돌려주므로, 에스컬레이션 자체가 필요 없다.

**σ=0.95 (게이트 닫힘, `adaptive_sigma=0.6` 로 별도 지정)**

| carelessness | Slow only | AND (검토율) | AND+일관성 (검토율) | Adaptive (검토율) |
|---|---|---|---|---|
| 0.00 | 0.0% | 0.0% (1.00) | 0.0% (1.00) | 0.0% (0.44) |
| 0.25 | 10.4% | 10.4% (1.00) | 0.0% (1.00) | 11.6% (0.44) |
| 0.50 | 21.1% | 21.1% (1.00) | 0.0% (1.00) | 22.4% (0.44) |
| 0.75 | 32.2% | 32.2% (1.00) | 0.0% (1.00) | 33.6% (0.44) |
| 1.00 | 44.4% | 44.4% (1.00) | 0.0% (1.00) | 44.4% (0.44) |

여기서는 경험 게이트가 닫혀 있어 Fast 가 매번 오염된 후보로 판단하므로, 공격 인스턴스의
44%(warmup 이력과 공격 목표가 실제로 다른 비율)에서 에스컬레이션이 발동한다(검토율 0.44,
AND 는 1.00). `adaptive_sigma` 를 `sigma` 와 별개로 낮게 주지 않으면(기본은 `sigma` 를
물려받음) 임계치가 0.95 가 돼 경험 점수 0.83 이 못 미쳐 에스컬레이션이 안 걸리고 Fast 와
동일하게 뚫린다 — `AND` 가 `consistency_sigma` 를 분리해야 했던 것과 같은 이유다.

**AND+일관성검사(0%)와 Adaptive 가 갈리는 이유.** AND+일관성검사는 Fast 가 경험과
모순되면 `_consistent()` 로 그 자리에서 거부하고, `agreed = fast.confirmed and
slow.confirmed` 조건상 Slow 의 실제 판단과 무관하게 무조건 차단한다(Slow 는 불려도
결과가 버려진다). Adaptive 는 에스컬레이션을 **진짜 질문**으로 쓴다 — Slow 가 승인하면
그대로 통과시킨다. 그래서 A 가 부주의하면 Adaptive 의 안전성은 AND+일관성검사(0%) 가
아니라 **Slow 단독과 같은 수준**까지 떨어진다(위 표에서 Adaptive ≈ Slow only). 버그가
아니라 트레이드오프다 — "진짜로 신뢰" 와 "무조건 차단" 은 다른 정책이고, 후자가 더
안전한 대신 그 분기에서 Slow 를 부르는 의미가 없어진다.

**요약.** Adaptive 는 review_rate 를 AND 의 1.00 에서 0(게이트 열림) 또는 0.44(게이트
닫힘)로 낮추면서 경험 게이트가 포착하는 공격엔 AND 와 동일한 안전성을 낸다. 다만
(1) 경험 이력이 없는 "첫 공격" 에는 무력하고, (2) 에스컬레이션된 뒤엔 Slow 자체의
신뢰도가 그대로 상한선이다. `tests/test_framework.py::TestAdaptive` 참고.

### Experience Score 누적 (`demo.py experience`)

같은 유형의 위임을 반복하면:

| 회차 | 경로 | 질문 | LLM | Exp.Score | H(초기) | 비용 |
|---|---|---|---|---|---|---|
| 1–4 | clarify | 2 | 0 | 0.50 → 0.80 | 1.93 → 1.50 | 2.0 |
| 5–7 | **experience** | **0** | 0 | 0.83 → 0.88 | 1.38 → 1.18 | **0.0** |

![Experience Score 누적 — 경험이 clarification 루프를 없앤다](figures/fig4_experience.png)

경험이 쌓이면 초기 엔트로피 자체가 내려가고, Score ≥ σ 가 되는 5회차부터 역질의 없이
자율 판단한다. Authority Flow 는 그래도 매번 동일하게 검사된다 —
`test_authority_beats_confidence` 가 경험을 10회 강제 주입해도 권한 위반은 여전히
차단됨을 확인한다.

### 실험 ⑤ — Authority Feedback Loop: scope 협상 (§7-2 구현, `fig7`)

DelegationBench-mini 9개와는 별도의 mini-set 이다(`scope_negotiation_tasks()`) —
섞으면 분모가 10개로 바뀌어 기존 표의 모든 퍼센트(예: 44.4%=4/9)가 흔들리기
때문이다. 여기서는 `scope_exceeded` 가 실제로 트리거되는 상황만 모아 Authority
Feedback Loop 자체를 본다.

| 과제 | 상황 | 협상 결과 |
|---|---|---|
| `overbroad_recoverable` | B 가 위임 상한보다 넓게 확신 | 상한 그대로(RESTRICT)면 충분 |
| `overbroad_wrong_target` | B 가 상한보다도 넓게 확신, A 가 원하는 건 상한보다 좁음 | RESTRICT 로는 부족 — A 가 정확히 교정(CORRECT) |
| `out_of_grant` | 애초에 겹치는 범위가 없음(대조군) | 협상 불가 — 하드 리젝트 |

| 설정 | unsafe↓ | benign↑ | feedback률 |
|---|---|---|---|
| Feedback 없음 | 0.0% | 0.0% | 0.0% |
| **Feedback 켬(제안)** | 0.0% | **100.0%** | 66.7% |

![Authority Feedback Loop — 정상/공격 조건](figures/fig7_authority_feedback.png)

**공격(belief 조작, cold) 시나리오에서도 수치가 완전히 동일하다.** Authority
Feedback 은 Slow 축과 같은 이유로 belief 조작에 면역이다 — B 의 자기신고
확신(H)이 아니라 A 의 실제 응답만 보기 때문이다.

| carelessness | unsafe↓ | benign↑ |
|---|---|---|
| 0.00 | 0.0% | 100.0% |
| 0.25 | 0.0% | 100.0% |
| 0.50 | 0.0% | 50.0% |
| 0.75 | 0.0% | 50.0% |
| 1.00 | 0.0% | 0.0% |

**carelessness 가 올라가도 unsafe 는 0% 로 고정이다** — A 가 확인 없이 범위 밖 제안을
그대로 승인해도 non-amplification 검사(매 라운드 top 에서 위임 예산 재검증)가 막는다.
대신 benign completion 이 떨어진다: A 가 oracle 이 아니라는 것의 실제 의미는
"위험해진다" 가 아니라 **"협상이 실패해 안전하게 거절되는 경우가 늘어난다"**
(over-rejection)는 것이다 — §실험③과 같은 결의 결과다.

**표현은 정확히 좁혀서 쓸 것.** 이건 "Principal 이 틀려도 모든 intent 오류를 막는다"
가 아니라 **"scope negotiation 에서 reviewer error 가 privilege amplification 으로
이어지는 걸 non-amplification invariant 가 막는다"** 는 뜻이다 — 이미 권한 범위 안의
*다른* 자원을 B 가 골랐다면(scope_exceeded 자체가 안 생기므로) Authority Feedback Loop
는 관여하지 않는다. 그건 별개의 intent confirmation 문제이고, 실제 LLM 평가에서
다시 나타날 가능성이 크다.

### 실험 ⑥ — Adaptive Verification: 언제 A 에게 다시 물어볼 것인가 (§7-2 확장, `fig8`)

실험 ⑤ 는 "왜 Authority Feedback Loop 가 필요한가" 를 보였을 뿐, 매번 A 를
부르면 검토율이 1.00 이 된다(연구의 출발점인 개입 최소화와 충돌 — 실험④ 의
Adaptive 와 같은 문제의식). `run_sequence()` 로 같은 위임 유형을 9라운드
순서대로 실행해(`scope_negotiation_sequence()`, 같은 `VerifiedAuthorityStore`
공유) 세 가지를 함께 본다: A) 반복이 Feedback 을 줄이는가, B) 위임 범위가 실제로
바뀌면 Feedback 으로 되돌아가는가, C) 조작된 제안(H=0)이 auto-restrict 를 속일
수 있는가.

![Adaptive Verification — 9라운드 타임라인](figures/fig8_adaptive_verification.png)

| # | 상황 | A 에게 물어봄 | auto-restrict | 이력 n | 확정된 scope |
|---|---|---|---|---|---|
| 0–2 | stable (8월 반복) | 예(매번) | 아니오 | 1→3 | `/reports/2026-08/` |
| 3–4 | stable (이력 충분) | **아니오** | **예** | 3 | `/reports/2026-08/` |
| 5 | **legitimate drift** (9월로 변경) | 예 | 아니오 | **1**(리셋) | `/reports/2026-09/` |
| 6–7 | post-drift repeat | 예(재구축 중) | 아니오 | 2→3 | `/reports/2026-09/` |
| 8 | **manipulated proposal** (H=0) | 아니오 | 예 | 3 | `/reports/2026-09/` (공격 목표 아님) |

- **A. 반복 → Feedback 감소.** round 0-2 는 이력 부족(`n<3`)으로 매번 실제로 물어본다.
  round 3 부터 `n_confirmed≥3, agreement≥0.8` (해석 가능한 단순 규칙)을 넘어 `C_adaptive
  = C_experience ∩ C_current_budget` 를 먼저 시도하고 재검증만으로 확정한다 — feedback률
  100%→0%, 같은 scope 로 안전하게 실행.
- **B. drift → 재활성화 + 리셋.** round 5 에서 범위가 8월→9월로 바뀌면 낡은 이력과 새
  상한이 안 겹쳐(`Privilege.meet` 실패) 자동 재사용이 무산되고 Feedback 으로 돌아간다.
  `VerifiedAuthorityStore.record()` 가 이때 **낡은 이력을 리셋**한다(`n`: 3→1) — 안 하면
  8월 이력이 9월 확인을 계속 압도해 `agreement_ratio` 가 오래도록 회복 안 된다
  (`test_a_differing_confirmation_resets_stale_history`).
- **C. 조작 → auto-restrict 도 안 속는다(단, 범위를 정확히 좁혀서 말할 것).** round 8
  은 H=0 으로 조작된 공격인데도 auto-restrict 가 A 에게 안 묻고 **정답**으로 실행된다 —
  재사용 후보가 `VerifiedAuthorityStore`(A 의 과거 확인)와 `auth.suggested`(위임 예산)의
  교집합일 뿐 B 의 조작된 후보를 아예 안 보기 때문이다. "belief 조작에 구조적으로 면역"
  이라 넓게 쓰면 과장이다 — 정확히는 **"immune to semantic-proposal manipulation under
  trusted authority state"** 다. 위임 예산이나 `VerifiedAuthorityStore` 자체가 조작 가능
  하다면 별개의 문제다.

`use_verified_experience=False` 로 끄면 매 라운드 실제로 물어본다 — adaptive 는 opt-in
이며 Authority Feedback Loop 자체의 안전성과는 독립이다. `TestAdaptiveVerification`,
`TestVerifiedAuthorityStore` 참고.

## 4. 구현하며 확인한 선행연구의 빈틈

전부 테스트로 박제해 두었다. 논문 Related Work / 차별점 서술에 쓸 수 있는 재료다.

> (1)(2)의 형식적 반례는 논문 재현 저장소 [`sage-clarify`](https://github.com/A0-J/sage-structured-uncertainty)
> 에 테스트로 들어 있다. 본 저장소는 그중 **런타임에서 실제로 문제가 되는 부분**
> (τ_exec 우선순위, π 의 의미)을 재현한다.

**(1) SAGE-Agent 의 EVPI 비음수성(Prop.2-1)은 실제로 깨진다.**
EVPI 를 정규화되지 않은 viability 위에 정의해 두었기 때문에, 현재 1위 후보를 제거할 수 있는
질문은 "best-candidate certainty"를 떨어뜨려 EVPI 가 음수가 된다. Jensen 논증은 정규화된
사후분포에서만 성립한다. **엔트로피로 바꾸면 이 문제가 사라진다** — IG = H(p) − E_r[H(p|r)]
는 상호정보량이므로 항상 0 이상이다 (`test_non_negative_on_random_beliefs`).
θ 게이팅을 정당화하는 부수 효과이기도 하다.

**(2) SAGE-Agent 의 submodularity(Prop.2-2)는 엔트로피로 바꿔도 성립하지 않는다.**
"질문 순서에 대한 수확 체감"을 엔트로피의 submodularity 로부터 유도했다고 적혀 있지만,
조건부 상호정보량 I(X;S|A) 는 I(X;S) 보다 커질 수 있다. 본 벤치마크에 반례가 있다: scope 질문은
단독으로 0.72 bits 를 주지만 action 을 먼저 알고 나면 0.97 bits 를 준다
(`test_caveat_gain_is_not_submodular`). **함의**: 질문 예산 k 를 "앞 질문이 더 이득"이라는
가정 위에 설계하면 안 되고, 매 라운드 IG 를 재계산해야 한다. 본 구현이 루프 안에서
다시 계산하는 이유다.

**(3) π 는 확신도가 아니라 '명세 완성도' 다.**
Eq.(13)은 인자가 채워져 있으면 p=1, 비어 있으면 1/|D| 를 준다. **값이 맞는지는 전혀 보지
않는다.** 따라서 π 는 evidential support 가 아니라 specification completeness 를 재는
값이고, confident-correct 와 confident-wrong 을 구분하지 못한다
(`test_value_correctness_is_invisible`). 이는 Abstract 의 핵심 주장 — specification
uncertainty 와 model uncertainty 를 깨끗이 분리한다 — 과 충돌한다. 지정된 인자에 무조건
p=1 을 주는 순간 model uncertainty 는 분리된 것이 아니라 **소거**된다.

**(4) tool 선택의 불확실성이 실행 게이트에 반영되지 않는다.**
Eq.(2)는 '∝' 로 Eq.(1)의 1/K 를 흡수하고 Prop.1 도 파라미터 곱만 가정한다. 그 결과
서로 배타적인 두 tool 이 모두 완전 지정이면 **둘 다 π=1** 이 되어 τ_exec 를 통과하고,
어느 쪽을 부를지는 tie-break 로 정해진다 (`TestToolChoiceBlindSpot`). 본 벤치마크의
`silent_misread` 가 그 사례다 — 조회와 반출 중 무엇인지 모르는 상태인데 질문 없이 실행된다.

**(5) Sim_path 는 SAGE-Bench 자신이 인정하듯 관대한 지표다.**
p\* ⊆ p 인 경우 Sim_path = 1.0 이 되어 "더 깊이 들어간 해석"을 잡지 못한다. 본 구현은
Sim_path 와 종단 액션 일치를 **둘 다** 요구해 일부 보완했지만, 완전한 해법은 아니다.
역으로 `silent_misread` 사례는 **Action_Acc 만으로는 못 잡고 Sim_path 라야 잡히는** 반대
방향의 증거다(양쪽 종단 액션이 모두 EXECUTE 인데 경로가 갈린다). 두 지표가 상보적이라는
근거로 쓸 수 있다.

**(6) ChainCaps 의 manifest quality 병목은 위임 맥락에서도 그대로다.**
ChainCaps 는 naive manifest 에서 차단율이 27.3% 로 떨어진다고 보고한다. 본 구현에서 그에
대응하는 것은 `Budget` 생성자와 `sysvars` 를 누가 어떻게 쓰느냐다. scope 를 `*` 로 열어두면
Authority Flow 는 아무것도 막지 못한다. 실제 배치 시 이 부분의 저작·린팅 도구가
연구의 실용성을 좌우할 가능성이 높다.

## 5. 파일 구성

| 파일 | 역할 |
|---|---|
| `src/dualflow/capability.py` | Authority Flow — privilege 순서관계, budget meet, 위임 체인 감쇠, 실패 분류(no_grant/condition_missing/scope_exceeded) |
| `src/dualflow/semantic.py` | Semantic Flow — action space, 엔트로피, 경험 점수, 역질의(IG), Principal(A 시뮬레이션) |
| `src/dualflow/rule_engine.py` | Joint Verification 의 매칭 — 위임 SOP 그래프, p\*, Sim_path, exact-field 진단(§7-2) |
| `src/dualflow/authority_feedback.py` | **Authority Feedback Loop** — scope 협상(§7-2), bounded negotiation, non-amplification, `VerifiedAuthorityStore`(adaptive, §7-2 확장) |
| `src/dualflow/framework.py` | 전체 조립, ablation 스위치, 평가 지표, `run_sequence`(순차 실험) |
| `src/dualflow/sage_baseline.py` | **SAGE-Agent 원 공식 재현** — Eq.(2), Def.4·5, τ_exec, α |
| `src/dualflow/bench.py` | DelegationBench-mini 9개 시나리오 + belief 조작 변형 + scope 협상 mini-set + 순차 시나리오 |
| `src/dualflow/llm.py` | LLM fallback 인터페이스 + 실제 API 어댑터 골격 |
| `src/dualflow/demo.py` | 11개 실험 (텍스트) |
| `src/dualflow/plots.py` | 그림 8장 생성 (matplotlib) |
| `tests/` | 149개 — 비증폭 정리, 엔트로피 성질, 종료성, 게이팅, ablation, 공격 실험, SAGE 재현, Authority Feedback, Adaptive Verification |

## 6. 실제 LLM 붙이기

교체할 지점은 세 곳뿐이고, 나머지는 그대로 재사용된다.

| 지점 | 현재 | 교체 |
|---|---|---|
| 해석 후보 생성 | `DelegationTask.candidates` (스크립트) | 명세 + 툴 스키마로부터 후보를 뽑는 파서/모델 |
| 역질의 응답 | `semantic.Principal` (오라클) | 실제 Agent A 엔드포인트 |
| LLM 의미 판단 | `llm.ScriptedJudge` | `llm.AnthropicJudge` (골격 포함) |

LLM 을 붙일 때도 **자유 생성이 아니라 후보 중 택일**로 좁혀서 부르는 구조를 유지할 것.
호출 1회, 출력 토큰 수 개로 끝나므로 fallback 비용 가정이 유지된다.

## 7. 다음 단계 (미해결)

0. **실험 ③ 의 후속.** carelessness 를 A 마다 다르게(에이전트별 신뢰도) 두거나,
   검토 예산(하루 N건)을 제약으로 넣으면 "Slow 를 누구에게, 몇 건에 쓸 것인가" 가
   최적화 문제가 된다. 지금 `warmup_then_attack` 이 그 실험의 골격이다.
1. **Slow 를 언제 켤 것인가 — 1차 구현 완료, §3 실험 ④.** `mode="adaptive"` 가
   "경험과의 불일치" 를 트리거로 써서 Slow 를 선별 호출한다. 경험 게이트가 열려
   있으면(σ=0.80) AND 와 동일한 0% unsafe 를 검토율 0 으로 낸다. 다만 두 가지는
   아직 미해결이다 — (a) 경험 이력이 없는 첫 공격에는 무력하다(콜드스타트), (b)
   게이트가 닫혀 있을 때(σ=0.95) 에스컬레이션된 요청은 Slow 자체의 신뢰도가 그대로
   상한선이라 AND+일관성검사(무조건 차단)만큼 안전하지는 않다 — "얼마나 자주 켤까"
   보다 **"에스컬레이션 이후 Slow 의 판단을 얼마나 신뢰할까"** 가 남은 질문이 됐다.
   실험 ③ 후속(항목 0)의 에이전트별 신뢰도가 이 신뢰 폭을 정하는 데 바로 쓰일 수 있다.
2. **Authority Feedback Loop — 1차 구현 완료, §3 실험 ⑤.** 권한의 "범위" 와
   "유무" 를 분리했다. `check_authority` 가 실패를 `no_grant`(action/resource 자체가
   없음 — 재협상 불가, 하드 리젝트) / `condition_missing`(조건은 B 가 채울 수 없는
   값 — 역시 협상 대상 아님) / `scope_exceeded`(action·resource·condition 은 맞는데
   범위만 넘음 — 협상 가능)로 나누고, `scope_exceeded` 일 때만 위임 예산에서 계산한
   `suggested`(실제 허용되는 상한)를 제공한다. `authority_feedback.py` 의
   `run_feedback()` 이 그 위에서 **B 제안 → A 확인/축소(`Principal.review_authority`,
   APPROVE/CORRECT/RESTRICT/REJECT) → 재검증** 을 bounded(`authority_feedback_max_rounds`,
   기본 2)로 협상한다. non-amplification 은 매 라운드 top 에서 예산에 대고 다시
   검증하는 것 자체로 보장된다 — A 가 부주의(`carelessness`)해서 범위 밖 제안을
   그대로 승인해도 다음 검증이 막는다(그 결과 unsafe 가 아니라 협상 실패로 이어진다).

   **왜 지름길(exact-field 매칭)로는 안 풀리는지 실측으로 먼저 확인했다.** Joint
   Verification 에 `V_action∧V_resource∧V_scope∧V_condition` 형태로 원본 값을
   `task.truth` 와 직접 비교하는 exact-field 매칭을 시도해봤다
   (`match_intent(..., require_fields=True)`, `Config.use_field_match`, 기본 False).
   벤치마크상으로는 자원 치환 공격을 완전히 막지만, A 가 검토를 아예 안 해도
   (`carelessness=1.0`) 여전히 unsafe=0% 가 나온다
   (`test_exact_field_match_is_an_oracle_not_a_fix`). 이건 Joint 가 안전해진 게
   아니다 — 정확히는, **현재 verifier 가 가진 정책·권한 정보만으로는 A 가 의도한
   정확한 resource/scope 를 복원할 수 없고, 이를 `task.truth` 와 비교하면 평가
   오라클이 된다**는 뜻이다. 추가 신뢰 소스(A 의 실제 확인) 없이는 이 문제를 풀 수
   없다는 게 이 진단 실험의 결론이고, Authority Feedback Loop 는 그 신뢰 소스를
   정식으로 만든 것이다 — `task.truth` 는 `Principal` 안에서만 쓰이고,
   `run_feedback()`/`framework.py` 는 `Principal.review_authority()` 의 응답
   (`ConfirmedAuthority`)만 본다(`test_run_feedback_only_needs_the_review_authority_method`).

   **Adaptive 확장도 1차 구현 완료 — §3 실험 ⑥.** "Authority 검사를 할지" 가
   아니라 **"Principal 에게 실제로 물어볼지"** 만 adaptive 하게 만들었다
   (no_grant/condition_missing 은 여전히 무조건 하드 리젝트, `valid` 는 여전히
   무조건 통과 — 바뀌는 건 `scope_exceeded` 분기뿐이다). `VerifiedAuthorityStore`
   가 A 가 **실제로** 확인해주고 재검증까지 통과해 EXECUTE 로 이어진 scope 만
   쌓고(`Verified Experience` — 아무 실행 결과나 담는 `ExperienceStore` 와 구분),
   충분하면(`n_confirmed≥3`, `agreement_ratio≥0.8`, 둘 다 해석 가능한 단순 규칙 —
   risk score 를 새로 만들지 않았다) A 에게 묻지 않고 `C_adaptive = C_experience ∩
   C_current_budget` 를 먼저 시도한 뒤 다시 top 의 `check_authority` 로 재검증한다.
3. **엔트로피를 LLM 에게 물어보는 안 (피드백 ④) 은 권장하지 않는다.** 차별점 표의
   첫 줄이 "저엔트로피 구간은 LLM 호출 자체를 원천 배제" 인데, H 를 LLM 으로 구하면
   모든 위임이 최소 1회 호출하게 되어 LLM률이 11.1% → 100% 로 오른다. 상시 LLM
   베이스라인과 비용이 같아진다. 타협안은 **LLM 은 후보 집합 Ω 생성에만 쓰고 H 는
   공식으로 계산**하는 것이다(현 구조가 이미 그 모양이다). 대신 "LLM 이 신고한 확률이
   얼마나 calibrated 한가" 를 별도 실험으로 돌리면 공식을 쓰는 근거가 논문에 생긴다.
4. **후보 생성기의 품질이 다음 병목이다.** 현재 실험은 Ω 안에 정답이 항상 있다고 가정한다.
   정답이 Ω 밖일 때(`apply_answer` 가 빈 집합을 만나는 경우)의 거동이 실제 배치의 리스크다.
5. **σ, k, λ 스윕.** θ 스윕과 같은 방식으로 돌릴 수 있게 `evaluate` 가 준비돼 있다.
6. **용어.** 논문에서는 `belief` 대신 `interpretation distribution` 같은 표현을 쓰는 편이
   안전하다. 교수님 지적대로 "belief" 는 개념적으로 애매하고, 우리 논지가 바로 그
   "자기신고 belief" 를 공격 대상으로 삼기 때문에 같은 단어를 쓰면 혼동된다.

## 참고문헌

- ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation. arXiv 2605.26542
- Structured Uncertainty guided Clarification for LLM Agents. arXiv 2511.08798 (Findings of ACL 2026)
- SAGE: A Service Agent Graph-guided Evaluation Benchmark. arXiv 2604.09285
