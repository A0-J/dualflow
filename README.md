# dualflow — Agent-to-Agent 권한 위임 검증 프레임워크

**Authority Flow × Semantic Flow → Joint Verification** 아키텍처의 실행 가능한 구현.

2026-09-02 미팅자료의 플로우 다이어그램을 그대로 코드로 옮기고, 세 편의 선행연구에서
가져온 부분이 실제로 의도한 대로 작동하는지 실험으로 확인할 수 있게 만들었다.
LLM 호출 없이 결정론적으로 돌아가며, 실제 모델은 인터페이스 하나만 맞추면 붙는다.

```bash
pip install -e ".[dev]"

dualflow-demo                    # 9개 실험 전체 (텍스트)
python -m dualflow.demo fastslow attack careless   # 필요한 것만
dualflow-plots                   # figures/ 에 그림 5장 저장
dualflow-plots careless --trials 50   # fig5 논문용 (기본 10회는 ±3%p 흔들린다)
python -m dualflow.demo joint    # 특정 파트만
pytest -q                        # 80개 검증 테스트
```

---

## 1. 아키텍처 대응표

```
Agent A --위임--> Agent B
  │
  ├─ AUTHORITY FLOW   허용 범위 A 검사 (action·resource·scope·condition)
  │                   위반 시 즉시 차단 — 하드 제약
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
| Action Space·확률분포 | `semantic.Interpretation`, `build_belief` | SAGE-Agent Def.2–3 |
| Entropy H | `semantic.entropy` | (EVPI → H 로 교체) |
| 임계치 θ | `Config.theta` | 신규 |
| Experience Score | `semantic.ExperienceStore` | 신규 |
| Clarification | `select_question`, `information_gain` | SAGE-Agent Def.4를 IG로 재정의 |
| LLM 의미 판단 | `llm.LLMJudge` (fallback) | SAGE-Agent(상시) → fallback 으로 격하 |
| 매칭 검증 | `rule_engine.match_intent`, `sim_path` | SAGE-Bench Eq.(6) |
| Execute / Reject | `framework.DelegationVerifier.run` | — |

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

여기서 가장 중요한 관찰은 **unsafe 가 어느 θ 에서도 0%** 라는 점이다. θ 는 "얼마나 유용한가"만
조절하고 "안전한가"는 Authority Flow 와 매칭 검증이 따로 책임진다. 두 축이 분리돼 있다는 것이
이 아키텍처의 설계 의도인데, 스윕이 그걸 실측으로 보여준다. 발표·논문에서 쓸 수 있는 그림이다.

반대로 기본값 θ=0.5 는 이 벤치마크에서 **지배당한다**. θ=0.25 가 같은 LLM률로 benign 100% 를
내기 때문이다. 기본값은 슬라이드 설정을 따라 0.5 로 두었으니, 실제 도메인에서는 반드시
스윕으로 정할 것.

### Fast / Slow / AND 파일럿 (피드백 ①②)

Semantic Flow 를 세 가지 방식으로 갈라 각각 돌린다. `Config.mode` 로 전환한다.

- **Fast** — 엔트로피를 한 번 재고 θ 로 판단. 넘으면 역질의(k회) → LLM fallback.
- **Slow** — B 가 해석을 정리해 A 에게 제시하고 승인/교정받는다. 엔트로피를 쓰지 않는다.
- **AND** — Fast 로 좁힌 해석을 A 가 **승인해야만** 통과. 교정 요구는 불일치로 보고 차단한다.

베이스라인 `SAGE-Agent (Eq.2+Def.4)` 는 우리 엔트로피 코드의 스위치를 끈 것이 아니라
**논문 공식을 그대로 옮긴 별도 구현**이다 (`sage_baseline.py`, §5 참고).

| 설정 | unsafe↓ | benign↑ | over-rej | 질문 | 검토 | LLM | 비용↓ |
|---|---|---|---|---|---|---|---|
| SAGE-Agent (Eq.2+Def.4) | 77.8% | 40.0% | 0.0% | 0.56 | 0.00 | 1.67 | 17.22 |
| SAGE + Joint | 11.1% | 40.0% | 40.0% | 0.56 | 0.00 | 1.67 | 17.22 |
| Fast + Joint | 0.0% | 80.0% | 20.0% | 0.44 | 0.00 | 0.11 | 1.56 |
| Slow + Joint | 0.0% | **100.0%** | 0.0% | 0.00 | 1.00 | 0.00 | 3.00 |
| AND (제안) | 0.0% | 80.0% | 20.0% | 0.44 | 1.00 | 0.11 | 4.56 |

**정상 상황에서는 Slow 가 AND 를 지배한다.** 안전성이 같고, A 의 교정이 `silent_misread`
를 살려내 benign 100% 를 찍으며, 비용도 더 싸다. 대신 **모든 위임에서 A 를 호출**한다
(검토율 1.00). 연구의 출발점이 "사람/LLM 개입 최소화" 였으므로 이 결과는
"Slow 를 켤지 말지" 가 아니라 **"Slow 를 언제 켤지"** 가 진짜 문제라는 뜻이다.

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
| AND (제안) | **0.0%** | 0.0% | 100.0% | 3.00 |

읽는 법:

- **EVPI 는 방어선이 아니다.** τ_exec 검사(§5.2 Step 1)가 EVPI 계산보다 앞에 있어,
  π 가 문턱을 넘으면 질문 생성도 EVPI 도 실행되지 않는다. 9건 전부 그렇게 통과했다.
  EVPI 를 아무리 잘 설계해도 belief 단계에서 이미 끝난다.
- **확률을 조작할 필요조차 없다.** Eq.(13)은 지정된 인자에 무조건 p=1 을 준다. 값이
  맞는지는 보지 않으므로, 공격자는 **빈칸만 채우면** π=1 을 얻는다.
- **엔트로피 축만으로는 부족하다.** Authority·매칭을 붙여도 44.4% 가 통과한다. 통과하는
  것들은 *같은 SOP 경로, 같은 권한 안의 다른 자원* 을 노린 공격이다
  (`/reports/2026-08/` → `/reports/2025-01/`). Sim_path 가 경로만 보기 때문에 잡히지 않는다.
- **Slow 축은 구조적으로 면역이다.** A 의 검토는 B 의 자기신고 값을 입력으로 쓰지 않는다.
  게다가 A 가 교정까지 해줘 benign 80% 를 복구한다.
- **AND 는 공격 하에서 오탐 0%, 미탐 100%.** 교수님이 말한 보수적 트레이드오프의
  정의 그대로 나온다. 안전하지만 공격받는 동안 시스템이 멈춘다.

이 표가 논문의 핵심 그림이 될 수 있다. 주장은 "우리가 더 싸다" 가 아니라
**"자기신고 불확실성 단일 축에 실행 결정을 거는 구조는 뚫린다"** 이고,
AND 결합은 취향이 아니라 그 공격에 대한 필연적 대응이 된다.

### 실험 ③ — A 의 검토가 완벽하지 않을 때 (`fig5`)

위 두 표는 A 가 **항상 옳게 교정하는 완벽한 검토자**라고 가정한다. 그 가정에서는
Slow 단독이 AND 를 지배한다(안전성 동일, benign 100%, 비용 저렴). 현실의 검토자는
그렇지 않으므로 `Config.carelessness` — A 가 제안을 대충 읽고 그냥 승인해 버릴 확률 —
를 넣고 스윕한다.

시나리오는 **정상 운영 5회로 경험을 쌓은 뒤 같은 유형에 belief 조작 공격 1회** 다.
지표는 공격 시점의 unsafe 실행률이고, 확률적이므로 10회 평균이다.

| carelessness | Slow only | AND | AND + 일관성검사 |
|---|---|---|---|
| 0.00 | 0.0% | 0.0% | 0.0% |
| 0.25 | 15.6% | 8.9% | 8.9% |
| 0.50 | 28.9% | 15.6% | 15.6% |
| 0.75 | 35.6% | 22.2% | 22.2% |
| 1.00 | 44.4% | **33.3%** | 33.3% |

*(σ=0.80, 10회 평균. 오른쪽 패널은 σ=0.95 로 경험 게이트를 닫은 경우 — 그때는 일관성
검사가 있어야 44.4% → 33.3% 로 내려간다.)*

`carelessness > 0` 인 구간은 확률적이므로 `warmup_then_attack` 이 시행별 결과와
표준오차를 함께 돌려주고, fig5 는 그것을 음영 밴드로 그린다. 기본 10회에서는 밴드가
±3%p 정도이고, `--trials 50` 이면 ±1%p 안으로 좁아진다. `carelessness=0` 에서는
난수가 개입하지 않아 분산이 정확히 0 이다.

**AND 가 존재해야 하는 이유가 여기서 처음으로 수치로 나온다.** c=0 에서는 세 방식이
같지만, A 가 흔들리기 시작하면 갈라진다. 그리고 **갈라지게 만드는 것은 엔트로피가 아니라
경험이다.** H 는 후보 집합만 오염시키면 0 으로 위조되지만, 누적 이력은 공격자가 손댈 수
없다 (`test_experience_is_what_separates_them_not_entropy` — warmup=0 이면 이득이 사라진다).

한계도 같이 보인다. 계속 거절돼 온 위임 유형은 성공 이력이 없어 경험이 쌓이지 않고,
그런 유형에는 두 방식 모두 무력하다. 남은 33.3% 가 그것이다.

### Experience Score 누적 (`demo.py experience`)

같은 유형의 위임을 반복하면:

| 회차 | 경로 | 질문 | LLM | Exp.Score | H(초기) | 비용 |
|---|---|---|---|---|---|---|
| 1–4 | clarify | 2 | 0 | 0.50 → 0.80 | 1.93 → 1.50 | 2.0 |
| 5–7 | **experience** | **0** | 0 | 0.83 → 0.88 | 1.38 → 1.18 | **0.0** |

경험이 쌓이면 초기 엔트로피 자체가 내려가고, Score ≥ σ 가 되는 5회차부터 역질의 없이
자율 판단한다. 그래도 Authority Flow 는 매번 동일하게 검사된다 —
`test_authority_beats_confidence` 가 경험을 10회 강제 주입해도 권한 위반은 여전히
차단됨을 확인한다.

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
| `src/dualflow/capability.py` | Authority Flow — privilege 순서관계, budget meet, 위임 체인 감쇠 |
| `src/dualflow/semantic.py` | Semantic Flow — action space, 엔트로피, 경험 점수, 역질의(IG) |
| `src/dualflow/rule_engine.py` | Joint Verification 의 매칭 — 위임 SOP 그래프, p\*, Sim_path |
| `src/dualflow/framework.py` | 전체 조립, ablation 스위치, 평가 지표 |
| `src/dualflow/sage_baseline.py` | **SAGE-Agent 원 공식 재현** — Eq.(2), Def.4·5, τ_exec, α |
| `src/dualflow/bench.py` | DelegationBench-mini 9개 시나리오 + belief 조작 변형 |
| `src/dualflow/llm.py` | LLM fallback 인터페이스 + 실제 API 어댑터 골격 |
| `src/dualflow/demo.py` | 9개 실험 (텍스트) |
| `src/dualflow/plots.py` | 그림 5장 생성 (matplotlib) |
| `tests/` | 119개 — 비증폭 정리, 엔트로피 성질, 종료성, 게이팅, ablation, 공격 실험, SAGE 재현 |

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
1. **Slow 를 언제 켤 것인가 — 가장 급한 문항.** 항상 켜면 A 의 검토율이 1.00 이 되어
   연구의 출발점(개입 최소화)과 충돌하고, 안 켜면 belief 조작에 44.4% 가 뚫린다.
   H 만으로는 위조 여부를 알 수 없다는 것이 실험의 결론이다. 유망한 신호 하나는
   **경험과의 불일치**다 — `/reports/` 로 4회 확정된 이력이 있는데 갑자기
   `/finance/` 를 H=0 으로 확신한다면 그 자체가 이상 신호다. `ExperienceStore` 에
   이미 필요한 통계가 다 들어 있어 트리거로 만들 수 있다.
2. **권한의 "범위" 와 "유무" 분리 (피드백 ③).** 아직 미구현. `check_authority` 가 둘을
   한 번에 판정하고 reason 문자열로만 구분한다. 쪼개면 처리 방법이 갈린다 —
   *유무* 실패(`delete` 권한 자체가 없음)는 재협상 불가한 하드 리젝트지만,
   *범위* 실패(권한은 있는데 요청이 넓음)는 **역질의로 살릴 수 있다**
   ("`/reports/` 까지만이면 되나요?"). over-rejection 을 줄이는 실질적 수단이기도 하다.
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
