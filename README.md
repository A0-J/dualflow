# dualflow — Agent-to-Agent 권한 위임 검증 프레임워크

**Authority Flow × Semantic Flow → Joint Verification** 아키텍처의 실행 가능한 구현.

2026-09-02 미팅자료의 플로우 다이어그램을 그대로 코드로 옮기고, 세 편의 선행연구에서
가져온 부분이 실제로 의도한 대로 작동하는지 실험으로 확인할 수 있게 만들었다.
LLM 호출 없이 결정론적으로 돌아가며, 실제 모델은 인터페이스 하나만 맞추면 붙는다.

```bash
pip install -e ".[dev]"

dualflow-demo                    # 6개 실험 전체
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

**(3) Sim_path 는 SAGE-Bench 자신이 인정하듯 관대한 지표다.**
p\* ⊆ p 인 경우 Sim_path = 1.0 이 되어 "더 깊이 들어간 해석"을 잡지 못한다. 본 구현은
Sim_path 와 종단 액션 일치를 **둘 다** 요구해 일부 보완했지만, 완전한 해법은 아니다.
역으로 `silent_misread` 사례는 **Action_Acc 만으로는 못 잡고 Sim_path 라야 잡히는** 반대
방향의 증거다(양쪽 종단 액션이 모두 EXECUTE 인데 경로가 갈린다). 두 지표가 상보적이라는
근거로 쓸 수 있다.

**(4) ChainCaps 의 manifest quality 병목은 위임 맥락에서도 그대로다.**
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
| `src/dualflow/bench.py` | DelegationBench-mini 9개 시나리오 |
| `src/dualflow/llm.py` | LLM fallback 인터페이스 + 실제 API 어댑터 골격 |
| `src/dualflow/demo.py` | 6개 실험 |
| `tests/` | 80개 — 비증폭 정리, 엔트로피 성질, 종료성, 게이팅, ablation |

## 6. 실제 LLM 붙이기

교체할 지점은 세 곳뿐이고, 나머지는 그대로 재사용된다.

| 지점 | 현재 | 교체 |
|---|---|---|
| 해석 후보 생성 | `DelegationTask.candidates` (스크립트) | 명세 + 툴 스키마로부터 후보를 뽑는 파서/모델 |
| 역질의 응답 | `semantic.Principal` (오라클) | 실제 Agent A 엔드포인트 |
| LLM 의미 판단 | `llm.ScriptedJudge` | `llm.AnthropicJudge` (골격 포함) |

LLM 을 붙일 때도 **자유 생성이 아니라 후보 중 택일**로 좁혀서 부르는 구조를 유지할 것.
호출 1회, 출력 토큰 수 개로 끝나므로 fallback 비용 가정이 유지된다.

## 7. 다음 단계로 제안

1. **후보 생성기의 품질이 다음 병목이다.** 현재 실험은 후보 집합에 정답이 항상 포함돼 있다고
   가정한다. 정답이 Ω 밖에 있을 때(`apply_answer` 가 빈 집합을 만나는 경우) 무슨 일이
   벌어지는지가 실제 배치의 핵심 리스크다.
2. **over-rejection 을 줄이는 경로.** 매칭 실패를 즉시 Reject 하지 말고 "매칭 실패 → 역질의
   1회 추가" 로 되돌리면 `silent_misread` 를 살릴 수 있다. 지금 구조에서 몇 줄이면 된다.
3. **σ, k, λ 스윕.** θ 스윕과 같은 방식으로 돌릴 수 있게 `evaluate` 가 준비돼 있다.
4. **Ω 밖 정답, 악의적 Agent A** 등 위협모델 확장. ChainCaps 도 manifest 를 신뢰한다고
   범위를 명시했으니, 본 연구도 같은 방식으로 scope 문장을 써두는 편이 안전하다.

## 참고문헌

- ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation. arXiv 2605.26542
- Structured Uncertainty guided Clarification for LLM Agents. arXiv 2511.08798 (Findings of ACL 2026)
- SAGE: A Service Agent Graph-guided Evaluation Benchmark. arXiv 2604.09285
