# 베이스라인 재현과 비판

`sage_baseline.py` 는 SAGE-Agent (Structured Uncertainty guided Clarification for
LLM Agents, arXiv 2511.08798) 의 Eq.(2), Def.4·5, τ_exec, α 를 원 공식 그대로
재현한 별도 구현이다. 이 저장소의 엔트로피 코드를 변형한 게 아니다 — 논문의 결함이
"우리 구현 탓" 이 아니라 원 공식 자체에서 나온다는 걸 보이기 위해서다.

구현하며 확인한 문제 6가지를 전부 테스트로 고정해 두었다. 논문 Related Work / 차별점
서술에 쓸 수 있는 재료다.

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

## 참고문헌

- ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation. arXiv 2605.26542
- Structured Uncertainty guided Clarification for LLM Agents. arXiv 2511.08798 (Findings of ACL 2026)
- SAGE: A Service Agent Graph-guided Evaluation Benchmark. arXiv 2604.09285
