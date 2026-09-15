# 설계 노트

시행착오와 구현 결정의 이유를 남겨둔 문서. [README](../README.md) 의 Current Scope
and Limitations / Roadmap 은 이 문서를 요약한 것이다.

## 지름길(exact-field 매칭)로는 안 풀리는 이유 — Authority Feedback Loop 의 동기

Joint Verification 에 `V_action∧V_resource∧V_scope∧V_condition` 형태로 원본 값을
`task.truth` 와 직접 비교하는 exact-field 매칭을 시도해봤다
(`match_intent(..., require_fields=True)`, `Config.use_field_match`, 기본 False).

벤치마크상으로는 자원 치환 공격을 완전히 막지만, A 가 검토를 아예 안 해도
(`carelessness=1.0`) 여전히 unsafe=0% 가 나온다
(`test_exact_field_match_is_an_oracle_not_a_fix`). 이건 Joint 가 안전해진 게
아니다 — 정확히는, **현재 verifier 가 가진 정책·권한 정보만으로는 A 가 의도한
정확한 resource/scope 를 복원할 수 없고, 이를 `task.truth` 와 비교하면 평가
오라클이 된다**는 뜻이다.

추가 신뢰 소스(A 의 실제 확인) 없이는 이 문제를 풀 수 없다는 게 이 진단 실험의
결론이고, Authority Feedback Loop 는 그 신뢰 소스를 정식으로 만든 것이다 —
`task.truth` 는 `Principal` 안에서만 쓰이고, `run_feedback()`/`framework.py` 는
`Principal.review_authority()` 의 응답(`ConfirmedAuthority`)만 본다
(`test_run_feedback_only_needs_the_review_authority_method`).

이 스위치는 라이브 파이프라인 기본값을 꺼두고(`use_field_match=False`)
진단/ablation 용으로만 남겼다.

## 미해결 항목

**Slow 리뷰어를 A 마다 다르게(에이전트별 신뢰도) 두거나, 검토 예산(하루 N건)을
제약으로 넣으면** "Slow 를 누구에게, 몇 건에 쓸 것인가" 가 최적화 문제가 된다.
지금 `warmup_then_attack` 이 그 실험의 골격이다. Adaptive Verification 이
에스컬레이션한 뒤 Slow 자체의 신뢰도가 상한선이 된다는 관찰(EXPERIMENTS.md)과
바로 연결된다.

**엔트로피를 LLM 에게 물어보는 안은 권장하지 않는다.** 차별점의 핵심이
"저엔트로피 구간은 LLM 호출 자체를 원천 배제" 인데, H 를 LLM 으로 구하면 모든
위임이 최소 1회 호출하게 되어 LLM률이 11.1% → 100% 로 오른다. 상시 LLM
베이스라인과 비용이 같아진다. 타협안은 **LLM 은 후보 집합 Ω 생성에만 쓰고 H 는
공식으로 계산**하는 것이다(현 구조가 이미 그 모양이다). 대신 "LLM 이 신고한
확률이 얼마나 calibrated 한가" 를 별도 실험으로 돌리면 공식을 쓰는 근거가
논문에 생긴다.

**후보 생성기의 품질이 다음 병목이다.** 현재 실험은 Ω 안에 정답이 항상 있다고
가정한다. 정답이 Ω 밖일 때(`apply_answer` 가 빈 집합을 만나는 경우)의 거동이
실제 배치의 리스크다.

**σ, k, λ 스윕.** θ 스윕과 같은 방식으로 돌릴 수 있게 `evaluate` 가 준비돼 있다.

**용어.** 논문에서는 `belief` 대신 `interpretation distribution` 같은 표현을
쓰는 편이 안전하다. "belief" 는 개념적으로 애매하고, 이 논지가 바로 그
"자기신고 belief" 를 공격 대상으로 삼기 때문에 같은 단어를 쓰면 혼동된다.

## 실제 LLM 붙이기

교체할 지점은 세 곳뿐이고, 나머지는 그대로 재사용된다.

| 지점 | 현재 | 교체 |
|---|---|---|
| 해석 후보 생성 | `DelegationTask.candidates` (스크립트) | 명세 + 툴 스키마로부터 후보를 뽑는 파서/모델 |
| 역질의 응답 | `semantic.Principal` (오라클) | 실제 Agent A 엔드포인트 |
| LLM 의미 판단 | `llm.ScriptedJudge` | `llm.AnthropicJudge` (골격 포함) |

LLM 을 붙일 때도 **자유 생성이 아니라 후보 중 택일**로 좁혀서 부르는 구조를
유지할 것. 호출 1회, 출력 토큰 수 개로 끝나므로 fallback 비용 가정이 유지된다.
