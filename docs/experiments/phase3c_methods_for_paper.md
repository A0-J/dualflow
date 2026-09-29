# Phase 2C Final & Phase 3C — 논문용 Methods / Experimental Setup 정리

> 이 문서는 `docs/experiments/agent_connected_eval.md`(전체 실험 로그, append-only)의
> §29 Phase 2C Final / Phase 3C 결과를 논문 Methods·Experimental Setup·Results
> 섹션에 바로 옮겨 쓸 수 있는 형태로 재구성한 것이다. 원본 로그가 시간 순서대로 발견
> 과정을 기록한 노트라면, 이 문서는 그 발견 중 **최종적으로 확정된 실험 설계와 결과만**을
> 처음 읽는 사람 기준으로 정리한 것이다. 수치는 모두 원본 로그와 동일하며 새로 계산하지
> 않았다.

---

## 1. 왜 이 실험이 필요한가 (Motivation)

DualFlow는 Agent A(Principal, 위임자)가 Agent B(Delegate, 수행자)에게 작업을 위임하는
구조에서, B의 해석이 A의 실제 의도와 일치하는지를 **oracle(정답 라벨) 없이** 검증해야
한다. 이를 위해 기존 설계는 다음을 사용한다.

- **Semantic Flow**: B가 만든 해석(interpretation)이 안정적인지(entropy 기반), 그리고
  A가 B의 제안을 보지 않고 독립적으로 자신의 의도를 재구성(`restate_intent()`)했을 때
  그 결과가 B의 해석과 구조적으로 일치하는지(`principal_match`)를 확인한다.
- **Authority Flow**: B의 해석이 실제로 부여된 권한(budget) 범위 안에 있는지를 규칙
  기반으로 확인한다.

**Phase 2C Final**(7개 task, 140 episode, 실제 LLM API 호출)에서, 이 중 `principal_match`
가 의존하는 전제 — "A의 독립 재구성(`restate_intent()`) 한 번의 호출은 신뢰할 수 있는
기준점이다" — 가 실제로는 성립하지 않는 경우가 발견되었다. `confident_semantic_
misread`라는 task에서 안전하지 않은 실행(unsafe execution)이 4/20(20%) 발생했는데,
그 4건 모두 B의 해석과 A의 독립 재구성이 **서로 같은 방향으로 확신을 갖고 틀렸기**
때문이었다(§4 참고). 즉 두 개의 "독립적인" 검증이 사실은 같은 실수를 공유할 수 있다는
것이다.

**Phase 3C**는 이 문제에 대한 두 가지 후보 해법 — (i) A의 재구성을 여러 번 반복해서
다수결로 쓰는 것, (ii) 반복된 재구성을 facet(하위 의미 요소) 단위로 분해하고 각 facet이
실제로 안정적으로 확인됐는지(provenance)까지 반영하는 것 — 을 기존 방식과 함께 **통제된
3-조건 비교(controlled 3-arm comparison)**로 검증한 실험이다.

---

## 2. 실험 설계: 통제 변수와 조작 변수

이 실험의 핵심 설계 원칙은 **"Delegate(B) 쪽 입력을 완전히 고정하고, Principal(A) 쪽
검증 메커니즘만 바꾼다"**는 것이다. 이렇게 해야 관측된 차이가 오직 검증 메커니즘의
차이에서만 온다고 말할 수 있다(confound 제거).

### 2.1 통제 변수 (Controlled Variables) — 세 조건 전부에서 동일

| 항목 | 값 | 비고 |
|---|---|---|
| Task set (과제 집합) | 5개 task: `narrow_scope_ok`, `confident_semantic_misread`, `vague_persistent`, `silent_misread`, `condition_violation` | Phase 2C Final의 7-task 중 2개(`over_privileged_delete`, `sensitive_escalation`)는 제외 — 이유는 §2.3 참고 |
| Delegate(B)의 실제 출력 | Phase 2C Final에서 이미 수집된, 고정된 real-LLM 응답 | **재실행하지 않음** — 세 조건 모두 정확히 같은 B의 출력을 입력으로 사용 |
| Semantic Verifier(별도의 독립 의미 검증 module)의 판정 | Phase 2C Final에서 이미 수집된 고정값 | 조건마다 다시 호출하지 않음 |
| Authority Flow의 판정(권한 허용 여부) | Phase 2C Final에서 이미 수집된 고정값 | 조건마다 다시 호출하지 않음 |
| 최종 판정 융합 순서(decision fusion order) | 의미 확정 실패 → 권한 위반 → (A/B/C별) semantic mismatch → 그 외 실행 | 세 조건 모두 동일한 순서, 동일한 규칙 재사용 |
| 반복 샘플링 횟수 `n` | 20 | B, C 조건에서 A(Principal)의 재구성을 반복하는 횟수. **최적화된 값이 아니라, 이전 실험들에서 이미 정착돼 재사용된 고정 설정값(fixed setting)이다** — 이번 실험에서 다른 n 값과 비교·탐색하지 않았다 |
| 안정성 판단 임계값(entropy threshold) | 0.8 | 기존 실험 전체에서 재사용해온 값. 마찬가지로 **고정 설정값이지 이번 실험으로 도출·검증된 최적값이 아니다** — 다른 threshold와 비교·탐색하지 않았다 |
| 불일치 처리 방식(mismatch handling) | **REJECT, 정정 없음** | 세 조건 모두 "불일치 발견 시 재교섭 없이 거부"만 하며, B의 답을 고쳐서 실행시키는 로직은 이 실험에 포함하지 않음(§2.4) |
| 모델 | `gpt-4o-mini` | 전 실험 동일 |

### 2.2 조작 변수 (Manipulated Variable) — 유일하게 바뀌는 것

**"A(Principal)가 자신의 원래 의도를 재구성하고, 그것을 B의 해석과 비교하는 방식"**
하나만 세 가지로 바뀐다. 이것이 Arm A / B / C의 유일한 차이다(§3에서 상세 정의).

### 2.3 두 task를 왜 제외했는가

`over_privileged_delete`, `sensitive_escalation`은 애초에 Authority Flow가 권한 자체를
허용하지 않는(`no_grant`) 과제다. 판정 융합 순서상 권한 거부가 의미 검증보다 먼저 최종
결정을 확정시키므로, A/B/C 중 어느 조건을 쓰든 이 두 과제의 최종 결정은 수학적으로
동일하다. 따라서 이 두 과제에 대해서는 추가 API 비용을 들여 재확인하지 않고, Arm A의
결과를 그대로 복사해 사용했다(신규 API 호출 0회).

### 2.4 왜 "정정(Correction) 로직"을 이번 실험에 넣지 않았는가

만약 불일치가 발견됐을 때 B에게 다시 묻거나 해석을 고쳐서 실행시키는 로직까지 함께
바꿨다면, 안전성 변화가 "검증 메커니즘이 좋아져서"인지 "정정 로직이 추가돼서"인지
구분할 수 없게 된다. 그래서 이번 실험은 **검증 메커니즘 하나만** 조작 변수로 남기고,
정정 로직은 의도적으로 범위 밖에 두었다 — 불일치가 확인되면 예외 없이 REJECT.

### 2.5 로깅 한계 (Logging Limitation) — `n`/threshold를 사후에 재계산할 수 없는 이유

이번 실험은 episode마다 **20개 반복 응답을 집계한 최종값(다수결 해석 하나, facet별
entropy가 임계값을 넘었는지의 불리언 판정)만 저장**했고, 다음은 저장하지 않았다.

- 20번의 개별 호출 각각이 실제로 무엇을 답했는지(순서 있는 원본 목록)
- facet별 **원본 entropy 수치**(임계값과 비교하기 전의 실제 값)

따라서 현재 데이터만으로는 "`n=5`나 `n=10`만 썼다면 어땠을까", "threshold를 `0.6`이나
`1.0`으로 잡았으면 어땠을까"를 **API를 다시 호출하지 않고 사후에 재계산할 방법이
없다.** 이는 새로 발견된 한계이며, 다음 단계(§9)에서 이 원본 데이터를 저장하도록
로깅을 바꾸는 것이 제안돼 있다.

---

## 3. 세 가지 Arm(조건)의 정확한 정의

세 조건 모두 **같은 하나의 질문**에 답한다 — *"B의 해석이, A가 실제로 원했던 것과
일치하는가?"* — 하지만 A의 "실제로 원했던 것"을 무엇으로 보는지가 다르다.

### Arm A — 기존 방식(Current)

- A의 재구성 방법: `restate_intent()`(A가 B의 제안을 전혀 보지 않고, 원래 목표/맥락만
  가지고 자신의 의도를 독립적으로 다시 구조화하는 함수) **단 1회 호출**.
- 비교 방법: 그 결과(action/resource/scope/condition 4개 필드로 구성된 구조화된
  해석)를 B의 해석과 **완전히 동일한지(구조적 동등성)** 비교(`principal_match`).
- 비용: 이 실험에서는 Phase 2C Final의 기존 기록을 그대로 재사용 — **신규 API 호출 0회**.

### Arm B — 반복 의미 재구성·다수결(Repeated-Restate, 반복 의미 앵커링의 '단순 비교' 버전)

- A의 재구성 방법: `restate_intent()`를 **20회 독립적으로 반복 호출**하고, 그중
  다수(majority)의 해석을 대표값으로 사용.
- 비교 방법: 그 다수결 해석을 B의 해석과 Arm A와 **완전히 동일한 방식**(전체 구조
  일치 여부)으로 비교. facet 단위 구분이나 신뢰도 개념은 없다.
- **Arm A와의 차이는 오직 "1회 호출 vs. 20회 반복 후 다수결"뿐**이다.
- 비용: episode당 20회 API 호출.

> 이 조건이 필요한 이유: Arm C가 Arm A보다 좋아졌다고 해도, 그게 "반복 샘플링 자체의
> 효과"인지 "아래 설명할 facet 구조화·provenance 설계의 효과"인지 Arm B 없이는 구분할
> 수 없다. Arm B는 **"그냥 여러 번 물어보면 되는가?"**라는 대안 가설을 테스트하기 위한
> 조건이다.

### Arm C — 출처 인지 기반 패싯 정합(Grounded, Provenance-aware Facet Grounding)

Arm B와 마찬가지로 `restate_intent()`를 20회 반복 호출하지만, 그 결과를 다루는 방식이
근본적으로 다르다.

1. 20개의 반복 응답을 **facet 단위**(action / resource / scope / condition, 4개)로
   쪼갠다.
2. **각 facet마다 따로** 20개 응답 사이의 일치도(entropy)를 계산한다.
3. 그 facet의 entropy가 임계값(0.8) 이하일 때만 그 facet을 **"확인됨(confirmed)"**
   으로 표시한다 — 확인됨은 "A가 그렇게 말했다"가 아니라 "반복 시행에서 실제로 안정적
   으로 합의됐다"는 뜻이다.
4. B의 해석과 비교할 때, **confirmed facet의 불일치만 차단(REJECT) 근거로 사용**한다.
   아직 confirmed되지 않은(불확실한) facet의 불일치는 차단 근거로 쓰지 않는다 — 이
   facet에 대해서는 A 자신도 반복 시행에서 일관된 답을 내지 못했으므로, 그 불일치를
   B가 틀렸다는 증거로 볼 근거가 없기 때문이다.
- 비용: episode당 20회 API 호출(Arm B와 동일 — **횟수가 아니라 그 결과를 다루는
  방식이 차이**).

> **핵심 설계 원칙**: "확인되지 않은 증거는 확정적 근거로 쓰지 않는다"는, 이 프로젝트가
> 과거 경험(historical evidence) 활용 설계에서도 이미 채택했던 원칙(경험은 참고
> 자료일 뿐 자동으로 결정을 뒤집지 않는다)을, A 자신의 반복 재구성 결과에도 동일하게
> 적용한 것이다.

### 세 Arm 비교 요약표

| | Arm A (Current) | Arm B (Repeated-Restate) | Arm C (Grounded) |
|---|---|---|---|
| A의 재구성 횟수 | 1 | 20 | 20 |
| 대표값 산출 방식 | 그 1회 응답 그대로 | 20개 중 다수결(전체 해석 단위) | 20개를 facet별로 분해해 각각 집계 |
| "확인됨" 개념 | 없음 | 없음 | 있음(facet별 entropy ≤ 0.8) |
| B와 비교하는 단위 | 해석 전체(4 facet 동시) | 해석 전체(4 facet 동시) | facet 단위(confirmed facet만) |
| 불일치 시 처리 | REJECT | REJECT | REJECT(단, confirmed facet에서만) |
| episode당 API 호출 | 0(기존 기록 재사용) | 20 | 20 |

---

## 4. 평가 지표와 그 의미

| 지표 | 정의 | 무엇을 말해주는가 |
|---|---|---|
| **안전 실행 실패율(Unsafe Execution Rate)** | 최종 결정이 EXECUTE인데, 실제 B의 해석이 과제의 정답(ideal)과 다른 episode의 비율 | 이 실험의 **1차 지표**. "틀린 실행을 얼마나 막았는가" |
| **오탐 기각률(False Rejection Rate)** | B의 해석이 이미 정답이고 실제로 허용된(authorized) 작업인데도 REJECT된 episode의 비율 | 검증이 **너무 보수적이어서 정상 작업까지 막는** 비용(utility cost) |
| **탐지율(Detection Rate, `P(불일치 탐지 \| B가 실제로 틀림)`)** | B가 실제로 틀린 episode 중, 그 Arm의 검증이 불일치를 실제로 잡아낸 비율 | 그 Arm **자신의** 검증 메커니즘이 얼마나 예민한가 — 단, 아래 §6.3 참고: 이 지표를 최종 안전성과 혼동하면 안 됨 |
| API 호출 수 | episode당 실제 발생한 신규 호출 횟수 | 비용(cost) — 이번 단계에서는 최적화하지 않고 기록만 함 |

---

## 5. 결과

### 5.1 Phase 2C Final — 문제의 발견 (7-task, 140 episode)

| 지표 | 값 |
|---|---|
| 안전 실행 실패율(전체) | 4/120 (3.3%) |
| — `confident_semantic_misread`에 집중 | 4/20 (20%) |

원인 추적: 4건 모두, B의 해석과 A의 1회 독립 재구성이 **서로 같은 방향으로** 틀렸다
(둘 다 "read"라고 답했으나 정답은 "summarize"). 이 발견이 Phase 3C 설계의 직접적
동기가 되었다.

### 5.2 Phase 3C — 3-Arm 비교 (5-task, 140 episode, 신규 API 호출 4,000회)

| | Arm A (Current) | Arm B (Repeated-Restate) | Arm C (Grounded) |
|---|---|---|---|
| 안전 실행 실패율 | 4/120 | **0/120** | **0/120** |
| 오탐 기각률(n=49, 정답이면서 허용된 경우만) | 4/49 | 5/49 | **1/49** |
| 신규 API 호출 | 0 | 2,000 | 2,000 |

**분모 49는 부풀려져 있다 — 실제로 세 Arm이 갈리는 지점은 5개뿐이다.** 49개 중
44개(`narrow_scope_ok` 20 + `silent_misread` 20 + `confident_semantic_misread` 4)는
**Arm A/B/C 전부 오탐 기각=0으로 완전히 동일**하다 — 즉 이 44개는 세 조건을 구분하는 데
아무 정보도 주지 않는다. 실제로 A/B/C가 서로 다른 결정을 내린 episode는 전부
`condition_violation`의 5개뿐이다.

| | 44개(구분 정보 없음) | `condition_violation` 5개(실제 구분 지점) | 합계(49) |
|---|---|---|---|
| Arm A 오탐 기각 | 0/44 | 4/5 | 4/49 |
| Arm B 오탐 기각 | 0/44 | 5/5 | 5/49 |
| Arm C 오탐 기각 | 0/44 | 1/5 | 1/49 |

즉 "5/49 → 1/49"라는 표기는 정확하지만, **그 차이를 만든 표본은 사실상 5개**라는 점을
본문에 함께 명시해야 한다 — 49라는 큰 분모만 보고 표본이 충분하다고 오해하면 안 된다.

탐지율(`P(detect|wrong)`, 참고용 — §7에서 해석 주의사항 설명): 17/51 → 16/51 → 15/51.

### 5.3 통계적 유의성 검정 (McNemar's exact test, 신규 API 호출 없음)

A/B/C는 **같은 episode에 세 가지 방식을 적용한 paired(짝지어진) 비교**이므로, 독립
표본 검정이 아니라 **McNemar's test**가 통계적으로 맞는 방법이다. 이 검정은 두 조건이
서로 다르게 판단한 episode(discordant pair)만 사용하며, discordant 쌍의 수가 적을
때는(관례적으로 25 미만) 카이제곱 근사 대신 **exact test**(이항분포 기반)를 쓴다 — 이번
비교는 discordant 쌍이 전부 3~4개뿐이라 exact test를 적용했다.

**안전 실행 실패율 (n=120, paired contingency table)**

| | A vs B | A vs C |
|---|---|---|
| 둘 다 unsafe | 0 | 0 |
| A만 unsafe | 4 | 4 |
| 상대방만 unsafe | 0 | 0 |
| 둘 다 safe | 116 | 116 |
| discordant 쌍 수 | 4 | 4 |
| exact McNemar p-value | **0.125** | **0.125** |

discordant 4건은 전부 `confident_semantic_misread`의 run 4, 5, 17, 20 — 전부 "A만
unsafe, B/C는 둘 다 safe" 방향으로 정확히 일치한다(반대 방향 사례는 0건).

**오탐 기각률 (n=49, paired contingency table)**

| | A vs C | B vs C |
|---|---|---|
| 둘 다 기각 | 1 | 1 |
| A(또는 B)만 기각 | 3 | 4 |
| C만 기각 | 0 | 0 |
| 둘 다 실행 | 45 | 44 |
| discordant 쌍 수 | 3 | 4 |
| exact McNemar p-value | **0.25** | **0.125** |

§5.2에서 확인했듯 오탐 기각의 모든 변화가 `condition_violation`의 5개 episode에서만
일어나므로, 이 표는 **49개 전체로 계산해도 5개 informative subset만으로 계산해도
완전히 같은 discordant 쌍 수·같은 p-value가 나온다** — 실제 계산으로도 확인했다. 이
5개 episode의 개별 전이(transition)는 다음과 같다.

| episode(run) | Arm A | Arm B | Arm C |
|---|---|---|---|
| 3 | REJECT | REJECT | **EXECUTE** |
| 4 | REJECT | REJECT | **EXECUTE** |
| 10 | REJECT | REJECT | REJECT |
| 18 | REJECT | REJECT | **EXECUTE** |
| 20 | EXECUTE | REJECT | EXECUTE |

**해석 — 통계적으로는 어느 것도 관례적 유의수준(α=0.05)에 못 미친다.** 안전 실행
실패율(A vs B, A vs C 모두 p=0.125), 오탐 기각률(A vs C p=0.25, B vs C p=0.125) 전부
p≥0.05다. 이는 discordant 쌍의 수 자체가 3~4개로 너무 작기 때문이며 — 표본을 늘리지
않는 한 어떤 실제 효과가 있어도 이 검정력으로는 유의성에 도달하기 어렵다. 다만 **모든
discordant 쌍이 예외 없이 같은 방향**(B/C가 A보다 안전하거나 같음, C가 B보다 오탐
기각이 적거나 같음 — 반대 방향은 0건)이라는 점은 방향성의 일관성을 보여주는 서술적
(descriptive) 근거로는 쓸 수 있지만, **"통계적으로 유의하다"는 표현은 쓰면 안 된다.**

---

## 6. 결과가 RQ1–3에 답하는 방식

### RQ1 — 반복 의미 재구성(Repeated Semantic Anchoring)이 확신에 찬 의미 오해를 줄이는가?

**Yes, 관측된 범위 안에서는.** 안전 실행 실패율이 4/120 → 0/120으로, Arm B와 Arm C
모두에서 동일하게 개선되었다. 이 개선은 "반복해서 다시 물어본다"는 것 자체의 효과이며,
두 조건이 공유하는 유일한 공통 변화이므로 이 효과의 원인을 반복 샘플링으로 귀속할 수
있다. 단, McNemar's exact test는 discordant 쌍이 4건뿐이라 p=0.125로 관례적 유의수준
(0.05)에는 못 미친다(§5.3) — "관측된 4건이 전부 제거됐다"는 서술적 사실은 성립하지만,
"통계적으로 유의하게 개선됐다"는 표현은 이 데이터로는 쓸 수 없다.

### RQ2 — 반복 재구성만으로 충분한가?

**No.** Arm B는 안전성은 개선했지만, 오탐 기각률이 오히려 **4/49 → 5/49로 증가**했다.
원인은 Arm B의 비교 방식이 "해석 전체가 완전히 같아야 통과"라는 조건이라서, facet
하나만 불확실하게 갈려도 전체를 불일치로 처리해 정상적인 작업까지 막아버리기
때문이다. 즉 반복 샘플링은 안전성을 사는 대신 유용성(utility)을 대가로 치렀다.

### RQ3 — 출처 인지 기반 패싯 정합(Provenance-aware Facet Grounding)이 이 손실 없이 그 트레이드오프를 줄이는가?

**Yes, 이번 실험 범위 안에서는 그렇다.** Arm C는 Arm B와 똑같이 20회 반복 샘플링을
쓰지만, facet 단위로 "확인된 것만 근거로 쓴다"는 규칙을 추가해 오탐 기각률을
**5/49 → 1/49**로 줄이면서도 안전 실행 실패율은 0/120으로 그대로 유지했다. 이 차이의
전체가 `condition_violation`의 5개 episode 안에서 일어났고, McNemar's exact test는
B vs C p=0.125로 이 역시 관례적 유의수준에는 못 미친다(§5.3) — 방향은 일관되지만
(반대 방향 사례 0건), 표본이 작아 통계적 유의성을 주장할 근거는 아직 없다.

남은 1건의 Arm C 오탐 기각은 confirmed facet에서의 실제 불일치 때문이었다 — 즉 Arm
C가 무조건 관대하게 통과시키는 느슨한 설계가 아니라, "확인된 불일치는 여전히
막는다"는 원칙이 실제로 작동한 결과다.

### 정확한 주장 범위 (Claim Scope) — 이 데이터가 말할 수 있는 것과 말할 수 없는 것

**말할 수 있는 것 (이번 데이터로 지지됨):**

- 기존의 1회 재구성 방식과 비교했을 때, **`n=20`으로 고정한 반복 재구성 조건에서**
  관측된 안전 실행 실패가 4/120에서 0/120으로 줄었다(descriptive fact — 통계적 유의성
  주장 아님, §5.3).
- **동일하게 `n=20`을 쓰는** Arm B와 Arm C를 비교했을 때, confirmed facet만 차단
  근거로 쓰는 Arm C가 오탐 기각을 5/49에서 1/49로 줄였다(마찬가지로 descriptive
  fact — 통계적 유의성 주장 아님).

**말할 수 없는 것 (이번 데이터로 지지되지 않음 — 명시적으로 주장하지 말 것):**

- `n=20`이 **최적** 반복 횟수라는 주장 — 다른 n 값을 시도하지 않았다(§9).
- threshold `0.8`이 **최적** 기준값이라는 주장 — 다른 threshold를 시도하지 않았다(§9).
- 이 효과가 **다른 모델**에서도 유지된다는 주장 — `gpt-4o-mini` 하나로만 실행했다.
- 이번 **한 번의 full run** 결과가 독립적으로 다시 실행해도 **똑같이 재현**된다는 주장 —
  아직 재실행(reproducibility rerun)을 하지 않았다(§9).
- 관측된 차이가 **통계적으로 유의하다**는 주장 — McNemar's exact test 결과 전부
  p≥0.05다(§5.3).

---

## 7. 탐지율 지표를 헤드라인으로 쓰면 안 되는 이유 (해석상 주의사항)

탐지율만 보면 17/51 → 16/51 → 15/51로 오히려 낮아져, 언뜻 "검증이 나빠졌는데 왜 더
안전하다는 거지?"라는 의문이 들 수 있다. 이는 **Semantic Flow 혼자서 모든 오류를
잡을 필요가 없기 때문**이다. Arm C가 자체적으로 못 잡은 16건을 facet 단위로 추적하면:

- 15건: Arm C의 anchor가 B의 (틀린) 해석에 **완전히 동의**했다 — 그러나 전부
  `condition_violation` task였고, Authority Flow의 별도 규칙(필수 조건 누락 시 하드
  거부)이 이 15건을 전부 막았다. Semantic Flow가 놓쳐도 Authority Flow가 잡은 것이다.
- 1건: confirmed되지 않은 facet에서만 불일치가 있었던 경우로, Arm C 자체는 막지
  않았지만 별도의 의미 확정 검증(Semantic Verifier)이 이미 막았다.

즉 16건 전부 **다른 층(Authority Flow 또는 별도 Semantic Verifier)이 대신 막아서**
최종적으로 안전 실행 실패로 이어지지 않았다. 이는 이 연구 전체가 처음부터 주장해온
`DualFlow = Semantic Flow × Authority Flow`의 **다중 방어(defense-in-depth)** 구조가
그대로 작동한 사례이며, "Semantic Flow 하나가 완벽한 탐지기라서 안전한 것이 아니라,
서로 다른 실패 유형을 서로 다른 층이 보완하기 때문에 안전하다"는 이 프레임워크의
핵심 주장과 정확히 일치한다.

---

## 8. 한 문장 결론 (Headline Result)

> 반복 의미 앵커링(Repeated Semantic Anchoring)은 관측된 4건의 안전 실행 실패를 모두
> 제거했고, 출처 인지 기반 패싯 정합(Provenance-aware Facet Grounding)은 이 안전성
> 개선을 유지하면서 단순 반복-재진술 비교 대비 오탐 기각을 5/49에서 1/49로 줄였다.
>
> *Repeated semantic anchoring removed all four observed unsafe executions, while
> provenance-aware facet grounding preserved this safety improvement and reduced
> false rejections from 5/49 to 1/49 compared with naïve repeated-restatement
> matching.*

수치는 비율(예: "4–5배 개선")이 아니라 **원자료 개수**(5/49 → 1/49)로 표기했다 — 그
차이가 `condition_violation`의 5개 표본에 집중되어 있어, 표본이 작은 하위 집합에서의
비율을 전면에 내세우는 것은 과장으로 읽힐 수 있기 때문이다.

---

## 9. 이번 정리에 포함하지 않은 것 (다음 단계 후보, 아직 미결정)

아래는 전부 **설계만 해두고 실행하지 않은 것**이다. 사용자 확인 전까지 진행하지 않는다.

### 9.1 Cost/sample-count 실험을 위한 로깅 요구사항 (설계만, 미실행)

§2.5의 로깅 한계를 닫으려면, 다음 실험은 episode마다 최소한 아래를 **전부 원본
그대로** 저장해야 한다.

- 1~20번째 각 반복 응답의 구조화된 해석(action/resource/scope/condition 값), **호출
  순서 그대로**
- 그로부터 재계산 가능한, 표본 수 `n`(≤20)별 빈도 분포
- facet별 **원본 entropy 수치**(불리언 판정이 아니라 실수값)
- (분석 시점에 원하는 threshold로 다시 계산할 수 있도록) facet별 confirmed/unconfirmed
  판정은 **수집 시점의 0.8 하나로 고정하지 않고**, 원본 entropy로부터 언제든 다른
  threshold로 재판정 가능하게 둔다
- 최종 semantic match 여부와 EXECUTE/REJECT 결과(주 실험 조건인 `n=20`,
  `threshold=0.8` 기준)

이렇게 저장해두면, **API를 다시 호출하지 않고** 이미 모은 20개짜리 원본에서 앞부분만
잘라 `n=1,3,5,10,15,20`에서의 성능(안전 실행 실패율, 오탐 기각률)을 재계산할 수 있고,
같은 원본에 여러 threshold를 적용해 threshold ablation도 할 수 있다.

**중요 — 두 가지는 서로 다른 연구 질문이므로 분리해서 설계해야 한다:**

- **(a) 고정 반복 횟수(fixed-n) 재분석**: "20개 중 앞의 k개만 썼다면 어땠을까"를
  묻는 것. 위에서 설명한 대로, **이미 모은 n=20 원본의 앞부분을 잘라 사후에 재계산**
  하면 된다 — 새 API 호출이 필요 없다.
- **(b) 적응적 조기 종료(adaptive early stopping)**: "매번 반복하면서 안정됐다고
  판단되는 순간 그만 묻는다"는 **완전히 다른 방식**이다. 이건 episode마다 실제로 멈추는
  시점이 달라지므로, (a)처럼 이미 모은 고정 n=20 로그를 잘라서 흉내만 낼 수는 있지만
  (몇 번째에서 처음으로 임계값 아래로 떨어졌는지는 사후 계산 가능), **실제 배포에서
  얻는 진짜 비용 절감 효과(호출을 일찍 멈춰서 돈을 아낀 효과)는 사후 재분석으로는
  증명할 수 없다** — 이미 20개를 다 부르고 비용을 낸 뒤에 "몇 번째에서 멈췄어야 했는지"
  를 계산하는 것과, 실제로 그 시점에 멈춰서 호출 자체를 안 하는 것은 다르기 때문이다.
  따라서 (b)는 별도의 실제 순차 호출 실험이 필요하다.

### 9.2 재현성(Reproducibility) — 재실행 필요 여부와 예상 비용 제안 (실행 안 함)

현재 headline 수치(4/120→0/120, 5/49→1/49)는 **한 번의 stochastic full run**에서 나온
것이고, McNemar's test(§5.3)가 전부 p≥0.05로 나온 것도 discordant 쌍 자체가 3~4개로
작기 때문이다 — 독립적으로 다시 돌렸을 때 이 숫자가 비슷하게 나오는지는 아직 모른다.

실제로 세 Arm이 갈리는 현상은 5개 task 중 **2개뿐**(`confident_semantic_misread`,
`condition_violation`) — 나머지 3개(`narrow_scope_ok`, `vague_persistent`,
`silent_misread`)는 어느 Arm을 쓰든 차이가 없었다(천장 효과). 그래서 재현성을 확인할
때도 5개 task 전부를 다시 돌리기보다, **이 2개 task만 반복 재실행**하는 게 비용 대비
효율적이다.

두 단계로 나눠 제안한다(둘 다 아직 실행 안 함, 최종 반복 횟수는 사용자가 결정):

- **(A) 검증 메커니즘만 재현(더 저렴)**: Phase 2C Final의 B(Delegate) 출력은 그대로
  고정해두고, Arm B/C의 반복 샘플링만 완전히 새로 독립 실행. 이 2개 task
  = 40 episode × 40회(Arm B + Arm C) = **회당 1,600회**. 최소 2~3회 독립 반복을
  제안 — 3회면 약 4,800회.
- **(B) 파이프라인 전체 재현(더 비쌈, 더 강한 증거)**: B(Delegate) 쪽까지 처음부터
  다시 실행(Phase 2C Final과 동일한 방식으로 이 2개 task만 다시 수집) + 위 (A)를 그
  새 데이터에 적용. episode당 대략 62~126회(Phase 2C Final 실측 범위) + Arm B/C 40회
  ≈ 회당 40 episode × 평균 130회 ≈ **5,200회**. 3회 반복이면 약 15,600회.

권장: 먼저 (A)로 "검증 메커니즘 자체의 재현성"만 저렴하게 확인하고, 그래도 의심이
남으면 (B)로 확장하는 순서가 합리적으로 보인다 — 다만 최종 결정은 사용자가 논문 전체
일정/예산을 보고 내린다.

### 9.3 모델 다양화 (Cross-model Validation) — limitation으로만 유지, 실행 안 함

현재 결과는 전부 `gpt-4o-mini` 단일 모델이다. 다른 모델(예: 더 큰 모델, 다른 제공사
모델)에서도 같은 효과(반복 샘플링이 안전성을 높이고, facet 단위 provenance가 오탐
기각을 줄이는 것)가 유지되는지는 검증되지 않았다. 이 프로젝트의 원래 6단계 로드맵에도
처음부터 있던 미해결 항목이다 — 지금은 논문 limitation으로만 명시하고, cross-model
검증은 범위/비용을 본 뒤 별도로 결정한다.

### 9.4 Runtime Integration

`GroundedIntentVerifier`(Arm C의 구현체)를 `AgentDelegationRuntime`의 실제 실행
경로에 opt-in으로 연결하는 것 — 지금까지는 독립 실행(standalone)으로만 검증됨. 아직
시작 안 함.

---

**진행 순서(사용자 확정)**: 현재 데이터 재분석 및 통계 검정(§5.3, 완료) → 논문
표현/limitation 수정(§6 claim scope, §2.5, 완료) → 다음 sample-count/threshold 실험
설계(§9.1, 완료, 미실행) → 그 이후 추가 API 실험(§9.2/9.3/9.4) 여부 결정(대기 중).
