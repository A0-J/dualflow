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
| 반복 샘플링 횟수 `n` | 20 | B, C 조건에서 A(Principal)의 재구성을 반복하는 횟수 |
| 안정성 판단 임계값(entropy threshold) | 0.8 | 기존 실험 전체에서 재사용해온 값, 이번 실험을 위해 새로 만들지 않음 |
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

`condition_violation` task의 "정답이면서 허용된" 5개 episode만 따로 보면 그 차이가
가장 뚜렷하다.

| | Arm A | Arm B | Arm C |
|---|---|---|---|
| 오탐 기각(5건 중) | 4/5 | 5/5 | 1/5 |

탐지율(`P(detect|wrong)`, 참고용 — §6.3에서 해석 주의사항 설명): 17/51 → 16/51 → 15/51.

---

## 6. 결과가 RQ1–3에 답하는 방식

### RQ1 — 반복 의미 재구성(Repeated Semantic Anchoring)이 확신에 찬 의미 오해를 줄이는가?

**Yes.** 안전 실행 실패율이 4/120 → 0/120으로, Arm B와 Arm C 모두에서 동일하게
개선되었다. 이 개선은 "반복해서 다시 물어본다"는 것 자체의 효과이며, 두 조건이
공유하는 유일한 공통 변화이므로 이 효과의 원인을 반복 샘플링으로 귀속할 수 있다.

### RQ2 — 반복 재구성만으로 충분한가?

**No.** Arm B는 안전성은 개선했지만, 오탐 기각률이 오히려 **4/49 → 5/49로 증가**했다.
원인은 Arm B의 비교 방식이 "해석 전체가 완전히 같아야 통과"라는 조건이라서, facet
하나만 불확실하게 갈려도 전체를 불일치로 처리해 정상적인 작업까지 막아버리기
때문이다. 즉 반복 샘플링은 안전성을 사는 대신 유용성(utility)을 대가로 치렀다.

### RQ3 — 출처 인지 기반 패싯 정합(Provenance-aware Facet Grounding)이 이 손실 없이 그 트레이드오프를 줄이는가?

**Yes, 이번 실험 범위 안에서는 그렇다.** Arm C는 Arm B와 똑같이 20회 반복 샘플링을
쓰지만, facet 단위로 "확인된 것만 근거로 쓴다"는 규칙을 추가해 오탐 기각률을
**5/49 → 1/49**로 줄이면서도 안전 실행 실패율은 0/120으로 그대로 유지했다.

남은 1건의 Arm C 오탐 기각은 confirmed facet에서의 실제 불일치 때문이었다 — 즉 Arm
C가 무조건 관대하게 통과시키는 느슨한 설계가 아니라, "확인된 불일치는 여전히
막는다"는 원칙이 실제로 작동한 결과다.

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

사용자 확인 전까지 진행하지 않음:

- **Cost ablation**: 현재 `n=20`으로 고정된 반복 샘플링 횟수를 10 → 5 → adaptive
  early-stop 등으로 줄였을 때도 이번 효과가 유지되는지.
- **Runtime integration**: `GroundedIntentVerifier`(Arm C의 구현체)를
  `AgentDelegationRuntime`의 실제 실행 경로에 opt-in으로 연결하는 것 — 지금까지는
  독립 실행(standalone)으로만 검증됨.

두 가지 모두 이번 정리 문서를 검토한 뒤 사용자가 진행 여부를 결정한다.
