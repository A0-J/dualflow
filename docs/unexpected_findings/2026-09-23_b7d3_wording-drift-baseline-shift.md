# B7d.3: 의미는 같았지만 baseline은 달라졌다

**날짜:** 2026-09-23
**experiment ID:** b7d3_representation_validation (오염된 pilot) → b7d3_canonical_rerun (재검증)

B7d.3에서는 v1과 v2 experience representation이 ambiguous delegation에 어떤 영향을 주는지 비교하려고 했다.

그런데 실험 도중 예상하지 못한 문제가 발견됐다.

현재 episode의 context 문장을 scenario에 저장된 문자열 그대로 사용하지 않고, 코드에서 의미상 동일한 문장으로 다시 조립하고 있었다.

원래 사용하던 문장은 다음과 같았다.

`The September 2026 report is located at /reports/2026-09/.`

하지만 diagnostic에서는 다음처럼 바뀌어 있었다.

`The report for the current period is located at /reports/2026-09/.`

사람이 읽으면 두 문장은 의미상 거의 동일하거나 매우 유사하다. 그래서 처음에는 이 차이가 실험 결과에 영향을 줄 것이라고 생각하지 않았다.

하지만 실제 결과는 크게 달랐다.

| Context wording | No-experience baseline | 관찰 |
| --- | --- | --- |
| Reconstructed wording | `H=0.000`, `P(export)=1.00` in 5/5 runs | baseline이 완전히 deterministic |
| Canonical wording | `H=0.469~0.811`, `P(summarize)=0.10~0.25` | ambiguity가 다시 나타남 |

즉, experience representation이나 scenario의 의미를 바꾼 것이 아니라 **context 문장 하나를 원래 표현으로 되돌렸을 뿐인데 baseline 자체가 완전히 달라졌다.**

reconstructed wording에서는 no-experience 조건조차 5/5 runs 모두 export로 수렴했다. 반대로 canonical wording으로 복원하자 같은 task에서 다시 uncertainty와 summarize 가능성이 나타났다.

## 왜 예상 밖이었나

당시 실험에서 측정하고 싶었던 것은 v1과 v2 experience의 차이였다.

context wording은 단순한 표현 차이일 뿐이고, 의미가 같다면 실험적으로도 같은 조건으로 취급할 수 있다고 생각했다.

하지만 실제로는 이 작은 표현 차이가 experience 효과보다 더 크게 baseline distribution을 바꿨다.

따라서 LLM diagnostic에서는 사람이 보기에 의미가 같은 paraphrase라도 **동일한 실험 입력이라고 가정하면 안 된다**는 점을 확인했다.

특히 DualFlow처럼 ambiguity와 probability distribution 자체를 측정하는 실험에서는 prompt wording의 작은 변화가 측정하려는 효과보다 더 큰 confound가 될 수 있었다.

## 이 결과가 바꾼 것

이 결과 이후 model-visible input은 의미적으로 재구성하지 않고 **실제로 검증된 문자열 자체를 고정해서 사용**하기로 했다.

구체적으로 다음 재현성 규칙이 추가됐다.

- scenario 파일에 model-visible delegation/context를 완성된 문자열 그대로 저장
- `build_current_context()`에서 문장을 다시 조립하지 않고 passthrough
- 실험 시작 시 delegation/context의 SHA256 fingerprint 기록
- regression test를 통해 canonical 문자열이 byte-identical하게 유지되는지 확인

이 과정이 현재 [`docs/REPRODUCIBILITY.md`](../REPRODUCIBILITY.md)와 scenario fingerprint 체계가 만들어진 직접적인 계기가 됐다.

또한 wording이 섞인 상태에서 수행했던 초기 B7d.3 pilot의 결과는 그대로 사용할 수 없다고 판단했다.

당시 pilot에서는 v2가 5/5 runs에서 일관되게 개선되는 것처럼 보였지만, canonical wording으로 다시 실행한 결과 그 효과는 훨씬 약하고 불안정하게 나타났다.

따라서 초기 결과는 representation 효과만을 측정한 결과가 아니라 wording confound가 섞인 결과였으며, 이후 B7d 계열 실험은 모두 canonical string과 fingerprint가 고정된 scenario 위에서 진행하게 됐다.

이 결과의 핵심은 다음과 같다.

> **의미가 같아 보이는 prompt paraphrase도 동일한 experimental condition이라고 가정할 수 없다.**

DualFlow의 이후 실험에서는 model-visible wording 자체를 하나의 통제 변수로 취급한다.

상세 실험 기록과 canonical rerun 결과는 [`docs/experiments/agent_connected_eval.md`](../experiments/agent_connected_eval.md)의 B7d.3 관련 섹션을 참조.
