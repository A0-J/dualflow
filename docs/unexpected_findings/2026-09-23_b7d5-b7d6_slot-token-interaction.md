# B7d.5–B7d.6: 재현된 효과는 의미만으로 설명되지 않았다

**날짜:** 2026-09-23
**experiment ID:** b7d5_v2_neutral_format_control, b7d5r_v2_neutral_format_control_replication, b7d6_lexical_vs_semantic_control

B7d.4에서는 `summarize-history`가 no-experience baseline보다 거의 나아지지 않았고, `export-history`만 기존 export prior 방향으로 완전히 수렴했다. 당시에는 v2 experience가 과거의 의미를 전달한다기보다 기존 prior를 강화하는 priming에 가까울 가능성이 있다고 보았다.

이를 확인하기 위해 B7d.5에서는 기존 v2와 같은 형식을 유지하면서 action 정보를 제거한 `neutral-history` 조건을 추가했다.

예상과 달리 neutral-history는 `P(export)=0.995`까지 올라가며 기존 export prior를 거의 결정론적으로 강화했다. 그런데 같은 구조에 summarize-history를 넣었을 때는 `P(summarize)=0.275`까지 상승했다.

즉, history block의 형식 자체는 export 방향을 강하게 밀었지만, summarize 관련 내용이 들어가면 그 강한 prior를 일부 거슬러 움직이는 현상이 나타났다.

처음에는 이 결과가 한 번의 batch에서 우연히 나타난 것일 가능성도 있었다. 그래서 B7d.5R에서는 동일한 조건과 설정으로 800-call 실험을 그대로 반복했다.

재실험에서도 같은 현상이 나타났다.

- B7d.5: `P(summarize|summarize-history)=0.275`, `P(summarize|neutral)=0.005`
- B7d.5R: `P(summarize|summarize-history)=0.225`, `P(summarize|neutral)=0.005`
- 두 실험 모두 10/10 runs에서 `summarize-history > neutral-history`

따라서 summarize-history가 강한 export prior를 거슬러 모델의 분포를 움직이는 현상 자체는 재현되는 것으로 확인했다.

하지만 이것만으로 semantic experience transfer라고 볼 수는 없었다.

기존 summarize-history에는 두 요소가 동시에 들어 있었기 때문이다.

1. 과거 Principal이 특정 의미를 확인했다는 structured relation
2. literal action label인 `summarize`

따라서 모델이 과거의 의미 관계를 활용한 것인지, 단순히 `summarize`라는 action token에 반응한 것인지 분리할 필요가 있었다.

## B7d.6: 의미와 literal token 분리

B7d.6에서는 이 두 요소를 분리하기 위해 네 가지 조건을 비교했다.

| Condition | 구성 | P(summarize) |
| --- | --- | ---: |
| Neutral (N) | summarize 의미 없음 / literal token 없음 | 0.010 |
| Lexical-only (L) | literal `summarize`만 존재 | 0.010 |
| Semantic paraphrase (P) | confirmed 의미는 유지하지만 `summarize` 계열 단어 없음 | 0.005 |
| Structured summarize (S) | confirmed relation + literal `summarize` | **0.230** |

결과는 예상보다 명확했다.

`summary`, `summarize`, `summarization`과 같은 단어를 의미 관계와 연결하지 않고 단순히 노출한 lexical-only 조건은 neutral과 차이가 없었다.

반대로 과거 Principal-confirmed 의미 관계를 그대로 유지하되 `summarize`라는 literal token을 제거하고 의미적으로 paraphrase한 조건도 neutral과 차이가 없었다.

효과는 기존 v2와 동일하게 **`Confirmed interpretation`이라는 구조적 관계 안에 literal `summarize` action label이 함께 존재한 경우에만** 나타났다.

2×2 구조로 보면 interaction contrast는 다음과 같다.

`0.230 - 0.010 - 0.005 + 0.010 = +0.225`

따라서 현재 v2에서 관찰된 효과는 단순히 `summarize`라는 단어가 등장해서 생긴 lexical priming만으로 설명되지 않는다. 동시에 literal action token 없이 의미만 paraphrase했을 때도 같은 효과가 나타나지 않았다.

현재 결과는 **특정 structural slot과 literal action label이 결합될 때 나타나는 slot-token interaction**에 더 가까운 패턴을 보여준다.

이 결과만으로 모든 형태의 semantic transfer가 불가능하다고 결론내릴 수는 없다. 다만 현재 v2 representation에서는 literal action token이 제거된 paraphrased semantic relation만으로는 동일한 효과가 나타나지 않았다.

따라서 현재 결과만으로 v2가 표현이 달라져도 동일한 과거 의미를 활용하는 paraphrase-invariant semantic experience transfer를 수행한다고 주장하기는 어렵다.

## 이 결과가 바꾼 방향

처음 B7d.5의 결과만 봤을 때는 summarize-history가 strong export prior를 거슬러 움직였기 때문에 v2가 실제 semantic content를 활용하고 있을 가능성이 생겼다.

하지만 B7d.5R과 B7d.6까지 확인한 결과, 이 효과는 단순한 의미 전달보다는 현재 v2 표현의 특정 구조와 action label의 결합에 크게 의존하는 것으로 나타났다.

따라서 v2 prompt를 계속 조정하기보다는 historical experience를 사용하는 방식 자체를 다시 검토하기로 했다.

다음 설계에서는 historical experience를 Delegate의 candidate generation prompt에 직접 넣는 방식보다, 현재 episode에서 독립적으로 생성된 interpretation을 검증하는 **semantic evidence**로 사용하는 구조를 우선 검토한다.

이 방향에서는 특히 다음 문제를 해결해야 한다.

- literal action-token에 대한 의존성
- structural slot/format priming
- neutral history 자체가 기존 prior를 강화하는 현상
- paraphrase에 대한 불변성 부족
- explicit current instruction과 historical experience가 충돌할 때 현재 지시를 우선하는 구조

B7d.6에서는 lexical-only 조건의 200개 sample 중 2개가 지정한 action vocabulary 밖의 `ACTION=prepare`를 반환했다. 전체 800 calls 중 2건이며 `P(summarize)`를 중심으로 한 위 결론에는 영향을 주지 않았다. 재실험이 필요한 수준으로 보지는 않지만, 이후 diagnostic에서는 out-of-vocabulary action도 함께 기록한다.

상세한 run-level 결과, experiment identity, token usage 및 structural checks는 [`docs/experiments/agent_connected_eval.md`](../experiments/agent_connected_eval.md)의 B7d.5, B7d.5R, B7d.6 기록을 참조.
