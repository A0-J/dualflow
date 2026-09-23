# B7d.4: 양의 delta가 semantic transfer를 의미하지 않았다

**날짜:** 2026-09-23
**experiment ID:** b7d4_semantic_content_ablation
**git SHA:** `3c03dda8894da6a1f42394add26682d1d051b2bd`

B7d.4에서는 v2 experience가 historical content의 방향에 실제로 반응하는지 확인하려고 했다.

같은 ambiguous September delegation과 같은 v2 format을 사용하면서 historical experience의 confirmed action만 바꿨다.

- no experience
- summarize-history
- export-history

예상은 단순했다.

v2가 historical experience의 내용을 현재 interpretation에 반영한다면, summarize-history에서는 `summarize` 쪽으로, export-history에서는 `export` 쪽으로 각각 대응되는 변화가 나타나야 했다.

실험 결과만 처음 보면 그 예상과 맞는 것처럼 보였다.

| Condition | P(summarize) | P(export) | Mean H |
| --- | ---: | ---: | ---: |
| No experience | 0.250 | 0.750 | 0.775 |
| Summarize-history | 0.230 | 0.770 | 0.744 |
| Export-history | 0.000 | 1.000 | 0.000 |

두 방향의 차이를 계산하면 다음과 같았다.

`delta_sum = +0.23`

`delta_export = +0.23`

숫자만 보면 summarize-history와 export-history가 서로 다른 방향으로 반응했고, 두 delta도 모두 양수이기 때문에 content-sensitive한 효과가 나타난 것처럼 해석할 수 있었다.

하지만 실제 분포를 보면 이야기가 달랐다.

## 실제로 움직인 것은 한쪽뿐이었다

summarize-history는 no-experience baseline을 개선하지 못했다.

`P(summarize)`는 오히려 `0.250 → 0.230`으로 소폭 낮아졌고, 10회 반복에서도 baseline과 비교해 4승 2무 4패였다.

즉 summarize-history가 모델을 summarize 방향으로 일관되게 움직였다고 보기 어려웠다.

반면 export-history에서는 전혀 다른 현상이 나타났다.

`P(export)=1.000`, `H=0.000`이 되었고, 10/10 runs에서 예외 없이 export로 완전히 수렴했다.

따라서 `delta_sum=+0.23`이 나온 이유는 summarize-history가 summarize 방향으로 성공적으로 이동했기 때문이 아니었다.

**export-history 한쪽만 100% export로 붕괴하면서 두 조건 사이의 차이가 커진 것이었다.**

즉,

> 양의 delta가 관찰됐다는 사실과 양방향 semantic transfer가 일어났다는 주장은 같은 의미가 아니었다.

이 점이 B7d.4에서 가장 예상 밖이었던 부분이다.

## 왜 중요했나

이 결과를 delta 하나만 보고 평가했다면 v2가 historical semantic content를 성공적으로 전달했다고 잘못 결론내릴 수 있었다.

하지만 run-level 분포와 baseline을 함께 보니, 관찰된 효과는 대칭적이지 않았다.

당시 ambiguous task 자체가 이미 export 쪽으로 기울어져 있었기 때문에, export-history의 강한 반응은 historical content가 제대로 전달된 결과일 수도 있었지만, 단순히 기존 export prior와 history block이 같은 방향으로 작용한 결과일 가능성도 있었다.

반대로 prior와 반대 방향인 summarize-history에서는 같은 효과가 나타나지 않았다.

따라서 B7d.4만으로는

- 실제 historical content에 대한 반응인지
- history block의 형식이나 존재가 기존 prior를 강화한 것인지
- 두 효과가 함께 섞여 있는지

구분할 수 없었다.

이 시점에서는 v2 semantic transfer가 확인됐다고 볼 수 없었고, **prior-congruent priming 또는 asymmetric content sensitivity 가능성**을 추가로 분리해야 했다(둘 다 이 단계에서는 확정된 원인이 아니라 가능한 해석/가설이었다).

## 이 결과가 바꾼 다음 실험

이 결과 때문에 v2를 바로 튜닝하거나 B7e로 넘어가지 않았다.

대신 다음 질문을 먼저 확인하기로 했다.

> export-history의 100% 수렴은 정말 `export`라는 historical content 때문인가, 아니면 action 정보가 없어도 v2 history block 자체가 기존 export prior를 강화하는가?

이를 확인하기 위해 B7d.5에서는 v2와 같은 구조를 유지하면서 action 방향만 제거한 `neutral-history` 조건을 추가했다.

즉 B7d.4의 핵심 발견은 v2의 성공 여부 자체가 아니라, **조건 간 delta만으로 semantic transfer를 판단해서는 안 되며 baseline과 각 방향의 실제 이동을 따로 봐야 한다는 것**이었다.

상세한 run-level 결과, experiment identity, token usage 및 structural checks는 [`docs/experiments/agent_connected_eval.md`](../experiments/agent_connected_eval.md)의 B7d.4 기록을 참조.
