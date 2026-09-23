# B7d.1–B7d.2: 불확실성은 줄었지만 의미는 더 틀려졌다

**날짜:** 2026-09-23
**experiment ID:** b7d1_semantic_transfer_diagnostic, b7d2_structure_only_control

B7d.1과 B7d.2에서는 Principal이 이전에 확인한 experience를 제공하면, ambiguous delegation을 해석할 때 그 경험이 현재 의미를 올바른 방향으로 유도할 것으로 예상했다.

특히 이전 episode에서 Principal이 `summarize`를 의도했다고 확인한 experience를 제공하면, 현재 ambiguous request에서도 `summarize`의 가능성이 높아지고 uncertainty가 낮아질 것으로 예상했다.

하지만 실제 결과는 반대였다.

| Condition | Entropy (H) | P(summarize) | 핵심 관찰 |
| --- | ---: | ---: | --- |
| No experience | 0.644 | 0.17 | 기존에는 export 쪽이 우세했지만 일부 ambiguity가 남아 있었음 |
| Summarize experience | 0.000 | 0.00 | 모든 sample이 export로 수렴 |
| Export experience | 0.000 | 0.00 | summarize experience와 동일하게 export로 수렴 |
| Read experience | 0.878 | 0.70 | 다른 experience에서는 분포가 크게 달라짐 |

가장 예상 밖이었던 것은 `summarize experience` 조건이었다.

Principal이 이전 episode에서 `summarize`로 확인한 경험을 제공했는데도 `P(summarize)`는 no-experience의 0.17에서 0.00으로 떨어졌다. 반대로 entropy는 0.644에서 0.000까지 감소했다.

즉, 모델의 uncertainty는 완전히 사라졌지만 그 결과는 Principal이 확인했던 의미와 일치하지 않았다. 모델은 더 확신하게 되었지만, 그 확신은 오히려 기존의 export 방향으로 굳어졌다.

이 결과는 처음 생각했던

> experience가 uncertainty를 줄이면 semantic alignment도 함께 좋아질 것이다

라는 가정이 성립하지 않을 수 있음을 보여줬다.

특히 summarize experience와 export experience는 서로 다른 confirmed action을 담고 있었음에도 둘 다 동일하게 `P(export)=1.0`으로 수렴했다.

따라서 현재 representation에서는 historical experience의 confirmed action이 현재 action으로 일관되게 전달된다고 보기 어려웠다.

반면 read experience에서는 전혀 다른 분포가 나타났기 때문에, experience의 내용이 항상 무의미했다고 볼 수도 없다. 문제는 **경험의 의미가 현재 interpretation에 안정적이고 일관된 방식으로 전달되지 않았다는 것**이었다.

## 왜 중요했나

이 실험 전에는 entropy 감소를 semantic verification이 잘 작동하고 있다는 긍정적인 신호로 해석할 가능성이 있었다.

하지만 B7d.1–B7d.2에서는 가장 낮은 entropy가 가장 올바른 interpretation을 의미하지 않았다.

오히려 잘못된 방향으로 분포가 완전히 수렴하면서 entropy가 0이 될 수 있었다.

이 결과 이후 DualFlow에서는 다음 원칙을 명확하게 분리해서 보기 시작했다.

**Low uncertainty ≠ Correct delegation**

즉, uncertainty는 모델이 얼마나 한 해석에 집중되어 있는지를 보여줄 수는 있지만, 그 해석이 Principal의 실제 의도와 맞는지는 별도로 검증해야 한다.

또한 historical experience가 존재한다는 사실만으로 semantic transfer가 일어났다고 볼 수 없으며, experience의 내용이 현재 interpretation에 어떤 방식으로 영향을 주는지를 별도의 실험으로 확인할 필요가 생겼다.

이 예상 밖의 결과가 이후 B7d 계열에서 experience representation, prompt wording, content sensitivity를 따로 분리해 확인하게 된 출발점이 되었다.

> 참고: B7d.1–B7d.2는 현재 사용 중인 exact-string scenario 및 fingerprint 기반 reproducibility 체계가 도입되기 전에 수행된 초기 diagnostic이다. 따라서 이 파일은 당시 관찰된 현상과 그로 인해 바뀐 연구 방향을 기록하기 위한 것이며, 최신 재현성 기준에 따른 최종 성능 결과로 사용하지 않는다.

상세 실험 기록과 당시 조건은 [`docs/experiments/agent_connected_eval.md`](../experiments/agent_connected_eval.md)의 B7d.1–B7d.2 관련 섹션을 참조.
