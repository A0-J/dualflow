# B7d.4 — summarize-history는 안 먹히는데 export-history만 100% 붕괴했다

**날짜:** 2026-09-23
**experiment ID:** b7d4_semantic_content_ablation
**git SHA:** `3c03dda8894da6a1f42394add26682d1d051b2bd`
자세한 원자료는
[`docs/experiments/agent_connected_eval.md`](../experiments/agent_connected_eval.md) §18
참고 (raw per-run 표, structural check, 실험 identity 전부 거기 있음 — 여기서는 반복하지
않는다).

## 무슨 실험이었나

B7d.3에서 재설계한 experience representation(v2)이 실제로 historical experience의
*내용*(confirmed action이 summarize였는지 export였는지)에 반응하는지 확인하려고 만든
ablation이다. 같은 v2 format, 같은 애매한 September delegation을 쓰되, historical
experience의 confirmed action만 summarize/export로 바꿔서 비교했다 (N=20, 10회 반복,
600 calls).

## 기대했던 결과

v2가 정말로 historical content를 읽고 있다면, summarize-history를 넣었을 때는
`P(summarize)`가 오르고, export-history를 넣었을 때는 `P(export)`가 오르는 식으로 **양쪽이
각자의 방향으로 대응해서 움직일 것**이라고 예상했다.

## 실제로 나온 결과

- no-experience baseline: `P(summarize)=0.25`
- summarize-history: `P(summarize)=0.23` — baseline과 거의 차이 없음 (10번 중 4승 2무
  4패, 평균은 오히려 baseline보다 낮음)
- export-history: `P(summarize)=0.00`, `P(export)=1.00`, `H=0.000` — **10번 run 전부
  예외 없이 완전히 결정론적으로 export에 수렴**

`delta_sum = P(summarize|summarize-hist) − P(summarize|export-hist) = 0.23 − 0.00 =
+0.23`, `delta_export`도 같은 크기로 +0.23. 숫자만 보면 "둘 다 양수니까 성공"처럼 보인다.

## 왜 의외였나

delta 값이 대칭적으로 양수인 걸 성공의 증거로 읽으려면, summarize-history가 summarize
쪽으로 실제로 올라가는 게 확인돼야 하는데 그렇지 않았다. delta_sum이 양수인 건
summarize-history가 잘 작동해서가 아니라, **export-history만 기존 모델 prior(export
쪽으로 기울어져 있음)와 같은 방향으로 완전히 붕괴했기 때문**이다. 즉 "둘 다 양수"라는
겉모습과 "대칭적인 semantic transfer가 일어났다"는 실제 내용이 서로 다른 이야기였다 —
숫자 하나(delta)만 보고 판단했다면 잘못된 결론에 도달할 뻔했다.

## 지금 해석

v2 format이 기존 prior와 같은 방향(export)의 historical content가 들어왔을 때는 강하게
반응하지만, prior와 반대 방향(summarize)의 content에는 거의 반응하지 않는다 — 즉
prior-congruent priming / asymmetric content sensitivity에 가깝다. 이건 B7d.1/B7d.2에서
확인했던 "block의 존재/형식 자체가 기존 prior를 강화한다"는 priming 메커니즘이 v2의
재설계된 format 아래에서도 여전히 지배적일 수 있다는 뜻이다. v2가 진짜 semantic content를
읽는지, 아니면 여전히 "historical block이 있다 + 그 내용이 우연히 prior와 같다"는
조합에만 반응하는지는 아직 분리되지 않았다.

## DualFlow 연구에 왜 중요한가

verified experience가 "Principal이 실제로 의도한 의미"를 전달한다고 주장하려면, 그 효과가
모델의 기존 prior 방향과 무관하게 양방향으로 작동해야 한다. 이번 결과는 그 주장의 핵심
전제(대칭적 content sensitivity)가 아직 성립하지 않는다는 걸 보여준다 — delta 지표
하나만으로 "transfer가 작동한다"고 결론 내리면 안 된다는 방법론적 경고이기도 하다.

## 다음 실험/결정에 어떤 영향을 줬나

export-history의 100% 붕괴가 "v2 block이 존재하기만 해도 생기는 format 효과"인지,
"export 내용이 prior와 일치해서 추가로 생긴 content 효과"인지 아직 구분이 안 된다. 이걸
분리하기 위해 B7d.5(v2-neutral-format control — action 방향을 전혀 담지 않은, 하지만
구조는 v2와 동일한 neutral history block을 넣어보는 실험)를 준비했다. `render_experience_
block_v2`는 수정하지 않았고, B7e는 계속 보류 상태다.
