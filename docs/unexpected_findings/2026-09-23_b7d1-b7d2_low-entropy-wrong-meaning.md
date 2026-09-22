# B7d.1/B7d.2 — Entropy는 0까지 떨어졌는데 의미는 틀렸다

**날짜:** 2026-09-23
**experiment ID:** b7d1_semantic_transfer_diagnostic, b7d2_structure_only_control
**관련 커밋:** `f572673` (B7d.1 기록), `c35770a` (B7d.2 기록) — 두 실험 모두 지금 쓰는
`scenario_fingerprint()`/exact-string scenario 체계가 생기기 *이전*의 ad-hoc 스크립트로
돌린 결과라, "이 SHA의 코드로 재현된다"는 의미의 git_sha는 없다. 여기 적은 커밋은 그
결과가 저장소에 기록된 시점이다. 자세한 원자료는
[`docs/experiments/agent_connected_eval.md`](../experiments/agent_connected_eval.md) §7–§11
참고.

## 무슨 실험이었나

`ExperienceAwareDelegate`가 과거 Principal-confirmed experience(예: "이 Principal은
이전에 summarize로 확인해줬다")를 현재 delegation의 context에 넣어주면, Delegate가 다음에
같은 종류의 애매한 delegation을 받았을 때 그 경험을 반영해서 더 정확하게(=Principal이
의도한 쪽으로) 해석할 거라고 기대하고 만든 기능이다. B7d.1/B7d.2는 이게 실제로 작동하는지
확인하려고 만든 첫 real-API 진단이다 — no-experience 조건과 summarize-experience/
export-experience/read-experience 조건, 그리고 (B7d.2에서 추가한) 아무 내용도 없는
neutral 조건을 비교했다.

## 기대했던 결과

historical experience의 confirmed action을 summarize로 넣으면 현재 delegation에서도
`P(summarize)`가 올라갈 것 — 즉 experience의 *내용*이 실제로 응답 방향을 바꿀 것이라고
기대했다.

## 실제로 나온 결과

- No experience: `H=0.644`, `P(summarize)=0.17`, `P(export)=0.83`
- summarize-experience: `H=0.000`, `P(summarize)=0.00` (5/5 runs 전부 export로 확정)
- export-experience: `H=0.000`, `P(summarize)=0.00` — **summarize-experience와 완전히
  동일한 결과**
- read-experience: `H=0.878`, `P(summarize)=0.70` (전혀 다른 방향)

summarize 조건만 놓고 보면 `H: 0.644 → 0.000`으로 entropy는 극적으로 줄었는데,
`P(summarize): 0.17 → 0.00`으로 오히려 더 나빠졌다 (ΔP = −0.17). 그리고
summarize-experience와 export-experience가 완전히 같은 분포를 만들어냈다는 건, historical
experience의 실제 내용(summarize였는지 export였는지)이 결과에 거의 영향을 주지 못했다는
뜻이다. 뒤이은 B7d.2(아무 내용 없는 neutral history block)도 summarize/export 조건과
거의 비슷하게 움직여서, 이게 "내용"이 아니라 "historical block이 있다는 사실 자체"가
모델의 기존 prior(export)를 강화하는 효과라는 걸 다시 확인했다.

## 왜 의외였나

entropy가 0으로 떨어지는 건 보통 "모델이 확신을 가졌다"는 신호로 읽기 쉽다. 그런데 여기선
그 확신이 틀린 방향으로 향했다 — 심지어 "정답(summarize)을 알려주는 experience"를 넣었을
때 오히려 정답 확률이 떨어졌다. entropy만 보고 있었다면 "experience가 효과가 있다"고
잘못 결론 내렸을 상황이다.

## 지금 해석

이 실험은 uncertainty(entropy `H`)와 semantic alignment(`P(Principal-confirmed action)`)가
서로 다른 것이고, 둘 중 하나만 보고 성공/실패를 판단하면 안 된다는 걸 보여주는 첫 번째
직접 증거다. entropy 감소는 "모델이 무언가로 수렴했다"는 뜻일 뿐, "그 수렴한 곳이 맞다"는
뜻이 아니다.

## DualFlow 연구에 왜 중요한가

이 발견 이후로 모든 B7d 계열 실험에서 entropy를 주 지표로 쓰지 않고, `P(target action)`을
primary, entropy를 secondary로 명시적으로 분리해서 보고하게 됐다. 이게 이 프로젝트의
평가 방법론 전체를 바꾼 첫 계기였다.

## 다음 실험/결정에 어떤 영향을 줬나

- B7c/B7d의 verified-experience representation을 "confirmed action을 한 줄 라벨로
  보여주는" 방식(v1)에서 재설계(v2, B7d.3)하도록 이끈 직접적인 동기가 됐다.
- 이후 모든 diagnostic 스크립트(`experience_transfer.py` 등)가 entropy와 P(action)을
  항상 같이, 그리고 별도로 보고하도록 설계됐다.
