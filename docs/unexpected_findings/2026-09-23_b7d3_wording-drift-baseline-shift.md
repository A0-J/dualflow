# B7d.3 — 문구 하나를 바꿨을 뿐인데 baseline 자체가 달라졌다

**날짜:** 2026-09-23
**experiment ID:** b7d3_representation_validation (오염된 pilot) → b7d3_canonical_rerun (재검증)
**git SHA:** 오염된 pilot은 `7081312`(스크립트 최초 커밋) 기준, 원인 확인·수정은 `490f5a3`,
재검증(canonical rerun)은 `a714e36` 기준.
자세한 원자료는
[`docs/experiments/agent_connected_eval.md`](../experiments/agent_connected_eval.md) §15–§16
참고.

## 무슨 실험이었나

B7d.3의 real-API pilot에서 `experience_transfer.py`의 `build_current_context()`가
scenario 파일의 개별 필드(resource, scope, temporal_label 등)로부터 현재 delegation의
environment context 문장을 코드로 조립하도록 짜여 있었다. 원래 B6.1/B7a 때부터 실제
검증에 쓰인 문장("The September 2026 report is located at /reports/2026-09/.")과
"의미상 동일하다"고 판단한 문장("The report for the current period is located at
/reports/2026-09/.")으로 재구성한 것이었다.

## 기대했던 결과

두 문장은 사람이 읽기엔 뜻이 완전히 같다. 그래서 이 재구성이 실험 결과에 유의미한 영향을
줄 거라고는 전혀 예상하지 않았다 — v1/v2 experience representation의 효과를 비교하는 게
목적이었지, context wording 자체를 실험 변수로 다룰 생각이 없었다.

## 실제로 나온 결과

- 오염된 wording으로 돌린 pilot: `no_experience` baseline이 **5번 run 전부** `H=0.000`,
  `P(export)=1.00`으로 완전히 결정론적이었다.
- 같은 코드/같은 scenario를 canonical wording("The September 2026 report is located
  at...")으로 되돌려서 다시 돌리니, `no_experience` baseline이 **더 이상 결정론적이지
  않았다** — 5번 run 모두 `H`가 0.469~0.811 사이로 실제 편차를 보였고, `P(summarize)`도
  0.10~0.25로 자연스러운 ambiguity가 살아 있었다.

즉 baseline 자체가 "결정론적인 export 확신"에서 "진짜 애매함이 남아있는 분포"로 완전히
바뀌었다 — 코드도, scenario의 의미도 안 바꿨는데, 딱 그 한 문장만 원래대로 되돌렸을 뿐인데.

## 왜 의외였나

이 실험의 목적은 experience representation(v1 vs v2)의 효과를 측정하는 것이었다. 그런데
실제로는 그보다 훨씬 작아 보이는 문구 재구성이 baseline 자체를 완전히 다른 분포로
바꿔놓을 만큼 더 큰 요인이었다. "의미상 동일한 paraphrase는 실험적으로도 동일하게
취급해도 된다"는 암묵적 가정이 틀렸다는 걸 실측으로 확인한 셈이다.

## 지금 해석

모델(gpt-4o-mini)은 겉보기에 의미가 같은 두 문장에도 서로 다르게 반응할 수 있다. 특히
지금처럼 애매함(ambiguity) 자체가 측정 대상인 실험에서는, prompt wording의 미세한 차이가
측정하려는 신호(experience의 효과)보다 더 큰 confound가 될 수 있다.

## DualFlow 연구에 왜 중요한가

이 발견이 `docs/REPRODUCIBILITY.md`를 만들게 된 직접적인 계기다. 이후로:

- scenario 파일이 model-visible한 delegation/context 문자열을 코드가 조립하는 필드가
  아니라 **문자열 그 자체로 저장**하게 됐다 (`external_audit_finance.json`
  `scenario_version: 1.1`).
- `build_current_context()`가 조립 로직에서 단순 passthrough로 바뀌었다.
- 모든 diagnostic 스크립트가 실행 시작 시 delegation/context의 SHA256 fingerprint를
  출력하게 됐다 (`scenario_fingerprint()`).
- `tests/test_scenario_reproducibility.py`로 scenario 파일의 문자열이
  `agent_smoke.py`의 검증된 상수와 byte-identical한지 회귀 테스트로 고정했다.

## 다음 실험/결정에 어떤 영향을 줬나

B7d.3의 v1-vs-v2 비교 자체를 canonical wording으로 다시 돌려야 했다(B7d.3 canonical
rerun, §16) — 오염된 pilot의 "5/5 일관된 개선" 수치는 무효화됐고, 재검증 결과는 훨씬 약하고
불안정한 신호로 나타났다(이 자체도 §16과 별도로 중요한 negative finding이지만, 이 노트의
핵심은 "wording이 confound였다"는 방법론적 교훈 쪽이다). 이후 B7d.4/B7d.5 등 모든 후속
실험이 이 canonical, fingerprint된 scenario 위에서만 진행되고 있다.
