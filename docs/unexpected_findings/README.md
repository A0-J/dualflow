# Unexpected Findings

이 폴더는 실험이 **사전에 예상/등록했던 결과와 실제로 다르게 나왔고, 그 차이가 해석·실험
설계·다음 연구 방향을 바꾼 경우에만** 기록을 남기는 곳이다. 일반적인 실패 로그가 아니다.

## 언제 여기에 기록하나

두 조건을 **모두** 만족할 때만 새 노트를 만든다.

- 관측된 결과가 preregistered/예상된 행동과 실질적으로 다르다, **그리고**
- 그 예상 밖 결과가 해석, 실험 설계, 또는 다음 연구 결정을 바꾼다.

## 언제 여기에 기록하지 않나

아래는 이 폴더에 남기지 않는다 (일반 실패 로그로 취급하지 않음):

- 단순히 실패한 시도
- 구현 실수
- 디버깅 과정
- 설정(config) 오류
- 중간에 폐기된 접근
- 연구적으로 의미 없는 통상적인 negative result

이런 것들은 필요하면 커밋 메시지나 `docs/experiments/agent_connected_eval.md`의 해당
실험 섹션 안에서 다뤄지고, 별도 파일로 만들지 않는다.

## 파일명 형식

```
YYYY-MM-DD_<experiment-id>_<short-title>.md
```

예: `2026-09-23_b7d4_asymmetric-prior-priming.md`

## 각 노트에 들어가는 내용

1. 날짜 / experiment ID / git SHA
2. 실험 의도
3. 예상했던 결과
4. 실제 결과 (핵심 수치만 — raw table/log는 반복하지 않고
   `docs/experiments/agent_connected_eval.md`의 해당 섹션을 참조)
5. 무엇이 왜 예상과 달랐는지
6. 현재 해석
7. DualFlow 연구에 왜 중요한지
8. 이 결과로 바뀐 결정 / 다음 실험

톤은 사실 위주로 담백하게 — negative result를 성공처럼 포장하지 않고, 그렇다고 모순되는
증거를 지우지도 않는다.
