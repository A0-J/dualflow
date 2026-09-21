[English](README.md) | 한국어

# DualFlow

**안전한 Agent-to-Agent 위임을 위한 Semantic·Authority 검증**

DualFlow는 실행 전에 agent-to-agent 위임을 검증하는 연구용 프로토타입이다. 하나의 confidence 점수로 뭉뚱그리면 안 되는 두 질문을 분리한다:

1. **Semantic 검증** — delegate가 principal의 요청을 정확히 이해했는가?
2. **Authority 검증** — 제안된 실행이 실제로 위임된 권한 안에서 허용되는가?

현재 core는 이 두 검증을 독립적인 verifier 컴포넌트로 노출한다:

- `SemanticVerifierAgent` → `SemanticVerdict`
- `AuthorityVerifierAgent` → `AuthorityVerdict`

두 결과는 `framework.py`의 결정론적(deterministic) joint-verification 레이어가 합친다.

> **현재 상태**
>
> Verifier-agent 아키텍처는 구현돼 있지만, 저장소는 아직 주로 결정론적 시뮬레이션 시나리오로 평가한다. **독립적인 실제 production LLM agent A/B로 하는 end-to-end 검증은 아직 아니다.** 다음 평가 단계에서 실제 agent 호출을 연결하고, runtime matching에 남아 있는 시뮬레이션 전용 intent oracle을 제거하고, 주요 실험 결과를 다시 생성할 예정이다.

---

## 동기

위임은 두 가지 다른 방식으로 실패할 수 있다.

### Semantic mismatch

delegate가 확신에 차서 요청을 잘못 이해할 수 있다.

```text
Principal:        "지난달 리포트를 읽어줘."
Delegate 해석:     /reports/2025-08/
```

불확실성이 낮다고 해석이 맞다는 보장은 없다.

### Authority mismatch

delegate가 요청을 정확히 이해했더라도, 실제로 위임된 권한 밖의 행동을 제안할 수 있다.

```text
위임된 scope:  read /reports/2026-08/*
제안된 scope:  read /reports/*
```

해석이 명확하다고 그 행동이 자동으로 승인되는 것은 아니다.

그래서 DualFlow는 실행 전에 semantic과 authority를 독립적으로 검증한다.

---

## 아키텍처

```text
                 Agent A (Principal)
                         |
                    delegation
                         v
                  Agent B (Delegate)
                         |
                  proposed action
                         |
             +-----------+-----------+
             |                       |
             v                       v
   SemanticVerifierAgent    AuthorityVerifierAgent
             |                       |
      SemanticVerdict          AuthorityVerdict
             |                       |
             +-----------+-----------+
                         |
                         v
               Deterministic Fusion
                  /             \
             EXECUTE           REJECT
```

### Semantic Verifier Agent

`SemanticVerifierAgent`는 `semantic.py`에 구현돼 있다.

기존 semantic 검증 전략을 그대로 소유한다:

- **Fast** — entropy/역질의 기반 검증
- **Slow** — principal 검토
- **AND** — 두 경로 모두 동의해야 함
- **Adaptive** — Fast를 먼저 시도하고, 필요할 때만 에스컬레이션

verifier는 `SemanticVerdict`를 반환한다. 이번 리팩터는 기존 Fast/Slow/AND/Adaptive 구현의 동작을 그대로 보존한다 — agent 추상화 자체가 새 LLM 호출을 추가하지는 않는다.

### Authority Verifier Agent

`AuthorityVerifierAgent`는 `authority_feedback.py`에 구현돼 있다.

authority 검증 시퀀스 전체를 소유한다:

1. 결정론적 authority 검사,
2. 협상이 허용될 때의 bounded authority feedback,
3. 수정된 해석의 재검증,
4. `AuthorityVerdict` 생성.

low-level authority primitive는 계속 결정론적으로 `capability.py`에 남아 있다.

### 결정론적 capability 레이어

`capability.py`는 Authority Verifier Agent가 쓰는 authority primitive를 제공한다:

- `Budget`
- budget intersection / attenuation
- delegation-chain 합성
- non-amplification
- `check_authority()`

위임된 capability는 delegation chain을 따라 내려가면서 더 넓어질 수 없다.

### Joint verification

`framework.py`가 orchestration과 최종 fusion을 소유한다. rule engine은 결정론적 structured-intent matching만 제공한다 — 세 번째 verifier agent가 아니다.

의도한 배포 아키텍처는 다음과 같다:

```text
Semantic evidence  -> SemanticVerifierAgent  -> SemanticVerdict
Authority evidence -> AuthorityVerifierAgent -> AuthorityVerdict
                                            \ /
                                  deterministic fusion
```

향후 배포에서는 각 verifier가 같은 API client를 공유하면서도 자신만의 model call과 tool context를 가질 수 있다.

---

## 지금 구현돼 있는 것

현재 저장소가 구현하는 것:

- 독립적인 Semantic/Authority verifier 인터페이스,
- 결정론적 capability 검사와 delegation attenuation,
- bounded authority feedback,
- semantic Fast / Slow / AND / Adaptive 전략,
- 최적화 레이어로서의 verified-authority 재사용,
- 결정론적 joint verification,
- SAGE-Agent 비교 구현,
- 결정론적 benchmark와 진단 실험.

이번 core 리팩터는 의도적으로 behavior-preserving이다. verifier-agent 추상화는 기존 구현에서 동작을 바꾸지 않고 추출한 것이다 — 현재 benchmark 출력을 바꾸지 않았다.

---

## 중요한 평가 경계 (Evaluation Boundary)

현재 실험은 **메커니즘 테스트**이지, 아직 production-agent 검증이 아니다.

통제된 benchmark는 시뮬레이션된 task 상태와 시뮬레이션된 principal을 쓴다. ground truth는 평가용으로, 그리고 principal이 실제로 아는 것을 시뮬레이션하는 용도로는 유효하다.

하지만 현재 브랜치의 최종 joint matching에는 아직 시뮬레이션 전용 의존성이 남아 있다 — runtime intent reference가 `DelegationTask.intent_fields`를 거쳐 `task.truth`에서 유도된다. 배포된 verifier는 그 숨겨진 truth에 접근할 수 있다고 가정할 수 없다.

이게 부수적인 게 아니라 실제로 그 탐지력의 근원이라는 것을, 직접 제거해봐서 확인했다. Semantic Verifier 자신이 resolve한 해석으로 이 reference를 대체하면 misread 탐지가 정확히 무너진다 — "delegate의 해석 vs delegate의 해석"이 되어, Authority가 제안을 독립적으로 바꾸지 않는 한 무조건 tautology가 된다. Experiment 1의 `Full = 0% unsafe`가 이 대체 아래에서는 11.1%(`silent_misread` 케이스)가 된다.

그래서 다음 research-behavior 변경은 runtime joint 검사가 `task.truth`가 아니라 resolve된 `SemanticVerdict`를 semantic reference로 쓰게 만드는 것이다 — `task.truth`는 평가용으로만 남긴다. 그리고 배포 환경에서 독립적인 두 번째 semantic 신호를 어디서 얻을지도 설계해야 한다(principal에게 다시 확인받는 것이 가장 명확한 후보이며, 이는 Adaptive Feedback이 이미 다루는 것과 같은 비용-안전성 트레이드오프를 다시 연다).

이 변경과 실제 Agent A/B 평가가 끝나기 전까지는, 기존 pilot 수치를 논문의 최종 main-result 증거가 아니라 통제된 sanity check/ablation으로 취급해야 한다.

---

## 빠른 시작

```bash
pip install -e ".[dev]"
pytest -q
```

현재 리팩터 기준선:

```text
183 passed
```

최소 데모 실행:

```bash
python experiments/demo.py
```

canonical 통제 benchmark 실행:

```bash
python experiments/benchmark.py
```

유지 중인 실험 그림 생성:

```bash
python experiments/plots.py --outdir figures
```

entropy probe 옵션 확인:

```bash
python experiments/entropy_probe.py --help
```

---

## 실험

실험 스크립트는 의도적으로 core package 밖에 둔다.

```text
experiments/
├── demo.py
├── benchmark.py
├── entropy_probe.py
└── plots.py
```

### `demo.py`

현재 DualFlow 파이프라인의 최소 end-to-end 예제.

### `benchmark.py`

core 리팩터 동안 쓰는 canonical 통제 benchmark.

### `entropy_probe.py`

semantic uncertainty 동작을 보는 독립 진단 실험.

### `plots.py`

유지 중인 실험 그림을 생성한다. plotting은 `dualflow` package와 분리돼 있어 core 구현이 시각화 코드에 의존하지 않는다.

더 이상 main 평가에 포함되지 않는 과거 pilot/진단 결과는 `docs/EXPERIMENTS.md`에 기록돼 있다.

---

## 저장소 구조

```text
src/dualflow/
├── capability.py
├── semantic.py
├── authority_feedback.py
├── rule_engine.py
├── framework.py
├── llm.py
├── sage_baseline.py
└── ...

experiments/
├── demo.py
├── benchmark.py
├── entropy_probe.py
└── plots.py

docs/
tests/
figures/
```

### Core 모듈

| 모듈 | 책임 |
| --- | --- |
| `semantic.py` | Semantic 표현과 `SemanticVerifierAgent` |
| `authority_feedback.py` | `AuthorityVerifierAgent`, bounded authority feedback, verified-authority 재사용 |
| `capability.py` | 결정론적 capability/budget/delegation primitive |
| `rule_engine.py` | 결정론적 structured-intent compatibility 검사 |
| `framework.py` | End-to-end orchestration과 결정론적 fusion |
| `llm.py` | 모델을 향한 인터페이스/어댑터 |
| `sage_baseline.py` | SAGE-Agent 비교 구현 |

리팩터 브랜치에 `bench.py`가 아직 남아 있다면, 이건 legacy v0 benchmark 스캐폴딩이고 canonical benchmark가 **아니다**. canonical 실험 진입점은 `experiments/benchmark.py`다.

---

## 현재 한계

현재 저장소는 아직 production-ready한 multi-agent 안전성의 증거로 읽으면 안 된다.

- canonical 실험에서 Agent A와 Agent B는 아직 독립적인 실제 LLM agent로 연결돼 있지 않다.
- verifier agent들은 구조적으로는 독립적이지만, 현재 통제된 실험은 아직 두 번의 독립적인 production model call을 요구하지 않는다.
- 통제된 benchmark에서 semantic candidate 생성과 principal 행동은 여전히 시뮬레이션이다.
- authority 검사는 현재 명시적인 `Budget`/capability 상태 위에서 동작한다 — 자연어 조직 정책에서 policy를 retrieve하는 것은 향후 과제다.
- 위에서 설명한 대로, 현재 runtime joint match에는 여전히 시뮬레이션에서 유도된 ground-truth 배선이 남아 있다.
- benchmark는 자연스러운 task 분포가 아니라 작고 통제된 규모다.
- 기존 pilot figure는 regression 테스트와 mechanism ablation에는 유용하지만, agent-pair 아키텍처에 대한 최종 main 실험 증거로 취급하면 안 된다.

---

## 다음 단계

1. **runtime ground-truth oracle 제거**
   - 최종 semantic matching을 `task.truth`가 아니라 `SemanticVerdict`에서 유도;
   - ground truth는 채점과 시뮬레이션용으로만 유지.

2. **실제 Agent A와 Agent B 연결**
   - 위임과 제안을 독립적인 model call로 생성;
   - Semantic과 Authority verifier의 context를 분리 유지.

3. **배포를 위한 AuthorityVerifierAgent 확장**
   - policy retrieval,
   - delegation/identity lookup,
   - resource-state 검사,
   - evidence-backed authority 판단,
   - 명시적인 unknown/escalation 처리.

4. **새 agent-connected benchmark 실행**
   - 검증 없음,
   - semantic-only,
   - authority-only,
   - 단일 verifier/joint-prompt baseline,
   - DualFlow 독립 verifier pair + 결정론적 fusion.

5. **논문용 figure 재생성**
   - 정답 실행률,
   - unsafe execution/attack 성공률,
   - false rejection률,
   - 역질의/escalation률,
   - LLM 호출 수,
   - latency,
   - token 비용,
   - mechanism ablation.

---

## 문서

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 아키텍처와 안전성 불변식
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — 통제된 실험과 과거 pilot 결과
- [`docs/BASELINES.md`](docs/BASELINES.md) — baseline 노트
- [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) — 구현·연구 설계 노트

일부 문서는 아직 agent-pair 이전 구현을 반영하고 있으며, 이번 리팩터의 일부로 갱신 중이다.

---

## 참고 문헌

이 저장소는 현재 다음 연구의 아이디어를 논의하거나 재현한다:

- ChainCaps: *Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation* (arXiv 2605.26542)
- SAGE-Agent: *Structured Uncertainty guided Clarification for LLM Agents* (arXiv 2511.08798, Findings of ACL 2026)
- SAGE-Bench: *A Service Agent Graph-guided Evaluation Benchmark* (arXiv 2604.09285)

선행 연구와 DualFlow 구현의 정확한 대응 관계는 문서를 참고할 것.

---

## 라이선스

MIT
