# Baselines and Related Work

이 문서는 DualFlow에서 사용하는 비교 베이스라인과 선행 연구의 역할을 정리한다.  
목적은 특정 논문을 비판하는 것이 아니라, **각 방법이 어떤 신호를 사용해 실행 여부를 판단하는지**, 그리고 **agent-to-agent delegation 환경에서 어떤 정보가 추가로 필요한지**를 명확히 구분하는 데 있다.

---

## 1. Overview

DualFlow는 세 계열의 선행 아이디어를 참고한다.

| Work | 핵심 아이디어 | DualFlow에서의 역할 |
|---|---|---|
| **SAGE-Agent** | 구조화된 uncertainty와 EVPI를 이용한 clarification | Semantic Flow의 비교 베이스라인 |
| **SAGE-Bench** | SOP graph 기반 path-level execution evaluation | Joint Verification의 path-level reference |
| **ChainCaps** | capability budget의 monotonic attenuation | Authority Flow의 non-amplification 원칙 |

세 방법은 서로 다른 문제를 다룬다.

- **SAGE-Agent**: 요청이 얼마나 불확실한가?
- **SAGE-Bench**: 실행 경로가 기준 SOP와 얼마나 일치하는가?
- **ChainCaps**: 위임 과정에서 권한이 확대되지 않는가?

DualFlow는 이들을 하나의 점수로 합치는 대신, **Semantic Verification과 Authority Verification을 분리한 뒤 실행 직전에 Joint Verification을 수행한다.**

---

## 2. SAGE-Agent baseline

### 2.1 Why this baseline?

SAGE-Agent는 tool-augmented LLM이 불완전한 요청을 받을 때, 구조화된 uncertainty를 계산하고 질문의 가치를 평가해 clarification 여부를 결정한다.

DualFlow의 Semantic Flow와 가장 직접적으로 비교되는 부분은 다음 세 요소다.

1. tool-call parameter의 불확실성 표현
2. 질문 선택을 위한 value-of-information 계산
3. uncertainty가 충분히 낮을 때 실행으로 전환하는 stopping rule

본 저장소의 `src/dualflow/sage_baseline.py`는 이 비교를 위해 SAGE-Agent의 핵심 수식을 별도 구현한다. DualFlow의 entropy 구현을 SAGE 방식으로 변형한 것이 아니라, **비교 대상의 실행 게이트를 독립적으로 재현**하는 것이 목적이다.

### 2.2 Reproduced components

구현 대응은 다음과 같다.

| SAGE-Agent component | Repository implementation |
|---|---|
| parameter viability score \(\pi_c\) | `pi()` |
| EVPI | `evpi()` |
| repeated-aspect cost | `cost()` |
| execution threshold \(\tau_{exec}\) | `SageAgentBaseline.run()` |
| iterative clarification | `SageAgentBaseline.run()` |

현재 구현에서 중요한 순서는 다음과 같다.

```text
candidate construction
        ↓
max π ≥ τ_exec ?
   ├─ yes → execute
   └─ no
        ↓
     compute EVPI
        ↓
 clarification / stopping decision
```

따라서 \(\tau_{exec}\) 조건을 이미 만족한 후보에 대해서는 clarification scoring이 실행되지 않는다.

---

## 3. Observations from the SAGE-Agent reproduction

아래 항목은 SAGE-Agent 전체의 일반적 실패를 주장하기 위한 것이 아니라, **논문에 기술된 수식을 본 저장소의 controlled scenarios에 적용했을 때 확인한 동작 특성**이다. 각 항목은 regression test로 고정해 두었다.

### 3.1 Parameter completeness and value correctness are different

재현된 viability score에서 지정된 parameter는 높은 점수를 받을 수 있지만, 그 값이 실제 의도와 맞는지는 별도 정보가 없으면 알 수 없다.

예를 들어 다음 두 후보가 모두 완전히 지정되어 있다면:

```text
read /reports/
read /etc/passwd/
```

parameter completeness만으로는 둘을 구분할 수 없다.

관련 테스트:

```text
test_value_correctness_is_invisible
```

DualFlow에서는 이 문제를 **semantic certainty와 delegated authority를 서로 다른 검증 축으로 분리해야 하는 이유**로 본다.

---

### 3.2 Tool-choice uncertainty can remain after parameters are fully specified

서로 다른 tool/action 후보가 모두 완전히 지정된 경우, parameter completeness만으로는 어떤 tool이 맞는지 결정되지 않을 수 있다.

본 benchmark의 `silent_misread`는 이러한 상황을 재현한다.

관련 테스트:

```text
TestToolChoiceBlindSpot
```

이는 다음을 의미한다.

> Parameter-level certainty does not necessarily imply intent-level certainty.

---

### 3.3 Early execution gates can bypass later clarification logic

현재 재현에서는 \(\max \pi \ge \tau_{exec}\) 조건이 만족되면 EVPI 계산 전에 실행이 결정된다.

따라서 후보 분포가 하나의 잘못된 해석으로 수렴하도록 조작된 controlled attack에서는 clarification 단계가 호출되지 않는다.

본 문서에서는 이를 **semantic-proposal manipulation**으로 부른다.

```text
wrong proposal
+ apparently complete specification
        ↓
execution gate satisfied
        ↓
no clarification
```

관련 테스트:

```text
test_evpi_is_never_computed_under_attack
test_attack_works_under_both_readings_of_eq2
```

이 관찰은 DualFlow에서 uncertainty score 하나를 최종 실행 권한으로 사용하지 않고, Authority Flow를 독립적으로 유지하는 동기다.

---

### 3.4 Information-gain question selection must be recomputed after feedback

DualFlow의 Semantic Flow는 Shannon entropy reduction을 기준으로 clarification question을 선택한다.

\[
IG(q) = H(p) - \mathbb{E}_{r}[H(p \mid r)]
\]

질문 하나에 대한 정보이득은 이전에 어떤 답을 얻었는지에 따라 달라질 수 있으므로, 질문 순서를 고정하지 않고 **매 clarification round에서 information gain을 다시 계산**한다.

이 항목은 특정 baseline을 대체하기 위한 주장이라기보다 DualFlow의 질문 선택 루프에 대한 설계 원칙이다.

---

## 4. SAGE-Bench and Joint Verification

SAGE-Bench는 SOP를 graph로 표현하고 agent trajectory를 graph path와 비교한다. DualFlow는 이 아이디어를 Joint Verification의 **path-level consistency signal**로 사용한다.

현재 live gate의 핵심은 다음 두 조건이다.

\[
Sim_{path}(p,p^*) \ge \tau
\]

그리고

\[
Terminal(p) = Terminal(p^*)
\]

여기서 terminal decision은 action type이 아니라 SOP trace의 최종 판정이다.

```text
EXECUTE
ESCALATE
REJECT
```

### Why both signals?

`Sim_path`만 사용할 경우 reference path를 포함하면서 더 깊이 진행한 경로가 높은 similarity를 받을 수 있다.

반대로 terminal decision만 비교하면 서로 다른 내부 경로가 같은 `EXECUTE`로 끝나는 경우를 구분하지 못한다.

따라서 현재 implementation은:

```text
path similarity
        AND
terminal decision match
```

를 사용한다.

### Important boundary

`V_action`, `V_resource`, `V_scope`, `V_condition`과 같은 exact-field equality는 구현되어 있지만 **live decision gate에는 사용하지 않는다**.

이 검사는 benchmark ground truth와 직접 비교할 경우 evaluation oracle이 될 수 있으므로, 현재 저장소에서는 ablation / diagnostic analysis에만 사용한다.

---

## 5. ChainCaps and Authority Flow

ChainCaps의 핵심 아이디어는 capability가 tool composition 또는 delegation을 거치면서 **확대되지 않아야 한다**는 것이다.

DualFlow의 Authority Flow는 이 원칙을 agent-to-agent delegation에 적용한다.

Delegation chain에서 authority budget은 매 hop마다 intersection으로 합성된다.

\[
C_{next} = C_{current} \cap C_{ceiling}
\]

따라서:

\[
C_n \subseteq C_{n-1} \subseteq \cdots \subseteq C_A
\]

가 구조적으로 유지된다.

DualFlow에서 non-amplification은 별도의 classifier가 아니라 다음 두 메커니즘으로 보장된다.

1. delegation budget composition은 intersection만 사용
2. negotiated / adaptively restricted proposal은 현재 budget에 대해 다시 authority check 수행

### Deployment implication

이 방식도 authority manifest 또는 budget 자체가 정확하다는 가정에 의존한다.

ChainCaps가 manifest quality를 주요 deployment bottleneck으로 보고한 것과 마찬가지로, DualFlow에서도 실제 시스템 적용 시 다음이 중요하다.

- budget authoring
- scope linting
- condition specification
- stale policy detection

즉 Authority Flow는 잘못 작성된 정책을 자동으로 정답 정책으로 바꾸는 메커니즘이 아니다.

---

## 6. Positioning of DualFlow

세 선행 방법과 DualFlow의 차이는 다음처럼 정리할 수 있다.

| Question | SAGE-Agent | SAGE-Bench | ChainCaps | DualFlow |
|---|:---:|:---:|:---:|:---:|
| 요청 의미의 불확실성 | ✓ |  |  | ✓ |
| clarification | ✓ |  |  | ✓ |
| SOP/path-level consistency |  | ✓ |  | ✓ |
| delegated authority bound |  |  | ✓ | ✓ |
| non-amplifying delegation |  |  | ✓ | ✓ |
| recoverable scope negotiation |  |  |  | ✓ |
| verified-history-based feedback reduction |  |  |  | ✓ |

DualFlow의 목표는 기존 방법 중 하나를 대체하는 것이 아니다.

핵심 구분은 다음이다.

```text
Semantic verification:
    "What does the request mean?"

Authority verification:
    "What is the delegate allowed to do?"

Joint verification:
    "Does the final proposal remain consistent with both?"
```

그리고 `scope_exceeded`처럼 **권한 자체는 존재하지만 제안 범위가 과도한 경우에만** bounded Authority Feedback을 허용한다.

```text
no_grant          → hard reject
condition_missing → hard reject
scope_exceeded    → feedback / restriction
valid             → continue
```

이 경계는 feedback이 새로운 권한을 만들어내는 통로가 되지 않도록 하기 위한 설계다.

---

## 7. How to interpret the baseline results

본 저장소의 baseline 실험은 다음 목적을 가진다.

- 논문 수식의 controlled reproduction
- DualFlow가 해결하려는 failure mode의 분리
- Semantic-only / authority-only 판단의 한계 확인
- Joint Verification과 Authority Feedback의 추가 효과 측정

현재 결과는 소규모 deterministic pilot benchmark를 기반으로 하므로, 실제 LLM 환경에서의 일반화를 주장하기 위한 최종 external evaluation은 아니다.

따라서 baseline 결과는 **mechanism-level evidence**로 해석한다.

향후 external evaluation에서는 실제 LLM proposal generation, 실제 multi-turn feedback, 더 큰 agent benchmark를 사용해 동일한 비교를 수행할 예정이다.

---

## References

- Suri, M. et al. **Structured Uncertainty guided Clarification for LLM Agents.** arXiv:2511.08798, 2025.
- Shi, L. et al. **SAGE: A Service Agent Graph-guided Evaluation Benchmark.** arXiv:2604.09285, 2026.
- Jiang, X. et al. **ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation.** arXiv:2605.26542, 2026.

---

## Repository pointers

- `src/dualflow/sage_baseline.py` — SAGE-Agent baseline reproduction
- `src/dualflow/rule_engine.py` — SOP graph tracing and path matching
- `src/dualflow/capability.py` — authority budget and delegation attenuation
- `src/dualflow/authority_feedback.py` — bounded scope negotiation and verified authority history
- `tests/test_sage_baseline.py` — SAGE-Agent reproduction tests
- `docs/EXPERIMENTS.md` — experimental protocol and results
- `docs/DESIGN_NOTES.md` — design decisions and oracle-analysis notes
