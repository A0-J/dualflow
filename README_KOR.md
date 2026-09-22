[English](README.md) | 한국어

# DualFlow

**LLM Agent 위임(delegation)을 위한 독립적인 Semantic·Authority 검증**

DualFlow는 **LLM agent 사이의 안전한 위임**을 연구하는 프레임워크다.

Principal Agent가 Delegate Agent에게 작업을 위임(delegation)할 때, 서로 다른 두 종류의 실패가 일어날 수 있다.

1. Delegate가 **Principal의 의도를 잘못 해석**하거나,
2. Delegate가 **위임받은 권한 밖의 행동**을 시도하는 경우다.

DualFlow는 이 두 문제를 독립적인 두 검증 경로로 분리한다.

- **Semantic Verifier** — Delegate가 제안한 행동이 위임된 의도와 실제로 일치하는지 검증한다.
- **Authority Verifier** — 제안된 행동이 위임된 capability와 scope 안에 있는지 검증한다.

두 verifier는 서로 다른 증거(evidence)만 보고 각자 독립적으로 판단하며, 그 결과는 결정론적(deterministic) 로직으로 합쳐진다.

---

## 아키텍처

```text
Principal Agent
      |
      | natural-language delegation
      v
Delegate Agent
      |
      | proposed interpretation / action
      v
+-------------------------------+
|                               |
v                               v
Semantic Verifier          Authority Verifier
|                               |
| SemanticVerdict               | AuthorityVerdict
|                               |
+---------------+---------------+
                |
                v
       Deterministic Fusion
                |
         EXECUTE / REJECT
```

핵심 설계 목표는 semantic 판단과 authority 집행을 철저히 분리하는 것이다. Semantic Verifier는 권한을 부여할 수 없고, Authority Verifier는 Principal의 의도를 새로 정의할 수 없다.

---

## 설계 원칙

### 독립적 검증

각 agent는 의도적으로 제한된 증거만 보고 판단한다. Delegate는 벤치마크의 정답(truth)도, Principal의 숨겨진 구조화된 intent도 받지 않는다.

Semantic Verifier는 delegation + Delegate의 proposal + runtime에서 보이는 context만으로 독립적으로 판단한다.

Authority Verifier는 Delegate의 proposal + 위임된 `Budget` + runtime context만으로 판단한다.

두 verifier의 출력은 fusion 이전에는 서로 공유되지 않는다.

### 결정론적 authority 경계

권한 여부는 LLM이 결정하지 않는다. DualFlow는 `Budget`, `Budget.meet()`, `delegation_chain()`, `check_authority()` 같은 결정론적 메커니즘 뒤에서 authority를 집행한다.

proposal이 위임된 scope를 넘으면 LLM이 더 좁은 proposal을 제안할 수는 있지만, 그 결과도 반드시 `check_authority()`를 다시 통과해야 승인된다. semantic experience는 authority를 절대 부여하거나 확장하지 않는다.

### Oracle-free agent runtime

실제 agent runtime은 실행 결정을 내릴 때 `task.truth`, ground-truth label, 기대되는 결정, evaluation label 등을 전혀 사용하지 않는다. 평가 전용 정보는 runtime 경로 바깥에 남는다.

기존 controlled benchmark는 mechanism 수준 평가와 regression 테스트를 위해 별도로 유지된다.

---

## Agent-Connected Evaluation

현재 연구는 Principal–Delegate 사이의 반복적인 워크플로우에 초점을 맞춘다. 모든 delegation이 곧바로 명확하다고 가정하는 대신, Delegate는 여러 후보 해석(interpretation) 사이의 불확실성을 직접 추정할 수 있다.

```text
Principal delegation
        |
        v
Delegate candidate sampling
        |
        v
Candidate distribution
        |
        v
Semantic entropy
        |
   +----+----+
   |         |
 low H      high H
   |         |
   |         v
   |    clarification request
   |         |
   |         v
   |      Principal
   |         |
   |         v
   |    clarified intent
   |         |
   +----+----+
        |
        v
verification / execution
```

더 장기적인 질문은, 이전 상호작용에서 얻은 Principal-confirmed experience가 semantic 정확도나 authority 안전성을 해치지 않으면서 이후의 clarification 비용을 줄일 수 있는가다.

### 현재 실험 시나리오

현재 pilot은 좁고 통제된 워크플로우 하나를 쓴다. **외부 감사용 재무 보고서 위임**(external-audit financial-report delegation)이다.

애매한(ambiguous) delegation 예시:

> Please prepare the September 2026 financial report for the external audit.

현재 실험 환경은 canonical action ontology `read` / `summarize` / `export`를 쓰고, resource/scope는 고정돼 있다. 예를 들어 `RESOURCE = file`, `SCOPE = /reports/2026-09/`다.

따라서 현재 측정되는 entropy는 이 canonical semantic action space 위에서의 불확실성을 나타내는 것이지, LLM이 생성할 수 있는 임의의 자유형(free-form) action 전체에 대한 것이 아니다.

### Candidate sampling과 semantic entropy

모델에게 자기평가(self-reported) confidence 점수를 묻는 대신, DualFlow는 독립적인 LLM 샘플을 여러 번 뽑아 불확실성을 추정한다.

```python
distribution = delegate.sample_candidates(
    delegation=delegation,
    context=context,
    n=N,
)
```

후보 확률은 경험적으로(empirically) 추정된다. semantic entropy는 다음과 같이 계산한다.

$$ H = -\sum_i p_i \log_2 p_i $$

각 후보의 기여도는 $-p_i \log_2 p_i$다. entropy 계산 자체는 로컬 연산이고 추가 API 호출이 필요 없다.

### 관측된 ambiguity

"Please prepare the August 2026 financial report for the external audit."라는 delegation에 대해, 10개 샘플씩 5번 독립 실행한 결과는 다음과 같다.

| Run | export | summarize | read | Entropy |
| --- | --- | --- | --- | --- |
| 1 | 0.50 | 0.50 | 0.00 | 1.000 |
| 2 | 0.70 | 0.30 | 0.00 | 0.881 |
| 3 | 0.60 | 0.40 | 0.00 | 0.971 |
| 4 | 0.60 | 0.40 | 0.00 | 0.971 |
| 5 | 0.50 | 0.50 | 0.00 | 1.000 |

50개 샘플 전체에서 resource/scope는 올바르게 고정된 채였다. ambiguity는 `summarize` vs. `export` 사이에 집중돼 있었다. 이는 실제 API로 확인한, 통제된 semantic ambiguity의 예시다.

### Principal clarification

후보 분포의 entropy가 설정된 threshold를 넘으면, Delegate는 Principal에게 역질의(clarification)를 할 수 있다.

> **Delegate:** Should I export the financial report for the external audit, or would you prefer a summary?
>
> **Principal:** Please summarize the August 2026 financial report.

Delegate는 이 clarification을 반영해서 다시 한 번 해석을 샘플링한다. 초기 real-API pilot에서는 5번 중 4번에서 clarification이 트리거됐다. clarification이 일어난 4번 모두 대략 `H = 0.88 ~ 0.97 bits`에서 `summarize = 1.00`, `H = 0.00`으로 바뀌었다. clarification이 일어난 run들의 평균 entropy 감소량은 **0.904 bits**였다.

이 pilot은 Principal clarification이 semantic uncertainty를 상당히 줄일 수 있음을 보여준다.

### 중요한 발견: entropy는 정확성이 아니다

clarification이 일어나지 않은 run 하나는 `export = 0.80`, `summarize = 0.20`, `H = 0.722`를 기록했다. pilot threshold가 0.8이었기 때문에 clarification은 생략됐고, `export`가 최종 해석으로 선택됐다.

이는 중요한 구분을 드러냈다. **entropy가 낮다고 해석이 맞다는 보장은 없다.** 그래서 DualFlow는 다음 두 가지를 별도로 평가한다.

- **semantic uncertainty (불확실성)** — entropy `H`.
- **semantic alignment (의미 정합성)** — Principal-confirmed intent에 부여된 확률.

모델은 Principal과의 정합성이 낮아지면서 동시에 더 확신을 가질 수도 있다.

### Verified experience

clarification의 결과는 Principal-confirmed semantic experience로 저장될 수 있다. 저장 key는 `(principal_id, task_category)`로 결정론적으로 정해진다. embedding 기반 검색, fuzzy matching, LLM 기반 검색 중 어느 것도 필요 없다.

verified experience에는 이전 delegation, clarification 질문, Principal의 답변, confirmed interpretation, clarification 전후의 분포(distribution)가 기록된다. Principal이 확인한 적 없는, entropy만 낮은 Delegate의 추측은 verified experience로 취급하지 않는다. experience record에는 authority grant나 capability 정보가 전혀 포함되지 않는다.

### Experience-transfer diagnostics

초기 experience representation은 과거 상호작용을 주로 구조화된 라벨 한 줄로만 노출했다. 예: `Principal-confirmed interpretation: ACTION=summarize`. 실제 API 실험 결과, 이 표현 방식은 Principal-confirmed semantic pattern을 안정적으로 전이(transfer)시키지 못했다.

400-call 진단 실험과 그 후의 500-call structure-only control은 다음을 보여줬다.

- `summarize`와 `export` 이력 라벨이 사실상 동일한 출력 분포를 만들어낼 수 있었다.
- action semantic이 전혀 없는 neutral한 이력 블록도 candidate distribution을 강하게 바꿀 수 있었다.
- entropy는 줄어드는데 semantic alignment는 오히려 나빠질 수 있었다.

이는 또 하나의 핵심 발견으로 이어졌다. **historical-context priming과 진짜 semantic transfer는 구분돼야 한다.** 이 부정적/예상 밖의 결과와 진단 조건들은 삭제되지 않고 실험 기록에 그대로 보존돼 있다 — [`docs/experiments/agent_connected_eval.md`](docs/experiments/agent_connected_eval.md)를 참고하라.

### Semantic-memory representation v2

현재 재설계는 experience를 단일 이력 라벨이 아니라 하나의 관계(relationship)로 표현한다.

```text
previous ambiguous delegation
        |
        v
Delegate clarification question
        |
        v
Principal clarification
        |
        v
confirmed meaning
```

모델에게는 verified 과거 상호작용을 현재 delegation에서 실제로 애매한 부분을 해소하는 증거로만 쓰라고 지시하며, 동시에 현재 task의 명시적 지시(explicit instruction)는 과거 experience보다 항상 우선해야 한다고 명시한다.

v2의 초기 pilot에서는 명시적 현재 지시를 그대로 유지하면서도 ambiguous task에서 부분적인 semantic-transfer 신호가 관측됐다. 다만 이 실험에서는 context-wording drift라는 confound도 함께 발견됐기 때문에, 이 pilot의 절대 수치를 이전 실험과 직접 비교해서는 안 된다. 그래서 다음 대규모 평가 전에는 정확한 prompt reproducibility 확보가 현재 최우선 과제다.

---

## 저장소 구조

```text
dualflow/
├── src/
│   └── dualflow/
│       ├── principal_agent.py
│       ├── delegate_agent.py
│       ├── semantic.py
│       ├── authority_feedback.py
│       ├── capability.py
│       ├── framework.py
│       ├── clarification.py
│       ├── agent_experience.py
│       ├── experience_aware_delegate.py
│       ├── agent_runtime.py
│       └── llm.py
│
├── experiments/
│   ├── agent_smoke.py
│   ├── demo.py
│   ├── benchmark.py
│   ├── bench.py
│   ├── plots.py
│   ├── entropy_probe.py
│   ├── baselines/
│   │   └── sage.py
│   │
│   ├── diagnostics/
│   │   ├── experience_transfer.py
│   │   └── experience_representation.py
│   │
│   └── scenarios/
│       └── external_audit_finance.json
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   ├── BASELINES.md
│   ├── DESIGN_NOTES.md
│   └── experiments/
│       └── agent_connected_eval.md
│
├── results/
│   └── agent/
│
├── figures/
│   └── agent/
│
└── tests/
```

- `src/dualflow/`는 runtime 메커니즘을 담는다.
- `experiments/scenarios/`는 재현 가능한 연구 시나리오를 담는다 — 코드가 여러 필드로부터 조립하는 prose가 아니라, 실제로 model에게 보이는 delegation/context 문자열 그 자체를 저장한다.
- `experiments/diagnostics/`는 특정 실패 모드(semantic-transfer priming, representation redesign)를 조사하는 실험을 담는다.
- `experiments/baselines/sage.py`는 SAGE-Agent 논문의 Algorithm 1을 비교 baseline으로 재현한 것이다 — `dualflow` core package의 일부가 아니다.
- `docs/experiments/agent_connected_eval.md`는 실제 API로 진행한 실험의 시간순 기록을 예상 밖/부정적 결과까지 포함해 보존한다.
- `results/agent/`와 `figures/agent/`는 향후 sequential agent evaluation과 최종 연구 결과물을 위해 예약돼 있다.

---

## 설치

core package 설치:

```bash
pip install -e .
```

테스트 실행과 figure 재생성을 위해:

```bash
pip install -e ".[dev]"
```

실제 LLM agent 실험을 위해:

```bash
pip install -e ".[agent]"
```

API key는 환경변수로 설정한다.

```bash
export OPENAI_API_KEY="..."
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

API key를 소스 코드나 커맨드라인 인자에 넣지 말 것.

---

## 빠른 확인

테스트 스위트 실행:

```bash
pytest -q
```

통제된 데모 실행:

```bash
python experiments/demo.py
```

통제된 benchmark 실행:

```bash
python experiments/benchmark.py
```

유지 중인 실험 figure 생성:

```bash
python experiments/plots.py --outdir figures
```

실제 agent smoke 모드 확인:

```bash
python experiments/agent_smoke.py --help
```

예시:

```bash
python experiments/agent_smoke.py --role principal
python experiments/agent_smoke.py --role delegate
python experiments/agent_smoke.py --role semantic
python experiments/agent_smoke.py --role authority
python experiments/agent_smoke.py --role runtime
python experiments/agent_smoke.py --role sample --n 10
python experiments/agent_smoke.py --role clarify --n 10 --entropy-threshold 0.8
python experiments/agent_smoke.py --role experience --n 10
```

### Experience-transfer diagnostics 재현

experience-transfer diagnostics는 runtime과 별도로 구현돼 있다.

```bash
python experiments/diagnostics/experience_transfer.py --help
python experiments/diagnostics/experience_representation.py --help
```

실험 시나리오는 `experiments/scenarios/external_audit_finance.json`에 저장돼 있다. 로컬 API 원본 출력은 의도적으로 Git에서 제외된다. 대신 저장소에는 실험 조건, 재현 가능한 진단 코드, 집계된 관측치, 알려진 confound, 시간순 연구 기록이 보존된다.

---

## Controlled Benchmark vs. Agent-Connected Evaluation

이 저장소에는 현재 두 가지 다른 형태의 평가가 함께 있다.

**Controlled benchmark** — mechanism 수준 테스트, regression 테스트, 통제된 semantic/authority 실패 케이스에 쓰인다. 일부 legacy controlled evaluation 경로는 내부적으로 oracle 정보를 쓰기 때문에, 실제 agent runtime 평가로 해석해서는 안 된다.

**Agent-connected evaluation** — 실제 Principal, Delegate, Semantic Verifier, Authority Verifier 상호작용을 쓴다. 실제 runtime은 결정을 내릴 때 벤치마크 정답을 쓰지 않는다.

두 평가 환경은 의도적으로 문서에서도 분리돼 있다.

---

## 현재 연구 상태

| 항목 | 상태 |
| --- | --- |
| LLM abstraction | 완료 |
| Principal Agent | 완료 |
| Delegate Agent | 완료 |
| Semantic Verifier | 완료 |
| Authority Verifier | 완료 |
| Oracle-free deterministic runtime | 완료 |
| Candidate sampling | 완료 |
| Entropy-based clarification | 완료 |
| Verified experience storage | 완료 |
| Experience-aware sampling | 구현됨 |
| v1 semantic-transfer diagnostic | 완료 |
| Structure-only control | 완료 |
| v2 semantic-memory redesign | 구현됨 |
| v2 real-API pilot | 부분적 증거(partial evidence) |
| 정확한 scenario/prompt reproducibility | 현재 최우선 과제 |
| Sequential multi-episode evaluation | 시작 전 |

semantic-transfer 동작이 정확히 고정된(fixed) model-visible scenario 아래에서 재현 가능해지기 전까지는, sequential 실험은 의도적으로 미룬다.

---

## 현재 연구 질문

1. verified된 Principal-specific experience가 이후의 ambiguous delegation에서 semantic alignment를 개선할 수 있는가?
2. 그러면서도 현재 지시의 명시적 변경(explicit change)을 덮어쓰지 않을 수 있는가?
3. 누적된 experience는 clarification 비율, LLM 호출 수, token 사용량, latency를 줄이는가?
4. semantic 정확도나 authority 안전성을 해치지 않으면서 이 효율성 향상을 얻을 수 있는가?
5. 결과는 prompt wording과 entropy threshold에 얼마나 민감한가?
6. 이 방법은 닫힌(closed) canonical action ontology를 넘어서도 잘 확장되는가?

---

## 문서

- [`docs/DESIGN_INVARIANTS.md`](docs/DESIGN_INVARIANTS.md) — 앞으로 어떤 변경으로도 절대 깨면 안 되는 조건 모음
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — 실제 API 결과가 재현 가능하려면 무엇을 고정하고 기록해야 하는지(scenario version, model, prompt fingerprint 등)
- [`experiments/README.md`](experiments/README.md) — `experiments/` 아래 각 스크립트가 무엇을 하는지, 어떤 걸 써야 하는지
- [`docs/experiments/agent_connected_eval.md`](docs/experiments/agent_connected_eval.md) — 실제 LLM 기반 agent-connected 평가 로그, 예상 밖/부정적 결과 포함
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 소스 코드 수준의 mechanism 설명
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — 통제된 실험과 과거 pilot 결과
- [`docs/BASELINES.md`](docs/BASELINES.md) — baseline 노트
- [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) — 폐기된 접근과 구현 결정
- [README.md](README.md) — English README

이 저장소는 현재 활발히 연구가 진행 중이다. 예상 밖의/부정적인 실험 결과는 문서에서 의도적으로 보존된다. 특히, semantic alignment를 함께 평가하지 않는 한 entropy가 낮다는 사실만으로 semantic 정확도가 개선됐다고 보지 않는다.

---

## 참고 문헌

이 저장소는 현재 다음 연구의 아이디어를 논의하거나 재현한다.

- ChainCaps: *Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation* (arXiv 2605.26542)
- SAGE-Agent: *Structured Uncertainty guided Clarification for LLM Agents* (arXiv 2511.08798, Findings of ACL 2026)
- SAGE-Bench: *A Service Agent Graph-guided Evaluation Benchmark* (arXiv 2604.09285)

선행 연구와 DualFlow 구현의 정확한 대응 관계는 문서를 참고할 것.

---

## 라이선스

MIT
