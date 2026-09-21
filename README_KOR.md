[English](README.md) | **한국어**

# DualFlow

**안전한 Agent-to-Agent 위임을 위한 의미·권한 검증**

```
Agent A (principal)
       │  위임 요청
       ▼
Agent B (delegate)
       │
   ┌───┴────────────────────────┐
   ▼                            ▼
Semantic Verification     Authority Verification
"A가 뭘 의도했나?"          "B가 뭘 할 수 있나?"
   └───────────┬────────────────┘
               ▼
       Joint Verification
               │
        ┌──────┴──────┐
        ▼             ▼
    EXECUTE      REJECT / ESCALATE
```

DualFlow 는 흔히 섞이는 두 질문 — "delegate 가 요청을 제대로 이해했는가"와
"제안된 행동이 위임된 권한 범위 안에 있는가" — 를 분리해서 각각 독립적으로
검증하고, 실행 전에 두 신호를 합친다. 위임 범위가 정책만으로는 안전하게
결정되지 않을 때는 principal 과 **bounded Authority Feedback Loop** 를 연다.
핵심 안전 메커니즘은 과거 이력에 의존하지 않는다 — **검증된 이력(verified
experience)은 반복되는 principal 확인 비용을 줄이는 최적화 계층으로만
쓰인다.**

구현은 결정론적이다(아래 실험을 돌리는 데 LLM 호출이 필요 없다). 실제 모델은
인터페이스 세 곳 뒤에 그대로 꽂힌다 — [현재 범위와 한계](#현재-범위와-한계) 참고.

## 왜 DualFlow 인가?

위임된 행동은 두 가지 서로 다른 방식으로 실패할 수 있다.

**의미 불일치(Semantic mismatch).** delegate 가 확신에 차서 요청을 잘못 읽을 수 있다.

```
Principal: "마케팅팀용으로 요약해줘. 원본 파일은 내보내지 마."
Delegate:   export /reports/2026-08/     ← 권한은 허용하지만 의미는 다르다
```

**권한 불일치(Authority mismatch).** delegate 가 요청을 정확히 이해했더라도,
실제로 위임된 범위 밖의 것을 제안할 수 있다.

```
Principal: "요약본을 이메일로 보내줘 — 사내(@corp.com) 주소로만."
Delegate:   send email *                 ← 의미는 맞지만 범위가 너무 넓다
```

**낮은 불확실성이 유효한 위임을 뜻하지 않고, 허용된 행동이 곧 의도된 행동도
아니다.** DualFlow 는 실행 전에 이 둘을 명시적으로, 각각 따로 검증한다.

## 아키텍처

DualFlow 는 그 자체로 안전해야 하는 **Core Safety Mechanism** 과, 사용 비용만
줄여주는 선택적 **Optimization Layer** 로 구성된다.

```
Core Safety Mechanism ── 과거 이력이 하나도 없어도 안전성이 유지된다
  1. Semantic Flow            모호함 해소: 엔트로피, information gain, 역질의(clarification)
  2. Authority Flow           capability / resource / scope / condition 검증
  3. Authority Feedback Loop  APPROVE / CORRECT / RESTRICT / REJECT, bounded rounds
  4. Joint Verification       확정된 의도·확인된 권한에 대한 최종 검증

Optimization Layer ── principal 과의 상호작용 비용만 줄인다. 안전성에는 영향 없음
  Verified Authority Store → Adaptive Feedback Gate → 필요할 때만 principal 호출
```

Optimization Layer 를 완전히 꺼도(`use_verified_experience=False`) Core 단독과
정확히 같은 안전성을 낸다 — 매번 principal 을 부를 뿐이다.

| 구성요소 | 구현 | 관련 선행연구 |
|---|---|---|
| 위임 예산, 위임 체인 | `capability.Budget`, `check_authority` | ChainCaps §3.2–3.3, Eq.(2) |
| Action space, 해석 후보 분포 | `semantic.Interpretation`, `build_belief` | SAGE-Agent Def.2–3 |
| 엔트로피 / 역질의 | `semantic.entropy`, `information_gain` | SAGE-Agent Def.4 를 information gain 으로 재정의 |
| LLM fallback | `llm.LLMJudge` | SAGE-Agent 의 상시 판단자를 fallback 으로 격하 |
| 매칭 검증 | `rule_engine.match_intent`, `sim_path` | SAGE-Bench Eq.(6) |
| Authority Feedback Loop | `authority_feedback.run_feedback` | 신규 |
| Verified Authority Store / Adaptive Gate | `authority_feedback.VerifiedAuthorityStore` | 신규 |
| 전체 조립 | `framework.DelegationVerifier.run` | — |

## 동작 방식

**의미 검증(Semantic Verification).** delegate 는 구조화된 해석
`(action, resource, scope, condition)` 을 만든다. DualFlow 는 해석 후보들에 대한
불확실성을 Shannon entropy 로 재고, 불확실성이 높으면 information gain 기준으로
역질의 질문을 고른다.

**권한 검증(Authority Verification).** 제안된 모든 실행은 현재 위임 예산에 대고
검사된다. 실패는 단순히 표시만 되는 게 아니라 분류된다:

- `no_grant` — action/resource 자체가 애초에 위임된 적 없음 → 하드 리젝트
- `condition_missing` — 범위는 맞지만 필수 조건(예: 비식별화)이 빠짐 → 하드
  리젝트, delegate 가 스스로 채울 수 있는 값이 아니다
- `scope_exceeded` — action/resource/condition 은 맞는데 범위만 넘음 →
  **협상 가능**
- 그 외 → 계속 진행

**Authority Feedback.** `scope_exceeded` 제안에 대해 principal 은 시스템이
계산한 제안(현재 예산이 실제로 커버하는 가장 넓은 범위)에 대해 `APPROVE`,
`CORRECT`, `RESTRICT`, `REJECT` 중 하나로 응답한다. 협상은 bounded 이고, 수정된
모든 제안은 승인되기 전에 위임 예산에 대해 다시 검증된다 — 예산 밖 제안에
부주의하게 `APPROVE` 해도 이 재검증이 걸러낸다(그대로 신뢰하지 않는다).

**Joint Verification.** 확정된 의미 해석과 확인된 권한이 원래 의도와 일치할
때만 — 경로 유사도와 **terminal decision match**(EXECUTE / ESCALATE / REJECT)
를 모두 만족할 때만 — 실행이 허용된다.

**Adaptive Feedback.** principal 에게 반복해서 확인받는 건 비용이 크다.
DualFlow 는 principal 이 **확인해주고** 시스템이 **재검증했고** 그 결과
**실행까지 성공한** — 이 세 조건을 모두 거친 권한 상태만 저장하고, 단순하고
해석 가능한 규칙으로 재사용한다:

```
n_confirmed >= 3  이고  agreement_ratio >= 0.8   =>  묻지 않고 재사용
```

재사용은 절대 이력을 그대로 재생하지 않는다. 재사용될 때마다 **현재** 예산과의
교집합으로 다시 계산되고 재검증된다:

```
C_adaptive = C_experience ∩ C_current_budget
C_adaptive ⊆ C_current_budget ⊆ C_A
```

**검증된 이력은 현재 권한을 좁힐 수 있을 뿐, 절대 새로 만들거나 넓히지 못한다.**

## 현재까지의 검증

하나의 통제된 8단계 위임 환경(`bench_single_env_v1`)과 두 개의 별도 mini-set
위에서, 각각 하나의 연구 질문에 답하는 4개 실험을 돌렸다:

| # | 실험 | 질문 |
|---|---|---|
| 1 | Core Ablation | Semantic과 Authority 검증은 서로 다른 실패를 잡는가? |
| 2 | Authority Feedback | Bounded negotiation이 권한을 넓히지 않고 utility를 회복하는가? |
| 3 | Adaptive Authority | 검증된 이력이 principal 호출을 안전하게 줄이는가? |
| 4 | Semantic Robustness | Entropy는 타당한 불확실성 신호인가, 그리고 확신에 찬 틀린 제안에도 안전한가? |

**대표 결과** (Experiment 1, 통제된 benchmark, proposal은 정답 해석 또는 코드에
고정 주입한 adversarial 값 중 하나다 — 실제 모델에서 샘플링한 게 아니다):

| 방식 | Unsafe ↓ (adversarial) |
|---|---:|
| Authority only | 33.3% |
| Semantic only | 66.7% |
| **Full DualFlow** | **0.0%** |

세 attack 각각을 **정확히 어떤 메커니즘이** 잡는지(Joint Verification, Authority
Feedback Loop, Authority의 하드 리젝트 — Semantic Flow 자신의 확신 게이트는
셋 다 스스로 못 잡는다)까지 포함한 전체 내용은 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
의 Experiment 1과 Mechanism Attribution에 있다.

### 1. 두 검증 축 모두 필요하다

![v1 Phase 2 — Correct vs. Adversarial Proposal](figures/fig10_v1_phase2.png)

각 축을 단독으로 빼면 서로 다른 이유로 서로 다른 실패가 통과한다 — Authority-only는
misread(제안된 행동 자체가 허용돼 있어서)를 놓치고, Semantic-only는 scope/condition
위반(예산 자체를 안 보므로)을 놓친다. 상세는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §2.

### 2. Authority Feedback 은 안전한 거절을 안전한 완료로 바꾼다

| 방식 | Unsafe ↓ | Benign ↑ | Feedback률 ↓ |
|---|---:|---:|---:|
| Feedback 없음 | 0.0% | 0.0% | 0.0% |
| **Feedback(제안)** | 0.0% | **100.0%** | 66.7% |

![Authority Feedback Loop — 정상/공격 조건](figures/fig7_authority_feedback.png)

Feedback 이 없으면 협상 가능한 모든 scope 위반이 안전하게 거절되지만 끝내
완료되지 않는다. Feedback 을 켜면 같은 사례가 안전하게 완료되고, principal 의
부주의(carelessness)가 1.0 까지 올라가도 안전하게 유지된다 — 수정된 모든
제안이 principal 이 뭐라고 답했든 매 라운드 재검증되기 때문이다. 단 이건
위임 예산과 principal feedback 채널 자체가 신뢰 가능하다는 전제에서만 성립한다
(자세한 건 [현재 범위와 한계](#현재-범위와-한계)). 상세는
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §3.

### 3. Adaptive 재사용은 반복·변경·조작을 모두 견딘다

![Adaptive Verification — 9라운드 타임라인](figures/fig8_adaptive_verification.png)

- **안정적 반복**: principal 확인이 3회 일관되게 쌓이면, 더 묻지 않고 검증된
  권한을 재사용한다 — 같은 위임 유형에서 feedback률이 100% 에서 0% 로 떨어진다.
- **정당한 변경(drift)**: 위임 범위가 실제로 바뀌면 낡은 이력이 새 예산과 안
  겹쳐 feedback 이 즉시 재활성화되고, 낡은 이력은 새 이력에 압도당하는 대신
  리셋된다.
- **조작**: 완전히 확신에 찬 조작된 제안이 와도 adaptive 재사용은 묻지 않고
  **정답** scope 로 해결된다 — delegate 의 후보를 전혀 보지 않고, principal 이
  확인해준 이력과 현재 예산의 교집합만 보기 때문이다.

라운드별 전체 데이터는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §4.

### 4. Entropy는 실제 신호이지만, 그것만으로는 부족하다

![Entropy validation — first real-LLM results](figures/fig9_entropy_probe.png)

첫 real-LLM 실행(GPT-4o-mini, N=20)에서, 모델이 100% 확신(H=0)으로 답한 요청
하나가 **미승인 스코프**를 가리켰다. 확신 게이트 혼자였으면 통과시켰을 것을
Authority Flow가 잡는다. 상세는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §5.

## 빠른 시작

```bash
pip install -e ".[dev]"

dualflow-demo              # 실험 11개 전체 (텍스트)
dualflow-plots             # 현재 그림은 figures/ 에, legacy 그림은 figures/legacy/ 에 저장
```

### 소프트웨어 검증 vs 실험적 검증

이 둘은 다른 것이다 — 앞은 구현에 대한 회귀 테스트, 뒤는 위의 mechanism-level 증거다.

```bash
pytest -q                 # 테스트 183개: 안전 invariant, 엔트로피 성질, 종료성,
                           # ablation — 소프트웨어 정확성이지 실험 결과가 아니다
```

### 특정 실험만

```bash
python -m dualflow.demo fastslow attack careless authfeedback adaptiveauth
dualflow-plots careless --trials 50   # 논문용 오차범위 (기본 10회는 흔들림)
OPENAI_API_KEY=... dualflow-entropy-probe --cases redesign   # real-LLM 시나리오 검증 재실행
```

## 실험 재현하기

이 README 의 모든 수치와 그림은 위 스크립트에서 그대로 나온 것이다 — 손으로 옮겨
적은 값은 없다. [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 는 위에 인용한 순서
그대로 각 실험을 그 결과가 나온 이유까지 전부 설명하고, 맨 끝 Appendix에는
초기(v0) 탐색 실험을 삭제하지 않고 내부 replication 증거로 남겨뒀다.
[docs/BASELINES.md](docs/BASELINES.md) 는 SAGE-Agent 베이스라인 재현과 그 과정에서
발견한 결함들을 다룬다. [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) 는
최종 결과에는 안 들어간 구현 결정들을 다룬다 — 지름길처럼 보였다가 알고 보니
평가용 오라클이었던 접근(정답과 직접 비교하는 매칭) 하나를 포함해서.

## 저장소 구성

| 파일 | 역할 |
|---|---|
| `capability.py` | 권한 모델 — privilege 순서관계, budget 교집합, 위임 체인 감쇠, 실패 분류 |
| `semantic.py` | Semantic Flow — 해석 후보, 엔트로피, 역질의, 경험 점수 |
| `authority_feedback.py` | Authority Feedback Loop 와 Verified Authority Store(adaptive 재사용) |
| `rule_engine.py` | Joint Verification — 위임 SOP 그래프, 경로 유사도 |
| `framework.py` | 전체 조립, ablation 스위치, 평가 지표 |
| `sage_baseline.py` | SAGE-Agent 베이스라인 — 논문 공식 그대로 재현 |
| `bench.py` | **v0(legacy)** — 9개 과제 벤치마크, 공격 변형, scope 협상·순차 실험 mini-set |
| `bench_single_env_v1.py` | **v1(현재)** — 단일 8단계 위임 환경, Correct/Adversarial Proposal 분리 |
| `entropy_probe.py` | Real-LLM entropy 검증 하네스(`dualflow-entropy-probe`) — 위 파일럿과 완전히 분리 |
| `llm.py` | LLM 인터페이스 — 실험용은 스크립트, 실제 모델용 어댑터 골격 포함 |
| `demo.py` / `plots.py` | 모든 실험의 텍스트/그림 출력 |
| `tests/` | 183개 — 소프트웨어 회귀 테스트이지 실험 결과가 아니다([빠른 시작](#빠른-시작) 참고) |

## 현재 범위와 한계

이 저장소는 현재 결정론적 파일럿 시나리오와 한 차례의 real-LLM 시나리오 검증으로
메커니즘을 검증한다. 실제 운영되는 LLM agent 전체에 대한 외부 타당성은 아직
end-to-end로 확립하지 못했다. 구체적으로:

- 메인 파일럿의 해석 후보 생성은 스크립트로 만든 것이지, 모델이 생성한 게 아니다
  (별도의 real-LLM 하네스 `entropy_probe.py`가 있지만, 아직 파일럿의 candidates를
  대체하진 않았다).
- principal 은 시뮬레이션이다(carelessness/overcaution 조절 가능) — 실제 사람이나
  agent 엔드포인트가 아니다.
- 파일럿 벤치마크는 소규모의 통제된 과제들이다 — 크거나 자연스러운 분포가 아니다.
- 검증된 권한 재사용은 현재 위임 예산과 `VerifiedAuthorityStore` 자체가 신뢰
  가능하다고 가정한다 — 둘 중 하나가 조작 가능하다면 지금 검증한 것과는 다른
  위협 모델이다.
- 권한 범위 안에 있지만 의도하지 않은 자원 선택(delegate 가 승인된 예산 안의
  *다른* 자원을 고르는 경우)은 `scope_exceeded` 자체가 발생하지 않으므로
  Authority Feedback 이 잡지 못한다 — 이건 별개의 intent-confirmation 문제로
  남아 있다.
- "Semantic Flow 혼자서는 세 adversarial case를 못 잡는다"는 코드에 고정 주입한
  proposal에 대한 서술이다 — 실제 모델이 이런 제안을 자연스럽게 얼마나 자주
  생성하는지에 대한 주장이 아니다.

실제 모델로 교체하는 지점은 정확히 세 곳(후보 생성, principal 응답, LLM
fallback)이고 [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) 에 문서화돼 있다.

## 로드맵

- **외부 검증** — 파일럿의 스크립트 candidates를 real-LLM 하네스(`entropy_probe.py`)로
  end-to-end 교체하고, 더 큰 벤치마크에서 검증.
- **에이전트별 principal 신뢰도** — carelessness 를 에이전트마다 다르게 두고
  검토 예산을 넣으면 "누구에게 Slow review 를 줄 것인가" 가 최적화 문제가 된다.
  `warmup_then_attack` 이 이미 그 골격을 제공한다.
- 기존 θ 스윕(Appendix)과 같은 방식의 σ / k / λ 스윕.

항목별 상세는 [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) 에 있다.

## 참고문헌

- ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation. arXiv 2605.26542
- Structured Uncertainty guided Clarification for LLM Agents. arXiv 2511.08798 (Findings of ACL 2026)
- SAGE: A Service Agent Graph-guided Evaluation Benchmark. arXiv 2604.09285
