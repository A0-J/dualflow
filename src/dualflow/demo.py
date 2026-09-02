"""
프레임워크 시연.

    dualflow-demo                    # 전체
    python -m dualflow.demo joint    # 특정 파트만
"""

from __future__ import annotations

import sys

from .bench import PRINCIPAL, adversarial_tasks, build_judge, build_tasks
from .capability import Budget, Privilege, check_authority, delegation_chain
from .framework import Config, DelegationVerifier, ExperienceStore, evaluate, outcome
from .sage_baseline import SageAgentBaseline
from .semantic import (
    Interpretation as I, Principal, build_belief, entropy, information_gain,
    candidate_questions,
)

BAR = "=" * 78
SUB = "-" * 78


def part1_authority():
    print(BAR)
    print("PART 1  AUTHORITY FLOW — 위임은 권한을 줄일 뿐 늘리지 못한다")
    print(BAR)
    task = next(t for t in build_tasks() if t.name == "chain_laundering")
    chain = delegation_chain(task.principal_budget, task.ceilings)
    names = ["Agent A (오케스트레이터)", "Agent B (리포트 담당)", "Agent C (배포 담당)"]
    for name, budget in zip(names, chain):
        print(f"\n  {name}\n    {budget}")
    print("\n  단조 감쇠 확인:", " ⊇ ".join(
        f"hop{i}" for i in range(len(chain))),
        "→", all(chain[i + 1] <= chain[i] for i in range(len(chain) - 1)))

    print(f"\n  요청: {task.truth}")
    r = check_authority(chain[-1], task.truth.privilege())
    print(f"  판정: {'통과' if r.allowed else '차단'} — {r.reason}")
    print("\n  각 홉은 개별적으로는 정당한 위임이었다. 그럼에도 합성 결과가 외부 발송")
    print("  권한을 되찾지 못하는 것이 ChainCaps 의 non-amplification 을 위임에 옮긴 효과다.")


def part2_semantic():
    print("\n" + BAR)
    print("PART 2  SEMANTIC FLOW — 엔트로피로 재고, 이득이 있는 것만 되묻는다")
    print(BAR)
    task = next(t for t in build_tasks() if t.name == "vague_clarifiable")
    print(f'\n  위임 명세: "{task.spec}"\n')
    belief = build_belief(task.candidates)
    for i, p in sorted(belief.items(), key=lambda kv: -kv[1]):
        print(f"    p={p:.2f}  {i}")
    print(f"\n  H = {entropy(belief):.3f} bits  (θ=0.5 → 수렴 실패, 역질의 필요)\n")
    print(f"  {'IG(bits)':>10}   질문 차원")
    print("  " + SUB[:60])
    for q in sorted(candidate_questions(belief), key=lambda q: -information_gain(belief, q)):
        print(f"  {information_gain(belief, q):>10.3f}   {q.dimension}")
    print("\n  이미 값이 하나뿐인 차원(resource)은 IG=0 이라 자동으로 탈락한다.")
    print("  SAGE-Agent 의 EVPI 와 달리 IG 는 상호정보량이므로 음수가 될 수 없다.")


def part3_joint():
    print("\n" + BAR)
    print("PART 3  JOINT VERIFICATION — 권한은 있는데 의도가 다른 경우")
    print(BAR)
    tasks = build_tasks()
    v = DelegationVerifier(Config(), judge=build_judge(tasks))
    for name in ("silent_misread", "over_privileged_delete", "vague_persistent"):
        task = next(t for t in tasks if t.name == name)
        r = v.run(task)
        print(f'\n[{task.category}] "{task.spec}"')
        for line in r.log:
            print("   " + line)
        print(f"   => {r.decision} ({r.route}, 질문 {r.n_questions}회, LLM {r.n_llm}회)")
    print("\n  silent_misread 가 핵심이다. 반출은 권한 안에 있고 B 의 확신도 높아")
    print("  (H=0.47 ≤ θ) 두 게이트를 모두 통과하지만, 원 의도가 '조회' 였으므로")
    print("  Sim_path 가 0.67 로 떨어져 걸린다. 종단 액션은 양쪽 다 EXECUTE 라")
    print("  Action_Acc 만 봤다면 놓쳤을 사례다.")


def part4_bench():
    print("\n" + BAR)
    print("PART 4  DelegationBench-mini — 트랙별 기여도 (ablation)")
    print(BAR)
    tasks = build_tasks()
    cfgs = [Config(name="Authority only", use_matching=False, use_semantic=False),
            Config(name="Semantic only", use_authority=False),
            Config(name="θ 게이트만 (LLM 없음)", use_llm=False),
            Config(name="상시 LLM (SAGE-Agent 형)", always_llm=True),
            Config(name="Full (제안)")]
    print(f"{'설정':<24}{'unsafe↓':>10}{'benign↑':>10}{'over-rej':>10}"
          f"{'LLM률↓':>9}{'질문':>7}{'비용↓':>8}")
    print(SUB)
    for c in cfgs:
        m = evaluate(c, tasks, judge=build_judge(tasks))
        print(f"{m['name']:<24}{m['unsafe_rate']*100:>9.1f}%{m['benign_completion']*100:>9.1f}%"
              f"{m['over_rejection']*100:>9.1f}%{m['llm_rate']*100:>8.1f}%"
              f"{m['avg_questions']:>7.2f}{m['avg_cost']:>8.2f}")
    print("\n  unsafe = 실행했는데 권한 밖이었거나 A 의 의도와 다른 해석이었던 비율")
    print("  benign = 정당한 위임을 올바른 해석으로 실행한 비율 (비용 c_q=1, c_llm=10)")
    print("\n  상시 LLM 은 '완벽한 LLM' 오라클을 가정한 상한선이다. 그 설정에서도")
    print("  제안 프레임워크는 같은 0% unsafe 를 1/6 비용으로 달성한다.")

    print("\n과제별 상세:")
    print(SUB)
    m = evaluate(Config(), tasks, judge=build_judge(tasks))
    for t, r, o in m["rows"]:
        print(f"  {t.name:<24}{t.category:<15}ideal={t.ideal_decision():<8}"
              f"got={r.decision:<8}{o}")


def part5_theta():
    print("\n" + BAR)
    print("PART 5  RQ2 — θ 를 어디에 둘 것인가 (비용 대비 이득)")
    print(BAR)
    tasks = build_tasks()
    print(f"{'θ':>6}{'unsafe↓':>10}{'benign↑':>10}{'LLM률':>9}{'평균질문':>10}{'평균비용':>10}")
    print(SUB)
    for th in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        m = evaluate(Config(theta=th), tasks, judge=build_judge(tasks))
        print(f"{th:>6.2f}{m['unsafe_rate']*100:>9.1f}%{m['benign_completion']*100:>9.1f}%"
              f"{m['llm_rate']*100:>8.1f}%{m['avg_questions']:>10.2f}{m['avg_cost']:>10.2f}")
    print("\n  θ 를 올릴수록 되묻지 않아 비용은 내려가지만 benign completion 이 무너진다.")
    print("  주목할 점은 unsafe 가 어느 θ 에서도 0% 이라는 것이다. θ 는 '얼마나 유용한가'")
    print("  를 조절할 뿐이고 '안전한가' 는 Authority Flow 와 매칭 검증이 따로 책임진다.")
    print("  두 축이 분리돼 있다는 것이 이 아키텍처의 설계 의도이자 실측 결과다.")


def part6_experience():
    print("\n" + BAR)
    print("PART 6  Experience Score — 같은 유형의 위임이 반복될 때")
    print(BAR)
    tasks = build_tasks()
    store = ExperienceStore()
    v = DelegationVerifier(Config(), store, build_judge(tasks))
    task = next(t for t in tasks if t.name == "vague_clarifiable")
    print(f'\n  위임 명세: "{task.spec}" 를 7회 반복\n')
    print(f"{'회차':>5}{'경로':>13}{'질문':>6}{'LLM':>5}{'Exp.Score':>11}{'H(초기)':>9}{'비용':>7}")
    print(SUB)
    for ep in range(1, 8):
        r = v.run(task)
        print(f"{ep:>5}{r.route:>13}{r.n_questions:>6}{r.n_llm:>5}"
              f"{store.score(task.key):>11.2f}{r.h_initial:>9.2f}{r.cost(v.cfg):>7.1f}")
    print("\n  판정 결과가 누적되면서 초기 엔트로피가 내려가고, Score ≥ σ=0.8 이 되는")
    print("  5회차부터는 역질의 없이 자율 판단한다. 그래도 Authority Flow 는 매번")
    print("  동일하게 검사된다 — 경험이 상한선을 밀어올리지는 못한다.")


def part7_fastslow():
    print("\n" + BAR)
    print("PART 7  Fast / Slow / AND — 세 결합 방식 파일럿")
    print(BAR)
    tasks = build_tasks()
    print("\n  Fast : 엔트로피 한 번 재고 θ 로 판단 (역질의 → LLM 순으로 fallback)")
    print("  Slow : B 가 해석을 정리해 A 에게 제시하고 승인/교정받음")
    print("  AND  : Fast 로 좁힌 해석을 A 가 승인해야만 통과 (교정 요구 = 불일치 = 차단)\n")
    _mode_table(tasks, build_judge(tasks))
    print("\n  Slow 는 silent_misread 를 A 의 교정으로 살려내 benign 100% 를 찍는다.")
    print("  대신 모든 위임에서 A 를 호출하므로 검토율이 1.00 이다.")
    print("  AND 는 안전성이 같고 비용이 가장 크다 — 이 벤치마크에서는 Slow 에 지배당한다.")
    print("  Slow 를 '언제' 태울지가 다음 실험 문항이다 (README §7).")


def part8_attack():
    print("\n" + BAR)
    print("PART 8  belief 조작 공격 — self-reported uncertainty 의 공격 표면")
    print(BAR)
    adv = adversarial_tasks()
    print("\n  위협 모델: B 의 후보 생성이 오염돼 후보가 목표 해석 하나로 좁혀진다.")
    print("  결과적으로 H=0 (SAGE-Agent 의 π_c=1) 이 되어 명확화가 통째로 생략된다.")
    print("  A 의 의도와 권한 정의는 신뢰하고, B 의 후보 생성은 신뢰하지 않는다.\n")
    for t in adv[:3]:
        print(f"    {t.name:<28} 목표 해석 = {t.attack}")
    print(f"    ... 총 {len(adv)}건\n")

    sage = SageAgentBaseline()
    routes = [sage.run(t.candidates, Principal(t.truth, t.refuses)) for t in adv]
    n_tau = sum(1 for r in routes if r.route == "sage:tau_exec")
    n_evpi = sum(1 for r in routes if r.evpi_computed)
    print(f"  SAGE-Agent 원 공식에서 무슨 일이 일어나는가:")
    print(f"    · max π = 1.0 ≥ τ_exec 로 Step 1 에서 즉시 실행: {n_tau}/{len(adv)}건")
    print(f"    · EVPI 가 한 번이라도 계산된 경우: {n_evpi}/{len(adv)}건")
    print(f"    · 명확화 질문: {sum(r.n_questions for r in routes)}회\n")
    _mode_table(adv, build_judge(build_tasks()))
    print("\n  SAGE-Agent 는 τ_exec 검사가 EVPI 보다 앞에 있어(§5.2 Step 1) 질문 생성도")
    print("  EVPI 계산도 일어나지 않는다. 공격자는 확률을 조작할 필요 없이 빈칸만 채우면 된다.")
    print("  Authority·매칭을 붙여도 '같은 SOP 경로,")
    print("  같은 권한 안의 다른 자원' 공격은 44.4% 가 통과한다 — 엔트로피 축만으로는")
    print("  부족하다는 뜻이다. A 의 검토는 B 의 자기신고 값을 입력으로 쓰지 않으므로")
    print("  구조적으로 면역이고, AND 는 공격 하에서도 오탐 0% 를 유지한다.")
    print("  그 대가로 미탐(over-rejection)이 100% 로 오른다 — 보수적 결합의 정의 그대로다.")


def _mode_table(tasks, judge):
    cfgs = [Config(name="SAGE-Agent (Eq.2+Def.4)", mode="sage",
                   use_authority=False, use_matching=False),
            Config(name="SAGE + Joint", mode="sage"),
            Config(name="Fast + Joint", mode="fast"),
            Config(name="Slow + Joint", mode="slow"),
            Config(name="AND (제안)", mode="and")]
    print(f"{'설정':<28}{'unsafe↓':>10}{'benign↑':>10}{'over-rej':>10}"
          f"{'질문':>7}{'검토':>7}{'LLM':>6}{'비용↓':>8}")
    print(SUB)
    for c in cfgs:
        m = evaluate(c, tasks, judge=judge)
        print(f"{m['name']:<28}{m['unsafe_rate']*100:>9.1f}%{m['benign_completion']*100:>9.1f}%"
              f"{m['over_rejection']*100:>9.1f}%{m['avg_questions']:>7.2f}"
              f"{m['review_rate']:>7.2f}{m['llm_rate']:>6.2f}{m['avg_cost']:>8.2f}")


PARTS = {"authority": part1_authority, "semantic": part2_semantic, "joint": part3_joint,
         "bench": part4_bench, "theta": part5_theta, "experience": part6_experience,
         "fastslow": part7_fastslow, "attack": part8_attack}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    unknown = [a for a in argv if a not in PARTS]
    if unknown:
        print(f"알 수 없는 파트: {unknown}\n사용 가능: {', '.join(PARTS)}")
        return 2
    for key in (argv or list(PARTS)):
        PARTS[key]()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
