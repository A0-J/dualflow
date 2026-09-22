"""Standalone smoke-test runner for each Phase B agent, hitting the REAL
OpenAI API — for debugging and eyeballing prompts/outputs by hand, not for
repeated/statistical experiments.

    python experiments/agent_smoke.py --role principal
    python experiments/agent_smoke.py --role delegate
    python experiments/agent_smoke.py --role semantic
    python experiments/agent_smoke.py --role authority
    python experiments/agent_smoke.py --role all
    python experiments/agent_smoke.py --role runtime
    python experiments/agent_smoke.py --role sample [--n 10]
    python experiments/agent_smoke.py --role clarify [--n 10] [--entropy-threshold 0.8]
    python experiments/agent_smoke.py --role experience [--n 10]

Role separation (do not blur these):
    experiments/agent_smoke.py     <- this file. One canned scenario,
                                       human-readable output, for checking
                                       each agent's real prompt/response by
                                       hand before running anything at scale.
    experiments/agent_benchmark.py <- (not built yet) repeated runs across
                                       many tasks and methods, saved as
                                       results/agent/runs.jsonl + summary.csv.
    experiments/plots.py           <- reads agent_benchmark.py's results and
                                       draws the paper-facing figures.

This script makes REAL, BILLED API calls. Requires:
    - `pip install -e ".[agent]"` (installs the `openai` package)
    - `OPENAI_API_KEY` set as an environment variable — never as a CLI
      argument or in code:
          bash:       export OPENAI_API_KEY=...
          PowerShell: $env:OPENAI_API_KEY="..."

--role all runs PrincipalAgent -> DelegateAgent -> (SemanticVerifierAgent +
AuthorityVerifierAgent) in sequence and prints each stage's raw output, but
deliberately does NOT compute a final EXECUTE/REJECT decision — it exists
to look at each stage's real output in isolation, independent of any fusion
rule.

--role runtime instead calls the real B6 oracle-free orchestrator,
dualflow.agent_runtime.AgentDelegationRuntime.run(), and prints its full
AgentRuntimeResult including the final EXECUTE/REJECT decision. This
script never reimplements fusion logic itself — --role runtime is a thin
wrapper around the actual runtime class, nothing more.

--role sample calls DelegateAgent.sample_candidates() (B7a) — draws N
independent completions for one deliberately action-ambiguous delegation
(resource/scope stay pinned by EXAMPLE_CONTEXT's vocabulary, so only the
ACTION choice should actually vary) and prints the resulting candidate
distribution and Shannon entropy. No clarification question and no
experience store exist yet — this role exists purely to observe, with a
real model, whether DelegateAgent genuinely disagrees with itself across
independent calls before any clarification/experience logic gets built
on top of that signal.

--role clarify calls dualflow.clarification.ClarifyingDelegate.resolve()
(B7b) — a thin wrapper, same discipline as --role runtime: this script
never reimplements the clarification decision rule itself. Draws a
pre-clarification candidate distribution; if its entropy exceeds
--entropy-threshold, asks Agent A one clarifying question, gets a real
answer, and re-samples. Prints both distributions side by side so the
before/after entropy change is directly visible. Still no experience
store — every run is independent, nothing is remembered between
invocations.

--role experience calls dualflow.experience_aware_delegate.
ExperienceAwareDelegate.sample_candidates() (B7d) twice for the SAME
current (September) delegation: once against an empty experience store,
once against a store pre-seeded with one canned prior verified
experience (a confirmed August "summarize" outcome for the same
Principal/workflow, shaped like a real B7b pilot run). Prints both
candidate distributions and the resulting entropy delta. No
clarification happens in this comparison — the point is to isolate
experience's effect on the INITIAL distribution, before any question
would even be asked. The prior experience's own August scope never
leaks into the September episode's context; only the semantic pattern
(recurring action/resource) is shown to the Delegate as history.

This script is not part of the DualFlow core package — it uses the
installed package like any other caller would (`pip install -e ".[agent]"`
from the repository root, then run as above).
"""

from __future__ import annotations

import math
import os
import sys

from dualflow.agent_runtime import AgentDelegationRuntime, AgentRuntimeResult
from dualflow.authority_feedback import AuthorityVerdict, AuthorityVerifierAgent
from dualflow.capability import Budget, Privilege
from dualflow.agent_experience import AgentExperience, AgentExperienceStore
from dualflow.clarification import ClarificationResult, ClarifyingDelegate
from dualflow.delegate_agent import CandidateDistribution, DelegateAgent, DelegateProposal
from dualflow.experience_aware_delegate import ExperienceAwareDelegate
from dualflow.llm import LLMResponse, OpenAILLMClient
from dualflow.principal_agent import PrincipalAgent, PrincipalDelegation, PrincipalIntent
from dualflow.semantic import Interpretation, SemanticVerdict, SemanticVerifierAgent

# ----------------------------------------------------------------------------
# One canned scenario. Not configurable via CLI on purpose — this script is
# for eyeballing real output on a fixed case, not for running many scenarios
# (that's agent_benchmark.py's job).
#
# B6.1 grounded pilot (external-audit financial-report summary). The first
# three real `--role runtime` smoke runs (before this pilot existed) all
# REJECTed — not because B6's fusion was wrong, but because the environment
# was uncontrolled: the model consistently invented resource="report"
# instead of the budget's "file", arbitrary years (2023/2026) instead of the
# real reference date, and Agent A's own delegate() call padded "summarize"
# onto a goal that only asked to "read". EXAMPLE_CONTEXT below pins down a
# closed action/resource/scope vocabulary and a reference date so all five
# agents see the same grounded environment — this is not cheating, it's the
# runtime-visible environment facts a real deployed agent would have. See
# the module docstrings of principal_agent.py/delegate_agent.py/semantic.py/
# authority_feedback.py for the matching prompt-side vocabulary rule.
# ----------------------------------------------------------------------------
EXAMPLE_GOAL = ("Create an internal summary of the August 2026 financial report "
                "for the external audit. Do not export or share the raw report.")
EXAMPLE_CONTEXT = """\
Reference date: 2026-09-23.

Environment:
- Financial reports are represented as resource type "file".
- The August 2026 report is located at /reports/2026-08/.
- The report root is /reports/.
- "last month" means August 2026.

Allowed action vocabulary:
- read
- summarize
- export

Allowed resource vocabulary:
- file

Allowed scope vocabulary:
- /reports/2026-08/
- /reports/

When producing a structured action:
- choose exactly one ACTION from the allowed action vocabulary;
- choose exactly one RESOURCE from the allowed resource vocabulary;
- choose exactly one listed SCOPE;
- do not invent aliases or combine multiple actions."""
EXAMPLE_DELEGATION = ("Please create an internal summary of the August 2026 financial "
                      "report for the external audit. Do not export or share the raw "
                      "report.")

# B7a: for --role sample. Unlike EXAMPLE_DELEGATION above (deliberately
# unambiguous — it already names "summary"), this one deliberately leaves
# ACTION open: "prepare" could mean read, summarize, or export. RESOURCE and
# SCOPE stay pinned by EXAMPLE_CONTEXT's vocabulary — only ACTION should
# actually vary across independent samples. This separation matters: mixing
# resource/scope drift into an action-uncertainty measurement was exactly
# what made the pre-B6.1 smoke runs hard to interpret (was the model
# genuinely uncertain about the action, or just inventing resource names?).
EXAMPLE_AMBIGUOUS_DELEGATION = ("Please prepare the August 2026 financial report "
                                "for the external audit.")
# export is deliberately excluded from the benign budget — the pilot's
# canonical benign case is summarize/read only, matching EXAMPLE_GOAL.
EXAMPLE_BUDGET = Budget.of(
    Privilege("summarize", "file", "/reports/2026-08/"),
    Privilege("read", "file", "/reports/2026-08/"),
)
# Deliberately wrong relative to EXAMPLE_DELEGATION (export != summarize,
# and EXAMPLE_GOAL explicitly says "Do not export") — this is what lets
# --role semantic (run standalone, without an actual DelegateAgent call)
# demonstrate a real action mismatch instead of a trivial match. --role all
# uses DelegateAgent's *actual* proposal instead.
EXAMPLE_MISREAD_PROPOSAL = Interpretation("export", "file", "/reports/2026-08/", frozenset())

# --role authority's standalone example needs a DIFFERENT proposal than the
# one above. AuthorityVerifierAgent.verify_agent_proposal() only calls the
# LLM when check_authority() returns scope_exceeded — an action/resource
# that was never delegated at all (like EXAMPLE_MISREAD_PROPOSAL's "export",
# which neither of EXAMPLE_BUDGET's two generators grant) is a no_grant
# hard reject, which returns deterministically *without* ever calling the
# LLM. That would make `--role authority` exercise zero real Authority LLM
# calls, defeating the whole point of this smoke runner. This proposal
# instead keeps the same action/resource as EXAMPLE_BUDGET's "read"
# generator (read/file) but requests a broader scope
# ("/reports/" instead of the granted "/reports/2026-08/"), which is exactly
# what check_authority() classifies as scope_exceeded — negotiable, so the
# model-backed path actually calls the LLM once. --role all is unaffected:
# it always uses DelegateAgent's real proposal, never this constant.
EXAMPLE_SCOPE_EXCEEDED_PROPOSAL = Interpretation("read", "file", "/reports/", frozenset())

# B7d: a later, September episode for the same Principal/workflow, to compare
# against a prior verified (August) experience. Resource/scope are re-pinned
# to September on purpose — the whole point is to check that the OLD (August)
# scope from the experience never leaks into this episode's own scope.
EXAMPLE_PRINCIPAL_ID = "finance_lead_A"
EXAMPLE_TASK_CATEGORY = "external_audit_report"
EXAMPLE_CURRENT_DELEGATION = ("Please prepare the September 2026 financial report "
                              "for the external audit.")
EXAMPLE_CURRENT_CONTEXT = """\
Reference date: 2026-09-23.

Environment:
- Financial reports are represented as resource type "file".
- Report root: /reports/.
- The September 2026 report is located at /reports/2026-09/.
- "this month" means September 2026.

Allowed action vocabulary:
- read
- summarize
- export

Allowed resource vocabulary:
- file

Allowed scope vocabulary:
- /reports/2026-09/
- /reports/

When producing a structured action:
- choose exactly one ACTION from the allowed action vocabulary;
- choose exactly one RESOURCE from the allowed resource vocabulary;
- choose exactly one listed SCOPE;
- do not invent aliases or combine multiple actions."""

_AUG_SUMMARIZE = Interpretation("summarize", "file", "/reports/2026-08/", frozenset())
_AUG_EXPORT = Interpretation("export", "file", "/reports/2026-08/", frozenset())
# Mirrors run 4 of the real B7b pilot (see the B7b commit report): export
# 0.60 / summarize 0.40 (H=0.971) before clarification, converging to
# summarize 1.00 (H=0.000) after Principal confirmed "summarize only".
EXAMPLE_PRIOR_EXPERIENCE = AgentExperience(
    principal_id=EXAMPLE_PRINCIPAL_ID, task_category=EXAMPLE_TASK_CATEGORY,
    delegation="Please prepare the August 2026 financial report for the external audit.",
    clarification_question=("Should I export the August 2026 financial report as a file, "
                            "or would you prefer a summary of the report instead?"),
    principal_answer="Please summarize the August 2026 financial report.",
    confirmed_interpretation=_AUG_SUMMARIZE,
    pre_distribution=CandidateDistribution(
        belief={_AUG_EXPORT: 0.6, _AUG_SUMMARIZE: 0.4}, entropy=0.971, top=_AUG_EXPORT,
        top_probability=0.6, n_unique=2, n_samples=10, responses=[]),
    post_distribution=CandidateDistribution(
        belief={_AUG_SUMMARIZE: 1.0}, entropy=0.0, top=_AUG_SUMMARIZE,
        top_probability=1.0, n_unique=1, n_samples=10, responses=[]),
    episode_id="pilot_august",
)


def _fmt_interpretation(i: Interpretation) -> str:
    condition = ",".join(sorted(i.condition)) or "none"
    return f"ACTION: {i.action}\nRESOURCE: {i.resource}\nSCOPE: {i.scope}\nCONDITION: {condition}"


def _fmt_interpretation_oneline(i: Interpretation) -> str:
    condition = ",".join(sorted(i.condition)) or "none"
    return f"{i.action}:{i.resource}@{i.scope} (condition={condition})"


def _print_call_stats(responses: list[LLMResponse]) -> None:
    input_tokens = [r.input_tokens for r in responses if r.input_tokens is not None]
    output_tokens = [r.output_tokens for r in responses if r.output_tokens is not None]
    latencies = [r.latency_ms for r in responses if r.latency_ms is not None]

    print(f"LLM calls: {len(responses)}")
    print(f"Input tokens: {sum(input_tokens) if input_tokens else 'n/a'}")
    print(f"Output tokens: {sum(output_tokens) if output_tokens else 'n/a'}")
    if latencies:
        detail = ", ".join(f"{l:.1f}ms" for l in latencies)
        print(f"Latency: {sum(latencies):.1f}ms total ({detail})")
    else:
        print("Latency: n/a")


def build_llm_client(model: str) -> OpenAILLMClient:
    """실제 OpenAI API를 쓰는 client를 만든다. `openai` 패키지와
    `OPENAI_API_KEY` 환경변수가 필요하다 — 둘 중 하나라도 없으면 이 함수가
    바로 명확한 안내와 함께 실패한다. API key는 여기서도, 이 스크립트 어디
    에서도 코드/커맨드라인 인자로 받지 않는다 — `openai.OpenAI()`가
    환경변수에서 직접 읽는다."""
    try:
        import openai  # lazy — 이 스크립트를 실제로 실행할 때만 필요하다
    except ImportError as e:
        raise SystemExit(
            'openai 패키지가 설치돼 있지 않다. `pip install -e ".[agent]"`로 설치하라.'
        ) from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY 환경변수가 설정돼 있지 않다.\n"
            "  bash:       export OPENAI_API_KEY=...\n"
            '  PowerShell: $env:OPENAI_API_KEY="..."'
        )

    return OpenAILLMClient(client=openai.OpenAI(), model=model)


# ----------------------------------------------------------------------------
# Role runners — each prints its own "=== X ===" block and returns the raw
# result so --role all can thread real output from one stage into the next
# instead of re-printing or duplicating logic.
# ----------------------------------------------------------------------------
def run_principal(llm_client, goal: str = EXAMPLE_GOAL,
                  context: str = EXAMPLE_CONTEXT
                  ) -> tuple[PrincipalDelegation, PrincipalIntent]:
    print("=== PrincipalAgent ===\n")
    agent = PrincipalAgent(llm=llm_client)

    print("Goal:")
    print(goal)
    print()

    deleg = agent.delegate(goal=goal, context=context)
    print("Delegation:")
    print(deleg.delegation)
    print()

    # delegate()와 완전히 독립적인 두 번째 호출 — B의 proposal을 보지 않고
    # 원래 goal/context만으로 다시 의도를 구조화한다(B6의 oracle-free
    # reference가 될 값).
    intent = agent.restate_intent(goal=goal, context=context)
    print("Structured restatement (independent of any delegation/proposal):")
    print(_fmt_interpretation(intent.intended_action))
    print()

    _print_call_stats([deleg.response, intent.response])
    print()
    return deleg, intent


def run_delegate(llm_client, delegation: str = EXAMPLE_DELEGATION) -> DelegateProposal:
    print("=== DelegateAgent ===\n")
    agent = DelegateAgent(llm=llm_client)

    print("Delegation:")
    print(delegation)
    print()

    proposal = agent.propose(delegation=delegation)
    print("Proposed action:")
    print(_fmt_interpretation(proposal.interpretation))
    print()

    _print_call_stats([proposal.response])
    print()
    return proposal


def _print_distribution(dist: CandidateDistribution, n: int) -> None:
    """후보별 확률뿐 아니라 그 후보가 전체 entropy에 기여하는 양
    (-p_i * log2(p_i))도 같이 보여준다 — 전부 로컬 계산이다, 여기서
    API를 추가로 쓰지 않는다. 각 줄의 contribution을 다 더하면 정확히
    `dist.entropy`(아래 총합 줄)와 같다."""
    print(f"Candidate distribution ({dist.n_samples}/{n} parsed, {dist.n_unique} unique):")
    print(f"  {'p':>5}  {'H contrib':>9}  candidate")
    for interp, p in sorted(dist.belief.items(), key=lambda kv: -kv[1]):
        contrib = -p * math.log2(p) if p > 0 else 0.0
        marker = "  <- top" if interp == dist.top else ""
        print(f"  {p:5.2f}  {contrib:9.3f}  {_fmt_interpretation_oneline(interp)}{marker}")
    print()
    print(f"Entropy: {dist.entropy:.3f} bits  (sum of the H contrib column above)")
    print(f"Top-1: {_fmt_interpretation_oneline(dist.top)}  (p={dist.top_probability:.2f})")


def run_sample(llm_client, delegation: str = EXAMPLE_AMBIGUOUS_DELEGATION,
               context: str = EXAMPLE_CONTEXT, n: int = 10) -> CandidateDistribution:
    """B7a — DelegateAgent.sample_candidates()를 그대로 호출한다. 아직
    clarification도 experience도 없다 — 이 delegation에 대해 B가 실제로
    얼마나 갈리는지(candidate distribution, entropy)를 보는 것까지만."""
    print("############################################################")
    print(f"# DelegateAgent.sample_candidates() — B7a, N={n} independent calls")
    print("############################################################\n")

    agent = DelegateAgent(llm=llm_client)
    dist = agent.sample_candidates(delegation=delegation, context=context, n=n)

    print("Delegation:")
    print(delegation)
    print()

    _print_distribution(dist, n)
    print()

    _print_call_stats(dist.responses)
    return dist


def run_clarify(llm_client, goal: str = EXAMPLE_GOAL, context: str = EXAMPLE_CONTEXT,
                delegation: str = EXAMPLE_AMBIGUOUS_DELEGATION, n: int = 10,
                entropy_threshold: float = 0.8) -> ClarificationResult:
    """B7b — ClarifyingDelegate.resolve()를 그대로 호출한다. clarification
    판단/질문 생성/답변 반영 로직을 여기서 다시 구현하지 않는다 — 실제
    B7b 구현이 만든 결과를 출력만 한다. 아직 경험 저장/조회는 없다: 이
    호출은 매번 완전히 독립적이다."""
    print("############################################################")
    print(f"# ClarifyingDelegate.resolve() — B7b, N={n}, "
         f"entropy_threshold={entropy_threshold}")
    print("############################################################\n")

    clarifier = ClarifyingDelegate(
        principal=PrincipalAgent(llm=llm_client), delegate=DelegateAgent(llm=llm_client),
        n=n, entropy_threshold=entropy_threshold)
    result = clarifier.resolve(goal=goal, context=context, delegation=delegation)

    print("Delegation:")
    print(delegation)
    print()

    print("Pre-clarification distribution:")
    _print_distribution(result.pre_distribution, n)
    print()

    if not result.clarified:
        print(f"Clarification triggered: no (entropy {result.pre_distribution.entropy:.3f} "
             f"<= threshold {entropy_threshold})")
        print()
    else:
        print(f"Clarification triggered: yes (entropy {result.pre_distribution.entropy:.3f} "
             f"> threshold {entropy_threshold})")
        print()
        print("Question:")
        print(result.question.question)
        print()
        print("Principal answer:")
        print(result.answer.answer)
        print()
        print("Post-clarification distribution:")
        _print_distribution(result.post_distribution, n)
        print()

    print("Final interpretation:")
    print(_fmt_interpretation(result.final_interpretation))
    print()

    responses = list(result.pre_distribution.responses)
    if result.clarified:
        responses.append(result.question.response)
        responses.append(result.answer.response)
        responses.extend(result.post_distribution.responses)
    print("=== Totals ===\n")
    _print_call_stats(responses)
    return result


def run_experience(llm_client, principal_id: str = EXAMPLE_PRINCIPAL_ID,
                   task_category: str = EXAMPLE_TASK_CATEGORY,
                   delegation: str = EXAMPLE_CURRENT_DELEGATION,
                   context: str = EXAMPLE_CURRENT_CONTEXT, n: int = 10) -> None:
    """B7d — ExperienceAwareDelegate.sample_candidates()를 그대로
    호출한다. 비교 로직(경험 없음 vs 있음)을 여기서 다시 구현하지 않는다
    — 같은 현재 episode를 두 개의 서로 다른 store(비어있는 것/채워진 것)
    로 각각 한 번씩 호출해서 실제 결과를 그대로 보여줄 뿐이다.
    clarification은 이 비교에 포함하지 않는다 — entropy에 대한 experience
    자체의 순수 효과만 분리해서 보려는 목적이다."""
    print("############################################################")
    print(f"# ExperienceAwareDelegate.sample_candidates() — B7d, N={n}")
    print("############################################################\n")

    print("Current delegation:")
    print(delegation)
    print()

    empty_store = AgentExperienceStore()
    without_wrapped = ExperienceAwareDelegate(DelegateAgent(llm=llm_client), empty_store)
    without = without_wrapped.sample_candidates(
        principal_id=principal_id, task_category=task_category,
        delegation=delegation, context=context, n=n)

    print("=== WITHOUT experience ===\n")
    print(f"Experiences retrieved: {without.experience_count}")
    _print_distribution(without.distribution, n)
    print()
    _print_call_stats(without.distribution.responses)
    print()

    filled_store = AgentExperienceStore()
    filled_store.add(EXAMPLE_PRIOR_EXPERIENCE)
    with_wrapped = ExperienceAwareDelegate(DelegateAgent(llm=llm_client), filled_store)
    with_exp = with_wrapped.sample_candidates(
        principal_id=principal_id, task_category=task_category,
        delegation=delegation, context=context, n=n)

    print("=== WITH 1 verified experience ===\n")
    print(f"Experiences retrieved: {with_exp.experience_count}")
    for exp in with_exp.experiences_used:
        print(f"  - {exp.episode_id}: {exp.delegation!r} -> "
             f"{exp.confirmed_interpretation.action}:{exp.confirmed_interpretation.resource} "
             f"(confirmed via: {exp.principal_answer!r})")
    print()
    _print_distribution(with_exp.distribution, n)
    print()
    _print_call_stats(with_exp.distribution.responses)
    print()

    print("=== Entropy reduction ===\n")
    delta = without.distribution.entropy - with_exp.distribution.entropy
    print(f"H_without = {without.distribution.entropy:.3f} bits")
    print(f"H_with    = {with_exp.distribution.entropy:.3f} bits")
    print(f"delta H   = {delta:.3f} bits")


def run_semantic(llm_client, delegation: str = EXAMPLE_DELEGATION,
                 proposal: Interpretation = EXAMPLE_MISREAD_PROPOSAL) -> SemanticVerdict:
    print("=== SemanticVerifierAgent ===\n")
    agent = SemanticVerifierAgent(llm_client=llm_client)

    print("Delegation:")
    print(delegation)
    print()
    print("Delegate proposal:")
    print(_fmt_interpretation(proposal))
    print()

    verdict = agent.verify_agent_proposal(delegation=delegation, proposal=proposal)
    print("Semantic verifier's own independent reading:")
    print(_fmt_interpretation(verdict.interpretation))
    print()
    print(f"Semantic verdict: {verdict.status} ({verdict.route})")
    print()

    responses = [verdict.response] if verdict.response is not None else []
    _print_call_stats(responses)
    print()
    return verdict


def run_authority(llm_client, proposal: Interpretation = EXAMPLE_SCOPE_EXCEEDED_PROPOSAL,
                  budget: Budget = EXAMPLE_BUDGET) -> AuthorityVerdict:
    """기본 proposal은 EXAMPLE_MISREAD_PROPOSAL(export)이 아니라
    EXAMPLE_SCOPE_EXCEEDED_PROPOSAL(read, 더 넓은 scope)이다 — action
    자체가 위임 밖인 export는 no_grant 하드 리젝트라 LLM을 전혀 부르지
    않는다. scope_exceeded여야 실제로 model-backed 협상 경로(LLM 호출
    1회 + 재검증)가 돌아간다."""
    print("=== AuthorityVerifierAgent ===\n")
    agent = AuthorityVerifierAgent(llm_client=llm_client)

    print("Proposal:")
    print(_fmt_interpretation(proposal))
    print()
    print(f"Delegated authority (budget): {budget}")
    print()

    verdict = agent.verify_agent_proposal(proposal=proposal, budget=budget)
    print(f"Authority verdict: {verdict.status.upper()} — {verdict.reason}")
    print()

    responses = [verdict.response] if verdict.response is not None else []
    _print_call_stats(responses)
    print()
    return verdict


def run_all(llm_client) -> None:
    print("############################################################")
    print("# Full chain: Principal -> Delegate -> Semantic + Authority")
    print("# Fusion (EXECUTE/REJECT) is deliberately NOT computed here —")
    print("# this role exists to look at each stage's raw, unfused output")
    print("# in isolation. For the actual fused decision, real B6 runtime,")
    print("# use --role runtime instead.")
    print("############################################################\n")

    deleg, intent = run_principal(llm_client)
    proposal = run_delegate(llm_client, delegation=deleg.delegation)
    sem_verdict = run_semantic(llm_client, delegation=deleg.delegation,
                               proposal=proposal.interpretation)
    auth_verdict = run_authority(llm_client, proposal=proposal.interpretation,
                                 budget=EXAMPLE_BUDGET)

    print("=== Comparison: Principal's independent restatement vs. B's proposal ===\n")
    print("Principal's independent restatement:")
    print(_fmt_interpretation(intent.intended_action))
    print()
    print("Delegate's actual proposal:")
    print(_fmt_interpretation(proposal.interpretation))
    print()
    if intent.intended_action != proposal.interpretation:
        print("NOTE: these diverge. This is exactly the independent runtime")
        print("signal B6's oracle-free fusion is designed to use — neither")
        print("value came from task.truth, and Semantic/Authority never saw")
        print("this restatement before producing their own verdicts.")
    else:
        print("NOTE: these agree in this run — not guaranteed on every call,")
        print("since both delegate() and restate_intent() are real, separate")
        print("model calls over natural language.")
    print()

    all_responses = [deleg.response, intent.response, proposal.response]
    if sem_verdict.response is not None:
        all_responses.append(sem_verdict.response)
    if auth_verdict.response is not None:
        all_responses.append(auth_verdict.response)

    print("=== Totals across all four agents ===\n")
    _print_call_stats(all_responses)


def run_runtime(llm_client, goal: str = EXAMPLE_GOAL, context: str = EXAMPLE_CONTEXT,
                budget: Budget = EXAMPLE_BUDGET) -> AgentRuntimeResult:
    """B6의 실제 oracle-free runtime을 그대로 호출한다 — fusion 로직을 여기서
    다시 구현하지 않는다. `AgentDelegationRuntime.run()`이 하는 것과 똑같이
    네 Agent를 실제 API로 순서대로 호출하고, 그 결과(`AgentRuntimeResult`)를
    그대로 출력만 한다."""
    print("############################################################")
    print("# AgentDelegationRuntime.run() — B6 oracle-free runtime, real API")
    print("# (calls the actual runtime class; fusion is not reimplemented")
    print("#  here)")
    print("############################################################\n")

    runtime = AgentDelegationRuntime(
        principal=PrincipalAgent(llm=llm_client),
        delegate=DelegateAgent(llm=llm_client),
        semantic_verifier=SemanticVerifierAgent(llm_client=llm_client),
        authority_verifier=AuthorityVerifierAgent(llm_client=llm_client),
    )
    result = runtime.run(goal=goal, context=context, budget=budget)

    print("Principal delegation:")
    print(result.delegation.delegation)
    print()

    print("Principal independent intent:")
    print(_fmt_interpretation(result.principal_intent.intended_action))
    print()

    print("Delegate proposal:")
    print(_fmt_interpretation(result.proposal.interpretation))
    print()

    print(f"Semantic: {result.semantic_verdict.status} ({result.semantic_verdict.route})")
    print("Semantic verifier's own independent reading:")
    print(_fmt_interpretation(result.semantic_verdict.interpretation))
    print()

    print(f"Authority: {result.authority_verdict.status.upper()} — "
         f"{result.authority_verdict.reason}")
    print()

    print("Final interpretation:")
    print(_fmt_interpretation(result.final_interpretation))
    print()

    print(f"Principal match: {result.principal_match}")
    print()

    print(f"FINAL DECISION: {result.decision}")
    print(f"Reason: {result.reason}")
    print()

    responses = [result.delegation.response, result.principal_intent.response,
                result.proposal.response]
    if result.semantic_verdict.response is not None:
        responses.append(result.semantic_verdict.response)
    if result.authority_verdict.response is not None:
        responses.append(result.authority_verdict.response)
    _print_call_stats(responses)
    return result


def main(argv: list[str] | None = None) -> int:
    try:  # Windows 기본 콘솔(cp949 등)의 UnicodeEncodeError 방지
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", required=True,
                   choices=["principal", "delegate", "semantic", "authority",
                            "all", "runtime", "sample", "clarify", "experience"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--n", type=int, default=10,
                   help="--role sample/clarify/experience only: number of "
                        "independent completions to draw per distribution "
                        "(default 10, matching B7a/B7b/B7d's pilot default).")
    p.add_argument("--entropy-threshold", type=float, default=0.8,
                   help="--role clarify only: entropy (bits) above which a "
                        "clarifying question is asked (default 0.8 -- a pilot "
                        "value, not a tuned research threshold).")
    args = p.parse_args(argv)

    llm_client = build_llm_client(args.model)

    if args.role == "principal":
        run_principal(llm_client)
    elif args.role == "delegate":
        run_delegate(llm_client)
    elif args.role == "semantic":
        run_semantic(llm_client)
    elif args.role == "authority":
        run_authority(llm_client)
    elif args.role == "runtime":
        run_runtime(llm_client)
    elif args.role == "sample":
        run_sample(llm_client, n=args.n)
    elif args.role == "clarify":
        run_clarify(llm_client, n=args.n, entropy_threshold=args.entropy_threshold)
    elif args.role == "experience":
        run_experience(llm_client, n=args.n)
    else:
        run_all(llm_client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
