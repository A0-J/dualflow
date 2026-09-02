"""Agent-to-Agent 권한 위임 검증 프레임워크.

Authority Flow (규칙 기반 상한선) ‖ Semantic Flow (Experience → Entropy →
Clarification → LLM) → Joint Verification (E = Authority ∩ Semantic + 매칭 검증).
"""

from .capability import (
    ANY, Agent, AuthorityResult, Budget, Privilege, check_authority,
    delegate, delegation_chain, scope_leq, scope_meet,
)
from .semantic import (
    Answer, Belief, DIMENSIONS, ExperienceStore, Interpretation, Principal, Question,
    apply_answer, build_belief, conditional_entropy, entropy, information_gain,
    normalize, normalized_entropy, select_question, top,
)
from .rule_engine import (
    ESCALATE, EXECUTE as SOP_EXECUTE, Fields, MatchResult, REJECT as SOP_REJECT,
    RuleEngine, classify, match_intent, sim_path,
)
from .sage_baseline import (
    SageAgentBaseline, SageCandidate, build_candidates as sage_build_candidates,
    evpi as sage_evpi, pi as sage_pi,
)
from .llm import AnthropicJudge, LLMJudge, ScriptedJudge, TopBeliefJudge
from .framework import (
    Config, DelegationTask, DelegationVerifier, EXECUTE, REJECT, Verdict,
    evaluate, outcome,
)
from .bench import PRINCIPAL, build_judge, build_tasks

__version__ = "0.1.0"
