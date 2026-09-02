"""
AUTHORITY FLOW — 규칙 기반 허용 범위 상한선.

ChainCaps (Monotonic Capability Attenuation, arXiv 2605.26542) 의 budget algebra 를
Agent-to-Agent 위임 맥락으로 확장한 구현.

원 논문과의 차이:
  ChainCaps          단일 에이전트의 도구 체인. 값(value)이 sink 로 이동하며 권한 감쇠.
                     sink privilege = (op, scope)
  본 구현            Agent A -> B -> C 위임 체인. 위임 명세가 권한 상한선을 감쇠.
                     privilege = (action, resource, scope, condition)  <- 미팅자료 슬라이드

핵심 불변식은 동일하다: 합성(위임)은 권한을 보존하거나 줄일 뿐, 늘릴 수 없다.
Theorem 1 (Non-amplification) 참고 및 tests/test_capability.py 참고.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

ANY = "*"


# --------------------------------------------------------------------------
# Scope 순서관계
# --------------------------------------------------------------------------
def scope_leq(a: str, b: str) -> bool:
    """a ⊆ b 인가. 경로 접두사와 도메인 접미사 두 형태를 지원한다."""
    if b == ANY:
        return True
    if a == ANY:
        return False
    if a == b:
        return True
    if b.startswith("*."):                       # 도메인: *.corp.com
        return a.endswith(b[1:])
    if b.endswith("/"):                          # 경로: /reports/
        return a.startswith(b)
    return False


def scope_meet(a: str, b: str) -> str | None:
    """두 scope 의 최대 하한. 겹치지 않으면 None(=공집합)."""
    if scope_leq(a, b):
        return a
    if scope_leq(b, a):
        return b
    return None


# --------------------------------------------------------------------------
# Privilege = (action, resource, scope, condition)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Privilege:
    """조건(condition)이 많을수록 '약한' 권한이다.

    read/file//reports/ 는 read/file// 보다 약하고(좁은 scope),
    send/email/*/{anonymized} 는 send/email/*/{} 보다 약하다(추가 조건).
    """
    action: str
    resource: str
    scope: str = ANY
    condition: frozenset[str] = frozenset()

    def __post_init__(self):
        if not isinstance(self.condition, frozenset):
            object.__setattr__(self, "condition", frozenset(self.condition))

    def __le__(self, other: "Privilege") -> bool:            # p ⪯ q
        return (self.action == other.action
                and self.resource == other.resource
                and scope_leq(self.scope, other.scope)
                and self.condition >= other.condition)

    def meet(self, other: "Privilege") -> "Privilege | None":
        if self.action != other.action or self.resource != other.resource:
            return None
        scope = scope_meet(self.scope, other.scope)
        if scope is None:
            return None
        return Privilege(self.action, self.resource, scope,
                         self.condition | other.condition)

    def __str__(self) -> str:
        cond = ("+" + ",".join(sorted(self.condition))) if self.condition else ""
        return f"{self.action}:{self.resource}@{self.scope}{cond}"


def _reduce(gens: Iterable[Privilege]) -> frozenset[Privilege]:
    """극대 원소만 남겨 downward-closed 집합을 정규형으로 표현."""
    gens = set(gens)
    keep = set()
    for p in gens:
        if not any(p != q and p <= q for q in gens):
            keep.add(p)
    return frozenset(keep)


# --------------------------------------------------------------------------
# Budget = downward-closed set of privileges (극대 생성자로 표현)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Budget:
    generators: frozenset[Privilege] = field(default_factory=frozenset)

    @staticmethod
    def of(*privileges: Privilege) -> "Budget":
        return Budget(_reduce(privileges))

    @property
    def is_empty(self) -> bool:
        return not self.generators

    def __contains__(self, p: Privilege) -> bool:
        return any(p <= g for g in self.generators)

    def meet(self, other: "Budget") -> "Budget":
        """B1 ∩ B2 — ChainCaps Eq.(2) 의 meet rule."""
        out = []
        for p in self.generators:
            for q in other.generators:
                m = p.meet(q)
                if m is not None:
                    out.append(m)
        return Budget(_reduce(out))

    def __le__(self, other: "Budget") -> bool:               # B1 ⊆ B2
        return all(g in other for g in self.generators)

    def __str__(self) -> str:
        if self.is_empty:
            return "{}"
        return "{" + ", ".join(sorted(str(g) for g in self.generators)) + "}"


TOP = Budget.of()   # 빈 예산 = 아무 것도 못 함 (fail-closed 기본값)


# --------------------------------------------------------------------------
# 위임 체인
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Agent:
    name: str
    budget: Budget


def delegate(principal: Budget, spec_ceiling: Budget | None = None) -> Budget:
    """A -> B 위임 1홉. 명세가 제시한 상한선과 위임자의 권한을 교집합한다.

    명세가 아무리 넓은 권한을 요구해도(spec_ceiling 이 넓어도) 결과는 principal 을
    넘지 못한다. 이것이 '경험/LLM 이 아무리 확신해도 상한선은 못 넘는다' 의 근거다.
    """
    if spec_ceiling is None:
        return principal
    return principal.meet(spec_ceiling)


def delegation_chain(principal: Budget, ceilings: Iterable[Budget | None]) -> list[Budget]:
    """다홉 위임의 각 단계별 유효 예산. 단조 감소가 보장된다."""
    out, cur = [principal], principal
    for c in ceilings:
        cur = delegate(cur, c)
        out.append(cur)
    return out


@dataclass
class AuthorityResult:
    allowed: bool
    effective_budget: Budget
    requested: Privilege
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


def check_authority(effective: Budget, requested: Privilege) -> AuthorityResult:
    """Sink rule — ChainCaps Eq.(3). 위반 시 즉시 차단(하드 제약)."""
    ok = requested in effective
    if ok:
        reason = "요청 권한이 유효 예산 안에 있음"
    elif effective.is_empty:
        reason = "위임 체인에서 유효 예산이 소진됨"
    else:
        same = [g for g in effective.generators
                if g.action == requested.action and g.resource == requested.resource]
        if not same:
            reason = f"허용되지 않은 action/resource: {requested.action}:{requested.resource}"
        elif any(not requested.condition >= g.condition for g in same):
            missing = set().union(*(g.condition for g in same)) - requested.condition
            reason = f"필수 조건 미충족: {sorted(missing)}"
        else:
            reason = f"scope 위반: {requested.scope} ⊄ " + \
                     "/".join(sorted(g.scope for g in same))
    return AuthorityResult(ok, effective, requested, reason)
