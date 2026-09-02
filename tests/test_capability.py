"""AUTHORITY FLOW 검증 — 순서관계, meet, 단조 감쇠, 비증폭."""

from __future__ import annotations

import itertools

import pytest

from dualflow.capability import (
    ANY, Budget, Privilege, check_authority, delegate, delegation_chain,
    scope_leq, scope_meet,
)

READ_ALL = Privilege("read", "file", "/")
READ_REPORTS = Privilege("read", "file", "/reports/")
SEND_CORP = Privilege("send", "email", "*.corp.com")
SEND_ANON = Privilege("send", "email", ANY, {"anonymized"})
DELETE_TMP = Privilege("delete", "file", "/tmp/")


class TestScope:
    def test_universal_scope_is_top(self):
        assert scope_leq("/reports/", ANY) and not scope_leq(ANY, "/reports/")

    def test_path_prefix(self):
        assert scope_leq("/reports/2026-08/", "/reports/")
        assert not scope_leq("/reports/", "/reports/2026-08/")
        assert not scope_leq("/hr/", "/reports/")

    def test_domain_suffix(self):
        assert scope_leq("mail.corp.com", "*.corp.com")
        assert not scope_leq("partner.example.com", "*.corp.com")

    def test_meet_is_the_narrower_or_empty(self):
        assert scope_meet("/reports/2026-08/", "/reports/") == "/reports/2026-08/"
        assert scope_meet("/reports/", ANY) == "/reports/"
        assert scope_meet("/hr/", "/reports/") is None


class TestPrivilegeOrder:
    def test_narrower_scope_is_weaker(self):
        assert READ_REPORTS <= READ_ALL and not READ_ALL <= READ_REPORTS

    def test_extra_condition_is_weaker(self):
        strong = Privilege("send", "email", ANY)
        weak = Privilege("send", "email", ANY, {"anonymized"})
        assert weak <= strong and not strong <= weak

    def test_different_action_is_incomparable(self):
        assert not READ_ALL <= DELETE_TMP and not DELETE_TMP <= READ_ALL
        assert READ_ALL.meet(DELETE_TMP) is None

    def test_meet_unions_conditions_and_narrows_scope(self):
        m = Privilege("send", "email", ANY, {"anonymized"}).meet(SEND_CORP)
        assert m == Privilege("send", "email", "*.corp.com", {"anonymized"})


class TestBudget:
    def test_membership_is_downward_closed(self):
        b = Budget.of(READ_ALL)
        assert READ_REPORTS in b
        assert Privilege("read", "file", "/reports/2026-08/") in b
        assert DELETE_TMP not in b

    def test_generators_are_reduced_to_maximal(self):
        b = Budget.of(READ_ALL, READ_REPORTS)
        assert b.generators == frozenset({READ_ALL})

    def test_meet_is_a_lower_bound(self):
        a = Budget.of(READ_ALL, SEND_CORP, SEND_ANON)
        c = Budget.of(READ_REPORTS, SEND_CORP)
        m = a.meet(c)
        assert m <= a and m <= c

    def test_meet_of_disjoint_is_empty(self):
        assert Budget.of(READ_ALL).meet(Budget.of(DELETE_TMP)).is_empty


class TestNonAmplification:
    """Theorem: 위임 합성은 권한을 늘릴 수 없다."""

    principal = Budget.of(READ_ALL, SEND_CORP, SEND_ANON)

    def test_delegation_never_widens(self):
        greedy = Budget.of(Privilege("delete", "file", ANY),
                           Privilege("send", "email", ANY))
        assert delegate(self.principal, greedy) <= self.principal

    def test_chain_is_monotonically_decreasing(self):
        ceilings = [Budget.of(READ_ALL, SEND_CORP), Budget.of(READ_REPORTS, SEND_CORP)]
        chain = delegation_chain(self.principal, ceilings)
        for a, b in itertools.pairwise(chain):
            assert b <= a

    def test_anything_allowed_downstream_was_allowed_upstream(self):
        ceilings = [Budget.of(READ_ALL, SEND_CORP), Budget.of(READ_REPORTS, SEND_CORP)]
        final = delegation_chain(self.principal, ceilings)[-1]
        for g in final.generators:
            assert g in self.principal

    def test_permission_laundering_is_blocked(self):
        """각 홉은 정당하지만 합성으로 외부 발송 권한을 되찾을 수 없다."""
        final = delegation_chain(self.principal,
                                 [Budget.of(SEND_CORP), Budget.of(SEND_CORP)])[-1]
        assert not check_authority(final, Privilege("send", "email", "partner.example.com"))

    def test_condition_cannot_be_dropped_by_composition(self):
        principal = Budget.of(SEND_ANON)
        final = delegate(principal, Budget.of(Privilege("send", "email", ANY)))
        assert Privilege("send", "email", "x.example.com") not in final
        assert Privilege("send", "email", "x.example.com", {"anonymized"}) in final


class TestAuthorityMessages:
    budget = Budget.of(READ_ALL, SEND_CORP, SEND_ANON)

    @pytest.mark.parametrize("req,fragment", [
        (DELETE_TMP, "허용되지 않은 action/resource"),
        (Privilege("send", "email", "partner.example.com"), "조건"),
        (Privilege("read", "db", "/x/"), "허용되지 않은 action/resource"),
    ])
    def test_reason_is_actionable(self, req, fragment):
        r = check_authority(self.budget, req)
        assert not r.allowed and fragment in r.reason

    def test_empty_budget_is_reported(self):
        r = check_authority(Budget.of(), READ_ALL)
        assert not r.allowed and "소진" in r.reason
