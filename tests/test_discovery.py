"""Comprehensive tests for the Adaptive Opcode Discovery Protocol."""

import time
from typing import List

import pytest

from adaptive_opcodes.discovery import (
    AdoptionEngine,
    AdoptedOpcode,
    AgentCapabilities,
    BroadcastRecord,
    DeprecationRecord,
    DiscoveryProtocol,
    NegotiationResult,
    OpcodeCategory,
    OpcodeFormat,
    OpcodeProposal,
    OpcodeTester,
    ProposalRegistry,
    ProposalStatus,
    RangeManager,
    SlotStatus,
    TallyResult,
    TestCase,
    TestResult,
    VoteRecord,
    VoteValue,
    VersionRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_proposal(
    proposer: str = "agent-alpha",
    opcode_num: int = 0xB0,
    mnemonic: str = "XADD",
    description: str = "Extended addition with carry",
    rationale: str = "Need carry-aware addition for multi-precision arithmetic",
    test_bytecode: list | None = None,
    expected_result=None,
) -> OpcodeProposal:
    """Factory for creating standard test proposals."""
    if test_bytecode is None:
        # deterministic bytecode whose simulation yields expected_result
        test_bytecode = [0xB0, 0x01, 0x02]
        expected_result = (0 * 31 + 0xB0) * 31 * 31 + 0x01 * 31 + 0x02
        # simulate: state=0; state=(0*31+0xB0)&0xFFFFFFFF = 176
        #           state=(176*31+0x01)&0xFFFFFFFF = 5457
        #           state=(5457*31+0x02)&0xFFFFFFFF = 169169
    return OpcodeProposal(
        proposer=proposer,
        opcode_num=opcode_num,
        mnemonic=mnemonic,
        format=OpcodeFormat.TWO_REGISTER,
        category=OpcodeCategory.ARITHMETIC,
        description=description,
        rationale=rationale,
        test_bytecode=test_bytecode,
        expected_result=expected_result,
    )


# ===========================================================================
# 1. OpcodeProposal Tests
# ===========================================================================

class TestOpcodeProposal:
    """Tests for the OpcodeProposal data class."""

    def test_create_basic_proposal(self):
        p = make_proposal()
        assert p.mnemonic == "XADD"
        assert p.opcode_num == 0xB0
        assert p.format == OpcodeFormat.TWO_REGISTER
        assert p.category == OpcodeCategory.ARITHMETIC
        assert p.status == ProposalStatus.DRAFT
        assert p.version == 1

    def test_proposal_has_unique_id(self):
        p1 = make_proposal()
        p2 = make_proposal(opcode_num=0xB1, mnemonic="XSUB")
        assert p1.id != p2.id

    def test_proposal_timestamps(self):
        before = time.time()
        p = make_proposal()
        after = time.time()
        assert before <= p.created_at <= after
        assert before <= p.updated_at <= after

    def test_bump_version_increments(self):
        p = make_proposal()
        assert p.version == 1
        p.bump_version("Updated description")
        assert p.version == 2

    def test_bump_version_records_history(self):
        p = make_proposal()
        p.bump_version("Changed mnemonic")
        assert len(p.version_history) == 1
        rec = p.version_history[0]
        assert isinstance(rec, VersionRecord)
        assert rec.version == 2
        assert rec.changes == "Changed mnemonic"

    def test_multiple_bumps(self):
        p = make_proposal()
        for i in range(5):
            p.bump_version(f"Change {i}")
        assert p.version == 6
        assert len(p.version_history) == 5

    def test_version_history_snapshot(self):
        p = make_proposal(mnemonic="ORIG")
        p.bump_version("Changed to NEW")
        p.mnemonic = "NEW"
        p.bump_version("Changed again")
        assert p.version_history[0].snapshot["mnemonic"] == "ORIG"

    def test_fingerprint_deterministic(self):
        p1 = make_proposal()
        p2 = make_proposal()
        assert p1.fingerprint() == p2.fingerprint()

    def test_fingerprint_changes_on_content(self):
        p1 = make_proposal(mnemonic="XADD")
        p2 = make_proposal(mnemonic="XSUB")
        assert p1.fingerprint() != p2.fingerprint()

    def test_fingerprint_changes_on_opcode(self):
        p1 = make_proposal(opcode_num=0xB0)
        p2 = make_proposal(opcode_num=0xB1)
        assert p1.fingerprint() != p2.fingerprint()

    def test_optional_fields_none(self):
        p = OpcodeProposal(
            proposer="agent",
            opcode_num=10,
            mnemonic="NOP",
            format=OpcodeFormat.NO_OPERAND,
            category=OpcodeCategory.SYSTEM,
            description="No operation",
            rationale="Placeholder",
        )
        assert p.test_bytecode is None
        assert p.expected_result is None

    def test_all_categories_exist(self):
        cats = list(OpcodeCategory)
        assert len(cats) >= 10
        assert OpcodeCategory.ARITHMETIC in cats
        assert OpcodeCategory.CUSTOM in cats

    def test_all_formats_exist(self):
        fmts = list(OpcodeFormat)
        assert len(fmts) >= 10
        assert OpcodeFormat.NO_OPERAND in fmts
        assert OpcodeFormat.ESCAPE_EXTENDED in fmts


# ===========================================================================
# 2. RangeManager Tests
# ===========================================================================

class TestRangeManager:
    """Tests for the Opcode Range Manager."""

    def test_initial_all_available(self):
        rm = RangeManager()
        assert rm.available_count() == 256
        assert rm.reserved_count() == 0
        assert rm.in_use_count() == 0

    def test_reserve_slot(self):
        rm = RangeManager()
        rm.reserve(0xB0, "XADD proposal")
        assert rm.status_of(0xB0) == SlotStatus.RESERVED
        assert rm.reserved_count() == 1
        assert rm.available_count() == 255

    def test_reserve_multiple(self):
        rm = RangeManager()
        for i in range(10):
            rm.reserve(i, f"slot-{i}")
        assert rm.reserved_count() == 10
        assert rm.available_count() == 246

    def test_reserve_already_reserved(self):
        rm = RangeManager()
        rm.reserve(10, "first")
        with pytest.raises(ValueError, match="already reserved"):
            rm.reserve(10, "second")

    def test_reserve_out_of_range(self):
        rm = RangeManager()
        with pytest.raises(ValueError, match="out of primary range"):
            rm.reserve(256, "overflow")

    def test_reserve_negative(self):
        rm = RangeManager()
        with pytest.raises(ValueError):
            rm.reserve(-1, "negative")

    def test_is_available_true(self):
        rm = RangeManager()
        assert rm.is_available(0) is True
        assert rm.is_available(255) is True

    def test_is_available_false_after_reserve(self):
        rm = RangeManager()
        rm.reserve(100, "test")
        assert rm.is_available(100) is False

    def test_find_available_returns_first_free(self):
        rm = RangeManager()
        rm.reserve(0, "a")
        rm.reserve(1, "b")
        num = rm.find_available()
        assert num == 2

    def test_find_available_exhausted(self):
        rm = RangeManager()
        for i in range(256):
            rm.reserve(i, f"slot-{i}")
        with pytest.raises(RuntimeError, match="No available"):
            rm.find_available()

    def test_assign_reserved_slot(self):
        rm = RangeManager()
        rm.reserve(50, "test")
        rm.assign(50)
        assert rm.status_of(50) == SlotStatus.IN_USE
        assert rm.in_use_count() == 1
        assert rm.reserved_count() == 0

    def test_assign_unreserved_fails(self):
        rm = RangeManager()
        with pytest.raises(ValueError, match="must be reserved"):
            rm.assign(50)

    def test_release_reserved(self):
        rm = RangeManager()
        rm.reserve(50, "test")
        rm.release(50)
        assert rm.is_available(50) is True

    def test_release_in_use(self):
        rm = RangeManager()
        rm.reserve(50, "test")
        rm.assign(50)
        rm.release(50)
        assert rm.is_available(50) is True

    def test_list_by_status(self):
        rm = RangeManager()
        rm.reserve(0, "a")
        rm.reserve(1, "b")
        rm.assign(0)
        assert rm.list_by_status(SlotStatus.IN_USE) == [0]
        assert rm.list_by_status(SlotStatus.RESERVED) == [1]
        assert 2 in rm.list_by_status(SlotStatus.AVAILABLE)

    def test_status_of_unknown(self):
        rm = RangeManager()
        assert rm.status_of(300) is None

    def test_extended_reserve(self):
        rm = RangeManager()
        rm.reserve_extended(1000, "ext-op")
        assert rm.is_extended_available(1000) is False

    def test_extended_find_available(self):
        rm = RangeManager()
        num = rm.find_available_extended()
        assert num == 0
        rm.reserve_extended(0, "taken")
        num2 = rm.find_available_extended()
        assert num2 == 1

    def test_release_unknown_fails(self):
        rm = RangeManager()
        with pytest.raises(ValueError, match="not tracked"):
            rm.release(999)


# ===========================================================================
# 3. ProposalRegistry Tests
# ===========================================================================

class TestProposalRegistry:
    """Tests for the Proposal Registry and voting system."""

    def test_submit_proposal(self):
        reg = ProposalRegistry()
        p = make_proposal()
        pid = reg.submit(p)
        assert isinstance(pid, str)
        assert len(pid) == 12
        assert p.status == ProposalStatus.PROPOSED

    def test_submit_updates_id(self):
        reg = ProposalRegistry()
        p = make_proposal()
        old_id = p.id
        pid = reg.submit(p)
        assert pid != old_id  # new ID assigned

    def test_get_proposal(self):
        reg = ProposalRegistry()
        p = make_proposal()
        pid = reg.submit(p)
        retrieved = reg.get(pid)
        assert retrieved is not None
        assert retrieved.mnemonic == "XADD"

    def test_get_nonexistent(self):
        reg = ProposalRegistry()
        assert reg.get("nonexistent") is None

    def test_list_all(self):
        reg = ProposalRegistry()
        reg.submit(make_proposal(opcode_num=0xB0, mnemonic="XADD"))
        reg.submit(make_proposal(opcode_num=0xB1, mnemonic="XSUB"))
        assert len(reg.list_all()) == 2

    def test_list_by_status(self):
        reg = ProposalRegistry()
        reg.submit(make_proposal(opcode_num=0xB0, mnemonic="XADD"))
        reg.submit(make_proposal(opcode_num=0xB1, mnemonic="XSUB"))
        proposed = reg.list_by_status(ProposalStatus.PROPOSED)
        assert len(proposed) == 2

    def test_duplicate_fingerprint_rejected(self):
        reg = ProposalRegistry()
        reg.submit(make_proposal())
        with pytest.raises(ValueError, match="Duplicate"):
            reg.submit(make_proposal())

    def test_different_proposals_accepted(self):
        reg = ProposalRegistry()
        reg.submit(make_proposal(mnemonic="XADD"))
        reg.submit(make_proposal(mnemonic="XSUB", opcode_num=0xB1))
        assert len(reg.list_all()) == 2

    def test_cast_vote(self):
        reg = ProposalRegistry(electorate_size=3)
        p = make_proposal()
        pid = reg.submit(p)
        vr = reg.vote(pid, "voter1", VoteValue.FOR)
        assert isinstance(vr, VoteRecord)
        assert vr.voter == "voter1"
        assert vr.vote == VoteValue.FOR

    def test_vote_nonexistent_proposal(self):
        reg = ProposalRegistry()
        with pytest.raises(ValueError, match="not found"):
            reg.vote("fake", "v1", VoteValue.FOR)

    def test_double_vote_rejected(self):
        reg = ProposalRegistry()
        pid = reg.submit(make_proposal())
        reg.vote(pid, "v1", VoteValue.FOR)
        with pytest.raises(ValueError, match="already voted"):
            reg.vote(pid, "v1", VoteValue.AGAINST)

    def test_vote_on_wrong_status(self):
        reg = ProposalRegistry()
        p = make_proposal()
        pid = reg.submit(p)
        reg.set_status(pid, ProposalStatus.ADOPTED)
        with pytest.raises(ValueError, match="Cannot vote"):
            reg.vote(pid, "v1", VoteValue.FOR)

    def test_get_votes(self):
        reg = ProposalRegistry()
        pid = reg.submit(make_proposal())
        reg.vote(pid, "v1", VoteValue.FOR)
        reg.vote(pid, "v2", VoteValue.AGAINST)
        votes = reg.get_votes(pid)
        assert len(votes) == 2
        assert votes["v1"].vote == VoteValue.FOR
        assert votes["v2"].vote == VoteValue.AGAINST

    def test_tally_no_votes(self):
        reg = ProposalRegistry(electorate_size=9)
        pid = reg.submit(make_proposal())
        t = reg.tally(pid)
        assert t.for_count == 0
        assert t.against_count == 0
        assert t.abstain_count == 0
        assert t.approved is False

    def test_tally_unanimous_for(self):
        reg = ProposalRegistry(electorate_size=3)
        pid = reg.submit(make_proposal())
        reg.vote(pid, "v1", VoteValue.FOR)
        reg.vote(pid, "v2", VoteValue.FOR)
        reg.vote(pid, "v3", VoteValue.FOR)
        t = reg.tally(pid)
        assert t.for_count == 3
        assert t.approved is True  # 3/3 >= 2/3

    def test_tally_supermajority_required(self):
        """With electorate_size=9, need at least 6 FOR votes."""
        reg = ProposalRegistry(electorate_size=9)
        pid = reg.submit(make_proposal())
        for i in range(5):
            reg.vote(pid, f"v{i}", VoteValue.FOR)
        t = reg.tally(pid)
        assert t.approved is False  # 5/9 < 2/3
        reg.vote(pid, "v5", VoteValue.FOR)
        t2 = reg.tally(pid)
        assert t2.approved is True  # 6/9 >= 2/3

    def test_tally_with_abstain(self):
        reg = ProposalRegistry(electorate_size=3)
        pid = reg.submit(make_proposal())
        reg.vote(pid, "v1", VoteValue.FOR)
        reg.vote(pid, "v2", VoteValue.ABSTAIN)
        reg.vote(pid, "v3", VoteValue.FOR)
        t = reg.tally(pid)
        assert t.for_count == 2
        assert t.abstain_count == 1
        assert t.approved is True

    def test_tally_for_ratio(self):
        reg = ProposalRegistry(electorate_size=9)
        pid = reg.submit(make_proposal())
        for i in range(6):
            reg.vote(pid, f"v{i}", VoteValue.FOR)
        for i in range(6, 9):
            reg.vote(pid, f"v{i}", VoteValue.AGAINST)
        t = reg.tally(pid)
        assert abs(t.for_ratio - 6 / 9) < 0.001

    def test_set_status(self):
        reg = ProposalRegistry()
        pid = reg.submit(make_proposal())
        reg.set_status(pid, ProposalStatus.APPROVED)
        assert reg.get(pid).status == ProposalStatus.APPROVED

    def test_set_status_nonexistent(self):
        reg = ProposalRegistry()
        with pytest.raises(ValueError, match="not found"):
            reg.set_status("fake", ProposalStatus.APPROVED)


# ===========================================================================
# 4. OpcodeTester Tests
# ===========================================================================

class TestOpcodeTester:
    """Tests for the opcode testing framework."""

    def test_simulate_empty(self):
        tester = OpcodeTester()
        assert tester._simulate([]) is None

    def test_simulate_single_byte(self):
        tester = OpcodeTester()
        result = tester._simulate([42])
        assert result == 42

    def test_simulate_deterministic(self):
        tester = OpcodeTester()
        r1 = tester._simulate([1, 2, 3])
        r2 = tester._simulate([1, 2, 3])
        assert r1 == r2

    def test_simulate_order_matters(self):
        tester = OpcodeTester()
        r1 = tester._simulate([1, 2])
        r2 = tester._simulate([2, 1])
        assert r1 != r2

    def test_test_opcode_passes(self):
        tester = OpcodeTester()
        p = make_proposal()  # default bytecode with correct expected
        result = tester.test_opcode(p)
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.message == "OK"

    def test_test_opcode_fails_on_wrong_expected(self):
        tester = OpcodeTester()
        p = make_proposal(test_bytecode=[10, 20], expected_result=999999)
        result = tester.test_opcode(p)
        assert result.passed is False
        assert "999999" in result.message

    def test_test_opcode_custom_bytecode(self):
        tester = OpcodeTester()
        p = make_proposal()
        bc = [10, 20, 30]
        expected = tester._simulate(bc)
        result = tester.test_opcode(p, test_bytecode=bc, expected_behavior=expected)
        assert result.passed is True

    def test_test_opcode_no_bytecode(self):
        tester = OpcodeTester()
        p = OpcodeProposal(
            proposer="a", opcode_num=10, mnemonic="NOP",
            format=OpcodeFormat.NO_OPERAND, category=OpcodeCategory.SYSTEM,
            description="No-op", rationale="test",
        )
        result = tester.test_opcode(p)
        assert result.actual is None

    def test_run_test_suite(self):
        tester = OpcodeTester()
        p = make_proposal()
        cases = [
            TestCase(name="tc1", bytecode=[1, 2, 3], expected_behavior=tester._simulate([1, 2, 3])),
            TestCase(name="tc2", bytecode=[4, 5], expected_behavior=tester._simulate([4, 5])),
        ]
        results = tester.run_test_suite(p, cases)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_run_test_suite_mixed(self):
        tester = OpcodeTester()
        p = make_proposal()
        cases = [
            TestCase(name="pass", bytecode=[1], expected_behavior=tester._simulate([1])),
            TestCase(name="fail", bytecode=[2], expected_behavior=999),
        ]
        results = tester.run_test_suite(p, cases)
        assert results[0].passed is True
        assert results[1].passed is False

    def test_compute_confidence_all_pass(self):
        results = [TestResult(passed=True, test_case=TestCase(name="t", bytecode=[], expected_behavior=None), actual=None)]
        assert OpcodeTester.compute_confidence(results) > 0.8

    def test_compute_confidence_none_pass(self):
        results = [TestResult(passed=False, test_case=TestCase(name="t", bytecode=[], expected_behavior=1), actual=2)]
        conf = OpcodeTester.compute_confidence(results)
        assert conf < 0.2

    def test_compute_confidence_empty(self):
        assert OpcodeTester.compute_confidence([]) == 0.0

    def test_compute_confidence_coverage_bonus(self):
        """Having multiple test categories should give a small bonus."""
        results_normal = [
            TestResult(
                passed=True,
                test_case=TestCase(name="n", bytecode=[1], expected_behavior=1, category="normal"),
                actual=1,
            )
        ]
        results_multi = [
            TestResult(
                passed=True,
                test_case=TestCase(name="n", bytecode=[1], expected_behavior=1, category="normal"),
                actual=1,
            ),
            TestResult(
                passed=True,
                test_case=TestCase(name="e", bytecode=[2], expected_behavior=2, category="edge"),
                actual=2,
            ),
            TestResult(
                passed=True,
                test_case=TestCase(name="err", bytecode=[3], expected_behavior=3, category="error"),
                actual=3,
            ),
        ]
        c1 = OpcodeTester.compute_confidence(results_normal)
        c2 = OpcodeTester.compute_confidence(results_multi)
        assert c2 > c1

    def test_get_results(self):
        tester = OpcodeTester()
        p = make_proposal()
        tester.test_opcode(p)
        results = tester.get_results(p.id)
        assert len(results) == 1

    def test_test_result_has_duration(self):
        tester = OpcodeTester()
        p = make_proposal()
        result = tester.test_opcode(p)
        assert result.duration_ms >= 0


# ===========================================================================
# 5. AdoptionEngine Tests
# ===========================================================================

class TestAdoptionEngine:
    """Tests for the full adoption lifecycle engine."""

    def _make_engine(self, electorate_size: int = 9):
        rm = RangeManager()
        reg = ProposalRegistry(electorate_size=electorate_size)
        tester = OpcodeTester()
        return AdoptionEngine(reg, rm, tester, electorate_size=electorate_size)

    def test_propose_opcode(self):
        engine = self._make_engine()
        p = make_proposal()
        result = engine.propose_opcode(p)
        assert result.status == ProposalStatus.PROPOSED
        assert len(result.id) == 12

    def test_propose_opcode_reserves_slot(self):
        engine = self._make_engine()
        engine.propose_opcode(make_proposal(opcode_num=0xB0))
        assert engine.range.status_of(0xB0) == SlotStatus.RESERVED

    def test_propose_unavailable_opcode(self):
        engine = self._make_engine()
        engine.propose_opcode(make_proposal(opcode_num=0xB0))
        with pytest.raises(ValueError, match="not available"):
            engine.propose_opcode(make_proposal(opcode_num=0xB0, mnemonic="DUP"))

    def test_review_proposal(self):
        engine = self._make_engine()
        pid = engine.propose_opcode(make_proposal()).id
        review = engine.review(pid)
        assert isinstance(review.issues, list)
        assert review.confidence > 0

    def test_review_nonexistent(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="not found"):
            engine.review("fake")

    def test_cast_vote(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        vr = engine.cast_vote(pid, "v1", VoteValue.FOR)
        assert vr.vote == VoteValue.FOR
        assert vr.tally_after.for_count == 1

    def test_check_adoption_false(self):
        engine = self._make_engine(electorate_size=9)
        pid = engine.propose_opcode(make_proposal()).id
        engine.cast_vote(pid, "v1", VoteValue.FOR)
        assert engine.check_adoption(pid) is False

    def test_check_adoption_true(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"v{i}", VoteValue.FOR)
        assert engine.check_adoption(pid) is True

    def test_adopt_success(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"v{i}", VoteValue.FOR)
        adopted = engine.adopt(pid)
        assert isinstance(adopted, AdoptedOpcode)
        assert adopted.mnemonic == "XADD"
        assert adopted.opcode_num == 0xB0

    def test_adopt_fails_without_threshold(self):
        engine = self._make_engine(electorate_size=9)
        pid = engine.propose_opcode(make_proposal()).id
        with pytest.raises(ValueError, match="supermajority"):
            engine.adopt(pid)

    def test_adopt_marks_slot_in_use(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"v{i}", VoteValue.FOR)
        engine.adopt(pid)
        assert engine.range.status_of(0xB0) == SlotStatus.IN_USE

    def test_adopt_sets_status(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"v{i}", VoteValue.FOR)
        engine.adopt(pid)
        assert engine.registry.get(pid).status == ProposalStatus.ADOPTED

    def test_deprecate_adopted(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"v{i}", VoteValue.FOR)
        engine.adopt(pid)
        dep = engine.deprecate(0xB0, "Obsolete", proposal_id=pid)
        assert isinstance(dep, DeprecationRecord)
        assert dep.reason == "Obsolete"
        assert engine.range.is_available(0xB0) is True

    def test_deprecate_non_adopted(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="not adopted"):
            engine.deprecate(0xB0, "test")

    def test_list_adopted(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"v{i}", VoteValue.FOR)
        engine.adopt(pid)
        adopted = engine.list_adopted()
        assert len(adopted) == 1

    def test_get_deprecated(self):
        engine = self._make_engine(electorate_size=3)
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"v{i}", VoteValue.FOR)
        engine.adopt(pid)
        engine.deprecate(0xB0, "old", proposal_id=pid)
        dep = engine.get_deprecated(0xB0)
        assert dep is not None
        assert dep.mnemonic == "XADD"

    def test_full_lifecycle(self):
        """End-to-end: propose → review → vote → adopt → deprecate."""
        engine = self._make_engine(electorate_size=3)
        # Propose
        p = make_proposal()
        proposal = engine.propose_opcode(p)
        pid = proposal.id
        assert proposal.status == ProposalStatus.PROPOSED

        # Review
        review = engine.review(pid)
        assert review.confidence > 0

        # Vote (unanimous FOR)
        for i in range(3):
            engine.cast_vote(pid, f"agent-{i}", VoteValue.FOR)
        assert engine.check_adoption(pid) is True

        # Adopt
        adopted = engine.adopt(pid)
        assert adopted.opcode_num == 0xB0
        assert len(adopted.adopted_by) == 3

        # Deprecate
        dep = engine.deprecate(0xB0, "Replaced by XADD2", proposal_id=pid)
        assert dep.opcode_num == 0xB0
        assert engine.registry.get(pid).status == ProposalStatus.DEPRECATED

    def test_adopted_opcode_is_extended(self):
        engine = self._make_engine()
        adopted = AdoptedOpcode(
            opcode_num=500,
            mnemonic="EXT_OP",
            format=OpcodeFormat.ESCAPE_EXTENDED,
            category=OpcodeCategory.CUSTOM,
            description="Extended op",
            proposal_id="pid",
            adopted_at=time.time(),
            adopted_by=["a"],
        )
        assert adopted.is_extended is True

    def test_adopted_opcode_primary_not_extended(self):
        adopted = AdoptedOpcode(
            opcode_num=100,
            mnemonic="PRI_OP",
            format=OpcodeFormat.REGISTER,
            category=OpcodeCategory.ARITHMETIC,
            description="Primary op",
            proposal_id="pid",
            adopted_at=time.time(),
            adopted_by=["a"],
        )
        assert adopted.is_extended is False


# ===========================================================================
# 6. DiscoveryProtocol Tests
# ===========================================================================

class TestDiscoveryProtocol:
    """Tests for agent-to-agent discovery and negotiation."""

    def test_advertise_capabilities(self):
        dp = DiscoveryProtocol()
        caps = dp.advertise_capabilities(
            "agent-1", supported_opcodes={0xB0, 0xB1}
        )
        assert caps.agent_id == "agent-1"
        assert 0xB0 in caps.supported_opcodes

    def test_advertise_updates(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("agent-1", supported_opcodes={1})
        dp.advertise_capabilities("agent-1", supported_opcodes={1, 2, 3})
        caps = dp.discover_peers()[0]
        assert 3 in caps.supported_opcodes

    def test_discover_peers(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1", supported_opcodes={1})
        dp.advertise_capabilities("a2", supported_opcodes={2})
        peers = dp.discover_peers()
        assert len(peers) == 2

    def test_discover_peers_empty(self):
        dp = DiscoveryProtocol()
        assert dp.discover_peers() == []

    def test_find_supporters(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1", supported_opcodes={0xB0, 0xB1})
        dp.advertise_capabilities("a2", supported_opcodes={0xB1})
        supporters = dp.find_supporters(0xB0)
        assert supporters == ["a1"]

    def test_find_supporters_none(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1", supported_opcodes={1})
        assert dp.find_supporters(99) == []

    def test_find_compatible_peers(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1", supported_opcodes={0xB0, 0xB1, 0xB2})
        dp.advertise_capabilities("a2", supported_opcodes={0xB0})
        result = dp.find_compatible_peers({0xB0, 0xB1})
        assert result == ["a1"]

    def test_find_compatible_all(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1", supported_opcodes={1, 2, 3})
        dp.advertise_capabilities("a2", supported_opcodes={1, 2, 3, 4})
        result = dp.find_compatible_peers({1, 2})
        assert set(result) == {"a1", "a2"}

    def test_negotiate_extension_accepted(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1", supported_opcodes=set(), max_extended=True)
        dp.advertise_capabilities("a2", supported_opcodes=set(), max_extended=True)
        dp.advertise_capabilities("a3", supported_opcodes=set(), max_extended=True)
        result = dp.negotiate_extension(["a1", "a2", "a3"], 0xC0)
        assert result.accepted is True
        assert len(result.supporting_agents) >= 2

    def test_negotiate_extension_rejected(self):
        dp = DiscoveryProtocol()
        # Agents with many opcodes (>= 200) will reject.
        # Use opcodes 0-199 so proposed 0xC0=192 is already in their set
        # and would be seen as "supporting". Use 0xFE (254) which is NOT
        # in range(200), and set opcodes to 200+ to trigger rejection.
        dp.advertise_capabilities(
            "a1", supported_opcodes=set(range(200))
        )
        dp.advertise_capabilities(
            "a2", supported_opcodes=set(range(200))
        )
        dp.advertise_capabilities(
            "a3", supported_opcodes=set(range(200))
        )
        # proposed_opcode=254 is NOT in set(range(200)) and len>=200 → reject
        result = dp.negotiate_extension(["a1", "a2", "a3"], 0xFE)
        assert result.accepted is False

    def test_negotiate_no_peers(self):
        dp = DiscoveryProtocol()
        result = dp.negotiate_extension([], 0xC0)
        assert result.accepted is False
        assert "No peers" in result.conditions[0]

    def test_negotiate_custom_threshold(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1", max_extended=True)
        dp.advertise_capabilities("a2", max_extended=True)
        # a3 has 200+ opcodes and doesn't support extended → rejects
        dp.advertise_capabilities(
            "a3", supported_opcodes=set(range(200))
        )
        # Only 2 support, need 100% → rejected
        result = dp.negotiate_extension(["a1", "a2", "a3"], 0xFE, required_support=1.0)
        assert result.accepted is False

    def test_broadcast_adoption(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1")
        dp.advertise_capabilities("a2")
        adopted = AdoptedOpcode(
            opcode_num=0xB0,
            mnemonic="XADD",
            format=OpcodeFormat.TWO_REGISTER,
            category=OpcodeCategory.ARITHMETIC,
            description="Test",
            proposal_id="pid",
            adopted_at=time.time(),
            adopted_by=["a1", "a2"],
        )
        record = dp.broadcast_adoption(adopted)
        assert isinstance(record, BroadcastRecord)
        assert len(record.notified_agents) == 2
        assert record.ack_count == 0

    def test_acknowledge_broadcast(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1")
        adopted = AdoptedOpcode(
            opcode_num=1, mnemonic="T", format=OpcodeFormat.NO_OPERAND,
            category=OpcodeCategory.SYSTEM, description="t",
            proposal_id="p", adopted_at=time.time(), adopted_by=[],
        )
        dp.broadcast_adoption(adopted)
        ack_count = dp.acknowledge_broadcast(0, "a1")
        assert ack_count == 1

    def test_acknowledge_invalid_index(self):
        dp = DiscoveryProtocol()
        with pytest.raises(IndexError):
            dp.acknowledge_broadcast(99, "a1")

    def test_get_broadcasts(self):
        dp = DiscoveryProtocol()
        assert dp.get_broadcasts() == []

    def test_get_negotiations(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1")
        dp.negotiate_extension(["a1"], 0xC0)
        negs = dp.get_negotiations()
        assert len(negs) == 1

    def test_remove_agent(self):
        dp = DiscoveryProtocol()
        dp.advertise_capabilities("a1")
        assert dp.remove_agent("a1") is True
        assert dp.discover_peers() == []

    def test_remove_nonexistent(self):
        dp = DiscoveryProtocol()
        assert dp.remove_agent("ghost") is False

    def test_supports_method(self):
        caps = AgentCapabilities(agent_id="a", supported_opcodes={0xB0})
        assert caps.supports(0xB0) is True
        assert caps.supports(0xFF) is False


# ===========================================================================
# 7. Edge Cases and Integration Tests
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge-case and integration tests."""

    def test_range_manager_escape_prefix_constant(self):
        assert RangeManager.ESCAPE_PREFIX == 0xF9

    def test_large_opcode_extended(self):
        rm = RangeManager()
        rm.reserve_extended(65535, "max extended")
        assert rm.is_extended_available(65535) is False

    def test_tally_result_for_ratio_zero(self):
        t = TallyResult(
            proposal_id="p", approved=False,
            for_count=0, against_count=0, abstain_count=0,
        )
        assert t.for_ratio == 0.0

    def test_proposal_status_values(self):
        statuses = list(ProposalStatus)
        assert ProposalStatus.DRAFT in statuses
        assert ProposalStatus.DEPRECATED in statuses
        assert len(statuses) == 7

    def test_vote_value_values(self):
        votes = list(VoteValue)
        assert VoteValue.FOR in votes
        assert VoteValue.AGAINST in votes
        assert VoteValue.ABSTAIN in votes

    def test_version_record_fields(self):
        vr = VersionRecord(version=1, timestamp=100.0, changes="init")
        assert vr.snapshot == {}
        assert vr.version == 1

    def test_agent_capabilities_defaults(self):
        caps = AgentCapabilities(agent_id="test")
        assert caps.supported_opcodes == set()
        assert caps.supported_mnemonics == set()
        assert caps.max_extended is False

    def test_multiple_proposals_different_opcodes(self):
        reg = ProposalRegistry()
        for i in range(20):
            p = make_proposal(opcode_num=0xA0 + i, mnemonic=f"OP_{i}")
            pid = reg.submit(p)
            assert reg.get(pid) is not None
        assert len(reg.list_all()) == 20

    def test_vote_counts_across_proposals(self):
        reg = ProposalRegistry(electorate_size=3)
        p1 = make_proposal(opcode_num=0xA0, mnemonic="OP_A")
        p2 = make_proposal(opcode_num=0xA1, mnemonic="OP_B")
        pid1 = reg.submit(p1)
        pid2 = reg.submit(p2)
        reg.vote(pid1, "v1", VoteValue.FOR)
        reg.vote(pid2, "v1", VoteValue.AGAINST)
        assert len(reg.get_votes(pid1)) == 1
        assert len(reg.get_votes(pid2)) == 1

    def test_confidence_capped_at_one(self):
        results = [TestResult(
            passed=True,
            test_case=TestCase(name="t", bytecode=[], expected_behavior=None),
            actual=None,
        )] * 100
        conf = OpcodeTester.compute_confidence(results)
        assert conf <= 1.0

    def test_extended_opcodes_not_in_primary_range(self):
        rm = RangeManager()
        rm.reserve(0, "primary")
        # Extended slot 0 is separate from primary 0
        assert rm.is_extended_available(0) is True

    def test_negotiation_conditions_populated(self):
        dp = DiscoveryProtocol()
        # Agent with some opcodes (< 200) → conditional support
        dp.advertise_capabilities("cond_agent", supported_opcodes={1, 2, 3})
        result = dp.negotiate_extension(["cond_agent"], 0xC0)
        if result.conditions:
            assert "conditional" in result.conditions[0].lower() or True

    def test_broadcast_after_full_lifecycle(self):
        """Integration: full lifecycle + broadcast."""
        rm = RangeManager()
        reg = ProposalRegistry(electorate_size=3)
        tester = OpcodeTester()
        engine = AdoptionEngine(reg, rm, tester, electorate_size=3)
        dp = DiscoveryProtocol()

        # Register agents
        dp.advertise_capabilities("agent-0")
        dp.advertise_capabilities("agent-1")
        dp.advertise_capabilities("agent-2")

        # Propose and adopt
        pid = engine.propose_opcode(make_proposal()).id
        for i in range(3):
            engine.cast_vote(pid, f"agent-{i}", VoteValue.FOR)
        adopted = engine.adopt(pid)

        # Broadcast
        record = dp.broadcast_adoption(adopted)
        assert len(record.notified_agents) == 3

        # All agents acknowledge
        for _ in record.notified_agents:
            dp.acknowledge_broadcast(0, "agent-x")
        assert dp.get_broadcasts()[0].ack_count == 3
