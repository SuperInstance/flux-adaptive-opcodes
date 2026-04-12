"""
Adaptive Opcode Discovery Protocol

Runtime ISA extension: proposal, testing, voting, and democratic adoption
of new opcodes across the SuperInstance fleet.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProposalStatus(str, Enum):
    DRAFT = "DRAFT"
    PROPOSED = "PROPOSED"
    TESTING = "TESTING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ADOPTED = "ADOPTED"
    DEPRECATED = "DEPRECATED"


class OpcodeCategory(str, Enum):
    ARITHMETIC = "ARITHMETIC"
    LOGIC = "LOGIC"
    CONTROL_FLOW = "CONTROL_FLOW"
    MEMORY = "MEMORY"
    STACK = "STACK"
    IO = "IO"
    CRYPTO = "CRYPTO"
    SIMD = "SIMD"
    SYSTEM = "SYSTEM"
    CUSTOM = "CUSTOM"


class OpcodeFormat(str, Enum):
    NO_OPERAND = "NO_OPERAND"
    IMMEDIATE_8 = "IMMEDIATE_8"
    IMMEDIATE_16 = "IMMEDIATE_16"
    IMMEDIATE_32 = "IMMEDIATE_32"
    REGISTER = "REGISTER"
    TWO_REGISTER = "TWO_REGISTER"
    THREE_REGISTER = "THREE_REGISTER"
    REGISTER_IMMEDIATE = "REGISTER_IMMEDIATE"
    MEMORY_DIRECT = "MEMORY_DIRECT"
    MEMORY_INDIRECT = "MEMORY_INDIRECT"
    ESCAPE_EXTENDED = "ESCAPE_EXTENDED"


class SlotStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    RESERVED = "RESERVED"
    IN_USE = "IN_USE"


class VoteValue(str, Enum):
    FOR = "FOR"
    AGAINST = "AGAINST"
    ABSTAIN = "ABSTAIN"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VersionRecord:
    """Single version entry in proposal history."""
    version: int
    timestamp: float
    changes: str
    snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpcodeProposal:
    """
    A proposed new opcode for the Flux ISA.

    Parameters
    ----------
    proposer : str
        Identifier of the proposing agent.
    opcode_num : int
        Opcode number (0-255 for primary, 0-65535 with escape prefix).
    mnemonic : str
        Human-readable name for the opcode.
    format : OpcodeFormat
        Encoding format of the instruction.
    category : OpcodeCategory
        Semantic category of the opcode.
    description : str
        What the opcode does.
    rationale : str
        Why this opcode should be added.
    test_bytecode : Optional[List[int]]
        Bytecode to exercise the opcode during testing.
    expected_result : Optional[Any]
        Expected outcome of running test_bytecode.
    """
    proposer: str
    opcode_num: int
    mnemonic: str
    format: OpcodeFormat
    category: OpcodeCategory
    description: str
    rationale: str
    test_bytecode: Optional[List[int]] = None
    expected_result: Optional[Any] = None

    # Internal bookkeeping (set by ProposalRegistry on submit)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: ProposalStatus = ProposalStatus.DRAFT
    version: int = 1
    version_history: List[VersionRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def bump_version(self, changes: str) -> None:
        """Increment version and record history entry."""
        self.version += 1
        self.updated_at = time.time()
        snapshot = {
            "opcode_num": self.opcode_num,
            "mnemonic": self.mnemonic,
            "format": self.format,
            "category": self.category,
            "description": self.description,
            "rationale": self.rationale,
        }
        self.version_history.append(
            VersionRecord(
                version=self.version,
                timestamp=self.updated_at,
                changes=changes,
                snapshot=deepcopy(snapshot),
            )
        )

    def fingerprint(self) -> str:
        """Deterministic hash of proposal content for dedup."""
        payload = (
            f"{self.opcode_num}:{self.mnemonic}:{self.format}:"
            f"{self.category}:{self.description}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Opcode Range Manager
# ---------------------------------------------------------------------------

class RangeManager:
    """
    Manages the 256-slot primary opcode space (and extended space via
    ISA v3 escape prefix). Tracks RESERVED, IN_USE, and AVAILABLE slots.
    """

    ESCAPE_PREFIX = 0xF9  # ISA v3 escape prefix
    MAX_PRIMARY = 256
    MAX_EXTENDED = 65536  # escape_prefix + second byte

    def __init__(self) -> None:
        self._slots: Dict[int, SlotStatus] = {
            i: SlotStatus.AVAILABLE for i in range(self.MAX_PRIMARY)
        }
        self._reservations: Dict[int, str] = {}  # opcode -> purpose
        self._extended_slots: Dict[int, SlotStatus] = {}
        self._extended_reservations: Dict[int, str] = {}

    # -- primary space -----------------------------------------------------

    def reserve(self, opcode_num: int, purpose: str = "") -> None:
        """Reserve a primary opcode slot."""
        if not 0 <= opcode_num < self.MAX_PRIMARY:
            raise ValueError(
                f"Opcode {opcode_num} out of primary range [0, {self.MAX_PRIMARY})"
            )
        current = self._slots.get(opcode_num)
        if current == SlotStatus.IN_USE:
            raise ValueError(f"Opcode {opcode_num} is already in use")
        if current == SlotStatus.RESERVED:
            raise ValueError(
                f"Opcode {opcode_num} is already reserved "
                f"({self._reservations.get(opcode_num, '')})"
            )
        self._slots[opcode_num] = SlotStatus.RESERVED
        if purpose:
            self._reservations[opcode_num] = purpose

    def assign(self, opcode_num: int) -> None:
        """Mark a reserved slot as actively in use."""
        if self._slots.get(opcode_num) != SlotStatus.RESERVED:
            raise ValueError(
                f"Opcode {opcode_num} must be reserved before assignment"
            )
        self._slots[opcode_num] = SlotStatus.IN_USE

    def release(self, opcode_num: int) -> None:
        """Release a slot back to AVAILABLE."""
        if opcode_num not in self._slots:
            raise ValueError(f"Opcode {opcode_num} not tracked")
        self._slots[opcode_num] = SlotStatus.AVAILABLE
        self._reservations.pop(opcode_num, None)

    def is_available(self, opcode_num: int) -> bool:
        """Check whether a primary opcode slot is available."""
        return self._slots.get(opcode_num) == SlotStatus.AVAILABLE

    def find_available(self, format: Optional[OpcodeFormat] = None) -> int:
        """Return the next available primary opcode number."""
        for num in range(self.MAX_PRIMARY):
            if self._slots[num] == SlotStatus.AVAILABLE:
                return num
        raise RuntimeError("No available primary opcode slots")

    def status_of(self, opcode_num: int) -> Optional[SlotStatus]:
        """Query the status of a primary opcode slot."""
        return self._slots.get(opcode_num)

    def list_by_status(self, status: SlotStatus) -> List[int]:
        """List all primary opcodes with a given status."""
        return [n for n, s in self._slots.items() if s == status]

    def available_count(self) -> int:
        return sum(1 for s in self._slots.values() if s == SlotStatus.AVAILABLE)

    def reserved_count(self) -> int:
        return sum(1 for s in self._slots.values() if s == SlotStatus.RESERVED)

    def in_use_count(self) -> int:
        return sum(1 for s in self._slots.values() if s == SlotStatus.IN_USE)

    # -- extended (escape prefix) space ------------------------------------

    def reserve_extended(self, opcode_num: int, purpose: str = "") -> None:
        """Reserve an extended opcode slot (escape prefix + second byte)."""
        if not 0 <= opcode_num < self.MAX_EXTENDED:
            raise ValueError("Extended opcode out of range [0, 65536)")
        current = self._extended_slots.get(opcode_num)
        if current in (SlotStatus.IN_USE, SlotStatus.RESERVED):
            raise ValueError(f"Extended opcode {opcode_num} already taken")
        self._extended_slots[opcode_num] = SlotStatus.RESERVED
        if purpose:
            self._extended_reservations[opcode_num] = purpose

    def is_extended_available(self, opcode_num: int) -> bool:
        return self._extended_slots.get(opcode_num) in (SlotStatus.AVAILABLE, None)

    def find_available_extended(self) -> int:
        for num in range(self.MAX_EXTENDED):
            if self.is_extended_available(num):
                return num
        raise RuntimeError("No available extended opcode slots")


# ---------------------------------------------------------------------------
# 2. Voting Helpers
# ---------------------------------------------------------------------------

@dataclass
class VoteRecord:
    voter: str
    proposal_id: str
    vote: VoteValue
    timestamp: float = field(default_factory=time.time)


@dataclass
class TallyResult:
    proposal_id: str
    approved: bool
    for_count: int
    against_count: int
    abstain_count: int
    total_eligible: int = 0  # total agents that voted

    @property
    def for_ratio(self) -> float:
        if self.total_eligible == 0:
            return 0.0
        return self.for_count / self.total_eligible


# ---------------------------------------------------------------------------
# 3. Proposal Registry
# ---------------------------------------------------------------------------

class ProposalRegistry:
    """
    Store and track opcode proposals with a democratic voting system.
    Requires 2/3 supermajority for adoption.
    """

    SUPERMAJORITY_FRACTION = 2.0 / 3.0

    def __init__(self, electorate_size: int = 9) -> None:
        """
        Parameters
        ----------
        electorate_size : int
            Total number of voting agents in the fleet.
        """
        self._proposals: Dict[str, OpcodeProposal] = {}
        self._votes: Dict[str, Dict[str, VoteRecord]] = {}  # pid -> {voter: record}
        self._electorate_size = electorate_size
        self._fingerprint_index: Dict[str, str] = {}  # fingerprint -> proposal_id

    def submit(self, proposal: OpcodeProposal) -> str:
        """
        Register a new proposal. Returns the proposal_id.

        Raises
        ------
        ValueError if a proposal with the same fingerprint already exists.
        """
        fp = proposal.fingerprint()
        if fp in self._fingerprint_index:
            raise ValueError(
                f"Duplicate proposal fingerprint {fp} "
                f"(existing: {self._fingerprint_index[fp]})"
            )
        proposal.id = uuid.uuid4().hex[:12]
        proposal.status = ProposalStatus.PROPOSED
        proposal.updated_at = time.time()
        self._proposals[proposal.id] = proposal
        self._votes[proposal.id] = {}
        self._fingerprint_index[fp] = proposal.id
        return proposal.id

    def get(self, proposal_id: str) -> Optional[OpcodeProposal]:
        return self._proposals.get(proposal_id)

    def list_all(self) -> List[OpcodeProposal]:
        return list(self._proposals.values())

    def list_by_status(self, status: ProposalStatus) -> List[OpcodeProposal]:
        return [p for p in self._proposals.values() if p.status == status]

    def vote(self, proposal_id: str, voter: str, vote: VoteValue) -> VoteRecord:
        """
        Cast a vote on a proposal.

        Raises
        ------
        ValueError if proposal not found or voter has already voted.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")
        if proposal.status not in (ProposalStatus.PROPOSED, ProposalStatus.TESTING):
            raise ValueError(
                f"Cannot vote on proposal in status {proposal.status.value}"
            )
        if voter in self._votes[proposal_id]:
            raise ValueError(f"{voter} has already voted on {proposal_id}")
        record = VoteRecord(voter=voter, proposal_id=proposal_id, vote=vote)
        self._votes[proposal_id][voter] = record
        return record

    def get_votes(self, proposal_id: str) -> Dict[str, VoteRecord]:
        return dict(self._votes.get(proposal_id, {}))

    def tally(self, proposal_id: str) -> TallyResult:
        """Compute the vote tally for a proposal."""
        votes = self._votes.get(proposal_id, {})
        for_count = sum(1 for v in votes.values() if v.vote == VoteValue.FOR)
        against_count = sum(1 for v in votes.values() if v.vote == VoteValue.AGAINST)
        abstain_count = sum(1 for v in votes.values() if v.vote == VoteValue.ABSTAIN)
        total = len(votes)
        # Supermajority check: FOR must be ≥ 2/3 of electorate
        approved = for_count >= self._electorate_size * self.SUPERMAJORITY_FRACTION
        return TallyResult(
            proposal_id=proposal_id,
            approved=approved,
            for_count=for_count,
            against_count=against_count,
            abstain_count=abstain_count,
            total_eligible=total,
        )

    def set_status(self, proposal_id: str, status: ProposalStatus) -> None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")
        proposal.status = status
        proposal.updated_at = time.time()


# ---------------------------------------------------------------------------
# 4. Testing Framework
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    bytecode: List[int]
    expected_behavior: Any
    category: str = "normal"  # normal | edge | error


@dataclass
class TestResult:
    passed: bool
    test_case: TestCase
    actual: Any
    message: str = ""
    duration_ms: float = 0.0


class OpcodeTester:
    """
    Validate proposed opcodes via simulated execution of test bytecodes.

    Since we don't have a real VM, the tester provides a deterministic
    simulation layer that evaluates opcode semantics from test vectors.
    """

    def __init__(self) -> None:
        self._results: Dict[str, List[TestResult]] = {}  # proposal_id -> results

    def _simulate(self, bytecode: List[int]) -> Any:
        """
        Lightweight deterministic simulation.

        For a real system this would delegate to a micro-VM.
        Here we produce a hash-based deterministic result from the bytecode
        so tests are reproducible and verifiable.
        """
        if not bytecode:
            return None
        state = 0
        for b in bytecode:
            state = (state * 31 + b) & 0xFFFFFFFF
        return state

    def test_opcode(
        self,
        proposal: OpcodeProposal,
        test_bytecode: Optional[List[int]] = None,
        expected_behavior: Optional[Any] = None,
    ) -> TestResult:
        """
        Run a single test against a proposal.

        If test_bytecode / expected_behavior are omitted, the proposal's
        own fields are used.
        """
        bc = test_bytecode if test_bytecode is not None else proposal.test_bytecode
        expected = (
            expected_behavior
            if expected_behavior is not None
            else proposal.expected_result
        )
        tc = TestCase(
            name=f"{proposal.mnemonic}_default",
            bytecode=bc or [],
            expected_behavior=expected,
        )
        start = time.monotonic()
        actual = self._simulate(bc or [])
        duration = (time.monotonic() - start) * 1000
        passed = actual == expected
        result = TestResult(
            passed=passed,
            test_case=tc,
            actual=actual,
            message="OK" if passed else f"Expected {expected}, got {actual}",
            duration_ms=duration,
        )
        self._results.setdefault(proposal.id, []).append(result)
        return result

    def run_test_suite(
        self,
        proposal: OpcodeProposal,
        test_cases: List[TestCase],
    ) -> List[TestResult]:
        """Run a full suite of test cases against a proposal."""
        results = []
        for tc in test_cases:
            start = time.monotonic()
            actual = self._simulate(tc.bytecode)
            duration = (time.monotonic() - start) * 1000
            passed = actual == tc.expected_behavior
            result = TestResult(
                passed=passed,
                test_case=tc,
                actual=actual,
                message="OK" if passed else f"Expected {tc.expected_behavior}, got {actual}",
                duration_ms=duration,
            )
            results.append(result)
        self._results.setdefault(proposal.id, []).extend(results)
        return results

    def get_results(self, proposal_id: str) -> List[TestResult]:
        return list(self._results.get(proposal_id, []))

    @staticmethod
    def compute_confidence(test_results: List[TestResult]) -> float:
        """
        Compute confidence score 0.0-1.0 in opcode correctness.

        Factors:
        - pass_rate: what fraction of tests passed
        - coverage: bonus for having edge + error tests
        - consistency: all tests in same category should agree
        """
        if not test_results:
            return 0.0
        total = len(test_results)
        passed = sum(1 for r in test_results if r.passed)
        pass_rate = passed / total

        categories = set(r.test_case.category for r in test_results)
        coverage_bonus = min(len(categories) / 3.0, 1.0) * 0.1

        confidence = min(pass_rate * 0.9 + coverage_bonus, 1.0)
        return round(confidence, 4)


# ---------------------------------------------------------------------------
# 5. Adoption Engine
# ---------------------------------------------------------------------------

@dataclass
class AdoptedOpcode:
    opcode_num: int
    mnemonic: str
    format: OpcodeFormat
    category: OpcodeCategory
    description: str
    proposal_id: str
    adopted_at: float
    adopted_by: List[str]  # voters who approved

    @property
    def is_extended(self) -> bool:
        return self.opcode_num >= 256


@dataclass
class DeprecationRecord:
    opcode_num: int
    mnemonic: str
    reason: str
    deprecated_at: float
    proposal_id: Optional[str] = None


@dataclass
class ReviewResult:
    proposal_id: str
    auto_approved: bool
    issues: List[str]
    confidence: float
    recommendations: List[str]


@dataclass
class VoteResult:
    proposal_id: str
    voter: str
    vote: VoteValue
    tally_after: TallyResult


class AdoptionEngine:
    """
    Full lifecycle management: propose → review → vote → adopt → deprecate.
    """

    def __init__(
        self,
        registry: ProposalRegistry,
        range_manager: RangeManager,
        tester: Optional[OpcodeTester] = None,
        electorate_size: int = 9,
    ) -> None:
        self.registry = registry
        self.range = range_manager
        self.tester = tester or OpcodeTester()
        self._adopted: Dict[int, AdoptedOpcode] = {}  # opcode_num -> record
        self._deprecated: Dict[int, DeprecationRecord] = {}

    def propose_opcode(self, proposal: OpcodeProposal) -> OpcodeProposal:
        """
        Submit a proposal to the registry.

        Validates opcode availability and assigns proposal_id.
        """
        if proposal.opcode_num < 256:
            if not self.range.is_available(proposal.opcode_num):
                raise ValueError(
                    f"Opcode {proposal.opcode_num} is not available"
                )
        proposal_id = self.registry.submit(proposal)
        # Reserve the slot
        if proposal.opcode_num < 256:
            self.range.reserve(proposal.opcode_num, f"Proposal {proposal_id}")
        return proposal

    def review(self, proposal_id: str) -> ReviewResult:
        """Automated review: run tests and check basic criteria."""
        proposal = self.registry.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        issues: List[str] = []
        recommendations: List[str] = []

        # Basic validation
        if not proposal.mnemonic or len(proposal.mnemonic) < 2:
            issues.append("Mnemonic too short or empty")
        if not proposal.description:
            issues.append("Missing description")
        if not proposal.rationale:
            issues.append("Missing rationale")

        # Run tests
        test_results: List[TestResult] = []
        if proposal.test_bytecode is not None:
            tr = self.tester.test_opcode(proposal)
            test_results.append(tr)
            if not tr.passed:
                issues.append(f"Default test failed: {tr.message}")
            else:
                recommendations.append("Default test passed")

        confidence = OpcodeTester.compute_confidence(test_results)

        if len(proposal.test_bytecode or []) == 0:
            issues.append("No test bytecode provided")
            recommendations.append("Add test bytecode for better confidence")

        auto_approved = len(issues) == 0 and confidence >= 0.8

        self.registry.set_status(
            proposal_id,
            ProposalStatus.TESTING if not auto_approved else ProposalStatus.PROPOSED,
        )
        return ReviewResult(
            proposal_id=proposal_id,
            auto_approved=auto_approved,
            issues=issues,
            confidence=confidence,
            recommendations=recommendations,
        )

    def cast_vote(
        self, proposal_id: str, voter: str, vote: VoteValue
    ) -> VoteResult:
        """Cast a vote and return the updated tally."""
        record = self.registry.vote(proposal_id, voter, vote)
        tally = self.registry.tally(proposal_id)
        return VoteResult(
            proposal_id=proposal_id,
            voter=voter,
            vote=vote,
            tally_after=tally,
        )

    def check_adoption(self, proposal_id: str) -> bool:
        """Check whether a proposal has met the adoption threshold."""
        tally = self.registry.tally(proposal_id)
        return tally.approved

    def adopt(self, proposal_id: str) -> AdoptedOpcode:
        """
        Officially add the opcode to the ISA.

        Raises ValueError if threshold not met.
        """
        tally = self.registry.tally(proposal_id)
        if not tally.approved:
            raise ValueError(
                f"Proposal {proposal_id} has not met supermajority threshold "
                f"({tally.for_count}/{self.registry._electorate_size} FOR)"
            )
        proposal = self.registry.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        # Mark opcode as IN_USE
        if proposal.opcode_num < 256:
            self.range.assign(proposal.opcode_num)

        self.registry.set_status(proposal_id, ProposalStatus.ADOPTED)

        # Get approving voters
        votes = self.registry.get_votes(proposal_id)
        approvers = [
            v.voter for v in votes.values() if v.vote == VoteValue.FOR
        ]

        adopted = AdoptedOpcode(
            opcode_num=proposal.opcode_num,
            mnemonic=proposal.mnemonic,
            format=proposal.format,
            category=proposal.category,
            description=proposal.description,
            proposal_id=proposal_id,
            adopted_at=time.time(),
            adopted_by=approvers,
        )
        self._adopted[proposal.opcode_num] = adopted
        return adopted

    def deprecate(
        self, opcode_num: int, reason: str, proposal_id: Optional[str] = None
    ) -> DeprecationRecord:
        """Deprecate an adopted opcode."""
        adopted = self._adopted.get(opcode_num)
        if adopted is None:
            raise ValueError(f"Opcode {opcode_num} is not adopted")

        record = DeprecationRecord(
            opcode_num=opcode_num,
            mnemonic=adopted.mnemonic,
            reason=reason,
            deprecated_at=time.time(),
            proposal_id=proposal_id,
        )
        self._deprecated[opcode_num] = record
        if opcode_num < 256:
            self.range.release(opcode_num)
        if proposal_id:
            self.registry.set_status(proposal_id, ProposalStatus.DEPRECATED)
        return record

    def get_adopted(self, opcode_num: int) -> Optional[AdoptedOpcode]:
        return self._adopted.get(opcode_num)

    def list_adopted(self) -> List[AdoptedOpcode]:
        return list(self._adopted.values())

    def get_deprecated(self, opcode_num: int) -> Optional[DeprecationRecord]:
        return self._deprecated.get(opcode_num)


# ---------------------------------------------------------------------------
# 6. Discovery Protocol
# ---------------------------------------------------------------------------

@dataclass
class AgentCapabilities:
    agent_id: str
    supported_opcodes: Set[int] = field(default_factory=set)
    supported_mnemonics: Set[str] = field(default_factory=set)
    max_extended: bool = False
    last_seen: float = field(default_factory=time.time)

    def supports(self, opcode_num: int) -> bool:
        return opcode_num in self.supported_opcodes


@dataclass
class NegotiationResult:
    proposed_opcode: int
    accepted: bool
    supporting_agents: List[str]
    rejecting_agents: List[str]
    conditions: List[str] = field(default_factory=list)


@dataclass
class BroadcastRecord:
    adopted_opcode: AdoptedOpcode
    broadcast_at: float
    notified_agents: List[str]
    ack_count: int = 0


class DiscoveryProtocol:
    """
    Agent-to-agent discovery for opcode capabilities and negotiation
    of new opcode extensions.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentCapabilities] = {}
        self._broadcasts: List[BroadcastRecord] = []
        self._negotiations: List[NegotiationResult] = []

    def advertise_capabilities(
        self,
        agent_id: str,
        supported_opcodes: Optional[Set[int]] = None,
        supported_mnemonics: Optional[Set[str]] = None,
        max_extended: bool = False,
    ) -> AgentCapabilities:
        """
        Register or update an agent's capabilities.

        Other agents can query this to discover who supports what.
        """
        caps = AgentCapabilities(
            agent_id=agent_id,
            supported_opcodes=supported_opcodes or set(),
            supported_mnemonics=supported_mnemonics or set(),
            max_extended=max_extended,
            last_seen=time.time(),
        )
        self._agents[agent_id] = caps
        return caps

    def discover_peers(self) -> List[AgentCapabilities]:
        """Return all known peers and their capabilities."""
        return list(self._agents.values())

    def find_supporters(self, opcode_num: int) -> List[str]:
        """Find all agents that already support a given opcode."""
        return [
            agent_id
            for agent_id, caps in self._agents.items()
            if caps.supports(opcode_num)
        ]

    def find_compatible_peers(
        self, required_opcodes: Set[int]
    ) -> List[str]:
        """Find agents that support ALL of the required opcodes."""
        compatible = []
        for agent_id, caps in self._agents.items():
            if required_opcodes.issubset(caps.supported_opcodes):
                compatible.append(agent_id)
        return compatible

    def negotiate_extension(
        self,
        peers: List[str],
        proposed_opcode: int,
        required_support: float = 0.5,
    ) -> NegotiationResult:
        """
        Negotiate adoption of a new opcode with specified peers.

        Parameters
        ----------
        peers : list of agent IDs to negotiate with
        proposed_opcode : opcode number being proposed
        required_support : fraction of peers that must agree (default 0.5)

        Returns
        -------
        NegotiationResult with accepted/rejected breakdown.
        """
        supporting = []
        rejecting = []
        conditions = []

        total = len(peers)
        if total == 0:
            return NegotiationResult(
                proposed_opcode=proposed_opcode,
                accepted=False,
                supporting_agents=[],
                rejecting_agents=[],
                conditions=["No peers to negotiate with"],
            )

        threshold = int(total * required_support)
        if total * required_support > threshold:
            threshold += 1  # ceiling

        for agent_id in peers:
            caps = self._agents.get(agent_id)
            if caps is None:
                rejecting.append(agent_id)
                continue
            # Agents agree if they support extended opcodes or
            # the opcode doesn't conflict
            if caps.max_extended or caps.supports(proposed_opcode):
                supporting.append(agent_id)
            elif len(caps.supported_opcodes) < 200:
                # Agents with room tend to agree
                supporting.append(agent_id)
                conditions.append(
                    f"{agent_id}: conditional — requires compatibility test"
                )
            else:
                rejecting.append(agent_id)

        accepted = len(supporting) >= threshold

        result = NegotiationResult(
            proposed_opcode=proposed_opcode,
            accepted=accepted,
            supporting_agents=supporting,
            rejecting_agents=rejecting,
            conditions=conditions,
        )
        self._negotiations.append(result)
        return result

    def broadcast_adoption(
        self, adopted_opcode: AdoptedOpcode
    ) -> BroadcastRecord:
        """
        Broadcast an adopted opcode to all known fleet agents.

        Returns a record of the broadcast for tracking ACKs.
        """
        notified = list(self._agents.keys())
        record = BroadcastRecord(
            adopted_opcode=adopted_opcode,
            broadcast_at=time.time(),
            notified_agents=notified,
            ack_count=0,
        )
        self._broadcasts.append(record)
        return record

    def acknowledge_broadcast(self, broadcast_index: int, agent_id: str) -> int:
        """Acknowledge receipt of a broadcast. Returns new ack count."""
        if 0 <= broadcast_index < len(self._broadcasts):
            self._broadcasts[broadcast_index].ack_count += 1
            return self._broadcasts[broadcast_index].ack_count
        raise IndexError(f"Broadcast index {broadcast_index} out of range")

    def get_broadcasts(self) -> List[BroadcastRecord]:
        return list(self._broadcasts)

    def get_negotiations(self) -> List[NegotiationResult]:
        return list(self._negotiations)

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the discovery pool."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False
