import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, List, Optional, Sequence, Set, Type, TypeVar

import numpy as np

# Type Definitions
BallotType = TypeVar("BallotType")
CandidateId = int
VoterId = int


@dataclass
class Candidate:
    id: CandidateId
    vector: np.typing.NDArray[np.float64]

    @classmethod
    def random(cls, id: CandidateId, dim: int) -> "Candidate":
        return cls(id, np.random.normal(loc=0.0, scale=1.0, size=dim))


@dataclass
class Ballot(Generic[BallotType]):
    voter_id: VoterId
    data: BallotType


# Concrete Voter class (no longer abstract)
@dataclass
class Voter:
    id: VoterId
    vector: np.typing.NDArray[np.float64]

    @classmethod
    def random(cls, id: VoterId, dim: int) -> "Voter":
        return cls(id, np.random.normal(loc=0.0, scale=1.0, size=dim))


class Election(ABC, Generic[BallotType]):
    def __init__(self, candidates: List[Candidate], winners: int = 1):
        self.candidates = candidates
        self.winners = winners
        self.rounds: List[Dict] = []

    @abstractmethod
    def cast_ballot(self, voter: Voter) -> Ballot[BallotType]:
        """Cast a ballot for a voter based on the election rules"""
        pass

    @abstractmethod
    def run(self, voters: Sequence[Voter]) -> List[Candidate]:
        pass


def rank_by_distance(
    voter_vector: np.ndarray, candidates: List[Candidate]
) -> List[Candidate]:
    return sorted(
        candidates, key=lambda c: float(np.linalg.norm(voter_vector - c.vector))
    )


########################################################
# First Past The Post (FPTP) System
########################################################


FPTPBallot = CandidateId


class FPTPElection(Election[FPTPBallot]):
    name: str = "FPTP"

    def cast_ballot(self, voter: Voter) -> Ballot[CandidateId]:
        """FPTP voter that chooses the closest candidate"""
        ranked_candidates = rank_by_distance(voter.vector, self.candidates)
        return Ballot(
            voter_id=voter.id,
            data=ranked_candidates[0].id,
        )

    def run(self, voters: Sequence[Voter]) -> List[Candidate]:
        ballots = [self.cast_ballot(v) for v in voters]
        candidate_ids = [b.data for b in ballots]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [self.candidates[candidate_id] for candidate_id, _ in winner_counts]


########################################################
# Ranked Choice Voting (RCV) System
########################################################


RankedBallot = Dict[CandidateId, int]


class RCVElection(Election[RankedBallot]):
    name: str = "RCV"

    def cast_ballot(self, voter: Voter) -> Ballot[RankedBallot]:
        """RCV/STV voter with preferences"""
        ranked_candidates = rank_by_distance(voter.vector, self.candidates)
        return Ballot(
            voter_id=voter.id,
            data={
                candidate.id: rank
                for rank, candidate in enumerate(ranked_candidates, 1)
            },
        )

    def run(self, voters: Sequence[Voter]) -> List[Candidate]:
        ballots = [self.cast_ballot(v) for v in voters]
        active_candidates = set(c.id for c in self.candidates)
        winners: List[Candidate] = []

        while len(winners) < self.winners and active_candidates:
            # Count current votes
            counts: Dict[CandidateId, int | float] = {
                cid: 0 for cid in active_candidates
            }
            for ballot in ballots:
                valid_ranks = {
                    cid: rank
                    for cid, rank in ballot.data.items()
                    if cid in active_candidates
                }
                if valid_ranks:
                    top_cid = min(valid_ranks, key=valid_ranks.get)  # type: ignore
                    counts[top_cid] += 1

            self.rounds.append(counts.copy())

            # Check for majority
            total = sum(counts.values())
            if total == 0:
                break

            if self.winners == 1:  # IRV Logic
                majority = total / 2
                for cid, count in counts.items():
                    if count > majority:
                        winners.append(next(c for c in self.candidates if c.id == cid))
                        return winners

                # Eliminate last place
                eliminate_cid = min(counts, key=counts.get)  # type: ignore
                active_candidates.remove(eliminate_cid)
            else:  # STV Logic
                quota = total / (self.winners + 1) + 1
                elected = [cid for cid, count in counts.items() if count >= quota]

                if elected:
                    for cid in elected:
                        winners.append(next(c for c in self.candidates if c.id == cid))
                        active_candidates.remove(cid)

                    # Transfer surplus votes (simplified)
                    transfer_factor = 0.5  # Actual STV uses precise calculations
                    for ballot in ballots:
                        if any(cid in ballot.data for cid in elected):
                            next_pref = next(
                                (
                                    cid
                                    for cid, rank in ballot.data.items()
                                    if cid in active_candidates
                                ),
                                None,
                            )
                            if next_pref:
                                counts[next_pref] += transfer_factor
                else:
                    eliminate_cid = min(counts, key=counts.get)  # type: ignore
                    active_candidates.remove(eliminate_cid)

        return winners


########################################################
# Single Transferable Vote (STV) System
########################################################


class STVElection(RCVElection):
    """Proper STV implementation with vote transfer"""

    name: str = "STV"

    def run(self, voters: Sequence[Voter]) -> List[Candidate]:
        ballots = [self.cast_ballot(v) for v in voters]
        active_candidates = {c.id: c for c in self.candidates}
        winners: List[Candidate] = []
        quota = len(ballots) / (self.winners + 1) + 1

        while len(winners) < self.winners and active_candidates:
            # Count current votes
            counts: Dict[CandidateId, float] = {cid: 0.0 for cid in active_candidates}
            for ballot in ballots:
                valid_ranks = {
                    cid: rank
                    for cid, rank in ballot.data.items()
                    if cid in active_candidates
                }
                if valid_ranks:
                    top_cid = min(valid_ranks, key=valid_ranks.get)  # type: ignore
                    counts[top_cid] += 1

            self.rounds.append(counts.copy())

            # Elect candidates meeting quota
            elected = [cid for cid, count in counts.items() if count >= quota]
            for cid in elected:
                winners.append(active_candidates.pop(cid))
                surplus = counts[cid] - quota

                # Transfer surplus votes
                transfer_factor = surplus / counts[cid]
                for ballot in ballots:
                    if ballot.data.get(cid, 0) == 1:  # First preference
                        next_pref = next(
                            (
                                cid
                                for cid, rank in ballot.data.items()
                                if cid in active_candidates
                            ),
                            None,
                        )
                        if next_pref:
                            counts[next_pref] += transfer_factor

            if not elected:
                # Eliminate lowest candidate
                eliminate_cid = min(counts, key=counts.get)  # type: ignore
                active_candidates.pop(eliminate_cid)

        return winners


########################################################
# Approval Voting System
########################################################


ApprovalBallot = Set[CandidateId]


class ApprovalVotingElection(Election[ApprovalBallot]):
    """Approval voting uses same counting as FPTP but different ballots"""

    name: str = "APPROVAL"

    def __init__(
        self, candidates: List[Candidate], winners: int = 1, cutoff: float = 0.5
    ):
        super().__init__(candidates, winners)
        self.cutoff = cutoff

    def cast_ballot(self, voter: Voter) -> Ballot[ApprovalBallot]:
        """Approval voter that chooses the closest candidates up to cutoff"""
        ranked_candidates = rank_by_distance(voter.vector, self.candidates)
        approved_candidates = ranked_candidates[
            : int(len(ranked_candidates) * self.cutoff)
        ]
        return Ballot(
            voter_id=voter.id,
            data={c.id for c in approved_candidates},
        )

    def run(self, voters: Sequence[Voter]) -> List[Candidate]:
        ballots = [self.cast_ballot(v) for v in voters]
        candidate_ids = [
            candidate_id for ballot in ballots for candidate_id in ballot.data
        ]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [self.candidates[candidate_id] for candidate_id, _ in winner_counts]


########################################################
# Limited Voting System
########################################################

LimitedBallot = List[CandidateId]


class LimitedVotingElection(Election[LimitedBallot]):
    """Limited Voting: Each voter can vote for up to k candidates"""

    name: str = "LIMITED"

    def __init__(
        self, candidates: List[Candidate], winners: int = 1, max_choices: int = 3
    ):
        super().__init__(candidates, winners)
        self.max_choices = max_choices

    def cast_ballot(self, voter: Voter) -> Ballot[Set[CandidateId]]:
        """Limited voter that selects up to max_choices candidates"""
        ranked_candidates = rank_by_distance(voter.vector, self.candidates)
        chosen = ranked_candidates[: self.max_choices]
        return Ballot(
            voter_id=voter.id,
            data={c.id for c in chosen},
        )

    def run(self, voters: Sequence[Voter]) -> List[Candidate]:
        ballots = [self.cast_ballot(v) for v in voters]
        candidate_ids = [
            candidate_id for ballot in ballots for candidate_id in ballot.data
        ]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [self.candidates[candidate_id] for candidate_id, _ in winner_counts]


# Usage Example
if __name__ == "__main__":
    DIMENSION = 3
    N_CANDIDATES = 10
    N_VOTERS = 10_000
    WINNERS = 2

    candidates = [Candidate.random(i, DIMENSION) for i in range(N_CANDIDATES)]
    voters = [Voter.random(i, DIMENSION) for i in range(N_VOTERS)]

    # Run elections
    fptp_election = FPTPElection(candidates, WINNERS)
    rcv_election = RCVElection(candidates, WINNERS)
    stv_election = STVElection(candidates, WINNERS)
    approval_election = ApprovalVotingElection(candidates, WINNERS)
    limited_election = LimitedVotingElection(candidates, WINNERS)

    print("FPTP Winners:", [candidate.id for candidate in fptp_election.run(voters)])
    print("RCV Winners:", [candidate.id for candidate in rcv_election.run(voters)])
    print("STV Winners:", [candidate.id for candidate in stv_election.run(voters)])
    print(
        "Approval Winners:",
        [candidate.id for candidate in approval_election.run(voters)],
    )
    print(
        "Limited Winners:", [candidate.id for candidate in limited_election.run(voters)]
    )
