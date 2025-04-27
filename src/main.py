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
VoterId = str


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


V = TypeVar("V", bound="Voter")


# Abstract Classes
@dataclass
class Voter(ABC, Generic[BallotType]):
    id: VoterId
    vector: np.typing.NDArray[np.float64]

    @classmethod
    def random(cls: Type[V], id: VoterId, dim: int) -> V:
        return cls(id, np.random.normal(loc=0.0, scale=1.0, size=dim))

    @abstractmethod
    def cast_ballot(self, candidates: List[Candidate]) -> Ballot[BallotType]:
        pass


class Election(ABC, Generic[BallotType]):
    def __init__(self, candidates: List[Candidate], winners: int = 1):
        self.candidates = candidates
        self.winners = winners
        self.rounds: List[Dict] = []

    @abstractmethod
    def run(self, voters: Sequence[Voter[BallotType]]) -> List[Candidate]:
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


class FPTPVoter(Voter[FPTPBallot]):
    """FPTP voter that chooses the closest candidate"""

    def cast_ballot(self, candidates: List[Candidate]) -> Ballot[CandidateId]:
        ranked_candidates = rank_by_distance(self.vector, candidates)
        return Ballot(
            voter_id=self.id,
            data=ranked_candidates[0].id,
        )


class FPTPElection(Election[FPTPBallot]):
    name: str = "FPTP"

    def run(self, voters: Sequence[Voter[FPTPBallot]]) -> List[Candidate]:
        ballots = [v.cast_ballot(self.candidates) for v in voters]
        candidate_ids = [b.data for b in ballots]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [self.candidates[candidate_id] for candidate_id, _ in winner_counts]


########################################################
# Ranked Choice Voting (RCV) System
########################################################


RankedBallot = Dict[CandidateId, int]


class RankedVoter(Voter[RankedBallot]):
    """RCV/STV voter with preferences"""

    def cast_ballot(self, candidates: List[Candidate]) -> Ballot[RankedBallot]:
        ranked_candidates = rank_by_distance(self.vector, candidates)
        return Ballot(
            voter_id=self.id,
            data={
                candidate.id: rank
                for rank, candidate in enumerate(ranked_candidates, 1)
            },
        )


class RCVElection(Election[RankedBallot]):
    name: str = "RCV"

    def run(self, voters: Sequence[Voter[RankedBallot]]) -> List[Candidate]:
        ballots = [v.cast_ballot(self.candidates) for v in voters]
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

    def run(self, voters: Sequence[Voter[RankedBallot]]) -> List[Candidate]:
        ballots = [v.cast_ballot(self.candidates) for v in voters]
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


@dataclass
class ApprovalVoter(Voter[ApprovalBallot]):
    """Approval voter that chooses the closest candidate"""

    cutoff: float

    def cast_ballot(self, candidates: List[Candidate]) -> Ballot[Set[CandidateId]]:
        ranked_candidates = rank_by_distance(self.vector, candidates)
        approved_candidates = ranked_candidates[
            : int(len(ranked_candidates) * self.cutoff)
        ]
        return Ballot(
            voter_id=self.id,
            data={c.id for c in approved_candidates},
        )


class ApprovalVotingElection(Election[ApprovalBallot]):
    """Approval voting uses same counting as FPTP but different ballots"""

    name: str = "APPROVAL"

    def run(self, voters: Sequence[Voter[ApprovalBallot]]) -> List[Candidate]:
        ballots = [v.cast_ballot(self.candidates) for v in voters]
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


class LimitedVoter(Voter[LimitedBallot]):
    """Limited voter that selects up to max_choices candidates"""

    max_choices: int

    def cast_ballot(self, candidates: List[Candidate]) -> Ballot[Set[CandidateId]]:
        ranked_candidates = rank_by_distance(self.vector, candidates)
        chosen = ranked_candidates[: self.max_choices]
        return Ballot(
            voter_id=self.id,
            data={c.id for c in chosen},
        )


class LimitedVotingElection(Election[LimitedBallot]):
    """Limited Voting: Each voter can vote for up to k candidates"""

    name: str = "LIMITED"

    def run(self, voters: Sequence[Voter[LimitedBallot]]) -> List[Candidate]:
        ballots = [v.cast_ballot(self.candidates) for v in voters]
        candidate_ids = [
            candidate_id for ballot in ballots for candidate_id in ballot.data
        ]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [self.candidates[candidate_id] for candidate_id, _ in winner_counts]


# Usage Example
if __name__ == "__main__":
    N_CANDIDATES = 10
    N_VOTERS = 10_000

    candidates = [Candidate.random(i, 3) for i in range(1, 10)]

    # STV Election
    stv_voters = [RankedVoter.random(f"v{i}", 3) for i in range(N_VOTERS)]
    stv_election = STVElection(candidates, winners=2)
    print("STV Winner:", stv_election.run(stv_voters)[0].id)
