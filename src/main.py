import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import numba
import numpy as np
from numba import jit, njit, prange

# Type Definitions
BallotType = TypeVar("BallotType")
CandidateId = int
VoterId = int


# Numba-compatible data structures
@dataclass
class Candidates:
    """Numba-compatible candidates data structure"""

    ids: np.ndarray  # Array of candidate IDs
    vectors: np.ndarray  # 2D array of candidate vectors [n_candidates, dimension]

    @classmethod
    def random(cls, n_candidates: int, dim: int) -> "Candidates":
        """Create random candidates"""
        ids = np.arange(n_candidates, dtype=np.int64)
        vectors = np.random.normal(loc=0.0, scale=1.0, size=(n_candidates, dim))
        return cls(ids, vectors)


@dataclass
class Voters:
    """Numba-compatible voters data structure"""

    ids: np.ndarray  # Array of voter IDs
    vectors: np.ndarray  # 2D array of voter vectors [n_voters, dimension]

    @classmethod
    def random(cls, n_voters: int, dim: int) -> "Voters":
        """Create random voters"""
        ids = np.arange(n_voters, dtype=np.int64)
        vectors = np.random.normal(loc=0.0, scale=1.0, size=(n_voters, dim))
        return cls(ids, vectors)

    def get_voter(self, idx: int) -> Tuple[VoterId, np.ndarray]:
        """Get voter by index"""
        return self.ids[idx], self.vectors[idx]

    def perturb(self, sigma: float) -> "Voters":
        """Create perturbed voters"""
        return Voters(
            self.ids,
            self.vectors
            + np.random.normal(loc=0.0, scale=sigma, size=self.vectors.shape),
        )


@dataclass
class Ballot(Generic[BallotType]):
    voter_id: VoterId
    data: BallotType


class Election(ABC, Generic[BallotType]):
    def __init__(self, candidates: Candidates, winners: int = 1):
        self.candidates = candidates
        self.winners = winners
        self.rounds: List[Dict] = []

    @abstractmethod
    def cast_ballot(
        self, voter_id: VoterId, voter_vector: np.ndarray
    ) -> Ballot[BallotType]:
        """Cast a ballot for a voter based on the election rules"""
        pass

    @abstractmethod
    def run(self, voters: Voters) -> List[Tuple[CandidateId, np.ndarray]]:
        pass


@njit
def rank_by_distance(
    voter_vector: np.ndarray, candidate_vectors: np.ndarray
) -> np.ndarray:
    """Numba-compatible function to rank candidates by distance"""
    n_candidates = candidate_vectors.shape[0]
    distances = np.zeros(n_candidates)
    for i in range(n_candidates):
        distances[i] = np.linalg.norm(voter_vector - candidate_vectors[i])
    return np.argsort(distances)


########################################################
# First Past The Post (FPTP) System
########################################################


FPTPBallot = CandidateId


class FPTPElection(Election[FPTPBallot]):
    name: str = "FPTP"

    def cast_ballot(
        self, voter_id: VoterId, voter_vector: np.ndarray
    ) -> Ballot[CandidateId]:
        """FPTP voter that chooses the closest candidate"""
        ranked_indices = rank_by_distance(voter_vector, self.candidates.vectors)
        return Ballot(
            voter_id=voter_id,
            data=int(self.candidates.ids[ranked_indices[0]]),
        )

    def run(self, voters: Voters) -> List[Tuple[CandidateId, np.ndarray]]:
        ballots = [
            self.cast_ballot(voters.ids[i], voters.vectors[i])
            for i in range(len(voters.ids))
        ]
        candidate_ids = [b.data for b in ballots]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [
            (cid, self.candidates.vectors[self.candidates.ids == cid][0])
            for cid, _ in winner_counts
        ]


########################################################
# Ranked Choice Voting (RCV) System
########################################################


RankedBallot = Dict[CandidateId, int]


class RCVElection(Election[RankedBallot]):
    name: str = "RCV"

    def cast_ballot(
        self, voter_id: VoterId, voter_vector: np.ndarray
    ) -> Ballot[RankedBallot]:
        """RCV/STV voter with preferences"""
        ranked_indices = rank_by_distance(voter_vector, self.candidates.vectors)
        return Ballot(
            voter_id=voter_id,
            data={
                int(self.candidates.ids[idx]): rank
                for rank, idx in enumerate(ranked_indices, 1)
            },
        )

    def run(self, voters: Voters) -> List[Tuple[CandidateId, np.ndarray]]:
        ballots = [
            self.cast_ballot(voters.ids[i], voters.vectors[i])
            for i in range(len(voters.ids))
        ]
        active_candidates = set(self.candidates.ids)
        winners: List[Tuple[CandidateId, np.ndarray]] = []

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
                        winners.append(
                            (
                                cid,
                                self.candidates.vectors[self.candidates.ids == cid][0],
                            )
                        )
                        return winners

                # Eliminate last place
                eliminate_cid = min(counts, key=counts.get)  # type: ignore
                active_candidates.remove(eliminate_cid)
            else:  # STV Logic
                quota = total / (self.winners + 1) + 1
                elected = [cid for cid, count in counts.items() if count >= quota]

                if elected:
                    for cid in elected:
                        winners.append(
                            (
                                cid,
                                self.candidates.vectors[self.candidates.ids == cid][0],
                            )
                        )
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

    def run(self, voters: Voters) -> List[Tuple[CandidateId, np.ndarray]]:
        ballots = [
            self.cast_ballot(voters.ids[i], voters.vectors[i])
            for i in range(len(voters.ids))
        ]
        active_candidates = {
            cid: self.candidates.vectors[self.candidates.ids == cid][0]
            for cid in self.candidates.ids
        }
        winners: List[Tuple[CandidateId, np.ndarray]] = []
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
                winners.append((cid, active_candidates.pop(cid)))
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

    def __init__(self, candidates: Candidates, winners: int = 1, cutoff: float = 0.5):
        super().__init__(candidates, winners)
        self.cutoff = cutoff

    def cast_ballot(
        self, voter_id: VoterId, voter_vector: np.ndarray
    ) -> Ballot[ApprovalBallot]:
        """Approval voter that chooses the closest candidates up to cutoff"""
        ranked_indices = rank_by_distance(voter_vector, self.candidates.vectors)
        approved_count = int(len(ranked_indices) * self.cutoff)
        approved_candidates = ranked_indices[:approved_count]
        return Ballot(
            voter_id=voter_id,
            data={int(self.candidates.ids[idx]) for idx in approved_candidates},
        )

    def run(self, voters: Voters) -> List[Tuple[CandidateId, np.ndarray]]:
        ballots = [
            self.cast_ballot(voters.ids[i], voters.vectors[i])
            for i in range(len(voters.ids))
        ]
        candidate_ids = [
            candidate_id for ballot in ballots for candidate_id in ballot.data
        ]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [
            (cid, self.candidates.vectors[self.candidates.ids == cid][0])
            for cid, _ in winner_counts
        ]


########################################################
# Limited Voting System
########################################################

LimitedBallot = List[CandidateId]


class LimitedVotingElection(Election[LimitedBallot]):
    """Limited Voting: Each voter can vote for up to k candidates"""

    name: str = "LIMITED"

    def __init__(self, candidates: Candidates, winners: int = 1, max_choices: int = 3):
        super().__init__(candidates, winners)
        self.max_choices = max_choices

    def cast_ballot(
        self, voter_id: VoterId, voter_vector: np.ndarray
    ) -> Ballot[Set[CandidateId]]:
        """Limited voter that selects up to max_choices candidates"""
        ranked_indices = rank_by_distance(voter_vector, self.candidates.vectors)
        chosen = ranked_indices[: self.max_choices]
        return Ballot(
            voter_id=voter_id,
            data={int(self.candidates.ids[idx]) for idx in chosen},
        )

    def run(self, voters: Voters) -> List[Tuple[CandidateId, np.ndarray]]:
        ballots = [
            self.cast_ballot(voters.ids[i], voters.vectors[i])
            for i in range(len(voters.ids))
        ]
        candidate_ids = [
            candidate_id for ballot in ballots for candidate_id in ballot.data
        ]
        counts = Counter(candidate_ids)
        winner_counts = counts.most_common(self.winners)
        return [
            (cid, self.candidates.vectors[self.candidates.ids == cid][0])
            for cid, _ in winner_counts
        ]


def run_single_winner_election(
    election: Election,
    true_voters: Voters,
    perturbed_voters: Sequence[Voters],
) -> float:
    true_winner = election.run(true_voters)[0][0]
    perturbed_winners = [election.run(voters)[0][0] for voters in perturbed_voters]
    return float(np.mean([winner == true_winner for winner in perturbed_winners]))


# Usage Example
if __name__ == "__main__":
    DIMENSION = 3
    N_CANDIDATES = 10
    N_VOTERS = 1_000
    WINNERS = 1
    SIGMA = 0.4
    ITERATIONS = 100

    candidates = Candidates.random(N_CANDIDATES, DIMENSION)
    voters = Voters.random(N_VOTERS, DIMENSION)
    perturbed_voters = [voters.perturb(SIGMA) for _ in range(ITERATIONS)]

    # single winner elections
    fptp_election = FPTPElection(candidates, 1)
    rcv_election = RCVElection(candidates, 1)
    approval_election = ApprovalVotingElection(candidates, 1)

    print(
        f"FPTP Match: {run_single_winner_election(fptp_election, voters, perturbed_voters)}"
    )
    print(
        f"RCV Match: {run_single_winner_election(rcv_election, voters, perturbed_voters)}"
    )
    print(
        f"Approval Match: {run_single_winner_election(approval_election, voters, perturbed_voters)}"
    )
