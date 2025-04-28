import random
from abc import ABC, abstractmethod
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
from numba import njit, prange

# Type Definitions
BallotType = TypeVar("BallotType")
CandidateId = int
VoterId = int


# Numba-compatible data structures
@dataclass
class Candidates:
    """Numba-compatible candidates data structure"""

    vectors: np.ndarray  # 2D array of candidate vectors [n_candidates, dimension]

    @classmethod
    def random(cls, n_candidates: int, dim: int) -> "Candidates":
        """Create random candidates"""
        vectors = np.random.normal(loc=0.0, scale=1.0, size=(n_candidates, dim))
        return cls(vectors)


@dataclass
class Voters:
    """Numba-compatible voters data structure"""

    vectors: np.ndarray  # 2D array of voter vectors [n_voters, dimension]

    @classmethod
    def random(cls, n_voters: int, dim: int) -> "Voters":
        """Create random voters"""
        vectors = np.random.normal(loc=0.0, scale=1.0, size=(n_voters, dim))
        return cls(vectors)

    def perturb(self, sigma: float) -> "Voters":
        """Create perturbed voters"""
        return Voters(
            self.vectors
            + np.random.normal(loc=0.0, scale=sigma, size=self.vectors.shape)
        )


class Election(ABC, Generic[BallotType]):
    def __init__(self, candidates: Candidates, winners: int = 1):
        self.candidates = candidates
        self.winners = winners
        self.rounds: List[Dict] = []

    @abstractmethod
    def cast_ballot(self, voter_vector: np.ndarray) -> BallotType:
        """Cast a ballot for a voter based on the election rules"""
        pass

    @abstractmethod
    def run(self, voters: Voters) -> List[CandidateId]:
        pass


@njit
def rank_by_distance(
    voter_vector: np.ndarray, candidate_vectors: np.ndarray
) -> np.ndarray:
    """
    Numba-compatible function to rank candidates by distance
    Args:
        voter_vector: The vector of the voter
        candidate_vectors: The vectors of the candidates (first index is the candidate index)
    Returns:
        The indices of the candidates sorted in descending preference
    """
    n_candidates = candidate_vectors.shape[0]
    distances = np.zeros(n_candidates)
    for i in prange(n_candidates):
        distances[i] = np.linalg.norm(voter_vector - candidate_vectors[i])
    return np.argsort(distances)


@njit
def count_occurrences(items):
    """
    Numba-compatible function to count occurrences of items in a list Returns a
    dictionary-like structure as a list of (item, count) tuples, sorted in
    descending order of count
    """
    # Create a list to store unique items and their counts
    unique_items = []
    counts = []

    # Count occurrences
    for item in items:
        found = False
        for i in range(len(unique_items)):
            if unique_items[i] == item:
                counts[i] += 1
                found = True
                break
        if not found:
            unique_items.append(item)
            counts.append(1)

    # Sort by count in descending order
    sorted_indices = np.argsort(np.array(counts))[::-1]
    result = []
    for i in sorted_indices:
        result.append((unique_items[i], counts[i]))

    return result


########################################################
# First Past The Post (FPTP) System
########################################################
@njit
def _fptp_cast_ballot(
    voter_vector: np.ndarray, candidate_vectors: np.ndarray
) -> CandidateId:
    ranked_indices = rank_by_distance(voter_vector, candidate_vectors)
    return int(ranked_indices[0])


@njit
def _fptp_run(
    voter_vectors: np.ndarray, candidate_vectors: np.ndarray, winners: int
) -> np.ndarray:
    n_voters = voter_vectors.shape[0]
    candidate_ids = np.zeros(n_voters, dtype=np.int64)
    for i in range(n_voters):
        candidate_ids[i] = _fptp_cast_ballot(voter_vectors[i], candidate_vectors)
    winner_counts = count_occurrences(candidate_ids)
    return np.array([cid for cid, _ in winner_counts[:winners]], dtype=np.int64)


class FPTPElection(Election[CandidateId]):
    name: str = "FPTP"

    def cast_ballot(self, voter_vector: np.ndarray) -> CandidateId:
        return _fptp_cast_ballot(voter_vector, self.candidates.vectors)

    def run(self, voters: Voters) -> List[CandidateId]:
        winners = _fptp_run(voters.vectors, self.candidates.vectors, self.winners)
        return [int(cid) for cid in winners]


########################################################
# Ranked Choice Voting (RCV) System
########################################################
@njit
def _rcv_cast_ballot(
    voter_vector: np.ndarray, candidate_vectors: np.ndarray
) -> Dict[CandidateId, int]:
    ranked_indices = rank_by_distance(voter_vector, candidate_vectors)
    return {int(idx): rank for rank, idx in enumerate(ranked_indices, 1)}


class RCVElection(Election[Dict[CandidateId, int]]):
    name: str = "RCV"

    def cast_ballot(self, voter_vector: np.ndarray) -> Dict[CandidateId, int]:
        return _rcv_cast_ballot(voter_vector, self.candidates.vectors)

    def run(self, voters: Voters) -> List[CandidateId]:
        ballots = [
            self.cast_ballot(voters.vectors[i]) for i in range(len(voters.vectors))
        ]
        active_candidates = set(range(len(self.candidates.vectors)))
        winners: List[CandidateId] = []

        while len(winners) < self.winners and active_candidates:
            counts: Dict[CandidateId, int | float] = {
                cid: 0 for cid in active_candidates
            }
            for ballot in ballots:
                valid_ranks = {
                    cid: rank
                    for cid, rank in ballot.items()
                    if cid in active_candidates
                }
                if valid_ranks:
                    top_cid = min(valid_ranks, key=lambda k: valid_ranks[k])
                    counts[top_cid] += 1

            self.rounds.append(counts.copy())

            total = sum(counts.values())
            if total == 0:
                break

            if self.winners == 1:
                majority = total / 2
                for cid, count in counts.items():
                    if count > majority:
                        winners.append(cid)
                        return winners

                eliminate_cid = min(counts, key=lambda k: counts[k])
                active_candidates.remove(eliminate_cid)
            else:
                quota = total / (self.winners + 1) + 1
                elected = [cid for cid, count in counts.items() if count >= quota]

                if elected:
                    for cid in elected:
                        winners.append(cid)
                        active_candidates.remove(cid)

                    transfer_factor = 0.5
                    for ballot in ballots:
                        if any(cid in ballot for cid in elected):
                            next_pref = next(
                                (
                                    cid
                                    for cid, rank in ballot.items()
                                    if cid in active_candidates
                                ),
                                None,
                            )
                            if next_pref:
                                counts[next_pref] += transfer_factor
                else:
                    eliminate_cid = min(counts, key=lambda k: counts[k])
                    active_candidates.remove(eliminate_cid)

        return winners


########################################################
# Single Transferable Vote (STV) System
########################################################
class STVElection(RCVElection):
    """Proper STV implementation with vote transfer"""

    name: str = "STV"

    def run(self, voters: Voters) -> List[CandidateId]:
        ballots = [
            self.cast_ballot(voters.vectors[i]) for i in range(len(voters.vectors))
        ]
        active_candidates = {
            cid: self.candidates.vectors[cid]
            for cid in range(len(self.candidates.vectors))
        }
        winners: List[CandidateId] = []
        quota = len(ballots) / (self.winners + 1) + 1

        while len(winners) < self.winners and active_candidates:
            counts: Dict[CandidateId, float] = {cid: 0.0 for cid in active_candidates}
            for ballot in ballots:
                valid_ranks = {
                    cid: rank
                    for cid, rank in ballot.items()
                    if cid in active_candidates
                }
                if valid_ranks:
                    top_cid = min(valid_ranks, key=lambda k: valid_ranks[k])
                    counts[top_cid] += 1

            self.rounds.append(counts.copy())

            elected = [cid for cid, count in counts.items() if count >= quota]
            for cid in elected:
                winners.append(cid)
                active_candidates.pop(cid)
                surplus = counts[cid] - quota

                transfer_factor = surplus / counts[cid]
                for ballot in ballots:
                    if ballot.get(cid, 0) == 1:
                        next_pref = next(
                            (
                                cid
                                for cid, rank in ballot.items()
                                if cid in active_candidates
                            ),
                            None,
                        )
                        if next_pref:
                            counts[next_pref] += transfer_factor

            if not elected:
                eliminate_cid = min(counts, key=lambda k: counts[k])
                active_candidates.pop(eliminate_cid)

        return winners


########################################################
# Approval Voting System
########################################################
@njit
def _approval_cast_ballot(
    voter_vector: np.ndarray, candidate_vectors: np.ndarray, cutoff: float
) -> np.ndarray:
    ranked_indices = rank_by_distance(voter_vector, candidate_vectors)
    approved_count = int(len(ranked_indices) * cutoff)
    return ranked_indices[:approved_count]


@njit
def _approval_run(
    voter_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    winners: int,
    cutoff: float,
) -> np.ndarray:
    n_voters = voter_vectors.shape[0]
    n_candidates = candidate_vectors.shape[0]
    candidate_ids = np.zeros(n_voters * n_candidates, dtype=np.int64)
    idx = 0
    for i in range(n_voters):
        approved_indices = _approval_cast_ballot(
            voter_vectors[i], candidate_vectors, cutoff
        )
        for j in range(len(approved_indices)):
            candidate_ids[idx] = int(approved_indices[j])
            idx += 1
    candidate_ids = candidate_ids[:idx]  # Trim to actual size
    winner_counts = count_occurrences(candidate_ids)
    return np.array([cid for cid, _ in winner_counts[:winners]], dtype=np.int64)


class ApprovalVotingElection(Election[Set[CandidateId]]):
    """Approval voting uses same counting as FPTP but different ballots"""

    name: str = "APPROVAL"

    def __init__(self, candidates: Candidates, winners: int = 1, cutoff: float = 0.5):
        super().__init__(candidates, winners)
        self.cutoff = cutoff

    def cast_ballot(self, voter_vector: np.ndarray) -> Set[CandidateId]:
        approved_indices = _approval_cast_ballot(
            voter_vector, self.candidates.vectors, self.cutoff
        )
        return {int(idx) for idx in approved_indices}

    def run(self, voters: Voters) -> List[CandidateId]:
        winners = _approval_run(
            voters.vectors, self.candidates.vectors, self.winners, self.cutoff
        )
        return [int(cid) for cid in winners]


########################################################
# Limited Voting System
########################################################
@njit
def _limited_cast_ballot(
    voter_vector: np.ndarray, candidate_vectors: np.ndarray, max_choices: int
) -> List[CandidateId]:
    ranked_indices = rank_by_distance(voter_vector, candidate_vectors)
    chosen = ranked_indices[:max_choices]
    return [int(idx) for idx in chosen]


@njit
def _limited_run(
    voter_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    winners: int,
    max_choices: int,
) -> np.ndarray:
    n_voters = voter_vectors.shape[0]
    candidate_ids = np.zeros(n_voters * max_choices, dtype=np.int64)
    idx = 0
    for i in range(n_voters):
        chosen = _limited_cast_ballot(voter_vectors[i], candidate_vectors, max_choices)
        for j in range(len(chosen)):
            candidate_ids[idx] = int(chosen[j])
            idx += 1
    candidate_ids = candidate_ids[:idx]  # Trim to actual size
    winner_counts = count_occurrences(candidate_ids)
    return np.array([cid for cid, _ in winner_counts[:winners]], dtype=np.int64)


class LimitedVotingElection(Election[List[CandidateId]]):
    """Limited Voting: Each voter can vote for up to k candidates"""

    name: str = "LIMITED"

    def __init__(self, candidates: Candidates, winners: int = 1, max_choices: int = 3):
        super().__init__(candidates, winners)
        self.max_choices = max_choices

    def cast_ballot(self, voter_vector: np.ndarray) -> List[CandidateId]:
        return _limited_cast_ballot(
            voter_vector, self.candidates.vectors, self.max_choices
        )

    def run(self, voters: Voters) -> List[CandidateId]:
        winners = _limited_run(
            voters.vectors, self.candidates.vectors, self.winners, self.max_choices
        )
        return [int(cid) for cid in winners]


def run_single_winner_election(
    election: Election,
    true_voters: Voters,
    perturbed_voters: Sequence[Voters],
) -> float:
    true_winner = election.run(true_voters)[0]
    perturbed_winners = [election.run(voters)[0] for voters in perturbed_voters]
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
