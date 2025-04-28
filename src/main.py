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
    for i in prange(n_voters):
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
) -> np.ndarray:
    """Returns array of [candidate_id, rank] pairs sorted by rank"""
    ranked_indices = rank_by_distance(voter_vector, candidate_vectors)
    n_candidates = len(ranked_indices)
    result = np.zeros((n_candidates, 2), dtype=np.int64)
    for i, idx in enumerate(ranked_indices):
        result[i] = [int(idx), i + 1]  # [candidate_id, rank]
    return result


@njit
def _rcv_run(
    voter_vectors: np.ndarray, candidate_vectors: np.ndarray, winners: int
) -> np.ndarray:
    n_voters = voter_vectors.shape[0]
    n_candidates = candidate_vectors.shape[0]

    # Create ballots array: [n_voters, n_candidates, 2] for [candidate_id, rank]
    ballots = np.zeros((n_voters, n_candidates, 2), dtype=np.int64)
    for i in prange(n_voters):
        ballots[i] = _rcv_cast_ballot(voter_vectors[i], candidate_vectors)

    # Track active candidates with boolean array
    active_candidates = np.ones(n_candidates, dtype=np.bool_)
    winners_array = np.zeros(winners, dtype=np.int64)
    winners_count = 0

    while winners_count < winners and np.any(active_candidates):
        # Count current votes
        counts = np.zeros(n_candidates, dtype=np.float64)
        for i in range(n_voters):
            # Find highest ranked active candidate
            for j in range(n_candidates):
                cid = int(ballots[i, j, 0])
                if active_candidates[cid]:
                    counts[cid] += 1
                    break

        total = np.sum(counts)
        if total == 0:
            break

        if winners == 1:  # IRV Logic
            majority = total / 2
            for cid in range(n_candidates):
                if active_candidates[cid] and counts[cid] > majority:
                    winners_array[winners_count] = cid
                    return winners_array[: winners_count + 1]

            # Eliminate last place
            min_count = np.inf
            eliminate_cid = -1
            for cid in range(n_candidates):
                if active_candidates[cid] and counts[cid] < min_count:
                    min_count = counts[cid]
                    eliminate_cid = cid
            active_candidates[eliminate_cid] = False

        else:  # STV Logic
            quota = total / (winners + 1) + 1
            elected = np.zeros(n_candidates, dtype=np.bool_)

            # Find candidates meeting quota
            for cid in range(n_candidates):
                if active_candidates[cid] and counts[cid] >= quota:
                    elected[cid] = True
                    winners_array[winners_count] = cid
                    winners_count += 1
                    active_candidates[cid] = False

            if np.any(elected):
                # Transfer surplus votes (simplified)
                for cid in range(n_candidates):
                    if elected[cid]:
                        surplus = counts[cid] - quota
                        transfer_factor = surplus / counts[cid]

                        # Update counts for next preferences
                        for i in range(n_voters):
                            if int(ballots[i, 0, 0]) == cid:  # First preference
                                # Find next active preference
                                for j in range(1, n_candidates):
                                    next_cid = int(ballots[i, j, 0])
                                    if active_candidates[next_cid]:
                                        counts[next_cid] += transfer_factor
                                        break
            else:
                # Eliminate lowest candidate
                min_count = np.inf
                eliminate_cid = -1
                for cid in range(n_candidates):
                    if active_candidates[cid] and counts[cid] < min_count:
                        min_count = counts[cid]
                        eliminate_cid = cid
                active_candidates[eliminate_cid] = False

    return winners_array[:winners_count]


class RCVElection(Election[Dict[CandidateId, int]]):
    name: str = "RCV"

    def cast_ballot(self, voter_vector: np.ndarray) -> Dict[CandidateId, int]:
        ballot_array = _rcv_cast_ballot(voter_vector, self.candidates.vectors)
        return {int(row[0]): int(row[1]) for row in ballot_array}

    def run(self, voters: Voters) -> List[CandidateId]:
        winners = _rcv_run(voters.vectors, self.candidates.vectors, self.winners)
        return [int(cid) for cid in winners]


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
    for i in prange(n_voters):
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
    for i in prange(n_voters):
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
