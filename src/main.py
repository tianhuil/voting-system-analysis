import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, List, Optional, Sequence, Set, TypeVar, Union

# Type Definitions
BallotType = TypeVar("BallotType")
CandidateId = int


@dataclass
class Candidate:
    id: CandidateId


@dataclass
class Ballot(Generic[BallotType]):
    voter_id: str
    data: BallotType


# Abstract Classes
class Voter(ABC, Generic[BallotType]):
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


# Voter Implementations
class RandomVoter(Voter[Dict[CandidateId, int]]):
    """FPTP/Approval voter that chooses randomly"""

    def __init__(self, voter_id: str):
        self.voter_id = voter_id

    def cast_ballot(
        self, candidates: List[Candidate]
    ) -> Ballot[Dict[CandidateId, int]]:
        # For Approval: Randomly approve 1-3 candidates
        approved = random.sample(candidates, k=random.randint(1, 3))
        return Ballot(voter_id=self.voter_id, data={c.id: 1 for c in approved})


class RankedVoter(Voter[Dict[CandidateId, int]]):
    """RCV/STV voter with preferences"""

    def __init__(self, voter_id: str, preferences: List[CandidateId]):
        self.voter_id = voter_id
        self.preferences = preferences

    def cast_ballot(
        self, candidates: List[Candidate]
    ) -> Ballot[Dict[CandidateId, int]]:
        remaining = [c.id for c in candidates if c.id not in self.preferences]
        random.shuffle(remaining)
        full_ranking = self.preferences + remaining
        return Ballot(
            voter_id=self.voter_id,
            data={cid: rank for rank, cid in enumerate(full_ranking, 1)},
        )


class StarVoter(Voter[Dict[CandidateId, float]]):
    """STAR voter that scores candidates 0-5"""

    def __init__(self, voter_id: str):
        self.voter_id = voter_id

    def cast_ballot(
        self, candidates: List[Candidate]
    ) -> Ballot[Dict[CandidateId, float]]:
        return Ballot(
            voter_id=self.voter_id,
            data={c.id: random.uniform(0, 5) for c in candidates},
        )


# Election Implementations
class FPTPElection(Election[Dict[CandidateId, int]]):
    name: str = "FPTP"

    def run(self, voters: Sequence[Voter[Dict[CandidateId, int]]]) -> List[Candidate]:
        ballots = [v.cast_ballot(self.candidates) for v in voters]
        votes: Dict[CandidateId, int] = {}

        # Count first-choice votes (FPTP) or sum approvals (Approval)
        for ballot in ballots:
            for cid, value in ballot.data.items():
                votes[cid] = votes.get(cid, 0) + value

        self.rounds.append(votes)
        sorted_candidates = sorted(
            self.candidates, key=lambda c: votes.get(c.id, 0), reverse=True
        )
        return sorted_candidates[: self.winners]


class RCVElection(Election[Dict[CandidateId, int]]):
    name: str = "RCV"

    def run(self, voters: Sequence[Voter[Dict[CandidateId, int]]]) -> List[Candidate]:
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


class STVElection(RCVElection):
    """Proper STV implementation with vote transfer"""

    name: str = "STV"

    def run(self, voters: Sequence[Voter[Dict[CandidateId, int]]]) -> List[Candidate]:
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


class STARVotingElection(Election[Dict[CandidateId, float]]):
    name: str = "STAR"

    def run(self, voters: Sequence[Voter[Dict[CandidateId, float]]]) -> List[Candidate]:
        ballots = [v.cast_ballot(self.candidates) for v in voters]

        # Scoring Round
        scores: Dict[CandidateId, float] = {}
        for ballot in ballots:
            for cid, score in ballot.data.items():
                scores[cid] = scores.get(cid, 0) + score

        self.rounds.append({"scores": scores})

        if len(self.candidates) <= 2:
            return sorted(
                self.candidates, key=lambda c: scores.get(c.id, 0), reverse=True
            )[: self.winners]

        # Automatic Runoff
        top_two = sorted(scores, key=scores.get, reverse=True)[:2]  # type: ignore
        runoff_counts = {cid: 0 for cid in top_two}

        for ballot in ballots:
            preferred = max(top_two, key=lambda cid: ballot.data.get(cid, 0))
            runoff_counts[preferred] += 1

        self.rounds.append({"runoff": runoff_counts})
        winner_id = max(runoff_counts, key=runoff_counts.get)  # type: ignore
        return [next(c for c in self.candidates if c.id == winner_id)]


class ApprovalVotingElection(FPTPElection):
    """Approval voting uses same counting as FPTP but different ballots"""

    name: str = "APPROVAL"

    pass


# Usage Example
if __name__ == "__main__":
    candidates = [
        Candidate(1),
        Candidate(2),
        Candidate(3),
    ]

    # STAR Voting
    star_voters = [StarVoter(f"v{i}") for i in range(100)]
    star_election = STARVotingElection(candidates)
    print("STAR Winner:", star_election.run(star_voters)[0].id)

    # STV Election
    stv_voters = [
        RankedVoter("v1", [1, 2]),
        RankedVoter("v2", [1, 3]),
        RankedVoter("v3", [2, 1]),
        RankedVoter("v4", [3, 2]),
        RankedVoter("v5", [3, 1]),
    ]
    stv_election = STVElection(candidates, winners=2)
