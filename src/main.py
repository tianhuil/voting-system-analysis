from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class ElectionType(Enum):
    SINGLE_WINNER = "single"
    MULTI_WINNER = "multi"
    MIXED = "mixed"


class VotingSystem(Enum):
    FPTP = "First-Past-the-Post"
    RCV = "Ranked Choice Voting"
    APPROVAL = "Approval"
    STAR = "STAR Voting"
    STV = "Single Transferable Vote"
    BORDA = "Borda Count"
    CONDORCET = "Condorcet"
    SCORE = "Score Voting"
    PHRAGMEN = "Phragmén's Method"


@dataclass
class Candidate:
    id: str
    name: str


@dataclass
class Ballot:
    voter_id: str
    rankings: Dict[str, int]  # candidate_id -> rank (1=first)
    scores: Dict[str, float]  # candidate_id -> score
    approvals: List[str]  # approved candidate_ids


class ElectionRound:
    def __init__(self, round_num: int):
        self.round_num = round_num
        self.ballots: List[Ballot] = []
        self.results: Dict[str, float] = {}  # candidate_id -> result_metric


class ElectionConfig:
    def __init__(
        self,
        system: VotingSystem,
        seats: int = 1,
        districts: int = 1,
        max_rounds: int = 1,
        scoring_range: Tuple[int, int] = (0, 5),
    ):
        self.system = system
        self.seats = seats
        self.districts = districts
        self.max_rounds = max_rounds
        self.scoring_range = scoring_range


class Election(ABC):
    def __init__(self, config: ElectionConfig):
        self.config = config
        self.candidates: List[Candidate] = []
        self.rounds: List[ElectionRound] = []
        self.current_round = ElectionRound(1)

    @abstractmethod
    def add_ballot(self, ballot: Ballot) -> bool:
        """Validate and add ballot to current round"""
        pass

    @abstractmethod
    def calculate_round(self) -> ElectionRound:
        """Process current round and return results"""
        pass

    def advance_round(self) -> ElectionRound:
        """Finalize current round and start new one"""
        finalized = self.calculate_round()
        self.rounds.append(finalized)

        if len(self.rounds) < self.config.max_rounds:
            self.current_round = ElectionRound(len(self.rounds) + 1)

        return finalized


# Concrete Implementations


class FPTPElection(Election):
    def add_ballot(self, ballot: Ballot) -> bool:
        if not ballot.rankings:
            return False
        self.current_round.ballots.append(ballot)
        return True

    def calculate_round(self) -> ElectionRound:
        counts: Dict[str, int] = {}
        for ballot in self.current_round.ballots:
            top_candidate = min(ballot.rankings.items(), key=lambda x: x[1])[0]
            counts[top_candidate] = counts.get(top_candidate, 0) + 1

        self.current_round.results = {k: float(v) for k, v in counts.items()}
        return self.current_round


class RCVElection(Election):
    def __init__(self, config: ElectionConfig):
        super().__init__(config)
        self.eliminated: List[str] = []

    def add_ballot(self, ballot: Ballot) -> bool:
        if len(ballot.rankings) < 1:
            return False
        self.current_round.ballots.append(ballot)
        return True

    def calculate_round(self) -> ElectionRound:
        active_candidates = set(c.id for c in self.candidates) - set(self.eliminated)
        counts: Dict[str, int] = {cid: 0 for cid in active_candidates}

        for ballot in self.current_round.ballots:
            active_ranks = {
                cid: rank
                for cid, rank in ballot.rankings.items()
                if cid in active_candidates
            }
            if active_ranks:
                top_candidate = min(active_ranks.items(), key=lambda x: x[1])[0]
                counts[top_candidate] += 1

        if self.config.seats == 1:  # IRV
            total = sum(counts.values())
            for cid, votes in counts.items():
                if votes > total / 2:
                    self.current_round.results = {cid: float(votes)}
                    return self.current_round

            # Eliminate last place
            elim = min(counts.items(), key=lambda x: x[1])[0]
            self.eliminated.append(elim)

        self.current_round.results = {k: float(v) for k, v in counts.items()}
        return self.current_round


class STARVotingElection(Election):
    def add_ballot(self, ballot: Ballot) -> bool:
        if not ballot.scores:
            return False
        self.current_round.ballots.append(ballot)
        return True

    def calculate_round(self) -> ElectionRound:
        # STAR Voting Logic
        scores: Dict[str, float] = {}
        for ballot in self.current_round.ballots:
            for cid, score in ballot.scores.items():
                scores[cid] = scores.get(cid, 0) + score

        if len(self.rounds) == 0:  # First round
            sorted_cands = sorted(scores.items(), key=lambda x: -x[1])
            top_two = [c[0] for c in sorted_cands[:2]]
            self.current_round.results = {cid: scores[cid] for cid in top_two}
        else:  # Runoff
            top_two = list(scores.keys())[:2]
            counts = {cid: 0 for cid in top_two}
            for ballot in self.current_round.ballots:
                preferred = max(top_two, key=lambda cid: ballot.scores.get(cid, 0))
                counts[preferred] += 1
            self.current_round.results = {k: float(v) for k, v in counts.items()}

        return self.current_round


class STVElection(Election):
    def __init__(self, config: ElectionConfig):
        super().__init__(config)
        self.quota: float = 0
        self.elected: List[str] = []
        self.transfers: Dict[str, List[Ballot]] = {}

    def add_ballot(self, ballot: Ballot) -> bool:
        if not ballot.rankings:
            return False
        self.current_round.ballots.append(ballot)
        return True

    def calculate_round(self) -> ElectionRound:
        if not self.quota:
            total = len(self.current_round.ballots)
            self.quota = total / (self.config.seats + 1) + 1

        counts: Dict[str, float] = {}
        for ballot in self.current_round.ballots:
            active = next(
                (cid for cid in ballot.rankings if cid not in self.elected), None
            )
            if active:
                counts[active] = counts.get(active, 0) + 1

        for cid, votes in counts.items():
            if votes >= self.quota:
                self.elected.append(cid)
                surplus = votes - self.quota
                transfer_factor = surplus / votes
                # Simplified transfer logic
                for ballot in self.current_round.ballots:
                    if ballot.rankings.get(cid, 0) == 1:
                        next_pref = next(
                            (
                                cid
                                for cid, rank in ballot.rankings.items()
                                if cid not in self.elected and rank > 1
                            ),
                            None,
                        )
                        if next_pref:
                            counts[next_pref] = (
                                counts.get(next_pref, 0) + transfer_factor
                            )

        self.current_round.results = {k: float(v) for k, v in counts.items()}
        return self.current_round


# Factory and Type Checking


class ElectionFactory:
    @staticmethod
    def create(config: ElectionConfig) -> Election:
        systems = {
            VotingSystem.FPTP: FPTPElection,
            VotingSystem.RCV: RCVElection,
            VotingSystem.STAR: STARVotingElection,
            VotingSystem.STV: STVElection,
        }
        return systems[config.system](config)


# Usage Example
if __name__ == "__main__":
    config = ElectionConfig(system=VotingSystem.RCV, seats=1, districts=1, max_rounds=3)

    election = ElectionFactory.create(config)
    candidates = [Candidate("1", "Alice"), Candidate("2", "Bob")]
    election.candidates = candidates

    # Add sample ballots
    election.add_ballot(Ballot("v1", {"1": 1, "2": 2}, {}, []))
    election.add_ballot(Ballot("v2", {"1": 1, "2": 2}, {}, []))
    election.add_ballot(Ballot("v3", {"2": 1}, {}, []))

    # Run rounds
    while len(election.rounds) < config.max_rounds:
        result = election.advance_round()
        print(f"Round {result.round_num} Results: {result.results}")
