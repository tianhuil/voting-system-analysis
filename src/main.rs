use ndarray::{Array1, Array2};
use num::Float;
use rand::prelude::*;
use rand_distr::Normal;
use std::collections::{HashMap, HashSet};

type CandidateId = i64;
type VoterId = i64;

struct Candidates {
    vectors: Array2<f64>,
}

impl Candidates {
    fn random(n_candidates: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let vectors = Array2::from_shape_fn((n_candidates, dim), |_| normal.sample(&mut rng));
        Self { vectors }
    }
}

struct Voters {
    vectors: Array2<f64>,
}

impl Voters {
    fn random(n_voters: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let vectors = Array2::from_shape_fn((n_voters, dim), |_| normal.sample(&mut rng));
        Self { vectors }
    }

    fn perturb(&self, sigma: f64) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, sigma).unwrap();
        let perturbation = Array2::from_shape_fn(self.vectors.dim(), |_| normal.sample(&mut rng));
        Self {
            vectors: &self.vectors + &perturbation,
        }
    }
}

trait Election {
    fn name(&self) -> &'static str;
    fn run(&self, voter_vectors: &Array2<f64>, candidate_vectors: &Array2<f64>, winners: usize) -> Vec<CandidateId>;
}

struct FPTPElection;

impl FPTPElection {
    fn cast_ballot(voter_vector: &Array1<f64>, candidate_vectors: &Array2<f64>) -> CandidateId {
        let mut min_dist = f64::INFINITY;
        let mut closest_candidate = 0;

        for (j, candidate_vector) in candidate_vectors.rows().into_iter().enumerate() {
            let candidate_vector = candidate_vector.to_owned();
            let dist = (voter_vector - &candidate_vector).mapv(|x| x * x).sum().sqrt();
            if dist < min_dist {
                min_dist = dist;
                closest_candidate = j as i64;
            }
        }
        closest_candidate
    }
}

impl Election for FPTPElection {
    fn name(&self) -> &'static str {
        "FPTP"
    }

    fn run(&self, voter_vectors: &Array2<f64>, candidate_vectors: &Array2<f64>, winners: usize) -> Vec<CandidateId> {
        let n_voters = voter_vectors.nrows();
        let mut candidate_ids = vec![0; n_voters];

        // Cast ballots
        for i in 0..n_voters {
            let voter_vector = voter_vectors.row(i).to_owned();
            candidate_ids[i] = Self::cast_ballot(&voter_vector, candidate_vectors);
        }

        // Count votes
        let mut counts: HashMap<CandidateId, usize> = HashMap::new();
        for &cid in &candidate_ids {
            *counts.entry(cid).or_insert(0) += 1;
        }

        // Get winners
        let mut count_vec: Vec<_> = counts.into_iter().collect();
        count_vec.sort_by(|a, b| b.1.cmp(&a.1));
        count_vec.into_iter().take(winners).map(|(cid, _)| cid).collect()
    }
}

struct RCVElection;

impl RCVElection {
    fn cast_ballot(voter_vector: &Array1<f64>, candidate_vectors: &Array2<f64>) -> Vec<(CandidateId, usize)> {
        let n_candidates = candidate_vectors.nrows();
        let mut distances: Vec<(usize, f64)> = (0..n_candidates)
            .map(|j| {
                let candidate_vector = candidate_vectors.row(j).to_owned();
                let dist = (voter_vector - &candidate_vector).mapv(|x| x * x).sum().sqrt();
                (j, dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        distances.iter()
            .enumerate()
            .map(|(rank, &(cid, _))| (cid as i64, rank + 1))
            .collect()
    }
}

impl Election for RCVElection {
    fn name(&self) -> &'static str {
        "RCV"
    }

    fn run(&self, voter_vectors: &Array2<f64>, candidate_vectors: &Array2<f64>, winners: usize) -> Vec<CandidateId> {
        let n_voters = voter_vectors.nrows();
        let n_candidates = candidate_vectors.nrows();
        let mut ballots = vec![vec![(0, 0); n_candidates]; n_voters];

        // Cast ballots
        for i in 0..n_voters {
            let voter_vector = voter_vectors.row(i).to_owned();
            ballots[i] = Self::cast_ballot(&voter_vector, candidate_vectors);
        }

        // Run RCV
        let mut active_candidates: HashSet<CandidateId> = (0..n_candidates).map(|i| i as i64).collect();
        let mut winners_vec = Vec::with_capacity(winners);

        while winners_vec.len() < winners && !active_candidates.is_empty() {
            // Count first preferences
            let mut counts: HashMap<CandidateId, usize> = HashMap::new();
            for ballot in &ballots {
                for &(cid, _) in ballot {
                    if active_candidates.contains(&cid) {
                        *counts.entry(cid).or_insert(0) += 1;
                        break;
                    }
                }
            }

            // Check for majority
            let total_votes: usize = counts.values().sum();
            let majority = total_votes / 2 + 1;

            let mut eliminated = None;
            for (&cid, &count) in &counts {
                if count >= majority {
                    winners_vec.push(cid);
                    active_candidates.remove(&cid);
                    eliminated = Some(cid);
                    break;
                }
            }

            if eliminated.is_none() {
                // Eliminate last place
                let min_count = counts.values().min().unwrap_or(&0);
                for (&cid, &count) in &counts {
                    if count == *min_count {
                        active_candidates.remove(&cid);
                        break;
                    }
                }
            }
        }

        winners_vec
    }
}

struct ApprovalVotingElection {
    cutoff: f64,
}

impl ApprovalVotingElection {
    fn cast_ballot(voter_vector: &Array1<f64>, candidate_vectors: &Array2<f64>, cutoff: f64) -> Vec<CandidateId> {
        let n_candidates = candidate_vectors.nrows();
        let mut distances: Vec<(usize, f64)> = (0..n_candidates)
            .map(|j| {
                let candidate_vector = candidate_vectors.row(j).to_owned();
                let dist = (voter_vector - &candidate_vector).mapv(|x| x * x).sum().sqrt();
                (j, dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let approved_count = (n_candidates as f64 * cutoff) as usize;
        distances.iter()
            .take(approved_count)
            .map(|&(cid, _)| cid as i64)
            .collect()
    }
}

impl Election for ApprovalVotingElection {
    fn name(&self) -> &'static str {
        "APPROVAL"
    }

    fn run(&self, voter_vectors: &Array2<f64>, candidate_vectors: &Array2<f64>, winners: usize) -> Vec<CandidateId> {
        let n_voters = voter_vectors.nrows();
        let mut candidate_ids = Vec::new();

        // Cast ballots
        for i in 0..n_voters {
            let voter_vector = voter_vectors.row(i).to_owned();
            let approved = Self::cast_ballot(&voter_vector, candidate_vectors, self.cutoff);
            candidate_ids.extend(approved);
        }

        // Count votes
        let mut counts: HashMap<CandidateId, usize> = HashMap::new();
        for &cid in &candidate_ids {
            *counts.entry(cid).or_insert(0) += 1;
        }

        // Get winners
        let mut count_vec: Vec<_> = counts.into_iter().collect();
        count_vec.sort_by(|a, b| b.1.cmp(&a.1));
        count_vec.into_iter().take(winners).map(|(cid, _)| cid).collect()
    }
}

fn run_single_winner_election<E: Election>(
    election: &E,
    candidates: &Candidates,
    true_voters: &Voters,
    perturbed_voters: &[Voters],
) -> f64 {
    let true_winner = election.run(&true_voters.vectors, &candidates.vectors, 1)[0];
    let matches: f64 = perturbed_voters
        .iter()
        .map(|voters| {
            let winner = election.run(&voters.vectors, &candidates.vectors, 1)[0];
            if winner == true_winner {
                1.0
            } else {
                0.0
            }
        })
        .sum();
    matches / perturbed_voters.len() as f64
}

fn main() {
    // Example usage
    let dimension = 3;
    let n_candidates = 10;
    let n_voters = 1000;
    let winners = 1;
    let sigma = 0.4;
    let iterations = 100;

    let candidates = Candidates::random(n_candidates, dimension);
    let voters = Voters::random(n_voters, dimension);
    let perturbed_voters: Vec<Voters> = (0..iterations)
        .map(|_| voters.perturb(sigma))
        .collect();

    // Run single winner elections
    let fptp = FPTPElection;
    let rcv = RCVElection;
    let approval = ApprovalVotingElection { cutoff: 0.5 };

    println!(
        "FPTP Match: {}",
        run_single_winner_election(&fptp, &candidates, &voters, &perturbed_voters)
    );
    println!(
        "RCV Match: {}",
        run_single_winner_election(&rcv, &candidates, &voters, &perturbed_voters)
    );
    println!(
        "Approval Match: {}",
        run_single_winner_election(&approval, &candidates, &voters, &perturbed_voters)
    );
} 