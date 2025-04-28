use itertools::Itertools;
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

type CandidateId = i64;

/// Normalizes a 2D array of vectors so each row has unit length
fn normalize_vectors(vectors: &mut Array2<f64>) {
    let norms = vectors.map_axis(Axis(1), |row| row.mapv(|x| x * x).sum().sqrt());
    vectors
        .rows_mut()
        .into_iter()
        .zip(norms.iter())
        .for_each(|(mut row, &norm)| {
            if norm > 0.0 {
                row.mapv_inplace(|x| x / norm);
            }
        });
}

struct Candidates {
    vectors: Array2<f64>,
}

impl Candidates {
    pub fn random(n_candidates: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut vectors = Array2::from_shape_fn((n_candidates, dim), |_| normal.sample(&mut rng));
        normalize_vectors(&mut vectors);
        Self { vectors }
    }

    pub fn normalize(raw_vectors: &Array2<f64>) -> Self {
        let mut vectors = raw_vectors.clone();
        normalize_vectors(&mut vectors);
        Self { vectors }
    }
}

struct Voters {
    vectors: Array2<f64>,
}

impl Voters {
    pub fn random(n_voters: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut vectors = Array2::from_shape_fn((n_voters, dim), |_| normal.sample(&mut rng));
        normalize_vectors(&mut vectors);
        Self { vectors }
    }

    pub fn perturb(&self, sigma: f64) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, sigma).unwrap();
        let perturbation = Array2::from_shape_fn(self.vectors.dim(), |_| normal.sample(&mut rng));
        let mut vectors = &self.vectors + &perturbation;
        normalize_vectors(&mut vectors);
        Self { vectors }
    }

    pub fn normalize(raw_vectors: &Array2<f64>) -> Self {
        let mut vectors = raw_vectors.clone();
        normalize_vectors(&mut vectors);
        Self { vectors }
    }
}

trait Election {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        voter_vectors: &Array2<f64>,
        candidate_vectors: &Array2<f64>,
        winners: usize,
    ) -> Vec<CandidateId>;
}

/// Ranks candidates by their alignment (dot product) with a voter's position vector
/// Assumes that the voter and candidate vectors are normalized
fn rank_by_alignment(
    voter_vector: &Array1<f64>,
    candidate_vectors: &Array2<f64>,
) -> Vec<(CandidateId, f64)> {
    let mut alignments: Vec<(CandidateId, f64)> = candidate_vectors
        .rows()
        .into_iter()
        .enumerate()
        .map(|(j, candidate_vector)| {
            let candidate_vector = candidate_vector.to_owned();
            let alignment = voter_vector.dot(&candidate_vector);
            (j as i64, alignment)
        })
        .collect();
    alignments.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    alignments
}

////////////////////////////////////////////////////////////
/// First Past the Post
////////////////////////////////////////////////////////////

struct FPTPElection;

impl FPTPElection {
    fn cast_ballot(voter_vector: &Array1<f64>, candidate_vectors: &Array2<f64>) -> CandidateId {
        rank_by_alignment(voter_vector, candidate_vectors)[0].0
    }
}

impl Election for FPTPElection {
    fn name(&self) -> &'static str {
        "FPTP"
    }

    fn run(
        &self,
        voter_vectors: &Array2<f64>,
        candidate_vectors: &Array2<f64>,
        winners: usize,
    ) -> Vec<CandidateId> {
        // Cast ballots using functional approach
        let candidate_ids: Vec<CandidateId> = voter_vectors
            .rows()
            .into_iter()
            .map(|voter_vector| Self::cast_ballot(&voter_vector.to_owned(), candidate_vectors))
            .collect();

        // Count votes
        let counts = candidate_ids.iter().counts();

        // Get winners
        let mut count_vec: Vec<_> = counts.into_iter().collect();
        count_vec.sort_by(|a, b| b.1.cmp(&a.1));
        count_vec
            .into_iter()
            .take(winners)
            .map(|(cid, _)| *cid)
            .collect()
    }
}

////////////////////////////////////////////////////////////
/// Ranked Choice Voting
////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct RCVRoundInfo {
    pub round_number: usize,
    pub active_candidates: HashSet<CandidateId>,
    pub counts: HashMap<CandidateId, usize>,
    pub total_votes: usize,
    pub majority_threshold: usize,
    pub winner: Option<CandidateId>,
    pub eliminated: Option<CandidateId>,
}

pub struct RCVResult {
    pub winner: Option<CandidateId>,
    pub rounds: Vec<RCVRoundInfo>,
}

struct RCVElection;

impl RCVElection {
    fn cast_ballot(
        voter_vector: &Array1<f64>,
        candidate_vectors: &Array2<f64>,
    ) -> Vec<(CandidateId, usize)> {
        rank_by_alignment(voter_vector, candidate_vectors)
            .into_iter()
            .enumerate()
            .map(|(rank, (cid, _))| (cid, rank + 1))
            .collect()
    }

    fn run_open(
        voter_vectors: &Array2<f64>,
        candidate_vectors: &Array2<f64>,
        winners: usize,
    ) -> RCVResult {
        if winners != 1 {
            panic!("RCV election only supports single winner elections");
        }

        // Cast ballots using functional approach
        let ballots: Vec<Vec<(CandidateId, usize)>> = voter_vectors
            .rows()
            .into_iter()
            .map(|voter_vector| Self::cast_ballot(&voter_vector.to_owned(), candidate_vectors))
            .collect();

        // Run RCV
        let mut active_candidates: HashSet<CandidateId> = candidate_vectors
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, _)| i as i64)
            .collect();
        let mut winner: Option<CandidateId> = None;
        let mut rounds = Vec::new();
        let mut round_number = 1;

        while winner.is_none() && !active_candidates.is_empty() {
            // Count first preferences
            let counts: HashMap<CandidateId, usize> = ballots
                .iter()
                .filter_map(|ballot| {
                    ballot
                        .iter()
                        .find(|&&(cid, _)| active_candidates.contains(&cid))
                        .map(|&(cid, _)| cid)
                })
                .counts();

            // Check for majority
            let total_votes: usize = counts.values().sum();
            let majority = total_votes / 2 + 1;

            let mut eliminated = None;

            // Check for majority winner
            if let Some((&cid, _)) = counts.iter().find(|(_, &count)| count >= majority) {
                winner = Some(cid);
                active_candidates.remove(&cid);
            } else {
                // If no winner, eliminate last place
                let min_count = counts.values().min().unwrap_or(&0);
                eliminated =
                    counts
                        .iter()
                        .find(|(_, &count)| count == *min_count)
                        .map(|(&cid, _)| {
                            active_candidates.remove(&cid);
                            cid
                        });
            }

            // Record round information
            rounds.push(RCVRoundInfo {
                round_number,
                active_candidates: active_candidates.clone(),
                counts,
                total_votes,
                majority_threshold: majority,
                winner,
                eliminated,
            });

            round_number += 1;
        }

        RCVResult { winner, rounds }
    }
}

impl Election for RCVElection {
    fn name(&self) -> &'static str {
        "RCV"
    }

    fn run(
        &self,
        voter_vectors: &Array2<f64>,
        candidate_vectors: &Array2<f64>,
        winners: usize,
    ) -> Vec<CandidateId> {
        Self::run_open(voter_vectors, candidate_vectors, winners)
            .winner
            .map_or_else(Vec::new, |w| vec![w])
    }
}

////////////////////////////////////////////////////////////
/// Approval Voting
////////////////////////////////////////////////////////////

struct ApprovalVotingElection {
    cutoff: f64,
}

impl ApprovalVotingElection {
    fn cast_ballot(
        voter_vector: &Array1<f64>,
        candidate_vectors: &Array2<f64>,
        cutoff: f64,
    ) -> Vec<CandidateId> {
        let n_candidates = candidate_vectors.nrows();
        // Ensure cutoff is at least 1/n_candidates to approve at least one candidate
        let min_cutoff = 1.0 / n_candidates as f64;
        let effective_cutoff = cutoff.max(min_cutoff);
        let approved_count = (n_candidates as f64 * effective_cutoff) as usize;
        if approved_count == 0 {
            panic!("approved_count is 0: cutoff too low or no candidates");
        }
        rank_by_alignment(voter_vector, candidate_vectors)
            .into_iter()
            .take(approved_count)
            .map(|(cid, _)| cid)
            .collect()
    }
}

impl Election for ApprovalVotingElection {
    fn name(&self) -> &'static str {
        "APPROVAL"
    }

    fn run(
        &self,
        voter_vectors: &Array2<f64>,
        candidate_vectors: &Array2<f64>,
        winners: usize,
    ) -> Vec<CandidateId> {
        if winners != 1 {
            panic!("Approval voting election only supports single winner elections");
        }

        // Cast ballots using functional approach
        let candidate_ids: Vec<CandidateId> = voter_vectors
            .rows()
            .into_iter()
            .flat_map(|voter_vector| {
                Self::cast_ballot(&voter_vector.to_owned(), candidate_vectors, self.cutoff)
            })
            .collect();

        // Count votes
        let counts = candidate_ids.iter().counts();

        // Get winners
        let mut count_vec: Vec<_> = counts.into_iter().collect();
        count_vec.sort_by(|a, b| b.1.cmp(&a.1));
        count_vec
            .into_iter()
            .take(winners)
            .map(|(cid, _)| *cid)
            .collect()
    }
}

////////////////////////////////////////////////////////////
/// Execution functions
////////////////////////////////////////////////////////////

/// Computes jackknife estimates of mean and standard error for a given aggregation function
///
/// # Arguments
/// * `data` - Vector of samples
/// * `f` - Aggregation function that takes a slice of samples and returns a f64
///
/// # Returns
/// Tuple of (mean, standard_error)
fn jackknife<T: Clone + Send + Sync, F>(data: &[T], f: F) -> (f64, f64)
where
    F: Fn(&[T]) -> f64 + Sync + Send,
{
    let n = data.len();

    // Compute leave-one-out estimates in parallel
    let theta_dots: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut leave_one_out: Vec<T> = data.to_vec();
            leave_one_out.remove(i);
            f(&leave_one_out)
        })
        .collect();

    // Compute jackknife mean
    let theta_dot = theta_dots.iter().sum::<f64>() / n as f64;

    // Compute jackknife standard error
    let variance = theta_dots
        .iter()
        .map(|&x| {
            let diff = x - theta_dot;
            diff * diff
        })
        .sum::<f64>()
        / (n - 1) as f64;

    let stderr = (variance / n as f64).sqrt();

    (theta_dot, stderr)
}

////////////////////////////////////////////////////////////
/// Run single winner elections
////////////////////////////////////////////////////////////
fn run_single_winner_election<E: Election + Sync>(
    election: &E,
    candidates: &Candidates,
    true_voters: &Voters,
    perturbed_voters: &[Voters],
) -> (f64, f64) {
    let true_winner = election.run(&true_voters.vectors, &candidates.vectors, 1)[0];
    let matches: Vec<f64> = perturbed_voters
        .par_iter()
        .map(|voters| {
            let winner = election.run(&voters.vectors, &candidates.vectors, 1)[0];
            if winner == true_winner {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let (mean, stderr) = jackknife(&matches, |slice| {
        slice.iter().sum::<f64>() / slice.len() as f64
    });

    println!("{}: {:.4} +/- {:.4}", election.name(), mean, stderr);

    (mean, stderr)
}

fn main() {
    let dimension = 3;
    let n_candidates = 10;
    let n_voters = 10_000;
    let sigma = 0.3;
    let iterations = 100;

    let candidates = Candidates::random(n_candidates, dimension);
    let voters = Voters::random(n_voters, dimension);
    let perturbed_voters: Vec<Voters> = (0..iterations).map(|_| voters.perturb(sigma)).collect();

    // Run single winner elections
    run_single_winner_election(&FPTPElection, &candidates, &voters, &perturbed_voters);
    run_single_winner_election(&RCVElection, &candidates, &voters, &perturbed_voters);
    run_single_winner_election(
        &ApprovalVotingElection { cutoff: 0.5 },
        &candidates,
        &voters,
        &perturbed_voters,
    );
}

#[cfg(test)]
mod tests;
