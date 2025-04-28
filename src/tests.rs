use super::{
    rank_by_alignment, ApprovalVotingElection, Candidates, Election, FPTPElection, RCVElection,
    Voters,
};
use ndarray::Array2;

/// Creates a Candidates object from a vector of values
///
/// # Arguments
/// * `values` - A flat vector of values to be reshaped into a 2D array
/// * `rows` - Number of rows (candidates)
/// * `cols` - Number of columns (dimensions)
fn mock_candidates_from_vec(values: Vec<f64>, rows: usize, cols: usize) -> Candidates {
    Candidates::normalize(&Array2::from_shape_vec((rows, cols), values).unwrap())
}

/// Creates a Voters object from a vector of values
///
/// # Arguments
/// * `values` - A flat vector of values to be reshaped into a 2D array
/// * `rows` - Number of rows (voters)
/// * `cols` - Number of columns (dimensions)
fn mock_voters_from_vec(values: Vec<f64>, rows: usize, cols: usize) -> Voters {
    Voters::normalize(&Array2::from_shape_vec((rows, cols), values).unwrap())
}

#[test]
fn test_rank_by_alignment() {
    let voter = mock_voters_from_vec(vec![0.0, 1.0], 1, 2);

    // Use Candidates.normalize and Voters.normalize
    let candidates = mock_candidates_from_vec(
        vec![
            0.0, 1.0, // Candidate 0: Perfect alignment (0,1)
            1.0, 1.0, // Candidate 1: Diagonal (1,1)
            0.0, -1.0, // Candidate 2: Opposite (0,-1)
        ],
        3,
        2,
    );

    // Get rankings using the normalized vectors
    let rankings = rank_by_alignment(&voter.vectors.row(0).to_owned(), &candidates.vectors);

    // Check the order is correct
    assert_eq!(rankings[0].0, 0); // First should be the perfectly aligned candidate
    assert_eq!(rankings[1].0, 1); // Second should be the diagonal candidate
    assert_eq!(rankings[2].0, 2); // Last should be the opposite candidate

    // Check alignment values
    assert!((rankings[0].1 - 1.0).abs() < 1e-10); // Perfect alignment should be ~1.0
    assert!(rankings[0].1 > rankings[1].1); // Perfect alignment should be higher than diagonal
    assert!(rankings[1].1 > rankings[2].1); // Diagonal should be higher than opposite
    assert!((rankings[2].1 + 1.0).abs() < 1e-10); // Opposite alignment should be ~-1.0
}

fn create_test_data() -> (Voters, Candidates) {
    // Create 3 candidates in 2D space
    let candidates = mock_candidates_from_vec(
        vec![
            0.0, 1.0, // Candidate 0: (0,1)
            1.0, 0.0, // Candidate 1: (1,0)
            -1.0, 0.0, // Candidate 2: (-1,0)
        ],
        3,
        2,
    );

    // Create 5 voters in 2D space with more voters aligned with candidate 0
    let voters = mock_voters_from_vec(
        vec![
            0.0, 1.0, // Voter 0: (0,1) - should prefer candidate 0
            0.0, 1.0, // Voter 1: (0,1) - should prefer candidate 0
            0.0, 1.0, // Voter 2: (0,1) - should prefer candidate 0
            1.0, 0.0, // Voter 3: (1,0) - should prefer candidate 1
            -1.0, 0.0, // Voter 4: (-1,0) - should prefer candidate 2
        ],
        5,
        2,
    );

    (voters, candidates)
}

#[test]
fn test_fptp_election() {
    let (voters, candidates) = create_test_data();
    let election = FPTPElection;

    // Test single winner
    let winners = election.run(&voters.vectors, &candidates.vectors, 1);
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0], 0); // Candidate 0 should win (3 votes)

    // Test multiple winners
    let winners = election.run(&voters.vectors, &candidates.vectors, 2);
    assert_eq!(winners.len(), 2);
    assert_eq!(winners[0], 0); // First place: Candidate 0 (3 votes)
    assert!(winners.contains(&1) || winners.contains(&2)); // Second place: Candidate 1 or 2 (1 vote each)
}

#[test]
fn test_rcv_election() {
    let (voters, candidates) = create_test_data();
    let election = RCVElection;

    // Test single winner
    let winners = election.run(&voters.vectors, &candidates.vectors, 1);
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0], 0); // Candidate 0 should win (3 votes)

    // Test ballot casting
    let voter_vector = voters.vectors.row(0).to_owned();
    let ballot = RCVElection::cast_ballot(&voter_vector, &candidates.vectors);
    assert_eq!(ballot[0].0, 0); // First preference should be candidate 0
    assert_eq!(ballot[0].1, 1); // Rank 1
}

#[test]
fn test_rcv_election_open() {
    // Candidates: 0 = Extreme Left, 1 = Consensus, 2 = Extreme Right
    let candidates = mock_candidates_from_vec(
        vec![
            -1.0, 0.0, // Candidate 0: Extreme Left
            0.0, 1.0, // Candidate 1: Consensus (centrist)
            1.0, 0.0, // Candidate 2: Extreme Right
        ],
        3,
        2,
    );

    // Voters:
    // 2 Left voters, 3 Center voters, 3 Right voters
    let voters = mock_voters_from_vec(
        vec![
            -1.0, 0.0, // Left
            -1.0, 0.0, // Left
            0.0, 1.0, // Center
            0.0, 1.0, // Center
            0.0, 1.0, // Center
            1.0, 0.0, // Right
            1.0, 0.0, // Right
            1.0, 0.0, // Right
        ],
        8,
        2,
    );

    let result = RCVElection::run_open(&voters.vectors, &candidates.vectors, 1);

    // Print detailed round information for debugging
    println!("\nDetailed election results:");
    for (i, round) in result.rounds.iter().enumerate() {
        println!("\nRound {}:", i + 1);
        println!("Active candidates: {:?}", round.active_candidates);
        println!("Vote counts: {:?}", round.counts);
        println!("Winner: {:?}", round.winner);
        println!("Eliminated: {:?}", round.eliminated);
        println!("Total votes: {}", round.total_votes);
        println!("Majority threshold: {}", round.majority_threshold);
    }

    // Check final result
    assert!(result.winner.is_some());
    assert_eq!(result.winner.unwrap(), 1, "Consensus candidate should win");

    // Check rounds
    assert_eq!(result.rounds.len(), 2, "Should take 2 rounds");

    // First round
    let round1 = &result.rounds[0];
    assert_eq!(round1.round_number, 1);
    assert_eq!(round1.total_votes, 8);
    assert_eq!(round1.majority_threshold, 5);
    assert!(round1.winner.is_none());
    assert!(round1.eliminated.is_some());

    // Check vote counts in first round
    assert_eq!(
        round1.counts,
        [(0, 2), (1, 3), (2, 3)].into_iter().collect(),
        "Round 1 should have Left: 2 votes, Center: 3 votes, Right: 3 votes"
    );

    // Verify active candidates in first round
    assert_eq!(
        round1.active_candidates,
        vec![1, 2].into_iter().collect(),
        "All candidates should be active in round 1"
    );

    // Verify which candidate was eliminated
    assert_eq!(
        round1.eliminated,
        Some(0),
        "Left candidate should be eliminated in round 1"
    );

    // Second round
    let round2 = &result.rounds[1];
    assert_eq!(round2.round_number, 2);
    assert_eq!(round2.total_votes, 8);
    assert_eq!(round2.majority_threshold, 5);
    assert_eq!(round2.winner, Some(1));
    assert!(round2.eliminated.is_none());

    // Check vote counts in second round
    assert_eq!(
        round2.counts,
        [(1, 5), (2, 3)].into_iter().collect(),
        "Only Center and Right should have votes in round 2"
    );

    // Verify active candidates in second round
    assert_eq!(
        round2.active_candidates,
        vec![2].into_iter().collect(),
        "Only Right should be active in round 2"
    );
}

#[test]
fn test_approval_voting() {
    let (voters, candidates) = create_test_data();
    let election = ApprovalVotingElection { cutoff: 0.5 };

    // Test single winner with all voters
    let winners = election.run(&voters.vectors, &candidates.vectors, 1);
    println!("Winners: {:?}", winners);
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0], 0); // Candidate 0 should win (most approvals)

    // Test ballot casting with first voter (aligned with candidate 0)
    let voter_vector = voters.vectors.row(0).to_owned();
    let alignments = rank_by_alignment(&voter_vector, &candidates.vectors);
    println!("Alignments: {:?}", alignments);

    let approved = ApprovalVotingElection::cast_ballot(&voter_vector, &candidates.vectors, 0.5);
    println!("Approved with cutoff 0.5: {:?}", approved);
    assert_eq!(approved.len(), 1); // With cutoff 0.5 and 3 candidates, should approve 1 candidate
    assert!(approved.contains(&0)); // Should approve candidate 0 (highest alignment)

    // Test with different cutoff
    let approved = ApprovalVotingElection::cast_ballot(&voter_vector, &candidates.vectors, 0.33);
    println!("Approved with cutoff 0.33: {:?}", approved);
    assert_eq!(approved.len(), 1); // With cutoff 0.33 and 3 candidates, should approve 1 candidate
    assert!(approved.contains(&0)); // Should still approve candidate 0 (highest alignment)

    // Test with higher cutoff
    let approved = ApprovalVotingElection::cast_ballot(&voter_vector, &candidates.vectors, 0.7);
    println!("Approved with cutoff 0.7: {:?}", approved);
    assert_eq!(approved.len(), 2); // With cutoff 0.7 and 3 candidates, should approve 2 candidates
    assert!(approved.contains(&0)); // Should definitely approve candidate 0 (highest alignment)
}

#[test]
fn test_edge_cases() {
    let (voters, candidates) = create_test_data();

    // Test with zero winners
    let election = FPTPElection;
    let winners = election.run(&voters.vectors, &candidates.vectors, 0);
    assert_eq!(winners.len(), 0);

    // Test with more winners than candidates
    let winners = election.run(&voters.vectors, &candidates.vectors, 5);
    assert_eq!(winners.len(), 3); // Should return all candidates

    // Test with empty voter set
    let empty_voters = Array2::from_shape_vec((0, 2), vec![]).unwrap();
    let winners = election.run(&empty_voters, &candidates.vectors, 1);
    assert_eq!(winners.len(), 0);
}
