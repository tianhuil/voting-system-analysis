use super::*;
use ndarray::Array1;

fn create_test_data() -> (Array2<f64>, Array2<f64>) {
    // Create 3 candidates in 2D space
    let candidate_vectors = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.0, 1.0, // Candidate 0: (0,1)
            1.0, 0.0, // Candidate 1: (1,0)
            -1.0, 0.0, // Candidate 2: (-1,0)
        ],
    )
    .unwrap();
    normalize_vectors(&mut candidate_vectors.clone());

    // Create 5 voters in 2D space with more voters aligned with candidate 0
    let voter_vectors = Array2::from_shape_vec(
        (5, 2),
        vec![
            0.0, 1.0, // Voter 0: (0,1) - should prefer candidate 0
            0.0, 1.0, // Voter 1: (0,1) - should prefer candidate 0
            0.0, 1.0, // Voter 2: (0,1) - should prefer candidate 0
            1.0, 0.0, // Voter 3: (1,0) - should prefer candidate 1
            -1.0, 0.0, // Voter 4: (-1,0) - should prefer candidate 2
        ],
    )
    .unwrap();
    normalize_vectors(&mut voter_vectors.clone());

    (voter_vectors, candidate_vectors)
}

#[test]
fn test_fptp_election() {
    let (voter_vectors, candidate_vectors) = create_test_data();
    let election = FPTPElection;

    // Test single winner
    let winners = election.run(&voter_vectors, &candidate_vectors, 1);
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0], 0); // Candidate 0 should win (3 votes)

    // Test multiple winners
    let winners = election.run(&voter_vectors, &candidate_vectors, 2);
    assert_eq!(winners.len(), 2);
    assert_eq!(winners[0], 0); // First place: Candidate 0 (3 votes)
    assert!(winners.contains(&1) || winners.contains(&2)); // Second place: Candidate 1 or 2 (1 vote each)
}

#[test]
fn test_rcv_election() {
    let (voter_vectors, candidate_vectors) = create_test_data();
    let election = RCVElection;

    // Test single winner
    let winners = election.run(&voter_vectors, &candidate_vectors, 1);
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0], 0); // Candidate 0 should win (3 votes)

    // Test ballot casting
    let voter_vector = voter_vectors.row(0).to_owned();
    let ballot = RCVElection::cast_ballot(&voter_vector, &candidate_vectors);
    assert_eq!(ballot[0].0, 0); // First preference should be candidate 0
    assert_eq!(ballot[0].1, 1); // Rank 1
}

#[test]
fn test_approval_voting() {
    let (voter_vectors, candidate_vectors) = create_test_data();
    let election = ApprovalVotingElection { cutoff: 0.5 };

    // Test single winner with all voters
    let winners = election.run(&voter_vectors, &candidate_vectors, 1);
    println!("Winners: {:?}", winners);
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0], 0); // Candidate 0 should win (most approvals)

    // Test ballot casting with first voter (aligned with candidate 0)
    let voter_vector = voter_vectors.row(0).to_owned();
    let alignments = rank_by_alignment(&voter_vector, &candidate_vectors);
    println!("Alignments: {:?}", alignments);

    let approved = ApprovalVotingElection::cast_ballot(&voter_vector, &candidate_vectors, 0.5);
    println!("Approved with cutoff 0.5: {:?}", approved);
    assert_eq!(approved.len(), 1); // With cutoff 0.5 and 3 candidates, should approve 1 candidate
    assert!(approved.contains(&0)); // Should approve candidate 0 (highest alignment)

    // Test with different cutoff
    let approved = ApprovalVotingElection::cast_ballot(&voter_vector, &candidate_vectors, 0.33);
    println!("Approved with cutoff 0.33: {:?}", approved);
    assert_eq!(approved.len(), 1); // With cutoff 0.33 and 3 candidates, should approve 1 candidate
    assert!(approved.contains(&0)); // Should still approve candidate 0 (highest alignment)

    // Test with higher cutoff
    let approved = ApprovalVotingElection::cast_ballot(&voter_vector, &candidate_vectors, 0.7);
    println!("Approved with cutoff 0.7: {:?}", approved);
    assert_eq!(approved.len(), 2); // With cutoff 0.7 and 3 candidates, should approve 2 candidates
    assert!(approved.contains(&0)); // Should definitely approve candidate 0 (highest alignment)
}

#[test]
fn test_edge_cases() {
    let (voter_vectors, candidate_vectors) = create_test_data();

    // Test with zero winners
    let election = FPTPElection;
    let winners = election.run(&voter_vectors, &candidate_vectors, 0);
    assert_eq!(winners.len(), 0);

    // Test with more winners than candidates
    let winners = election.run(&voter_vectors, &candidate_vectors, 5);
    assert_eq!(winners.len(), 3); // Should return all candidates

    // Test with empty voter set
    let empty_voters = Array2::from_shape_vec((0, 2), vec![]).unwrap();
    let winners = election.run(&empty_voters, &candidate_vectors, 1);
    assert_eq!(winners.len(), 0);
}
