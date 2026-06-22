/// consensus.rs — pBFT Proof-of-Stake consensus for FLoBC.
///
/// Implements the paper's Byzantine Fault Tolerance requirement:
///   "Strictly more than 2/3 of validators are non-Byzantine."
///
/// Each block proposal must carry signed votes from validators holding
/// collectively more than 2/3 of total stake.  Only then is the block
/// committed to the chain — identical to the pBFT safety threshold.

use crate::block::VoteData;

/// Result of a consensus round.
#[derive(Debug)]
pub struct ConsensusResult {
    pub accepted:     bool,
    pub yes_stake:    f64,
    pub total_stake:  f64,
    pub yes_ratio:    f64,
    pub threshold:    f64,
    pub voter_count:  usize,
    pub yes_count:    usize,
}

/// Run a pBFT-style PoS vote.
///
/// * `votes`     — vote records from each validator (address, yes/no, stake)
/// * `threshold` — minimum yes-stake / total-stake ratio to accept (2/3 ≈ 0.6667)
///
/// Returns a `ConsensusResult` with `accepted = true` iff yes-stake > threshold.
pub fn pbft_vote(votes: &[VoteData], threshold: f64) -> ConsensusResult {
    let total_stake: f64 = votes.iter().map(|v| v.stake).sum();
    let yes_stake:   f64 = votes.iter().filter(|v| v.vote).map(|v| v.stake).sum();
    let yes_count        = votes.iter().filter(|v| v.vote).count();

    // Guard against zero-stake edge case
    let yes_ratio = if total_stake > 1e-12 {
        yes_stake / total_stake
    } else {
        0.0
    };

    // pBFT rule: strictly more than threshold (default 2/3)
    let accepted = total_stake > 0.0 && yes_ratio > threshold;

    ConsensusResult {
        accepted,
        yes_stake,
        total_stake,
        yes_ratio,
        threshold,
        voter_count: votes.len(),
        yes_count,
    }
}
