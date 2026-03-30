use crate::data;
use crate::data::BuildOptionId;
use crate::search_handler::{LocalState, SearchResult, SharedState};
use crate::searcher::Searcher;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub struct OptimizationSearcher {
    target: BuildOptionId,
    target_time: f32,
    sequence: Vec<BuildOptionId>,
}

#[derive(PartialOrd)]
struct OptimizationTarget {
    pub num_targets: usize,
    pub time: f32,
}

impl OptimizationTarget {
    pub fn inf() -> Self {
        Self {
            num_targets: usize::MAX,
            time: f32::MAX,
        }
    }
}

impl OptimizationSearcher {
    pub fn new(sequence: Vec<BuildOptionId>, target: BuildOptionId, time: u32) -> Self {
        Self {
            target,
            target_time: time as f32,
            sequence,
        }
    }

    fn apply_insertion(
        &self,
        insertion: SequenceInsertion,
        sequence: &Vec<BuildOptionId>,
        initial_state: LocalState,
        max_time: f32,
    ) -> OptimizationTarget {
        let suffix = sequence[..insertion.idx]
            .iter()
            .copied()
            .chain(std::iter::once(insertion.building))
            .chain(sequence[insertion.idx..].iter().copied());

        OptimizationSearcher::compute_target(suffix, initial_state, self.target, max_time)
    }

    fn apply_deletion(
        &self,
        deletion: SequenceDeletion,
        sequence: &Vec<BuildOptionId>,
        initial_state: LocalState,
        max_time: f32,
    ) -> OptimizationTarget {
        let suffix = sequence[..deletion.idx]
            .iter()
            .copied()
            .chain(sequence[(deletion.idx + 1)..].iter().copied());

        OptimizationSearcher::compute_target(suffix, initial_state, self.target, max_time)
    }

    fn compute_target<Iter: Iterator<Item = BuildOptionId>>(
        sequence: Iter,
        mut state: LocalState,
        target: BuildOptionId,
        max_time: f32,
    ) -> OptimizationTarget {
        let mut num_targets = 0;
        for build_id in sequence {
            if build_id == target {
                num_targets += 1;
            }

            match state.compute_next(build_id, max_time) {
                None => return OptimizationTarget::inf(),
                Some(new_state) => state = new_state,
            }
        }
        OptimizationTarget {
            num_targets,
            time: state.time,
        }
    }
}
struct SequenceInsertion {
    idx: usize,
    building: BuildOptionId,
}

impl SequenceInsertion {
    /// generates every possible insertion of 1 building in `sequence`
    pub fn generate(
        sequence: &Vec<BuildOptionId>,
        initial_state: LocalState,
    ) -> Vec<SequenceInsertion> {
        let mut state_to_i = initial_state;
        let mut insertions = Vec::new();

        // the last building in the sequence is a target, we do not need to insert after that
        for i in 0..sequence.len() {
            let build_options = data::get_build_options(&state_to_i.has_built);
            for option in build_options.ids() {
                insertions.push(SequenceInsertion {
                    idx: i,
                    building: option,
                })
            }

            if i < sequence.len() {
                state_to_i = state_to_i
                    .compute_next(sequence[i], f32::MAX)
                    .expect("sequence should be valid");
            }
        }

        insertions
    }
}

struct SequenceDeletion {
    idx: usize,
}

impl SequenceDeletion {
    pub fn generate(
        sequence: &Vec<BuildOptionId>,
        initial_state: LocalState,
    ) -> Vec<SequenceDeletion> {
        let mut state_to_i = initial_state;
        let mut insertions = Vec::new();

        // the last building in the sequence is our target, do not delete it
        for i in 0..sequence.len() {
            insertions.push(SequenceDeletion {
                idx: 0,
            });

            if i < sequence.len() {
                state_to_i = state_to_i
                    .compute_next(sequence[i], f32::MAX)
                    .expect("sequence should be valid");
            }
        }

        insertions
    }
}

impl PartialEq for OptimizationTarget {
    fn eq(&self, other: &Self) -> bool {
        self.num_targets == other.num_targets
            && self.time.total_cmp(&other.time) == std::cmp::Ordering::Equal
    }
}

impl Eq for OptimizationTarget {}

impl Ord for OptimizationTarget {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.num_targets
            .cmp(&other.num_targets)
            .then_with(|| self.time.total_cmp(&other.time).reverse())
    }
}
