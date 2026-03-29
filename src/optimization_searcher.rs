use crate::data;
use crate::data::BuildOptionId;
use crate::search_handler::{LocalState, SearchResult, SharedState};
use crate::searcher::Searcher;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub struct OptimizationSearcher {
    target: BuildOptionId,
    sequence: Vec<BuildOptionId>,
}

impl OptimizationSearcher {
    pub fn new(sequence: Vec<BuildOptionId>, target: BuildOptionId) -> Self {
        Self { target, sequence }
    }

    fn compute_time<Iter: Iterator<Item = BuildOptionId>>(
        &self,
        sequence: Iter,
        mut state: LocalState,
        max_time: f32,
    ) -> f32 {
        for build_id in sequence {
            match state.compute_next(build_id, max_time) {
                None => return f32::MAX,
                Some(new_state) => state = new_state,
            }
        }
        state.time
    }

    fn try_swaps(
        &self,
        sequence: &Vec<BuildOptionId>,
        initial_state: LocalState,
        max_time: f32,
    ) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::new();
        let mut state_to_i = initial_state;

        // we do not need to swap the target building
        for i in 0..(sequence.len() - 2) {
            if sequence[i] == sequence[i + 1] {
                // swap would be ineffective
                continue;
            }

            let suffix = std::iter::once(sequence[i + 1])
                .chain(std::iter::once(sequence[i]))
                .chain(sequence[(i + 2)..].iter().copied());

            let time = self.compute_time(suffix, state_to_i.clone(), best_time);
            if time < best_time {
                best_time = time;
                best_sequence = sequence.clone();
            }

            if i < sequence.len() {
                state_to_i = state_to_i
                    .compute_next(sequence[i], f32::MAX)
                    .expect("sequence should be valid");
            }
        }

        (best_sequence, best_time)
    }

    fn try_insertions(
        &self,
        sequence: &Vec<BuildOptionId>,
        initial_state: LocalState,
        max_time: f32,
    ) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::with_capacity(sequence.len() + 1);
        let mut state_to_i = initial_state;

        // the last building in the sequence is our target, we do not need to insert after that
        for i in 0..sequence.len() {
            let build_options = data::get_build_options(&state_to_i.has_built);
            for option in build_options.ids() {
                let suffix = std::iter::once(option).chain(sequence[i..].iter().copied());
                let time = self.compute_time(suffix, state_to_i.clone(), best_time);

                if time < best_time {
                    best_time = time;

                    best_sequence.clear();
                    best_sequence.extend_from_slice(&sequence[..i]);
                    best_sequence.push(option);
                    best_sequence.extend_from_slice(&sequence[i..]);
                }
            }

            if i < sequence.len() {
                state_to_i = state_to_i
                    .compute_next(sequence[i], f32::MAX)
                    .expect("sequence should be valid");
            }
        }

        (best_sequence, best_time)
    }

    fn try_deletions(
        &self,
        sequence: &Vec<BuildOptionId>,
        initial_state: LocalState,
        max_time: f32,
    ) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::with_capacity(sequence.len() - 1);
        let mut state_to_i = initial_state;

        // the last building in the sequence is our target, do not delete it
        for i in 0..sequence.len() {
            let time = self.compute_time(
                sequence[(i + 1)..].iter().copied(),
                state_to_i.clone(),
                best_time,
            );

            if time < best_time {
                best_time = time;

                best_sequence.clear();
                best_sequence.extend_from_slice(&sequence[..i]);
                best_sequence.extend_from_slice(&sequence[(i + 1)..]);
            }

            if i < sequence.len() {
                state_to_i = state_to_i
                    .compute_next(sequence[i], f32::MAX)
                    .expect("sequence should be valid");
            }
        }

        (best_sequence, best_time)
    }
}

impl Searcher for OptimizationSearcher {
    fn search(
        &mut self,
        shared_state: &Arc<SharedState>,
        initial_state: LocalState,
    ) -> SearchResult {
        let mut best_sequence = Vec::new();
        let mut last_best_time;
        let mut best_time = f32::MAX;

        loop {
            last_best_time = best_time;
            let (new_sequence, new_time) = self.try_swaps(&self.sequence, initial_state.clone(), best_time);
            let (new_sequence, new_time) = self.try_insertions(&new_sequence, initial_state.clone(), new_time);
            let (new_sequence, new_time) = self.try_deletions(&new_sequence, initial_state.clone(), new_time);

            best_time = new_time;
            if best_time < last_best_time {
                best_sequence = new_sequence;
                shared_state
                    .best_time
                    .store(f32::ceil(best_time) as u32, Ordering::Relaxed);
            } else {
                break;
            }
        }

        SearchResult {
            time: best_time,
            sequence: best_sequence,
        }
    }
}
