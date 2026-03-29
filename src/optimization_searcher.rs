use crate::data;
use crate::data::BuildOptionId;
use crate::search_handler::{LocalState, SearchResult, SharedState};
use std::sync::Arc;

pub struct OptimizationSearcher {
    initial_state: LocalState,
    sequence: Vec<BuildOptionId>,
}

impl OptimizationSearcher {
    pub fn new(initial_state: LocalState, sequence: Vec<BuildOptionId>) -> Self {
        Self {
            initial_state,
            sequence,
        }
    }

    pub fn search(&mut self, s: &Arc<SharedState>) -> SearchResult {
        let mut best_sequence = Vec::new();
        let mut last_best_time;
        let mut best_time = f32::MAX;

        loop {
            last_best_time = best_time;
            let (new_sequence, new_time) = self.try_swaps(&self.sequence, best_time);
            let (new_sequence, new_time) = self.try_insertions(&new_sequence, new_time);
            let (new_sequence, new_time) = self.try_deletions(&new_sequence, new_time);

            best_time = new_time;
            if best_time < last_best_time {
                best_sequence = new_sequence;
            } else {
                break;
            }
        }

        SearchResult {
            time: best_time,
            sequence: best_sequence,
        }
    }

    fn compute_time<Iter: Iterator<Item = BuildOptionId>>(
        &self,
        sequence: Iter,
        mut state: LocalState,
        max_time: f32,
    ) -> f32{
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
        max_time: f32,
    ) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::new();
        let mut state_to_i = self.initial_state.clone();

        for i in 0..(sequence.len() - 1) {
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
        max_time: f32,
    ) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::with_capacity(sequence.len() + 1);
        let mut state_to_i = self.initial_state.clone();

        for i in 0..=sequence.len() {
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
        max_time: f32,
    ) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::with_capacity(sequence.len() - 1);
        let mut state_to_i = self.initial_state.clone();

        for i in 0..=sequence.len() {
            let time = self.compute_time(sequence[(i + 1)..].iter().copied(), state_to_i.clone(), best_time);

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
