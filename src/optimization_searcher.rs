use crate::data::BuildOptionId;
use crate::search_handler::{LocalState, SearchResult, SharedState};
use std::sync::Arc;
use crate::data;

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
        let old_sequence = self.sequence.clone();

        let (new_sequence, new_time) = self.try_swaps(old_sequence, f32::MAX);

        todo!()
    }

    fn compute_time(&self, sequence: &Vec<BuildOptionId>, max_time: f32) -> f32 {
        let mut state = self.initial_state.clone();
        for build_id in sequence {
            match state.compute_next(*build_id, max_time) {
                None => return f32::MAX,
                Some(new_state) => state = new_state,
            }
        }
        state.time
    }

    fn try_swaps(&self, mut sequence: Vec<BuildOptionId>, max_time: f32) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::new();

        for i in 1..sequence.len() {
            sequence.swap(i - 1, i);
            let time = self.compute_time(&sequence, best_time);
            if time < best_time {
                best_time = time;
                best_sequence = sequence.clone();
            }
            // swap back
            sequence.swap(i - 1, i);
        }

        (best_sequence, best_time)
    }

    fn try_insertions(&self, mut sequence: Vec<BuildOptionId>, max_time: f32) -> (Vec<BuildOptionId>, f32) {
        let mut best_time = max_time;
        let mut best_sequence = Vec::new();

        for i in 0..sequence.len() {
            // sequence.insert(i);
            let time = self.compute_time(&sequence, best_time);
            if time < best_time {
                best_time = time;
                best_sequence = sequence.clone();
            }
            // remove
            // sequence.insert(i);
        }

        (best_sequence, best_time)
    }
}
