use crate::data;
use crate::data::BuildOptionId::{
    ConstructionVehicleT1, ConstructionVehicleT2,
};
use crate::data::{BuildOptionId, BuildSet};
use crate::search_handler::{LocalState, SearchResult, SharedState};
use crate::searcher::Searcher;
use std::sync::atomic::Ordering;
use std::sync::Arc;
pub struct BruteForceSearcher {
    target: BuildOptionId,
    max_buildings: u32,
    best_time: f32,
}

impl BruteForceSearcher {
    pub fn new(target: BuildOptionId, max_buildings: u32) -> Self {
        Self {
            target,
            max_buildings,
            best_time: f32::MAX,
        }
    }
    
    fn search_inner(
        &mut self,
        sequence: &mut Vec<BuildOptionId>,
        remaining_depth: u32,
        l: LocalState,
        s: &SharedState,
    ) -> SearchResult {
        if remaining_depth == 0 {
            s.sequences_checked.fetch_add(1, Ordering::Relaxed);
            return SearchResult {
                time: l.time,
                sequence: sequence.clone(),
            };
        }

        let options = data::get_build_options(&l.has_built);

        let options = if remaining_depth == 1 {
            if options.contains(self.target) {
                BuildSet::of(self.target)
            } else {
                return SearchResult {
                    time: f32::MAX,
                    sequence: Vec::new(),
                };
            }
        } else if options.contains(ConstructionVehicleT1)
            && !l.has_built.contains(ConstructionVehicleT1)
        {
            // force building a T1` constructor (always optimal)
            BuildSet::of(ConstructionVehicleT1)
        } else if options.contains(ConstructionVehicleT2)
            && !l.has_built.contains(ConstructionVehicleT2)
        {
            // force building a T2 constructor (always optimal)
            BuildSet::of(ConstructionVehicleT2)
        } else {
            options
        };

        let mut best = SearchResult {
            time: f32::MAX,
            sequence: Vec::new(),
        };

        for option in options.ids() {
            let new_local = match l.compute_next(option, self.best_time) {
                Some(value) => value,
                None => {
                    s.sequences_skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

            sequence.push(option);
            let candidate = self.search_inner(sequence, remaining_depth - 1, new_local, s);
            sequence.pop();

            if candidate.time < best.time {
                best = candidate;
            }
        }

        if best.time < self.best_time {
            let best_time_u32 = f32::ceil(best.time) as u32;
            s.best_time.store(best_time_u32, Ordering::Relaxed)
        }

        best
    }
}


impl Searcher for BruteForceSearcher {
    fn search(
        &mut self,
        shared_state: &Arc<SharedState>,
        initial_state: LocalState,
    ) -> SearchResult {
        let mut best = SearchResult {
            time: f32::MAX,
            sequence: Vec::new(),
        };

        let mut current_sequence = Vec::new();

        for search_depth in 2..self.max_buildings {
            let candidate = self.search_inner(
                &mut current_sequence,
                search_depth,
                initial_state.clone(),
                Arc::as_ref(&shared_state),
            );

            println!("\nBest {} sequence: {:?}", search_depth, candidate.sequence);

            if candidate.time < best.time {
                best = candidate;
            }
        }

        best
    }
}