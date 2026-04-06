use crate::data;
use crate::data::BuildOptionId::{ConstructionVehicleT1, ConstructionVehicleT2};
use crate::data::{BuildOptionId, BuildSet};
use crate::search_handler::{LocalState, SearchResult};
use crate::searcher::Searcher;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
pub struct BruteForceSearcher {
    target: BuildOptionId,
    max_buildings: u32,
    best_time: f32,
    state: Arc<BruteForceState>,
}

struct BruteForceState {
    pub best_score: AtomicU32,
    pub sequences_checked: AtomicU32,
    pub sequences_skipped: AtomicU32,
}

impl BruteForceSearcher {
    pub fn new(target: BuildOptionId, max_buildings: u32) -> Self {
        Self {
            target,
            max_buildings,
            best_time: f32::MAX,
            state: Arc::new(BruteForceState {
                best_score: AtomicU32::default(),
                sequences_checked: AtomicU32::default(),
                sequences_skipped: AtomicU32::default(),
            })
        }
    }

    fn search_inner(
        &mut self,
        sequence: &mut Vec<BuildOptionId>,
        remaining_depth: u32,
        l: LocalState,
    ) -> SearchResult {
        if remaining_depth == 0 {
            self.state.sequences_checked.fetch_add(1, Ordering::Relaxed);
            return SearchResult {
                score: l.time,
                sequence: sequence.clone(),
            };
        }

        let options = data::get_build_options(&l.has_built);

        let options = if remaining_depth == 1 {
            if options.contains(self.target) {
                BuildSet::of(self.target)
            } else {
                return SearchResult {
                    score: f32::MAX,
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
            score: f32::MAX,
            sequence: Vec::new(),
        };

        for option in options.ids() {
            let new_local = match l.compute_next(option, self.best_time) {
                Some(value) => value,
                None => {
                    self.state.sequences_skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

            sequence.push(option);
            let candidate = self.search_inner(sequence, remaining_depth - 1, new_local);
            sequence.pop();

            if candidate.score < best.score {
                best = candidate;
            }
        }

        if best.score < self.best_time {
            let best_time_u32 = f32::ceil(best.score) as u32;
            self.state
                .best_score
                .store(best_time_u32, Ordering::Relaxed)
        }

        best
    }
}

impl Searcher for BruteForceSearcher {
    fn search(&mut self, initial_state: LocalState) -> SearchResult {
        let mut best = SearchResult {
            score: f32::MAX,
            sequence: Vec::new(),
        };

        let mut current_sequence = Vec::new();

        for search_depth in 2..self.max_buildings {
            let candidate =
                self.search_inner(&mut current_sequence, search_depth, initial_state.clone());

            println!("\nBest {} sequence: {:?}", search_depth, candidate.sequence);

            if candidate.score < best.score {
                best = candidate;
            }
        }

        best
    }

    fn new_progress_updater(&self) -> Arc<dyn Display + Send + Sync> {
        self.state.clone()
    }
}

impl Display for BruteForceState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let best_score = self.best_score.load(Ordering::Relaxed);
        let checked = self.sequences_checked.load(Ordering::Relaxed);
        let skipped = self.sequences_skipped.load(Ordering::Relaxed);

        write!(f,
               "best time: {}, checked: {}, skipped: {}",
               if best_score == u32::MAX {
                   "n/a".to_string()
               } else {
                   best_score.to_string()
               },
               checked,
               skipped,
        )
    }
}
