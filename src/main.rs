use crate::brute_force_search::BruteForceSearcher;
use crate::data::{BuildOptionId, BuildSet};
use crate::search_handler::LocalState;

mod data;
mod build_option;
mod search_handler;
mod brute_force_search;
mod optimization_searcher;

fn main() {
    let initial_state = LocalState {
        time: 0_f32,
        metal: 1000_f32,
        energy: 1000_f32,
        energy_generation: 3f32,
        metal_generation: 2f32,
        build_power: 300,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 1000,
        has_built: BuildSet::new(),
    };

    let searcher = BruteForceSearcher::new(BuildOptionId::AdvancedVehicleLab, 15);
    let result = search_handler::search(searcher, initial_state);
    
    println!("Best sequence: {:?} in {} seconds", result.sequence.iter().map(|i| i.data()), result.time);
}

