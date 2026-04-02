#![allow(dead_code)]

use crate::machine_learning::reinforcement_learning::ReinforcementLearning;
use crate::machine_learning::reward::ResourceGenerationReward;
use crate::search_handler::LocalState;

mod data;
mod build_option;
mod search_handler;
mod brute_force_search;
mod optimization_searcher;
mod searcher;
mod policy;
pub mod machine_learning;
mod random;

fn main() {
    let initial_state = LocalState::initial();

    // let mut searcher = BruteForceSearcher::new(BuildOptionId::AdvancedVehicleLab, 15);
    let mut searcher = ReinforcementLearning::new(10000, 0, 3000.0, Box::from(ResourceGenerationReward));

    let result = search_handler::search(&mut searcher, initial_state);

    println!();
    println!("Best sequence: {:?} in {} seconds", result.sequence.iter().map(|i| i.data()), result.time);
}

