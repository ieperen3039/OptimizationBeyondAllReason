#![allow(dead_code)]

use crate::machine_learning::neat_trainer::{NeatTrainer, NeatTrainerConfig};
use crate::machine_learning::reinforcement_learning::ReinforcementLearning;
use crate::machine_learning::reward::{MetalGenerationReward, ResourceGenerationReward, TierReward};
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
    let initial_state = LocalState {
        time: 0.0,
        metal: 100_f32,
        energy: 100_f32,
        energy_generation: 1f32,
        metal_generation: 1f32,
        build_power: 300,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 1000,
        has_built: data::BuildSet::new(),
    };

    // let mut searcher = BruteForceSearcher::new(BuildOptionId::AdvancedVehicleLab, 15);
    // let mut searcher = ReinforcementLearning::new(2000, 0x3039, 30.0 * 600.0, Box::from(MetalGenerationReward));
    let config = NeatTrainerConfig {
        population_size: 100,
        num_generations: 100,
        reward_model: Box::from(ResourceGenerationReward),
        crossover_probability: 0.9,
        add_connection_probability: 0.02,
        add_node_probability: 0.01,
    };
    let mut searcher = NeatTrainer::new_with_config(config, 0x3039, 10.0 * 600.0);

    let result = search_handler::search(&mut searcher, initial_state);

    println!();
    println!("Best sequence: {:?} with a score of {}", result.sequence.iter().map(|i| i.data()), result.score);
}

