use std::cell::Cell;
use crate::data::BuildOptionId;
use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::policy::Policy;
use crate::random::MyRandom;
use crate::search_handler::LocalState;

const INPUT_SIZE: usize = 17;
const LAYER_0_SIZE: usize = 128;
const LAYER_1_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 32;

struct ReinforcementPolicy {
    first: [[f32; LAYER_0_SIZE]; INPUT_SIZE],
    second: [[f32; LAYER_1_SIZE]; LAYER_0_SIZE],
    third: [[f32; OUTPUT_SIZE]; LAYER_1_SIZE],
    rng: MyRandom
}

impl ReinforcementPolicy {
    pub fn new() -> Self {
        todo!()
    }

    pub fn train(&mut self, num_trajectories: u32, max_time: f32) {
        for i in 0..num_trajectories {
            let mut state = LocalState::initial();
            let mut sequence = Vec::new();
            let mut sum_reward = 0.0;
            let mut probabilities = Vec::new();
            let mut rewards = Vec::new();

            //  while not done:
            //         action_probabilities = policy(state)
            //         action = sample_from(action_probabilities)
            //         state, reward, done = environment_step(action)
            //         probabilities[t] = action_probabilities[action]
            //         rewards[t] = reward
            let mut t = 0;
            loop {
                let output_tensor = self.calculate_outputs(&state);
                let chosen_index = self.choose(output_tensor);
                let next_build = BuildOptionId::from(chosen_index as u8);
                let next_state = state.compute_next(next_build, max_time);

                if next_state.is_none() { break }
                let next_state = next_state.unwrap();

                probabilities[t] = output_tensor[chosen_index];

                let energy_generation_gain = next_state.energy_generation - state.energy_generation;
                let metal_generation_gain = next_state.compute_potential_metal_production() - state.compute_potential_metal_production();
                rewards[t] = energy_generation_gain + metal_generation_gain * 50.0;
                sum_reward += rewards[t];

                state = next_state;
                sequence.push(next_build);
                t += 1;
            }
            let mut num_timesteps = t;

            //     for each timestep t in episode:
            //          returns = sum(rewards[t..])
            let mut returns : Vec<f32> = Vec::new();
            for t in 0..num_timesteps {
                returns[t] = rewards[t..].iter().sum()
            }

            //     for each timestep t in episode:
            //         log_prob = log_probability_of(probabilities[t])
            //         gradient = derivative_of(log_prob with respect to policy parameters)
            //         update_amount = returns[t] * gradient
            //         adjust policy parameters to increase update_amount
            for t in 0..num_timesteps {
                let log_prob = f32::ln(probabilities[t]);
                let update_amount = returns[t] * gradient_of(log_prob);
            }

            todo!()
        }
    }

    fn convert_to_float(has_constructor_t2: bool) -> f32 {
        if has_constructor_t2 { 1.0 } else { 0.0 }
    }

    /// Build input tensor, run inference, normalize output.
    /// The returned tensor is normalized; its elements sum to 1
    fn calculate_outputs(&self, state: &LocalState) -> [f32; OUTPUT_SIZE] {
        let fraction_of_energy_converted = state.energy_generation / state.conversion_drain;
        let fraction_of_storage_generated_per_second = (state.energy_storage as f32) / state.energy_generation;
        let metal_per_build_power = state.metal_generation / state.build_power as f32;
        let energy_per_build_power = state.energy_generation / state.build_power as f32;

        let input_tensor: [f32; INPUT_SIZE] = [
            // raw state
            state.time, // measured in seconds, can go as high as 1_800
            state.metal, // can go as high as 1_000_000
            state.energy, // can go as high as 10_000_000
            state.energy_generation,
            state.metal_generation,
            state.build_power as f32, // increments in steps of 200
            state.conversion_drain,
            state.conversion_result,
            state.energy_storage as f32,
            // relational values
            fraction_of_energy_converted,
            fraction_of_storage_generated_per_second,
            metal_per_build_power,
            energy_per_build_power,
            // build options
            Self::convert_to_float(state.has_built.contains(VehicleLab)),
            Self::convert_to_float(state.has_built.contains(ConstructionVehicleT1)),
            Self::convert_to_float(state.has_built.contains(AdvancedVehicleLab)),
            Self::convert_to_float(state.has_built.contains(ConstructionVehicleT2)),
        ];

        let mut layer_0_tensor = [0.0; LAYER_0_SIZE];

        for x in 0..INPUT_SIZE {
            for y in 0..LAYER_0_SIZE {
                layer_0_tensor[y] = f32::max(self.first[x][y] * input_tensor[x], 0.0);
            }
        }

        let mut layer_1_tensor = [0.0; LAYER_1_SIZE];

        for x in 0..LAYER_0_SIZE {
            for y in 0..LAYER_1_SIZE {
                layer_1_tensor[y] = f32::max(self.second[x][y] * layer_0_tensor[x], 0.0);
            }
        }

        let mut output_tensor = [0.0; OUTPUT_SIZE];

        for x in 0..OUTPUT_SIZE {
            for y in 0..LAYER_1_SIZE {
                output_tensor[x] = f32::max(self.third[x][y] * layer_1_tensor[y], 0.0);
            }
        }

        let mut sum_of_exponentials = 0.0;
        for x in 0..OUTPUT_SIZE {
            output_tensor[x] = f32::exp(output_tensor[x]);
            sum_of_exponentials += output_tensor[x];
        }

        for x in 0..OUTPUT_SIZE {
            output_tensor[x] /= sum_of_exponentials;
        }

        output_tensor
    }

    /// picks an action based on the given probabilities
    fn choose(&self, probabilities: [f32; OUTPUT_SIZE]) -> usize {
        let mut choice = self.rng.next_f32();
        for x in 0..OUTPUT_SIZE {
            if probabilities[x] < choice {
                choice -= probabilities[x];
            } else {
                return x;
            }
        }

        unreachable!("the sum of probabilities should be exactly 1, and choice is [0, 1)")
    }
}

impl Policy for ReinforcementPolicy {
    fn get_next(&self, state: &LocalState, sequence: &Vec<BuildOptionId>) -> BuildOptionId {
        let output_tensor = self.calculate_outputs(state);
        BuildOptionId::from(self.choose(output_tensor) as u8)
    }
}
