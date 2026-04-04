use std::ops::Index;
use crate::data;
use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::data::{BuildOptionId, BuildSet};
use crate::machine_learning::common;
use crate::machine_learning::neat::{InputTensor, NeatNetwork, OutputTensor};
use crate::machine_learning::reward::Reward;
use crate::random::MyRandom;
use crate::search_handler::LocalState;

pub struct NeatTrainer {
    reward_model: Box<dyn Reward>,
    next_innovation_id: usize,
    rng: MyRandom,
    max_game_time: f32,
}

impl NeatTrainer {
    fn evaluate(&mut self, model: &mut NeatNetwork, initial_state: LocalState) -> f32 {
        let mut state = initial_state.clone();

        // create trajectory
        loop {
            // track gradients for the backward pass later
            let input = Self::build_input_tensor(&state);
            let logits = model.forward(&input);

            let build_options = data::get_build_options(&state.has_built);
            let next_build = Self::get_max(logits, build_options);
            let next_state = state.compute_next(next_build, self.max_game_time);

            if next_state.is_none() {
                break;
            }

            state = next_state.unwrap();
        }

        self.reward_model.calculate(&initial_state, &state)
    }

    /// Build input tensor based on state
    pub fn build_input_tensor(state: &LocalState) -> InputTensor {
        let fraction_of_energy_converted = if state.conversion_drain <= 0.0 {
            0.0
        } else {
            state.energy_generation / state.conversion_drain
        };
        let fraction_of_storage_generated_per_second =
            state.energy_generation / (state.energy_storage as f32);
        let metal_per_build_power = state.metal_generation / state.build_power as f32;
        let energy_per_build_power = state.energy_generation / state.build_power as f32;

        [
            // raw state
            state.time,
            state.metal,
            state.energy,
            state.energy_generation,
            state.metal_generation,
            state.build_power as f32,
            state.conversion_drain,
            state.energy_storage as f32,
            // relational values
            fraction_of_energy_converted,
            fraction_of_storage_generated_per_second,
            (metal_per_build_power + 1.0) / 5.0,
            (energy_per_build_power + 1.0) / 5.0,
            // build options
            common::convert_to_float(state.has_built.contains(VehicleLab)),
            common::convert_to_float(state.has_built.contains(ConstructionVehicleT1)),
            common::convert_to_float(state.has_built.contains(AdvancedVehicleLab)),
            common::convert_to_float(state.has_built.contains(ConstructionVehicleT2)),
        ]
    }

    fn get_max(outputs: OutputTensor, allowed_options: BuildSet) -> BuildOptionId {
        let mut max_value = f32::MIN;
        let mut best_id = BuildOptionId::Invalid;
        for id in allowed_options.ids() {
            let value = outputs[id as usize];
            if value > max_value {
                best_id = id;
                max_value = value;
            }
        }
        best_id
    }

    #[allow(dead_code)]
    fn filter_softmax(outputs: OutputTensor, allowed_options: BuildSet) -> OutputTensor {
        /* From dfdx:
         * Let's denote `t - t.max()` expression tm:
         * `(tm - tm.exp().sum().ln()).exp()`
         *
         * Another reduction is the identity of the form `e^(x - y)` = `e^x / e^y`.
         * `tm.exp() / tm.exp().sum().ln().exp()`
         *
         * First we can re-use the `tm.exp()` calculation - lets call it tme
         * `tme / tme.sum().ln().exp()`
         *
         * And finally we know that `t.ln().exp()` is equivalent to `t`. I.e. they are fused
         * `tme / tme.sum()`
         */

        let mut max = f32::MIN;
        for idx in allowed_options.ids().map(|b| b as usize) {
            if outputs[idx] > max {
                max = outputs[idx];
            }
        }
        let mut result = OutputTensor::default();
        let mut sum = 0.0;
        for idx in allowed_options.ids().map(|b| b as usize) {
            let t = outputs[idx];
            let tm = t - max;
            let tme = tm.exp();
            sum += tme;
            result[idx] = tme;
        }

        for out in &mut result {
            *out /= sum
        }

        result
    }

    #[allow(dead_code)]
    pub fn select(distribution: OutputTensor, chosen_fraction: f32) -> usize {
        let mut cumulative = 0.0;
        for i in 0..distribution.len() {
            cumulative += distribution[i];
            if chosen_fraction < cumulative {
                return i;
            }
        }

        unreachable!("probabilities should sum to 1.0");
    }
}
