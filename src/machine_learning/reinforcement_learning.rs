use crate::data::BuildOptionId;
use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::machine_learning::reward::{EnergyGenerationReward, Reward};
use crate::random::MyRandom;
use crate::search_handler::LocalState;
use dfdx::nn::{DeviceBuildExt, Module, ZeroGrads};
use dfdx::optim::{Optimizer, SgdConfig};
use dfdx::prelude::{CpuError, Linear, ReLU};
use dfdx::tensor::{Cpu, Tensor, TensorFrom, Trace};
use dfdx::tensor_ops::{Backward, SelectTo};
use dfdx::{optim::Sgd, shapes::Rank1};
use std::ops::Index;

pub struct ReinforcementLearning {
    device: Cpu,
    reward_model: Box<dyn Reward>,
    num_trajectories: u32,
    rng: MyRandom,
}

const INPUT_SIZE: usize = 17;
const LAYER_0_SIZE: usize = 128;
const LAYER_1_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 32;

pub type InputTensor = Tensor<Rank1<INPUT_SIZE>, f32, Cpu>;
pub type OutputTensor = Tensor<Rank1<OUTPUT_SIZE>, f32, Cpu>;

type Layers = (
    Linear<INPUT_SIZE, LAYER_0_SIZE>,
    ReLU,
    Linear<LAYER_0_SIZE, LAYER_1_SIZE>,
    ReLU,
    Linear<LAYER_1_SIZE, OUTPUT_SIZE>,
);

struct Step {
    state: LocalState,
    action: usize,
    reward: f32,
}


impl ReinforcementLearning {
    pub fn new(num_trajectories: u32, random_seed: u32) -> Self {
        let device = Cpu::default();
        Self {
            device,
            reward_model: Box::from(EnergyGenerationReward {}),
            num_trajectories,
            rng: MyRandom::new_from_u32(random_seed),
        }
    }

    pub fn get_device(&self) -> Cpu {
        self.device.clone()
    }

    pub fn train(
        &mut self,
        max_time: f32,
    ) -> Box<dyn Module<InputTensor, Error = CpuError, Output = OutputTensor>> {
        let mut model = self.device.build_module::<Layers, f32>();

        let mut optimizer = Sgd::new(
            &model,
            SgdConfig {
                lr: 1e-3,
                momentum: None,
                weight_decay: None,
            },
        );

        for _ in 0..self.num_trajectories {
            let mut state = LocalState::initial();
            let mut steps = Vec::new();

            // create trajectory
            loop {
                // track gradients for the backward pass later
                let input = Self::build_input_tensor(&state, &self.device);
                let logits = model.forward(input);

                // Get probabilities for sampling
                let probabilities = logits.softmax();
                let chosen_index = Self::select(probabilities, self.rng.next_f32());

                let next_build = BuildOptionId::from(chosen_index as u8);
                let next_state = state.compute_next(next_build, max_time);

                if next_state.is_none() {
                    break;
                }
                let next_state = next_state.unwrap();

                let reward = self.reward_model.calculate(&state, &next_state);
                steps.push(Step {
                    state,
                    action: chosen_index,
                    reward,
                });

                state = next_state;
            }

            // convert rewards into returns
            let mut running_return = 0.0;
            let reward_gamma = self.reward_model.gamma();
            for step in steps.iter_mut().rev() {
                running_return = step.reward + (reward_gamma * running_return);
                step.reward = running_return;
            }

            for step in steps {
                let input = Self::build_input_tensor(&step.state, &self.device);
                let logits = model.forward(input.traced(model.alloc_grads()));

                let log_prob = logits.log_softmax().select(self.device.tensor(step.action));

                let loss = log_prob * (-step.reward);
                let gradients = loss.backward();
                optimizer.update(&mut model, &gradients).unwrap();
            }
        }

        Box::from(model)
    }

    pub fn select(distribution: OutputTensor, chosen_fraction: f32) -> usize {
        let mut cumulative = 0.0;
        for i in 0..OUTPUT_SIZE {
            cumulative += distribution.index([i]);
            if chosen_fraction < cumulative {
                return i;
            }
        }

        unreachable!("probabilities should sum to 1.0");
    }

    fn convert_to_float(has_constructor_t2: bool) -> f32 {
        if has_constructor_t2 { 1.0 } else { 0.0 }
    }

    /// Build input tensor based on state
    pub fn build_input_tensor(state: &LocalState, device: &Cpu) -> InputTensor {
        let fraction_of_energy_converted = state.energy_generation / state.conversion_drain;
        let fraction_of_storage_generated_per_second =
            (state.energy_storage as f32) / state.energy_generation;
        let metal_per_build_power = state.metal_generation / state.build_power as f32;
        let energy_per_build_power = state.energy_generation / state.build_power as f32;

        device.tensor([
            // raw state
            state.time,   // measured in seconds, can go as high as 1_800
            state.metal,  // can go as high as 1_000_000
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
        ])
    }
}
