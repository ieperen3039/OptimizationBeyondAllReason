use crate::data::BuildOptionId;
use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::policy::Policy;
use crate::random::MyRandom;
use crate::search_handler::LocalState;
use dfdx::nn::{DeviceBuildExt, Module, ZeroGrads};
use dfdx::optim::{Optimizer, OptimizerUpdateError, SgdConfig};
use dfdx::prelude::{CpuError, Linear, ReLU};
use dfdx::tensor::{AsArray, Cpu, Tensor, TensorFrom, Trace};
use dfdx::tensor_ops::{Backward, SelectTo};
use dfdx::{optim::Sgd, shapes::Rank1};

const INPUT_SIZE: usize = 17;
const LAYER_0_SIZE: usize = 128;
const LAYER_1_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 32;

type InputTensor = Tensor<Rank1<INPUT_SIZE>, f32, Cpu>;
type OutputTensor = Tensor<Rank1<OUTPUT_SIZE>, f32, Cpu>;

trait Reward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32;
    fn gamma(&self) -> f32 {
        1.0
    }
}

struct ReinforcementPolicy {
    device: Cpu,
    reward_model: Box<dyn Reward>,
    rng: MyRandom,
}

type Layers = (
    Linear<INPUT_SIZE, LAYER_0_SIZE>,
    ReLU,
    Linear<LAYER_0_SIZE, LAYER_1_SIZE>,
    ReLU,
    Linear<LAYER_1_SIZE, OUTPUT_SIZE>,
);

struct EnergyGenerationReward;
impl Reward for EnergyGenerationReward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32 {
        let energy_generation_gain = after.energy_generation - before.energy_generation;
        let metal_generation_gain = after.compute_potential_metal_production()
            - before.compute_potential_metal_production();
        energy_generation_gain + metal_generation_gain * 50.0
    }
}

struct Step {
    state: LocalState,
    action: usize,
    reward: f32,
}

impl ReinforcementPolicy {
    pub fn new() -> Self {
        let device: Cpu = Default::default();
        Self {
            device,
            reward_model: Box::from(EnergyGenerationReward {}),
            rng: MyRandom::new(0),
        }
    }

    pub fn train(&mut self, num_trajectories: u32, max_time: f32) -> Box<dyn Module<InputTensor, Error=CpuError, Output=OutputTensor>> {
        let mut model = self.device.build_module::<Layers, f32>();

        for _ in 0..num_trajectories {
            let mut state = LocalState::initial();
            let mut steps = Vec::new();

            let mut optimizer = Sgd::new(
                &model,
                SgdConfig {
                    lr: 1e-3,
                    momentum: None,
                    weight_decay: None,
                },
            );

            // create trajectory
            loop {
                // track gradients for the backward pass later
                let input = self.build_input_tensor(&state);
                let logits = model.forward(input);

                // Get probabilities for sampling
                let probabilities = logits.softmax();
                let chosen_index = self.choose(probabilities.array());

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
                let input = self.build_input_tensor(&step.state);
                let logits = model.forward(input.traced(model.alloc_grads()));

                let log_prob = logits.log_softmax().select(self.device.tensor(step.action));

                let loss = log_prob * (-step.reward);
                let gradients = loss.backward();
                let result = optimizer.update(&mut model, &gradients);
                match result {
                    Ok(_) => {}
                    Err(OptimizerUpdateError::DeviceError(CpuError::OutOfMemory)) => {
                        // ideally we toss some data and dump the model to disk
                        panic!("OutOfMemory")
                    }
                    Err(OptimizerUpdateError::DeviceError(CpuError::WrongNumElements)) => {
                        panic!("Not enough elements were provided when creating a tensor")
                    }
                    Err(OptimizerUpdateError::UnusedParams(e)) => panic!("{:?}", e),
                }
            }
        }

        Box::from(model)
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

    fn convert_to_float(has_constructor_t2: bool) -> f32 {
        if has_constructor_t2 { 1.0 } else { 0.0 }
    }

    /// Build input tensor based on state
    fn build_input_tensor(&self, state: &LocalState) -> InputTensor {
        let fraction_of_energy_converted = state.energy_generation / state.conversion_drain;
        let fraction_of_storage_generated_per_second =
            (state.energy_storage as f32) / state.energy_generation;
        let metal_per_build_power = state.metal_generation / state.build_power as f32;
        let energy_per_build_power = state.energy_generation / state.build_power as f32;

        self.device.tensor([
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

impl Policy for ReinforcementPolicy {
    fn get_next(&self, state: &LocalState, sequence: &Vec<BuildOptionId>) -> BuildOptionId {
        let input = self.build_input_tensor(&state);
        let logits = model.forward(input);
        // Get probabilities for sampling
        let probabilities = logits.softmax();
        let chosen_index = self.choose(probabilities.array());
        assert!(chosen_index <= (u8::MAX as usize));
        BuildOptionId::from(chosen_index as u8)
    }
}
