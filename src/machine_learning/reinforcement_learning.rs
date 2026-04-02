use crate::data::BuildOptionId;
use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::machine_learning::reinforcement_policy::DeterministicReinforcementPolicy;
use crate::machine_learning::reward::Reward;
use crate::policy::Policy;
use crate::random::MyRandom;
use crate::search_handler::{LocalState, SearchResult, SharedState};
use crate::searcher::Searcher;
use dfdx::nn::{BuildOnDevice, DeviceBuildExt, Module, ResetParams, SaveToNpz, ZeroGrads};
use dfdx::optim::{Optimizer, SgdConfig};
use dfdx::prelude::{Linear, ReLU};
use dfdx::tensor::{Cpu, Tensor, TensorFrom, Trace};
use dfdx::tensor_ops::{Backward, SelectTo};
use dfdx::{optim::Sgd, shapes::Rank1};
use std::ops::Index;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub struct ReinforcementLearning {
    device: Cpu,
    reward_model: Box<dyn Reward>,
    num_trajectories: u32,
    rng: MyRandom,
    pub max_game_time: f32,
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

pub type Model = <Layers as BuildOnDevice<Cpu, f32>>::Built;

struct Step {
    state: LocalState,
    action: usize,
    reward: f32,
}

impl ReinforcementLearning {
    pub fn new(num_trajectories: u32, random_seed: u32, max_game_time: f32, reward_model: Box<dyn Reward>) -> Self {
        let device = Cpu::default();
        Self {
            device,
            reward_model,
            num_trajectories,
            max_game_time,
            rng: MyRandom::new_from_u32(random_seed),
        }
    }

    pub fn get_device(&self) -> Cpu {
        self.device.clone()
    }

    pub fn train(&mut self) -> Model {
        let mut model = self.device.build_module::<Layers, f32>();
        model.reset_params();

        let mut optimizer = Sgd::new(
            &model,
            SgdConfig {
                lr: 1e-3,
                momentum: None,
                weight_decay: None,
            },
        );

        for _ in 0..self.num_trajectories {
            let steps = self.create_trajectory(&mut model, LocalState::initial());
            self.evaluate(&mut model, &mut optimizer, steps);
        }

        model
    }

    fn evaluate(
        &mut self,
        mut model: &mut Model,
        optimizer: &mut Sgd<Model, f32, Cpu>,
        mut steps: Vec<Step>,
    ) {
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

    fn create_trajectory(&mut self, model: &mut Model, initial_state: LocalState) -> Vec<Step> {
        let mut state = initial_state;
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
            let next_state = state.compute_next(next_build, self.max_game_time);

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

        steps
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

impl Searcher for ReinforcementLearning {
    fn search(
        &mut self,
        shared_state: &Arc<SharedState>,
        initial_state: LocalState,
    ) -> SearchResult {
        let mut model = self.device.build_module::<Layers, f32>();
        model.reset_params();

        let mut optimizer = Sgd::new(
            &model,
            SgdConfig {
                lr: 1e-3,
                momentum: None,
                weight_decay: None,
            },
        );

        let mut best_time = f32::MAX;

        for _ in 0..self.num_trajectories {
            let steps = self.create_trajectory(&mut model, initial_state.clone());
            let sequence_time = steps.last().map(|s| s.state.time).unwrap_or(f32::MAX);

            if sequence_time < best_time {
                best_time = sequence_time;
                shared_state.best_time.store(f32::ceil(sequence_time) as u32, Ordering::Relaxed);
            }

            self.evaluate(&mut model, &mut optimizer, steps);
            shared_state
                .sequences_checked
                .fetch_add(1, Ordering::Relaxed);
        }

        let result = model.save("my_model_snapshot.npz");
        if let Err(error) = result {
            eprintln!("{}", error);
        }

        let policy = DeterministicReinforcementPolicy::from_model(model);

        let mut state = initial_state;
        let mut sequence = Vec::new();
        loop {
            let build_choice = policy.get_next(&state, &sequence);
            let next_state = state.compute_next(build_choice, self.max_game_time);
            if next_state.is_none() {
                break;
            }
            sequence.push(build_choice);
            state = next_state.unwrap();
        }

        SearchResult {
            time: state.time,
            sequence,
        }
    }
}
