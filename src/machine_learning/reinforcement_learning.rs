use crate::data;
use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::data::{BuildOptionId, BuildSet};
use crate::machine_learning::reinforcement_policy::DeterministicReinforcementPolicy;
use crate::machine_learning::reward::Reward;
use crate::policy::Policy;
use crate::random::MyRandom;
use crate::search_handler::{LocalState, SearchResult, SharedState};
use crate::searcher::Searcher;
use dfdx::nn::{
    BuildOnDevice, DeviceBuildExt, LoadFromNpz, Module, ResetParams, SaveToNpz, ZeroGrads,
};
use dfdx::optim::{Optimizer, SgdConfig};
use dfdx::prelude::{Linear, ReLU};
use dfdx::tensor::{Cpu, Tensor, TensorFrom, Trace};
use dfdx::tensor_ops::{Backward, ChooseFrom, SelectTo};
use dfdx::{optim::Sgd, shapes::Rank1};
use simple_error::SimpleError;
use std::ops::Index;
use std::path::Path;
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
    pub fn new(
        num_trajectories: u32,
        random_seed: u32,
        max_game_time: f32,
        reward_model: Box<dyn Reward>,
    ) -> Self {
        let device = Cpu::default();
        Self {
            device,
            reward_model,
            num_trajectories,
            max_game_time,
            rng: MyRandom::new_from_u32(random_seed),
        }
    }

    pub fn load(device: &Cpu, path: &Path) -> Result<Model, SimpleError> {
        let mut model = device.build_module::<Layers, f32>();
        model
            .load(path)
            .map_err(|e| SimpleError::new(e.to_string()))?;
        Ok(model)
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

        let zero_logits: OutputTensor = self.device.tensor([-1.0e9; 32]);

        for step in steps {
            let input = Self::build_input_tensor(&step.state, &self.device);
            let logits = model.forward(input.traced(model.alloc_grads()));

            let can_build_tensor =
                self.buildset_to_tensor(data::get_build_options(&step.state.has_built));
            let logits = can_build_tensor.choose(logits, zero_logits.clone());

            let log_prob = logits.log_softmax().select(self.device.tensor(step.action));

            let loss = log_prob * (-step.reward);
            let gradients = loss.backward();
            optimizer.update(&mut model, &gradients).unwrap();
        }
    }

    fn create_trajectory(&mut self, model: &mut Model, initial_state: LocalState) -> Vec<Step> {
        let mut state = initial_state;
        let mut steps = Vec::new();
        let zero_logits: OutputTensor = self.device.tensor([-1.0e9; 32]);

        // create trajectory
        loop {
            // track gradients for the backward pass later
            let input = Self::build_input_tensor(&state, &self.device);
            let logits = model.forward(input);

            let can_build_tensor =
                self.buildset_to_tensor(data::get_build_options(&state.has_built));
            let logits = can_build_tensor.choose(logits, zero_logits.clone());

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
            assert!(reward >= 0.0);
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

    fn buildset_to_tensor(&self, build_set: BuildSet) -> Tensor<Rank1<32>, bool, Cpu> {
        let mut array = [false; 32];
        for idx in build_set.ids() {
            array[idx as usize] = true;
        }
        self.device.tensor(array)
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

        let mut best_score = 0.0;

        for _ in 0..self.num_trajectories {
            let steps = self.create_trajectory(&mut model, initial_state.clone());

            if steps.is_empty() {
                shared_state
                    .sequences_skipped
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let score = self
                .reward_model
                .calculate(&initial_state, &steps.last().unwrap().state);

            if score > best_score {
                best_score = score;
                shared_state
                    .best_score
                    .store(f32::ceil(score) as u32, Ordering::Relaxed);
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
        let mut built = [0; data::NUM_BUILD_OPTIONS];
        loop {
            let build_choice = policy.get_next(&state, &built);
            let next_state = state.compute_next(build_choice, self.max_game_time);
            if next_state.is_none() {
                break;
            }
            sequence.push(build_choice);
            built[build_choice as usize] += 1;
            state = next_state.unwrap();
        }

        SearchResult {
            time: state.time,
            sequence,
        }
    }
}
