use crate::data;
use crate::data::BuildOptionId::*;
use crate::data::{BuildOptionId, BuildSet};
use crate::machine_learning::common;
use crate::machine_learning::reinforcement_policy::DeterministicReinforcementPolicy;
use crate::machine_learning::reward::Reward;
use crate::policy::Policy;
use crate::random::MyRandom;
use crate::search_handler::{LocalState, SearchResult};
use crate::searcher::Searcher;
use dfdx::nn::{
    BuildOnDevice, DeviceBuildExt, LoadFromNpz, Module, ResetParams, SaveToNpz, ZeroGrads,
};
use dfdx::optim::{Optimizer, SgdConfig, WeightDecay};
use dfdx::prelude::{Linear, Sigmoid};
use dfdx::tensor::{Cpu, Tensor, TensorFrom, Trace};
use dfdx::tensor_ops::{Backward, ChooseFrom, SelectTo};
use dfdx::{optim::Sgd, shapes::Rank1};
use simple_error::SimpleError;
use std::fmt::{Display, Formatter};
use std::ops::Index;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub struct ReinforcementLearning {
    device: Cpu,
    reward_model: Box<dyn Reward>,
    num_trajectories: u32,
    rng: MyRandom,
    max_game_time: f32,
    state: Arc<ReinforcementLearningState>,
}

struct ReinforcementLearningState {
    pub last_score: AtomicU32,
    pub current_trajectory: AtomicU32,
}

const INPUT_SIZE: usize = 16;
const LAYER_0_SIZE: usize = 32;
const LAYER_1_SIZE: usize = 32;
const OUTPUT_SIZE: usize = data::NUM_BUILD_OPTIONS;

pub type InputTensor = Tensor<Rank1<INPUT_SIZE>, f32, Cpu>;
pub type OutputTensor = Tensor<Rank1<OUTPUT_SIZE>, f32, Cpu>;

type Layers = (
    Linear<INPUT_SIZE, LAYER_0_SIZE>,
    Sigmoid,
    Linear<LAYER_0_SIZE, OUTPUT_SIZE>,
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
            state: Arc::new(ReinforcementLearningState {
                last_score: AtomicU32::default(),
                current_trajectory: AtomicU32::default(),
            }),
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
                lr: 1e-5,
                momentum: None,
                weight_decay: Some(WeightDecay::Decoupled(1e-3)),
            },
        );

        for t in 0..self.num_trajectories {
            self.state.current_trajectory.store(t, Ordering::Relaxed);

            let steps = self.create_trajectory(&mut model, LocalState::initial());
            if steps.is_empty() {
                continue;
            }

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
        let reward_gamma = self.reward_model.future_reward_gamma();
        for step in steps.iter_mut().rev() {
            running_return = step.reward + (reward_gamma * running_return);
            step.reward = running_return;
        }

        self.state
            .last_score
            .store(f32::floor(running_return) as u32, Ordering::Relaxed);

        let zero_logits: OutputTensor = self.device.tensor([-1.0e9; OUTPUT_SIZE]);
        let mut gradients = model.alloc_grads();

        for step in steps {
            let input = Self::build_input_tensor(&step.state, &self.device);
            let logits = model.forward(input.traced(gradients));

            let can_build_tensor =
                self.buildset_to_tensor(data::get_build_options(&step.state.has_built));
            let logits = can_build_tensor.choose(logits, zero_logits.clone());

            let log_prob = logits.log_softmax().select(self.device.tensor(step.action));

            let loss = log_prob * (-step.reward);
            gradients = loss.backward();
        }
        optimizer.update(&mut model, &gradients).unwrap();
    }

    fn create_trajectory(&mut self, model: &mut Model, initial_state: LocalState) -> Vec<Step> {
        let mut state = initial_state;
        let mut steps = Vec::new();
        let zero_logits: OutputTensor = self.device.tensor([-1.0e9; OUTPUT_SIZE]);

        // create trajectory
        loop {
            // track gradients for the backward pass later
            let input = Self::build_input_tensor(&state, &self.device);
            let logits = model.forward(input);

            let build_options = data::get_build_options(&state.has_built);
            let can_build_tensor = self.buildset_to_tensor(build_options.clone());
            let logits = can_build_tensor.choose(logits, zero_logits.clone());

            // Get probabilities for sampling
            let probabilities = logits.softmax();
            let chosen_index = Self::select(probabilities, self.rng.next_f32());

            let next_build = BuildOptionId::from(chosen_index as u8);
            if !build_options.contains(next_build) {
                // disqualify this trajectory; should rarely happen
                return Vec::new();
            }

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

    /// Build input tensor based on state
    pub fn build_input_tensor(state: &LocalState, device: &Cpu) -> InputTensor {
        let fraction_of_energy_converted = if state.conversion_drain <= 0.0 {
            0.0
        } else {
            state.energy_generation / state.conversion_drain
        };
        let fraction_of_storage_generated_per_second =
            state.energy_generation / (state.energy_storage as f32);
        let metal_per_build_power = state.metal_generation / state.build_power as f32;
        let energy_per_build_power = state.energy_generation / state.build_power as f32;

        device.tensor([
            // raw state
            f32::ln(state.metal + 1.0) / 14.0,
            f32::ln(state.energy + 1.0) / 15.0,
            f32::ln(state.energy_generation + 1.0) / 20.0,
            f32::ln(state.metal_generation + 1.0),
            state.build_power as f32 / 20000.0, // increments in steps of 200
            f32::ln(state.conversion_drain + 1.0) / 20.0,
            f32::ln(state.energy_storage as f32 + 1.0) / 15.0,
            // relational values
            fraction_of_energy_converted,
            fraction_of_storage_generated_per_second,
            f32::ln(metal_per_build_power + 1.0) / 5.0,
            f32::ln(energy_per_build_power + 1.0) / 5.0,
            1.0, // dummy
            // build options
            common::convert_to_float(state.has_built.contains(VehicleLab)),
            common::convert_to_float(state.has_built.contains(ConstructionVehicleT1)),
            common::convert_to_float(state.has_built.contains(AdvancedVehicleLab)),
            common::convert_to_float(state.has_built.contains(ConstructionVehicleT2)),
        ])
    }

    fn buildset_to_tensor(&self, build_set: BuildSet) -> Tensor<Rank1<OUTPUT_SIZE>, bool, Cpu> {
        let mut array = [false; OUTPUT_SIZE];
        for idx in build_set.ids() {
            if idx as usize >= OUTPUT_SIZE {
                break;
            }
            array[idx as usize] = true;
        }
        self.device.tensor(array)
    }
}

impl Searcher for ReinforcementLearning {
    fn search(&mut self, initial_state: LocalState) -> SearchResult {
        let model = self.train();

        let result = model.save("my_model_snapshot.npz");
        if let Err(error) = result {
            eprintln!("Could not save model: {}", error);
        }

        let policy = DeterministicReinforcementPolicy::from_model(model);

        let mut state = initial_state.clone();
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

        let score = self.reward_model.calculate(&initial_state, &state);

        SearchResult { score, sequence }
    }

    fn new_progress_updater(&self) -> Arc<dyn Display + Send + Sync> {
        self.state.clone()
    }
}

impl Display for ReinforcementLearningState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let last_score = self.last_score.load(Ordering::Relaxed);
        let checked = self.current_trajectory.load(Ordering::Relaxed);

        write!(f, "checked: {}, last_score: {}", checked, last_score)
    }
}
