use crate::data::BuildOptionId;
use crate::machine_learning::reinforcement_learning as rl;
use crate::machine_learning::reinforcement_learning::ReinforcementLearning;
use crate::policy::Policy;
use crate::random::MyRandom;
use crate::search_handler::LocalState;
use dfdx::nn::Module;
use dfdx::prelude::CpuError;
use dfdx::tensor::Cpu;

pub struct ReinforcementPolicy {
    model: Box<dyn Module<rl::InputTensor, Error = CpuError, Output = rl::OutputTensor>>,
    device: Cpu,
    rng: MyRandom,
}

impl ReinforcementPolicy {
    pub fn new(learner: ReinforcementLearning, random_seed: u32) -> ReinforcementPolicy {
        let device = learner.get_device();
        let rng = MyRandom::new_from_u32(random_seed);
        let mut learner = ReinforcementLearning::new(1000, rng.next_u32());
        let model = learner.train(1000.0);
        ReinforcementPolicy { model, device, rng }
    }
}

impl Policy for ReinforcementPolicy {
    fn get_next(&self, state: &LocalState, _sequence: &Vec<BuildOptionId>) -> BuildOptionId {
        let input = ReinforcementLearning::build_input_tensor(state, &self.device);
        let logits = self.model.forward(input);
        // Get probabilities for sampling
        let probabilities = logits.softmax();
        let chosen_index = ReinforcementLearning::select(probabilities, self.rng.next_f32());
        assert!(chosen_index <= (u8::MAX as usize));
        BuildOptionId::from(chosen_index as u8)
    }
}
