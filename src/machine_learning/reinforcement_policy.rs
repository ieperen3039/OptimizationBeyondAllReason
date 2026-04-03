use crate::data;
use crate::data::BuildOptionId;
use crate::machine_learning::reinforcement_learning as rl;
use crate::machine_learning::reinforcement_learning::ReinforcementLearning;
use crate::machine_learning::reward::ResourceGenerationReward;
use crate::policy::Policy;
use crate::random::MyRandom;
use crate::search_handler::LocalState;
use dfdx::nn::{LoadFromNpz, Module};
use dfdx::tensor::{AsArray, Cpu};
use simple_error::SimpleError;
use std::path::Path;

pub struct DeterministicReinforcementPolicy {
    model: rl::Model,
    device: Cpu,
}

impl DeterministicReinforcementPolicy {
    pub fn new(learner: ReinforcementLearning, random_seed: u32) -> Self {
        let device = learner.get_device();
        let rng = MyRandom::new_from_u32(random_seed);
        let mut learner = ReinforcementLearning::new(
            1000,
            rng.next_u32(),
            3000.0,
            Box::from(ResourceGenerationReward),
        );
        let model = learner.train();
        DeterministicReinforcementPolicy { model, device }
    }

    pub fn from_model(model: rl::Model) -> Self {
        Self {
            model,
            device: Cpu::default(),
        }
    }

    pub fn from_file(path: &Path) -> Result<Self, SimpleError> {
        let device = Cpu::default();

        Ok(Self {
            model: ReinforcementLearning::load(&device, path)?,
            device,
        })
    }
}

impl Policy for DeterministicReinforcementPolicy {
    fn get_next(
        &self,
        state: &LocalState,
        _built: &[usize; data::NUM_BUILD_OPTIONS],
    ) -> BuildOptionId {
        let input = ReinforcementLearning::build_input_tensor(state, &self.device);
        let logits = self.model.forward(input);
        // Get probabilities for sampling
        let probabilities = logits.softmax().array();
        let (index, _value) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, v1), (_, v2)| f32::total_cmp(v1, v2))
            .unwrap();
        println!("Picking {index} from {probabilities:?}");
        assert!(index <= (u8::MAX as usize));

        BuildOptionId::from(index as u8)
    }
}
