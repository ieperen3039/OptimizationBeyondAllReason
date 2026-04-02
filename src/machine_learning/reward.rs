use crate::search_handler::LocalState;

pub trait Reward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32;
    fn gamma(&self) -> f32 {
        1.0
    }
}
pub struct ResourceGenerationReward;
impl Reward for ResourceGenerationReward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32 {
        let energy_generation_gain = after.energy_generation - before.energy_generation;
        let metal_generation_gain = after.compute_potential_metal_production()
            - before.compute_potential_metal_production();
        energy_generation_gain + metal_generation_gain * 50.0
    }
}