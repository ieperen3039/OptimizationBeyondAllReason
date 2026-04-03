use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::search_handler::LocalState;

pub trait Reward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32;

    fn future_reward_gamma(&self) -> f32 {
        1.0
    }
}

pub struct CompoundReward;
impl Reward for CompoundReward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32 {
        let energy_generation_gain = after.energy_generation - before.energy_generation;
        let metal_generation_gain = after.compute_potential_metal_production()
            - before.compute_potential_metal_production();

        let tier_factor = if after.has_built.contains(ConstructionVehicleT1) {
            2.0
        } else if after.has_built.contains(VehicleLab) {
            4.0
        } else if after.has_built.contains(ConstructionVehicleT2) {
            8.0
        } else if after.has_built.contains(AdvancedVehicleLab) {
            16.0
        } else {
            0.0
        };

        (energy_generation_gain + metal_generation_gain * 80.0) * tier_factor
    }
}

pub struct ResourceGenerationReward;
impl Reward for ResourceGenerationReward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32 {
        let energy_generation_gain = after.energy_generation - before.energy_generation;
        let metal_generation_gain = after.compute_potential_metal_production()
            - before.compute_potential_metal_production();
        energy_generation_gain + metal_generation_gain * 80.0
    }
}

pub struct TierReward;
impl Reward for TierReward {
    fn calculate(&self, before: &LocalState, after: &LocalState) -> f32 {
        if after.has_built.contains(ConstructionVehicleT1) {
            1.0
        } else if after.has_built.contains(VehicleLab) {
            2.0
        } else if after.has_built.contains(ConstructionVehicleT2) {
            3.0
        } else if after.has_built.contains(AdvancedVehicleLab) {
            4.0
        } else {
            0.0
        }
    }
    fn future_reward_gamma(&self) -> f32 {
        0.0
    }
}
