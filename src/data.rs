use crate::build_option::BuildOption;
use std::mem;
use crate::data;
use crate::data::BuildOptionId::{ConstructionVehicleT1, ConstructionVehicleT2};

pub const NUM_BUILD_OPTIONS : usize = 13;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum BuildOptionId {
    WindTurbine,
    SolarCollector,
    AdvancedSolarCollector,
    EnergyConverter,
    AdvancedEnergyConverter,
    FusionReactor,
    AdvancedFusionReactor,
    BuildTurret,
    VehicleLab,
    ConstructionVehicleT1,
    AdvancedVehicleLab,
    ConstructionVehicleT2,
    Juggernaut,
}

impl From<u8> for BuildOptionId {
    fn from(value: u8) -> Self {
        assert!((value as usize) < NUM_BUILD_OPTIONS);
        unsafe { mem::transmute(value) }
    }
}

impl BuildOptionId {
    pub fn data(self) -> &'static BuildOption {
        &BUILD_OPTIONS[self as usize]
    }
}

pub const BUILD_OPTIONS: [BuildOption; NUM_BUILD_OPTIONS] = [
    BuildOption {
        name: "Wind",
        cost_metal: 43,
        cost_energy: 175,
        cost_bp: 1680,
        energy_generation: 10,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 0,
    },
    BuildOption {
        name: "Solar",
        cost_metal: 150,
        cost_energy: 0,
        cost_bp: 2800,
        energy_generation: 20,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 50,
    },
    BuildOption {
        name: "Asol",
        cost_metal: 370,
        cost_energy: 4000,
        cost_bp: 8150,
        energy_generation: 80,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 100,
    },
    BuildOption {
        name: "Conv",
        cost_metal: 1,
        cost_energy: 1250,
        cost_bp: 2680,
        energy_generation: 0,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 70.0,
        conversion_result: 1.0,
        energy_storage: 0,
    },
    BuildOption {
        name: "Aconv",
        cost_metal: 370,
        cost_energy: 21000,
        cost_bp: 31300,
        energy_generation: 0,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 70.0,
        conversion_result: 1.0,
        energy_storage: 0,
    },
    BuildOption {
        name: "Fus",
        cost_metal: 3600,
        cost_energy: 22000,
        cost_bp: 59000,
        energy_generation: 850,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 2500,
    },
    BuildOption {
        name: "Afus",
        cost_metal: 9700,
        cost_energy: 48000,
        cost_bp: 329200,
        energy_generation: 3000,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 9000,
    },
    BuildOption {
        name: "Nano",
        cost_metal: 230,
        cost_energy: 3200,
        cost_bp: 5300,
        energy_generation: 0,
        metal_generation: 0,
        build_power: 200,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 0,
    },
    BuildOption {
        name: "T1",
        cost_metal: 570,
        cost_energy: 1550,
        cost_bp: 5650,
        energy_generation: 0,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 100,
    },
    BuildOption {
        name: "Builder_T1",
        cost_metal: 145,
        cost_energy: 2100,
        cost_bp: 4160,
        energy_generation: 10,
        metal_generation: 0,
        build_power: 95,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 50,
    },
    BuildOption {
        name: "T2",
        cost_metal: 2600,
        cost_energy: 16000,
        cost_bp: 28000,
        energy_generation: 0,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 200,
    },
    BuildOption {
        name: "Builder_T2",
        cost_metal: 580,
        cost_energy: 7000,
        cost_bp: 28000,
        energy_generation: 20,
        metal_generation: 0,
        build_power: 310,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 100,
    },
    BuildOption {
        name: "Jugg",
        cost_metal: 29000,
        cost_energy: 615000,
        cost_bp: 730000,
        energy_generation: 0,
        metal_generation: 0,
        build_power: 0,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 0,
    },
];

#[derive(Clone)]
pub struct BuildSet {
    bit_mask: u32,
}

impl BuildSet {
    pub const fn new() -> Self {
        Self { bit_mask: 0 }
    }
    pub const fn of(element: BuildOptionId) -> Self {
        Self { bit_mask: (1 << element as u32) }
    }
    pub const fn contains(&self, building: BuildOptionId) -> bool {
        (self.bit_mask & (1 << building as u32)) != 0
    }
    pub const fn contains_all(&self, other: BuildSet) -> bool {
        (self.bit_mask & other.bit_mask) != 0
    }
    #[allow(unused_parens)]
    pub const fn add(&mut self, building: BuildOptionId) {
        self.bit_mask |= (1 << building as u32);
    }
    pub const fn add_all(&mut self, other: BuildSet) {
        self.bit_mask |= other.bit_mask;
    }
    pub const fn remove(&mut self, building: BuildOptionId) {
        self.bit_mask &= !(1 << building as u32);
    }
    pub const fn remove_all(&mut self, building: BuildSet) {
        self.bit_mask &= !building.bit_mask;
    }
    pub const fn with(self, building: BuildOptionId) -> Self {
        Self {
            bit_mask: self.bit_mask | (1 << building as u32),
        }
    }
    pub fn ids(&self) -> impl Iterator<Item = BuildOptionId> {
        (0_u8..32_u8)
            .filter(|i| self.bit_mask & (1 << i) != 0)
            .map(u8::into)
    }
    pub fn iter(&self) -> impl Iterator<Item = &BuildOption> {
        (0_u8..32_u8)
            .filter(|i| self.bit_mask & (1 << i) != 0)
            .map(|i| &BUILD_OPTIONS[i as usize])
    }
    pub fn len(&self) -> usize {
        self.bit_mask.count_ones() as usize
    }
}

pub const BUILD_OPTIONS_CON: BuildSet = BuildSet::new()
    .with(BuildOptionId::WindTurbine)
    .with(BuildOptionId::SolarCollector)
    .with(BuildOptionId::EnergyConverter)
    .with(BuildOptionId::VehicleLab);
pub const BUILD_OPTIONS_T1: BuildSet = BuildSet::new()
    .with(BuildOptionId::WindTurbine)
    .with(BuildOptionId::SolarCollector)
    .with(BuildOptionId::AdvancedSolarCollector)
    .with(BuildOptionId::EnergyConverter)
    .with(BuildOptionId::BuildTurret)
    .with(BuildOptionId::AdvancedVehicleLab);
pub const BUILD_OPTIONS_T2: BuildSet = BuildSet::new()
    .with(BuildOptionId::AdvancedSolarCollector)
    .with(BuildOptionId::AdvancedEnergyConverter)
    .with(BuildOptionId::FusionReactor)
    .with(BuildOptionId::AdvancedFusionReactor);


pub fn get_build_options(build_set: &BuildSet) -> BuildSet {
    if build_set.contains(ConstructionVehicleT2) {
        data::BUILD_OPTIONS_T2
    } else if build_set.contains(ConstructionVehicleT1) {
        data::BUILD_OPTIONS_T1
    } else {
        data::BUILD_OPTIONS_CON
    }
}