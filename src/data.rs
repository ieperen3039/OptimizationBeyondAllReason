use crate::build_option::BuildOption;

pub const BUILD_OPTIONS_CON: &[BuildOption] = &[
    WIND_TURBINE,
    SOLAR_COLLECTOR,
    ENERGY_CONVERTER,
    VEHICLE_LAB,
];
pub const BUILD_OPTIONS_T1: &[BuildOption] = &[
    WIND_TURBINE,
    SOLAR_COLLECTOR,
    ADVANCED_SOLAR_COLLECTOR,
    ENERGY_CONVERTER,
    BUILD_TURRET,
    CONSTRUCTION_VEHICLE_T1,
    ADVANCED_VEHICLE_LAB,
];
pub const BUILD_OPTIONS_T2: &[BuildOption] = &[
    // no options to build WIND_TURBINE and SOLAR_COLLECTOR
    ADVANCED_SOLAR_COLLECTOR,
    ENERGY_CONVERTER,
    ADVANCED_ENERGY_CONVERTER,
    FUSION_REACTOR,
    ADVANCED_FUSION_REACTOR,
    BUILD_TURRET,
    CONSTRUCTION_VEHICLE_T1,
    CONSTRUCTION_VEHICLE_T2,
];

pub const WIND_TURBINE: BuildOption = BuildOption::new_generator("Wind", 43, 175, 1680, 10, 0);

pub const SOLAR_COLLECTOR: BuildOption = BuildOption::new_generator("Solar", 150, 0, 2800, 20, 50);

pub const ADVANCED_SOLAR_COLLECTOR: BuildOption =
    BuildOption::new_generator("Asol", 370, 4000, 8150, 80, 100);

pub const ENERGY_CONVERTER: BuildOption =
    BuildOption::new_converter("Conv", 1, 1250, 2680, 70.0, 1.0);

pub const ADVANCED_ENERGY_CONVERTER: BuildOption =
    BuildOption::new_converter("Aconv", 370, 21000, 31300, 70.0, 1.0);

pub const FUSION_REACTOR: BuildOption =
    BuildOption::new_generator("Fus", 3600, 22000, 59000, 850, 2500);

pub const ADVANCED_FUSION_REACTOR: BuildOption =
    BuildOption::new_generator("Afus", 9700, 48000, 329200, 3000, 9000);

pub const BUILD_TURRET: BuildOption = BuildOption {
    name: "Nano",
    cost_metal: 230,
    cost_energy: 3200,
    cost_bp: 5300,
    energy_generation: 0,
    build_power: 200,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    energy_storage: 0,
};

pub const VEHICLE_LAB: BuildOption = BuildOption {
    name: "T1",
    cost_metal: 570,
    cost_energy: 1550,
    cost_bp: 5650,
    energy_generation: 0,
    build_power: 0,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    energy_storage: 100,
};

pub const CONSTRUCTION_VEHICLE_T1: BuildOption = BuildOption {
    name: "Builder_T1",
    cost_metal: 145,
    cost_energy: 2100,
    cost_bp: 4160,
    energy_generation: 10,
    build_power: 95,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    energy_storage: 50,
};

pub const ADVANCED_VEHICLE_LAB: BuildOption = BuildOption {
    name: "T2",
    cost_metal: 2600,
    cost_energy: 16000,
    cost_bp: 28000,
    energy_generation: 0,
    build_power: 0,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    energy_storage: 200,
};

pub const CONSTRUCTION_VEHICLE_T2: BuildOption = BuildOption {
    name: "Builder_T2",
    cost_metal: 580,
    cost_energy: 7000,
    cost_bp: 28000,
    energy_generation: 20,
    build_power: 310,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    energy_storage: 100,
};

pub const JUGGERNAUT: BuildOption = BuildOption {
    name: "Jugg",
    cost_metal: 29000,
    cost_energy: 615000,
    cost_bp: 730000,
    energy_generation: 0,
    build_power: 0,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    energy_storage: 0,
};
