use crate::build_option::BuildOption;

const WIND_TURBINE: BuildOption = BuildOption::new_generator("Wind", 43, 175, 1680, 10, 0);

const SOLAR_COLLECTOR: BuildOption = BuildOption::new_generator("Solar", 150, 0, 2800, 20, 50);

const ADVANCED_SOLAR_COLLECTOR: BuildOption = BuildOption::new_generator("Asol", 370, 4000, 8150, 80, 100);

const ENERGY_CONVERTER: BuildOption = BuildOption::new_converter("Conv", 1, 1250, 2680, 70.0, 1.0);

const ADVANCED_ENERGY_CONVERTER: BuildOption =
    BuildOption::new_converter("Aconv", 370, 21000, 31300, 70.0, 1.0);

const FUSION_REACTOR: BuildOption = BuildOption::new_generator("Fus", 3600, 22000, 59000, 850, 2500);

const ADVANCED_FUSION_REACTOR: BuildOption = BuildOption::new_generator("Afus", 9700, 48000, 329200, 3000, 9000);

const BUILD_TURRET: BuildOption = BuildOption {
    name: "Nano",
    cost_metal: 230,
    cost_energy: 3200,
    cost_bp: 5300,
    make_energy: 0,
    build_power: 200,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    storage_energy: 0,
};

const CONSTRUCTION_VEHICLE_T1: BuildOption = BuildOption {
    name: "Builder_T1",
    cost_metal: 145,
    cost_energy: 2100,
    cost_bp: 4160,
    make_energy: 10,
    build_power: 95,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    storage_energy: 50,
};

const ADVANCED_VEHICLE_PLANT: BuildOption = BuildOption {
    name: "T2",
    cost_metal: 2600,
    cost_energy: 16000,
    cost_bp: 28000,
    make_energy: 0,
    build_power: 0,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    storage_energy: 200,
};

const CONSTRUCTION_VEHICLE_T2: BuildOption = BuildOption {
    name: "Builder_T2",
    cost_metal: 580,
    cost_energy: 7000,
    cost_bp: 28000,
    make_energy: 20,
    build_power: 310,
    conversion_drain: 0.0,
    conversion_result: 0.0,
    storage_energy: 100,
};