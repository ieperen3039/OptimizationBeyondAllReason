use crate::build_option::BuildOption;
use crate::data;

pub const BUILD_OPTIONS_CON: &[BuildOption] = &[
    data::WIND_TURBINE,
    data::SOLAR_COLLECTOR,
    data::ENERGY_CONVERTER,
    data::VEHICLE_PLANT,
];
pub const BUILD_OPTIONS_T1: &[BuildOption] = &[
    data::WIND_TURBINE,
    data::SOLAR_COLLECTOR,
    data::ADVANCED_SOLAR_COLLECTOR,
    data::ENERGY_CONVERTER,
    data::BUILD_TURRET,
    data::CONSTRUCTION_VEHICLE_T1,
    data::ADVANCED_VEHICLE_PLANT,
];
pub const BUILD_OPTIONS_T2: &[BuildOption] = &[
    // no options to build WIND_TURBINE and SOLAR_COLLECTOR
    data::ADVANCED_SOLAR_COLLECTOR,
    data::ENERGY_CONVERTER,
    data::ADVANCED_ENERGY_CONVERTER,
    data::FUSION_REACTOR,
    data::ADVANCED_FUSION_REACTOR,
    data::BUILD_TURRET,
    data::CONSTRUCTION_VEHICLE_T1,
    data::CONSTRUCTION_VEHICLE_T2,
];

#[derive(Clone)]
struct LocalState {
    pub time: u32,
    pub metal: u32,
    pub energy: u32,
    pub make_energy: u32,
    pub build_power: u32,
    pub conversion_drain: f32,
    pub conversion_result: f32,
    pub storage_energy: u32,
    pub has_t1_con: bool,
    pub has_t2: bool,
    pub has_t2_con: bool,
}

struct GlobalState {}

pub struct SearchResult {
    pub time: u32,
    pub sequence: Vec<&'static BuildOption>,
}

pub fn search() -> SearchResult {
    let mut sequence = Vec::new();
    let mut global_state = GlobalState {};
    let initial_state = LocalState {
        time: 0,
        metal: 1000,
        energy: 1000,
        make_energy: 3,
        build_power: 300,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        storage_energy: 0,
        has_t1_con: false,
        has_t2: false,
        has_t2_con: false,
    };

    let mut best = SearchResult {
        time: u32::MAX,
        sequence: Vec::new(),
    };

    for i in 10..50 {
        let candidate = search_inner(&mut sequence, i, initial_state.clone(), &mut global_state);

        if candidate.time < best.time {
            best = candidate;
        }
    }

    best
}

fn search_inner(
    sequence: &mut Vec<&'static BuildOption>,
    remaining_depth: u32,
    l: LocalState,
    g: &mut GlobalState,
) -> SearchResult {
    if remaining_depth == 0 {
        return SearchResult {
            time: l.time,
            sequence: sequence.clone(),
        };
    }

    let options = if remaining_depth == 0 {
        &[data::JUGGERNAUT]
    } else if l.has_t2_con {
        BUILD_OPTIONS_T1
    } else if l.has_t2 {
        // force building a T2 constructor (always optimal)
        &[data::CONSTRUCTION_VEHICLE_T2]
    } else if l.has_t1_con {
        BUILD_OPTIONS_T1
    } else {
        BUILD_OPTIONS_CON
    };

    let mut best = SearchResult {
        time: u32::MAX,
        sequence: Vec::new(),
    };

    for option in options {
        let metal_shortage = l.metal - option.cost_metal;
        let conversions_to_do = (metal_shortage as f32) / l.conversion_result;

        let final_build_time = 1; // TODO

        let energy_to_convert = f32::ceil(conversions_to_do * l.conversion_drain) as u32;
        let metal_from_convert = f32::ceil(conversions_to_do * l.conversion_result) as u32;

        let new_local = LocalState {
            time: l.time + final_build_time,
            metal: l.metal - option.cost_metal + metal_from_convert * final_build_time,
            energy: l.energy - option.cost_energy - energy_to_convert
                + l.make_energy * final_build_time,
            make_energy: l.make_energy + option.make_energy,
            build_power: l.build_power + option.build_power,
            conversion_drain: l.conversion_drain + option.conversion_drain,
            conversion_result: l.conversion_result + option.conversion_result,
            storage_energy: l.storage_energy + option.storage_energy,
            has_t1_con: l.has_t1_con || option == &data::CONSTRUCTION_VEHICLE_T1,
            has_t2: l.has_t2 || option == &data::ADVANCED_VEHICLE_PLANT,
            has_t2_con: l.has_t2_con || option == &data::CONSTRUCTION_VEHICLE_T2,
        };

        sequence.push(option);
        let candidate = search_inner(sequence, remaining_depth - 1, new_local, g);
        sequence.pop();

        if candidate.time < best.time {
            best = candidate;
        }
    }

    best
}
