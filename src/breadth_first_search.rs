use crate::build_option::BuildOption;
use crate::data;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone)]
struct LocalState {
    pub time: u32,
    pub metal: u32,
    pub energy: u32,
    pub energy_generation: u32,
    pub build_power: u32,
    pub conversion_drain: f32,
    pub conversion_result: f32,
    pub energy_storage: u32,
    pub has_t1_con: bool,
    pub has_t2: bool,
    pub has_t2_con: bool,
}

struct GlobalState {
    pub best_time: AtomicU32,
    pub sequences_checked: AtomicU32,
    pub sequences_skipped_time: AtomicU32,
}

pub struct SearchResult {
    pub time: u32,
    pub sequence: Vec<&'static BuildOption>,
}

pub fn search() -> SearchResult {
    let mut sequence = Vec::new();
    let mut global_state = GlobalState {
        best_time: AtomicU32::new(u32::MAX),
        sequences_checked: AtomicU32::default(),
        sequences_skipped_time: AtomicU32::default(),
    };
    let initial_state = LocalState {
        time: 0,
        metal: 1000,
        energy: 1000,
        energy_generation: 3,
        build_power: 300,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 0,
        has_t1_con: false,
        has_t2: false,
        has_t2_con: false,
    };

    let mut best = SearchResult {
        time: u32::MAX,
        sequence: Vec::new(),
    };

    for i in 10..15 {
        let candidate = search_inner(&mut sequence, i, initial_state.clone(), &mut global_state);

        println!("Best {} sequence: {:?}", i, candidate.sequence);

        if candidate.time < best.time {
            best = candidate;
        }
    }

    println!("Checked: {}, Skipped: {}", global_state.sequences_checked.load(Ordering::Relaxed), global_state.sequences_skipped_time.load(Ordering::Relaxed));

    best
}

fn search_inner(
    sequence: &mut Vec<&'static BuildOption>,
    remaining_depth: u32,
    l: LocalState,
    g: &mut GlobalState,
) -> SearchResult {
    if remaining_depth == 0 {
        g.sequences_checked.fetch_add(1, Ordering::Relaxed);
        return SearchResult {
            time: l.time,
            sequence: sequence.clone(),
        };
    }

    let options = if remaining_depth == 1 {
        &[data::JUGGERNAUT]
    } else if l.has_t2_con {
        data::BUILD_OPTIONS_T2
    } else if l.has_t2 {
        // force building a T2 constructor (always optimal)
        &[data::CONSTRUCTION_VEHICLE_T2]
    } else if l.has_t1_con {
        data::BUILD_OPTIONS_T1
    } else {
        data::BUILD_OPTIONS_CON
    };

    let mut best = SearchResult {
        time: u32::MAX,
        sequence: Vec::new(),
    };

    for option in options {
        let metal_shortage = (option.cost_metal as i32) - (l.metal as i32);
        let conversion_time_f = if metal_shortage <= 0 {
            0f32
        } else if l.conversion_result > 0f32 {
            (metal_shortage as f32) / l.conversion_result
        } else {
            // not enough metal stored, no metal conversion
            continue;
        };
        let energy_to_convert = f32::ceil(conversion_time_f * l.conversion_drain) as i32;
        let conversion_time = f32::ceil(conversion_time_f) as u32;

        let energy_shortage = (option.cost_energy as i32) + energy_to_convert - (l.energy as i32);

        let energy_generation_time = if energy_shortage < 0 {
            0u32
        } else if l.energy_generation > 0 {
            f32::ceil((energy_shortage as f32) / (l.energy_generation as f32)) as u32
        } else {
            // not enough energy stored, no energy generation
            continue;
        };

        let build_power_time = option.cost_bp / l.build_power;

        let final_build_time =
            u32::max(conversion_time, build_power_time).max(energy_generation_time);

        if l.time + final_build_time > g.best_time.load(Ordering::Relaxed) {
            g.sequences_skipped_time.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        // non-zero if we were not limited by energy generation
        let energy_surplus = (l.energy_generation * final_build_time) as i32 - energy_shortage;
        assert!(energy_surplus >= 0);

        let final_conversion_time = if l.conversion_drain == 0f32 {
            0f32
        } else {
            conversion_time_f + ((energy_surplus as f32) / l.conversion_drain)
        };
        assert!(final_conversion_time >= 0f32);

        let final_metal_gained = f32::ceil(final_conversion_time * l.conversion_result) as i32;

        let new_local = LocalState {
            time: l.time + final_build_time,
            metal: (final_metal_gained - metal_shortage) as u32,
            energy: u32::clamp(energy_surplus as u32, 0, l.energy_storage),
            energy_generation: l.energy_generation + option.energy_generation,
            build_power: l.build_power + option.build_power,
            conversion_drain: l.conversion_drain + option.conversion_drain,
            conversion_result: l.conversion_result + option.conversion_result,
            energy_storage: l.energy_storage + option.energy_storage,
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

    if best.time < g.best_time.load(Ordering::Relaxed) {
        g.best_time.store(best.time, Ordering::Relaxed)
    }

    best
}
