use crate::build_option::BuildOption;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use crate::data;

#[derive(Clone)]
struct LocalState {
    pub time: f32,
    pub metal: f32,
    pub energy: f32,
    pub energy_generation: f32,
    pub build_power: u32,
    pub conversion_drain: f32,
    pub conversion_result: f32,
    pub energy_storage: u32,
    pub has_t1: bool,
    pub has_t1_con: bool,
    pub has_t2: bool,
    pub has_t2_con: bool,
}

struct SharedState {
    pub best_time: AtomicU32,
    pub sequences_checked: AtomicU32,
    pub sequences_skipped_last: AtomicU32,
    pub sequences_skipped_time: AtomicU32,
}

struct GlobalState {
}

pub struct SearchResult {
    pub time: u32,
    pub sequence: Vec<&'static BuildOption>,
}

pub fn search() -> SearchResult {
    let mut sequence = Vec::new();
    let shared_state = Arc::new(SharedState {
        best_time: AtomicU32::new(u32::MAX),
        sequences_checked: AtomicU32::default(),
        sequences_skipped_last: AtomicU32::default(),
        sequences_skipped_time: AtomicU32::default(),
    });
    let done = Arc::new(AtomicBool::new(false));

    let progress_state = Arc::clone(&shared_state);
    let progress_done = Arc::clone(&done);
    let progress_handle = thread::spawn(move || progress_updater(progress_state, progress_done));

    let mut global_state = GlobalState {
    };

    let initial_state = LocalState {
        time: 0_f32,
        metal: 1000_f32,
        energy: 1000_f32,
        energy_generation: 3f32,
        build_power: 300,
        conversion_drain: 0.0,
        conversion_result: 0.0,
        energy_storage: 1000,
        has_t1: false,
        has_t1_con: false,
        has_t2: false,
        has_t2_con: false,
    };

    let mut best = SearchResult {
        time: u32::MAX,
        sequence: Vec::new(),
    };

    for i in 14..20 {
        let candidate = search_inner(
            &mut sequence,
            i,
            initial_state.clone(),
            &mut global_state,
            Arc::as_ref(&shared_state),
        );

        println!("\nBest {} sequence: {:?}", i, candidate.sequence);

        if candidate.time < best.time {
            best = candidate;
        }
    }

    done.store(true, Ordering::Relaxed);
    progress_handle.join().unwrap();

    println!(
        "Checked: {}, Shortened: {}, Skipped: {}",
        shared_state.sequences_checked.load(Ordering::Relaxed),
        shared_state.sequences_skipped_last.load(Ordering::Relaxed),
        shared_state.sequences_skipped_time.load(Ordering::Relaxed),
    );

    best
}

fn progress_updater(progress_state: Arc<SharedState>, progress_done: Arc<AtomicBool>) {
    use std::time::Duration;

    while !progress_done.load(Ordering::Relaxed) {
        let best_time = progress_state.best_time.load(Ordering::Relaxed);
        let checked = progress_state.sequences_checked.load(Ordering::Relaxed);
        let skipped = progress_state
            .sequences_skipped_time
            .load(Ordering::Relaxed);

        print!(
            "\rProgress: best_time={}, checked={}, skipped={}",
            if best_time == u32::MAX {
                "n/a".to_string()
            } else {
                best_time.to_string()
            },
            checked,
            skipped
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());

        thread::sleep(Duration::from_millis(200));
    }
}

fn search_inner(
    sequence: &mut Vec<&'static BuildOption>,
    remaining_depth: u32,
    l: LocalState,
    g: &mut GlobalState,
    s: &SharedState,
) -> SearchResult {
    if remaining_depth == 0 {
        s.sequences_checked.fetch_add(1, Ordering::Relaxed);
        return SearchResult {
            time: f32::ceil(l.time) as u32,
            sequence: sequence.clone(),
        };
    }

    let options = if remaining_depth == 1 {
        if l.has_t2_con {
            &[data::JUGGERNAUT]
        } else {
            return SearchResult {
                time: u32::MAX,
                sequence: Vec::new(),
            };
        }
    } else if l.has_t2_con {
        data::BUILD_OPTIONS_T2
    } else if l.has_t2 {
        // force building a T2 constructor (always optimal)
        &[data::CONSTRUCTION_VEHICLE_T2]
    } else if l.has_t1_con {
        data::BUILD_OPTIONS_T1
    } else if l.has_t1 {
        // force building a T1` constructor (always optimal)
        &[data::CONSTRUCTION_VEHICLE_T1]
    } else {
        data::BUILD_OPTIONS_CON
    };

    let mut best = SearchResult {
        time: u32::MAX,
        sequence: Vec::new(),
    };

    for option in options {
        let metal_shortage = (option.cost_metal as f32) - l.metal;
        let conversion_time = if metal_shortage <= 0_f32 {
            0_f32
        } else if l.conversion_result > 0_f32 {
            (metal_shortage) / l.conversion_result
        } else {
            // not enough metal stored, no metal conversion
            continue;
        };
        let energy_to_convert = conversion_time * l.conversion_drain;
        let energy_shortage = (option.cost_energy as f32) + energy_to_convert - (l.energy);

        let energy_generation_time = if energy_shortage < 0_f32 {
            0_f32
        } else if l.energy_generation > 0_f32 {
            energy_shortage / l.energy_generation
        } else {
            // not enough energy stored, no energy generation
            continue;
        };

        let build_power_time = (option.cost_bp / l.build_power) as f32;

        let final_build_time =
            f32::max(conversion_time, build_power_time).max(energy_generation_time);

        let total_time_u32 = f32::ceil(l.time + final_build_time) as u32;
        if total_time_u32 > s.best_time.load(Ordering::Relaxed) {
            if remaining_depth == 0 {
                s.sequences_skipped_last.fetch_add(1, Ordering::Relaxed);
            } else {
                s.sequences_skipped_time.fetch_add(1, Ordering::Relaxed);
            }
            continue;
        }

        // non-zero if we were not limited by energy generation
        let energy_surplus = if final_build_time == energy_generation_time {
            0_f32
        } else {
            (l.energy_generation * final_build_time) - energy_shortage
        };
        assert!(energy_surplus >= 0_f32);

        let final_conversion_time = if l.conversion_drain == 0_f32 {
            0_f32
        } else {
            conversion_time + (energy_surplus / l.conversion_drain)
        };
        assert!(final_conversion_time >= 0_f32);

        let final_metal_gained = f32::ceil(final_conversion_time * l.conversion_result);

        let new_local = LocalState {
            time: l.time + final_build_time,
            metal: final_metal_gained - metal_shortage,
            energy: f32::clamp(energy_surplus, 0_f32, l.energy_storage as f32),
            energy_generation: l.energy_generation + option.energy_generation as f32,
            build_power: l.build_power + option.build_power,
            conversion_drain: l.conversion_drain + option.conversion_drain,
            conversion_result: l.conversion_result + option.conversion_result,
            energy_storage: l.energy_storage + option.energy_storage,
            has_t1: l.has_t1 || option == &data::VEHICLE_LAB,
            has_t1_con: l.has_t1_con || option == &data::CONSTRUCTION_VEHICLE_T1,
            has_t2: l.has_t2 || option == &data::ADVANCED_VEHICLE_LAB,
            has_t2_con: l.has_t2_con || option == &data::CONSTRUCTION_VEHICLE_T2,
        };

        sequence.push(option);
        let candidate = search_inner(sequence, remaining_depth - 1, new_local, g, s);
        sequence.pop();

        if candidate.time < best.time {
            best = candidate;
        }
    }

    if best.time < s.best_time.load(Ordering::Relaxed) {
        s.best_time.store(best.time, Ordering::Relaxed)
    }

    best
}
