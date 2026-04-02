use crate::data::{BuildOptionId, BuildSet};
use crate::searcher::Searcher;
use crate::data;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

#[derive(Clone)]
pub struct LocalState {
    pub time: f32,
    pub metal: f32,
    pub energy: f32,
    pub energy_generation: f32,
    pub metal_generation: f32,
    pub build_power: u32,
    pub conversion_drain: f32,
    pub conversion_result: f32,
    pub energy_storage: u32,
    pub has_built: BuildSet,
}

pub struct SharedState {
    pub done: AtomicBool,
    pub best_time: AtomicU32,
    pub sequences_checked: AtomicU32,
    pub sequences_skipped: AtomicU32,
}

pub struct SearchResult {
    pub time: f32,
    pub sequence: Vec<BuildOptionId>,
}

impl LocalState {
    pub fn initial() -> Self {
        LocalState {
            time: 0.0,
            metal: 1000_f32,
            energy: 1000_f32,
            energy_generation: 3f32,
            metal_generation: 2f32,
            build_power: 300,
            conversion_drain: 0.0,
            conversion_result: 0.0,
            energy_storage: 1000,
            has_built: BuildSet::new(),
        }
    }
}

pub fn search(searcher: &mut dyn Searcher, initial_state: LocalState) -> SearchResult {
    let shared_state = Arc::new(SharedState {
        done: AtomicBool::new(false),
        best_time: AtomicU32::new(u32::MAX),
        sequences_checked: AtomicU32::default(),
        sequences_skipped: AtomicU32::default(),
    });

    let progress_state = Arc::clone(&shared_state);
    let progress_handle = thread::spawn(move || progress_updater(progress_state));

    let result = searcher.search(&shared_state, initial_state);
    let best = result;

    shared_state.done.store(true, Ordering::Relaxed);
    progress_handle.join().unwrap();

    best
}
fn progress_updater(progress_state: Arc<SharedState>) {
    use std::time::Duration;

    while !progress_state.done.load(Ordering::Relaxed) {
        let best_time = progress_state.best_time.load(Ordering::Relaxed);
        let checked = progress_state.sequences_checked.load(Ordering::Relaxed);
        let skipped = progress_state.sequences_skipped.load(Ordering::Relaxed);

        print!(
            "\rProgress: best_time: {}, checked: {}, skipped: {}",
            if best_time == u32::MAX {
                "n/a".to_string()
            } else {
                best_time.to_string()
            },
            checked,
            skipped,
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());

        thread::sleep(Duration::from_millis(200));
    }
}

impl LocalState {
    pub fn compute_potential_metal_production(&self) -> f32 {
        if self.conversion_drain <= self.energy_generation {
            self.metal_generation + self.conversion_result
        } else {
            let fraction_of_conversion_used = self.conversion_drain / self.energy_generation;
            self.metal_generation + self.conversion_result * fraction_of_conversion_used
        }
    }

    pub fn compute_next(&self, option_id: BuildOptionId, maximum_time: f32) -> Option<LocalState> {
        let option = &data::BUILD_OPTIONS[option_id as usize];

        let metal_shortage = (option.cost_metal as f32) - self.metal;
        let conversion_time = if metal_shortage <= 0.0 {
            0.0
        } else if (self.conversion_result > 0.0) || (self.metal_generation > 0.0) {
            (metal_shortage) / (self.conversion_result + self.metal_generation)
        } else {
            // not enough metal stored, no metal conversion
            return None;
        };
        let energy_to_convert = conversion_time * self.conversion_drain;
        let energy_shortage = (option.cost_energy as f32) + energy_to_convert - (self.energy);

        let energy_generation_time = if energy_shortage < 0.0 {
            0.0
        } else if self.energy_generation > 0.0 {
            energy_shortage / self.energy_generation
        } else {
            // not enough energy stored, no energy generation
            return None;
        };

        let build_power_time = (option.cost_bp / self.build_power) as f32;

        let final_build_time =
            f32::max(conversion_time, build_power_time).max(energy_generation_time);

        if self.time + final_build_time > maximum_time {
            return None;
        }

        // non-zero if we were not limited by energy generation
        let energy_surplus = if final_build_time == energy_generation_time {
            0.0
        } else {
            (self.energy_generation * final_build_time) - energy_shortage
        };
        assert!(energy_surplus >= 0.0);

        let final_conversion_time = if self.conversion_drain == 0.0 {
            0.0
        } else {
            conversion_time + (energy_surplus / self.conversion_drain)
        };
        assert!(final_conversion_time >= 0.0);

        let final_metal_gained = f32::ceil(final_conversion_time * self.conversion_result);

        Some(LocalState {
            time: self.time + final_build_time,
            metal: final_metal_gained - metal_shortage,
            energy: f32::clamp(energy_surplus, 0.0, self.energy_storage as f32),
            energy_generation: self.energy_generation + option.energy_generation as f32,
            metal_generation: self.metal_generation + option.metal_generation as f32,
            build_power: self.build_power + option.build_power,
            conversion_drain: self.conversion_drain + option.conversion_drain,
            conversion_result: self.conversion_result + option.conversion_result,
            energy_storage: self.energy_storage + option.energy_storage,
            has_built: self.has_built.clone().with(option_id),
        })
    }
}
