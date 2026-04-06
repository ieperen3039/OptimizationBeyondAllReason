use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::data::{BuildOptionId, BuildSet};
use crate::machine_learning::common;
use crate::machine_learning::neat::{Gene, InputTensor, NeatNetwork, OutputTensor};
use crate::machine_learning::reward::Reward;
use crate::random::MyRandom;
use crate::search_handler::{LocalState, SearchResult};
use crate::searcher::Searcher;
use std::fmt::{Display, Formatter};
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};

const COMPATIBLITY_THRESHOLD: f32 = 3.0;

pub struct NeatTrainer {
    population: Vec<NeatNetwork>,
    best_network: NeatNetwork,
    best_score: f32,
    next_innovation_id: usize,
    next_species_id: usize,
    rng: MyRandom,
    max_game_time: f32,
    generation_number: usize,
    species: Vec<Specie>,
    config: NeatTrainerConfig,
    state: Arc<NeatStateWrapper>,
}

pub struct NeatTrainerConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub reward_model: Box<dyn Reward>,
    pub crossover_probability: f32,
    pub add_connection_probability: f32,
    pub add_node_probability: f32,
}

struct Specie {
    id: usize,
    base_genome: Vec<Gene>,
}

struct NeatStateWrapper(Mutex<NeatState>);
struct NeatState {
    generation_number: usize,
    max_survior_genome_size: usize,
    best_survior_genome_size: usize,
    num_created_species: usize,
    num_extinct_species: usize,
    best_score: f32,
}

impl Searcher for NeatTrainer {
    fn search(&mut self, initial_state: LocalState) -> SearchResult {
        for generation_idx in 0..self.config.num_generations {
            self.state.update(|s| s.generation_number = generation_idx);

            self.create_next_generation(&initial_state);
        }

        SearchResult {
            score: self.best_score,
            sequence: self.get_sequence(&self.best_network, initial_state),
        }
    }

    fn new_progress_updater(&self) -> Arc<dyn Display + Send + Sync> {
        self.state.clone()
    }
}

impl NeatTrainer {
    pub fn new_with_config(
        config: NeatTrainerConfig,
        random_seed: u32,
        max_game_time: f32,
    ) -> Self {
        let rng = MyRandom::new_from_u32(random_seed);
        let population = (0..config.population_size)
            .map(|i| NeatNetwork::new().add_connection(i, &rng).unwrap())
            .collect();

        NeatTrainer {
            population,
            best_network: NeatNetwork::new(),
            best_score: 0.0,
            next_innovation_id: 0,
            next_species_id: 1, // we start with one
            rng,
            max_game_time,
            generation_number: 0,
            species: vec![Specie {
                id: 0,
                base_genome: Vec::new(),
            }],
            config,
            state: Arc::new(NeatStateWrapper(Mutex::new(NeatState {
                generation_number: 0,
                max_survior_genome_size: 0,
                best_survior_genome_size: 0,
                num_created_species: 0,
                num_extinct_species: 0,
                best_score: 0.0,
            }))),
        }
    }

    pub fn create_next_generation(&mut self, initial_state: &LocalState) {
        let last_generation = std::mem::take(&mut self.population);

        let mut survivor_scores = Vec::new();
        for idx in 0..last_generation.len() {
            let score = self.evaluate(&last_generation[idx], initial_state.clone());
            survivor_scores.push((idx, score));
        }

        survivor_scores.sort_by(|(_, score1), (_, score2)| score1.total_cmp(score2).reverse());
        let num_survivors = survivor_scores.len() / 10;
        let survivor_idx: Vec<_> = survivor_scores
            .iter()
            .take(num_survivors)
            .map(|(idx, _)| *idx)
            .collect();

        self.population = Vec::new();
        for parent_idx in 0..num_survivors {
            for _ in 0..10 {
                let mut child = if self.rng.next_f32() < self.config.crossover_probability {
                    let crossover_target =
                        f32::floor(parent_idx as f32 * self.rng.next_f32()) as usize;
                    if crossover_target == parent_idx {
                        last_generation[survivor_idx[parent_idx]].mutate(&self.rng)
                    } else {
                        last_generation[survivor_idx[crossover_target]]
                            .cross_with(&last_generation[survivor_idx[parent_idx]], &self.rng)
                    }
                } else {
                    last_generation[survivor_idx[parent_idx]].mutate(&self.rng)
                };

                if self.rng.next_f32() < self.config.add_connection_probability {
                    let new_child = child.add_connection(self.next_innovation_id, &self.rng);
                    if let Some(new_child) = new_child {
                        child = new_child;
                    }

                    self.next_innovation_id += 1;
                }

                if self.rng.next_f32() < self.config.add_node_probability {
                    child = child.add_node(
                        self.next_innovation_id,
                        self.next_innovation_id + 1,
                        &self.rng,
                    );
                    self.next_innovation_id += 2;
                }

                self.population.push(child);
            }
        }

        let longest_genome = survivor_scores
            .iter()
            .map(|(idx, _)| last_generation[*idx].num_connections())
            .max()
            .unwrap();

        let (idx_of_best, score_of_best) = survivor_scores[*survivor_idx.first().unwrap()];
        if score_of_best > self.best_score {
            self.best_score = score_of_best;
            self.best_network = last_generation.into_iter().nth(idx_of_best).unwrap();
        }

        for network in &mut self.population {
            let gene = network.sequence();
            let mut least_distance = 0.0;
            let mut closest_specie = self.species.first().unwrap();
            for s in &self.species {
                let distance = NeatNetwork::get_genome_distance(&s.base_genome, &gene);
                if distance < least_distance {
                    least_distance = distance;
                    closest_specie = s;
                }
            }

            if least_distance < COMPATIBLITY_THRESHOLD {
                network.specie = closest_specie.id;
            } else {
                let id = self.next_species_id;
                self.next_species_id += 1;
                self.species.push(Specie {
                    id,
                    base_genome: gene,
                });
                network.specie = id;
            }
        }

        self.state.update(|s| {
            s.best_score = self.best_score;
            s.best_survior_genome_size = self.best_network.num_connections();
            s.max_survior_genome_size = usize::max(s.max_survior_genome_size, longest_genome);
            s.num_created_species = self.next_species_id;
            s.num_extinct_species = self.next_species_id - self.species.len();
        });
    }

    fn get_sequence(&self, model: &NeatNetwork, initial_state: LocalState) -> Vec<BuildOptionId> {
        let mut state = initial_state.clone();
        let mut sequence = Vec::new();

        loop {
            let input = Self::build_input_tensor(&state);

            let (next_build, next_state) = model.run(&state, &input, self.max_game_time);

            if next_state.is_none() {
                return sequence;
            }

            sequence.push(next_build);
            state = next_state.unwrap();
        }
    }

    fn evaluate(&self, model: &NeatNetwork, initial_state: LocalState) -> f32 {
        let mut state = initial_state.clone();

        loop {
            let input = Self::build_input_tensor(&state);

            let (_, next_state) = model.run(&state, &input, self.max_game_time);

            if next_state.is_none() {
                break;
            }

            state = next_state.unwrap();
        }

        self.config.reward_model.calculate(&initial_state, &state)
    }

    /// Build input tensor based on state
    pub fn build_input_tensor(state: &LocalState) -> InputTensor {
        let fraction_of_energy_converted = if state.conversion_drain <= 0.0 {
            0.0
        } else {
            state.energy_generation / state.conversion_drain
        };
        let fraction_of_storage_generated_per_second =
            state.energy_generation / (state.energy_storage as f32);
        let metal_per_build_power = state.metal_generation / state.build_power as f32;
        let energy_per_build_power = state.energy_generation / state.build_power as f32;

        [
            // bias
            1.0,
            // raw state
            f32::ln(state.metal + 1.0) / 14.0,
            f32::ln(state.energy + 1.0) / 15.0,
            f32::ln(state.energy_generation + 1.0) / 18.0,
            f32::ln(state.metal_generation + 1.0),
            state.build_power as f32 / 20000.0, // increments in steps of 200
            f32::ln(state.conversion_drain + 1.0) / 20.0,
            f32::ln(state.energy_storage as f32 + 1.0) / 15.0,
            // relational values
            fraction_of_energy_converted,
            fraction_of_storage_generated_per_second,
            f32::ln(metal_per_build_power + 1.0) / 5.0,
            f32::ln(energy_per_build_power + 1.0) / 5.0,
            // build options
            common::convert_to_float(state.has_built.contains(VehicleLab)),
            common::convert_to_float(state.has_built.contains(ConstructionVehicleT1)),
            common::convert_to_float(state.has_built.contains(AdvancedVehicleLab)),
            common::convert_to_float(state.has_built.contains(ConstructionVehicleT2)),
        ]
    }

    #[allow(dead_code)]
    fn filter_softmax(outputs: OutputTensor, allowed_options: BuildSet) -> OutputTensor {
        /* From dfdx:
         * Let's denote `t - t.max()` expression tm:
         * `(tm - tm.exp().sum().ln()).exp()`
         *
         * Another reduction is the identity of the form `e^(x - y)` = `e^x / e^y`.
         * `tm.exp() / tm.exp().sum().ln().exp()`
         *
         * First we can re-use the `tm.exp()` calculation - lets call it tme
         * `tme / tme.sum().ln().exp()`
         *
         * And finally we know that `t.ln().exp()` is equivalent to `t`. I.e. they are fused
         * `tme / tme.sum()`
         */

        let mut max = f32::MIN;
        for idx in allowed_options.ids().map(|b| b as usize) {
            if outputs[idx] > max {
                max = outputs[idx];
            }
        }
        let mut result = OutputTensor::default();
        let mut sum = 0.0;
        for idx in allowed_options.ids().map(|b| b as usize) {
            let t = outputs[idx];
            let tm = t - max;
            let tme = tm.exp();
            sum += tme;
            result[idx] = tme;
        }

        for out in &mut result {
            *out /= sum
        }

        result
    }

    #[allow(dead_code)]
    pub fn select(distribution: OutputTensor, chosen_fraction: f32) -> usize {
        let mut cumulative = 0.0;
        for i in 0..distribution.len() {
            cumulative += distribution[i];
            if chosen_fraction < cumulative {
                return i;
            }
        }

        unreachable!("probabilities should sum to 1.0");
    }
}

impl NeatStateWrapper {
    pub fn update<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut NeatState),
    {
        let mut guard = self.0.lock().unwrap();
        update_fn(guard.deref_mut());
    }
}

impl Display for NeatStateWrapper {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let state = match self.0.lock() {
            Ok(state) => state,
            Err(e) => return e.fmt(f),
        };

        write!(
            f,
            "generation: {}, best score: {}, best survior genome size: {}, max survior genome size: {}, \
            num created species: {}, num extinct species: {}",
            state.generation_number,
            state.best_score,
            state.best_survior_genome_size,
            state.max_survior_genome_size,
            state.num_created_species,
            state.num_extinct_species,
        )
    }
}
