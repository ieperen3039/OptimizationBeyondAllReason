use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::data::{BuildOptionId, BuildSet};
use crate::machine_learning::neat::{Gene, InputTensor, NeatNetwork, OutputTensor};
use crate::machine_learning::reward::Reward;
use crate::machine_learning::{common, neat};
use crate::random::MyRandom;
use crate::search_handler::{LocalState, SearchResult};
use crate::searcher::Searcher;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Write;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};

const COMPATIBLITY_THRESHOLD: f32 = 3.0;
const OFFSPRING_PER_PARENT: usize = 10;

pub struct NeatTrainer {
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
    population: Vec<NeatNetwork>,
    base_genome: Vec<Gene>,
}

#[derive(Clone)]
struct NetworkScore {
    species_idx: usize,
    member_idx: usize,
    score: f32,
}

struct NeatStateWrapper(Mutex<NeatState>);
struct NeatState {
    generation_number: usize,
    largest_genome_length: usize,
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

        let mut out_file = File::create("my_neat_model.json").unwrap();
        let best_as_json = serde_json::to_string(&self.best_network).unwrap();
        println!("{}", best_as_json);
        out_file.write(best_as_json.as_bytes()).unwrap();

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
        // we seed the network with all possible input-output combinations rather than respecting
        // the population size. This may cause the first
        // generation to take excessively long to compute, but by exhaustively searching every
        // direct connection, we kick off the search in every promising direction, without hurting
        // dimensionality minimalisation and reducing the chance that a good direction is never found
        let initial_population_size = neat::INPUT_SIZE * neat::OUTPUT_SIZE * 2;
        let population = (0..initial_population_size)
            .map(|i| Self::create_on_index(i, initial_population_size))
            .collect();

        NeatTrainer {
            best_network: NeatNetwork::new(),
            best_score: 0.0,
            // every initial member has its own innovation
            next_innovation_id: initial_population_size,
            next_species_id: 1, // we start with one
            rng,
            max_game_time,
            generation_number: 0,
            species: vec![Specie {
                id: 0,
                base_genome: Vec::new(),
                population,
            }],
            config,
            state: Arc::new(NeatStateWrapper(Mutex::new(NeatState {
                generation_number: 0,
                largest_genome_length: 0,
                best_survior_genome_size: 0,
                num_created_species: 0,
                num_extinct_species: 0,
                best_score: 0.0,
            }))),
        }
    }

    fn create_on_index(i: usize, total_population_size: usize) -> NeatNetwork {
        assert!(i < total_population_size);
        let weight = if i < (total_population_size / 2) {
            1.0
        } else {
            -1.0
        };
        let output_idx = (i / neat::INPUT_SIZE) % neat::OUTPUT_SIZE;
        NeatNetwork::new_with_connection(
            i % neat::INPUT_SIZE,
            neat::INPUT_SIZE + output_idx,
            weight,
            i,
        )
    }

    pub fn create_next_generation(&mut self, initial_state: &LocalState) {
        // Every species is assigned a potentially different number of offspring in proportion to
        // sum of adjusted fitnesses f′_i of its member organisms. Species then reproduce by first
        // eliminating the lowest performing members from the population. The entire population is
        // then replaced by the offspring of the remaining organisms in each species.
        let mut total_species_score = 0.0;
        let mut species_scores = Vec::new();
        let mut best = NetworkScore::default();
        for species_idx in 0..self.species.len() {
            let s = &self.species[species_idx];
            let mut member_scores = Vec::new();
            let mut species_score = 0.0;
            let num_members = s.population.len();
            for member_idx in 0..num_members {
                let network = &s.population[member_idx];
                let score = self.evaluate(network, initial_state.clone());

                species_score += score;
                member_scores.push(NetworkScore {
                    species_idx,
                    member_idx,
                    score,
                });
            }
            species_score /= num_members as f32;
            member_scores.sort();
            let best_of_species = member_scores.first().unwrap();
            if best_of_species < &best {
                best = best_of_species.clone();
            }
            species_scores.push((species_score, member_scores));
            total_species_score += species_score;
        }

        if best.score > self.best_score {
            self.best_score = best.score;
            self.best_network = self.get(&best).clone()
        }

        let num_offspring_total = self.config.population_size as f32;
        let mut new_population = Vec::new();
        for (score, members) in species_scores {
            let score_fraction_of_total = score / total_species_score;
            let num_offspring = f32::ceil(score_fraction_of_total * num_offspring_total) as usize;
            // extremely promising innovations may sometimes be assigned more than 10 offspring per parent;
            // we limit this explosive growth to optimize for the expected case
            let num_offspring = num_offspring.clamp(0, members.len() * OFFSPRING_PER_PARENT);
            for child_idx in 0..num_offspring {
                // assign all offspring to the first few parents (the best ones) until we run out of budget
                let parent_idx = child_idx / OFFSPRING_PER_PARENT;
                let mut child = self.get(&members[parent_idx]).clone();

                if self.rng.next_f32() < self.config.crossover_probability {
                    // if parent_idx is 0, this will always return 0
                    let crossover_target_idx =
                        f32::floor(parent_idx as f32 * self.rng.next_f32()) as usize;
                    if crossover_target_idx == parent_idx {
                        child.mutate(&self.rng)
                    } else {
                        child.cross_with(self.get(&members[crossover_target_idx]), &self.rng)
                    }
                } else {
                    child.mutate(&self.rng)
                };

                if self.rng.next_f32() < self.config.add_connection_probability {
                    let success = child.add_connection(self.next_innovation_id, &self.rng);
                    if success {
                        self.next_innovation_id += 1;
                    }
                }

                if self.rng.next_f32() < self.config.add_node_probability {
                    child.add_node(
                        self.next_innovation_id,
                        self.next_innovation_id + 1,
                        &self.rng,
                    );
                    self.next_innovation_id += 2;
                }

                new_population.push(child);
            }
        }

        // species
        self.species.iter_mut().for_each(|s| s.population.clear());

        let mut longest_genome_length = 0;
        for network in new_population {
            let genome = network.sequence();
            if genome.len() > longest_genome_length {
                longest_genome_length = genome.len()
            }

            let mut least_distance = f32::MAX;
            let mut closest_specie_idx = 0;
            for specie_idx in 0..self.species.len() {
                let s = &self.species[specie_idx];
                let distance = NeatNetwork::get_genome_distance(&s.base_genome, &genome);
                if distance < least_distance {
                    least_distance = distance;
                    closest_specie_idx = specie_idx;
                }
            }

            if least_distance < COMPATIBLITY_THRESHOLD {
                let specie = &mut self.species[closest_specie_idx];
                specie.population.push(network);
            } else {
                let id = self.next_species_id;
                self.next_species_id += 1;
                self.species.push(Specie {
                    id,
                    population: vec![network],
                    base_genome: genome,
                });
            }
        }

        self.species.retain(|s| !s.population.is_empty());

        self.state.update(|s| {
            s.best_score = self.best_score;
            s.best_survior_genome_size = self.best_network.num_connections();
            s.largest_genome_length = usize::max(s.largest_genome_length, longest_genome_length);
            s.num_created_species = self.next_species_id;
            s.num_extinct_species = self.next_species_id - self.species.len();
        });
    }

    fn get(&self, score: &NetworkScore) -> &NeatNetwork {
        &self.species[score.species_idx].population[score.member_idx]
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

impl Default for NetworkScore {
    fn default() -> Self {
        NetworkScore {
            species_idx: 0,
            member_idx: 0,
            score: f32::NEG_INFINITY,
        }
    }
}

impl Eq for NetworkScore {}

impl PartialEq<Self> for NetworkScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl PartialOrd for NetworkScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NetworkScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score).reverse()
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
            "generation: {}, best score: {}, best survior genome size: {}, length of longest genome: {}, \
            num created species: {}, num extinct species: {}",
            state.generation_number,
            state.best_score,
            state.best_survior_genome_size,
            state.largest_genome_length,
            state.num_created_species,
            state.num_extinct_species,
        )
    }
}
