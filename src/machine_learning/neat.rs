use crate::data;
use crate::data::{BuildOptionId, BuildSet};
use crate::random::MyRandom;
use crate::search_handler::LocalState;

pub const INPUT_SIZE: usize = 16;
pub const OUTPUT_SIZE: usize = 12;
pub const OUTPUT_MAPPING: [BuildOptionId; OUTPUT_SIZE] = [
    BuildOptionId::WindTurbine,
    BuildOptionId::SolarCollector,
    BuildOptionId::AdvancedSolarCollector,
    BuildOptionId::EnergyConverter,
    BuildOptionId::AdvancedEnergyConverter,
    BuildOptionId::FusionReactor,
    BuildOptionId::AdvancedFusionReactor,
    BuildOptionId::BuildTurret,
    BuildOptionId::VehicleLab,
    BuildOptionId::ConstructionVehicleT1,
    BuildOptionId::AdvancedVehicleLab,
    BuildOptionId::ConstructionVehicleT2,
];

const DISABLED_GENE_PROBABILITY: f32 = 0.75;
const MINIMUM_CONNECTION_WEIGHT: f32 = 0.75;

pub type InputTensor = [f32; INPUT_SIZE];
pub type OutputTensor = [f32; OUTPUT_SIZE];

#[derive(Clone, Debug)]
pub struct NeatNetwork {
    /// sorted on input node
    connections: Vec<Connection>,
    num_hidden_nodes: usize,
}

#[derive(Clone, Debug)]
struct Connection {
    input: usize,
    output: usize,
    weight: f32,
    enabled: bool,
    innovation_id: usize,
}

const WEIGHT_PERTUBATION_PROBABILITY: f32 = 0.9;
const WEIGHT_PERTUBATION_MAX: f32 = 0.2;
const WEIGHT_CUT_PROBABILITY: f32 = 0.01;

const GENOME_DISTANCE_EXCESS_FACTOR: f32 = 1.0;
const GENOME_DISTANCE_DISJOINT_FACTOR: f32 = 1.0;
const GENOME_DISTANCE_WEIGHT_FACTOR: f32 = 0.4;

impl NeatNetwork {
    pub fn new() -> Self {
        NeatNetwork {
            connections: Vec::new(),
            num_hidden_nodes: 0,
        }
    }

    pub fn new_with_connection(input: usize, output: usize, weight: f32, innovation_id: usize) -> Self {
        assert!(input < INPUT_SIZE);
        assert!(output >= INPUT_SIZE);
        assert!(output < INPUT_SIZE + OUTPUT_SIZE);
        NeatNetwork {
            connections: vec![Connection{
                input,
                output,
                weight,
                enabled: true,
                innovation_id,
            }],
            num_hidden_nodes: 0,
        }
    }

    pub fn run(&self, state: &LocalState, input: &InputTensor, max_game_time: f32) -> (BuildOptionId, Option<LocalState>) {
        let logits = self.forward(&input);

        let build_options = data::get_build_options(&state.has_built);
        let next_build = NeatNetwork::get_max(logits, build_options);
        let next_state = state.compute_next(next_build, max_game_time);
        (next_build, next_state)
    }

    pub fn forward(&self, inputs: &InputTensor) -> OutputTensor {
        let mut node_values = vec![0.0; INPUT_SIZE + self.num_hidden_nodes + OUTPUT_SIZE];
        node_values[..INPUT_SIZE].copy_from_slice(inputs);

        let mut last_processed_node_idx = INPUT_SIZE;

        for conn in &self.connections {
            if !conn.enabled {
                continue;
            }

            while conn.input >= last_processed_node_idx {
                last_processed_node_idx += 1;
                node_values[last_processed_node_idx] =
                    Self::sigmoid(node_values[last_processed_node_idx])
            }

            node_values[conn.output] += node_values[conn.input];
        }

        let mut outputs = [0.0; OUTPUT_SIZE];
        let start_of_output = INPUT_SIZE + self.num_hidden_nodes;
        outputs.clone_from_slice(&node_values[start_of_output..]);
        outputs
    }

    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    pub fn sequence(&self) -> Vec<Gene> {
        let mut this_sequence: Vec<Gene> = self.connections.iter().map(|c| Gene { id: c.innovation_id, weight: c.weight }).collect();
        this_sequence.sort_by_key(|g| g.id);
        this_sequence
    }

    pub fn get_genome_distance(this_sequence: &Vec<Gene>, other_sequence: &Vec<Gene>) -> f32 {
        let mut this_idx = 0;
        let mut other_idx = 0;
        let mut num_disjoint = 0;
        let num_excess;
        let mut num_equal = 0;
        let mut total_weight_difference = 0.0;
        loop {
            if this_idx >= this_sequence.len() {
                num_excess = other_sequence.len() - other_idx;
                break;
            } else if other_idx >= other_sequence.len() {
                num_excess = this_sequence.len() - this_idx;
                break;
            }

            let this_gene = &this_sequence[this_idx];
            let other_gene = &other_sequence[other_idx];
            if this_gene.id == other_gene.id {
                num_equal += 1;
                total_weight_difference += (this_gene.weight - other_gene.weight).abs();
                this_idx += 1;
                other_idx += 1;
            } else if this_gene.id < other_gene.id {
                num_disjoint += 1;
                other_idx += 1;
            } else {
                num_disjoint += 1;
                this_idx += 1;
            }
        }

        let lenght_of_longest = std::cmp::max(this_sequence.len(), other_sequence.len());

        (GENOME_DISTANCE_EXCESS_FACTOR * num_excess as f32) / (lenght_of_longest as f32)
            + (GENOME_DISTANCE_DISJOINT_FACTOR * num_disjoint as f32) / (lenght_of_longest as f32)
            + GENOME_DISTANCE_WEIGHT_FACTOR * (total_weight_difference / num_equal as f32)
    }

    pub fn add_connection(&mut self, innovation_id: usize, rng: &MyRandom) -> bool {
        let input = rng.random_index(INPUT_SIZE + self.num_hidden_nodes);
        let output = rng.random_between(input + 1, INPUT_SIZE + self.num_hidden_nodes + OUTPUT_SIZE - 1);

        let first_higher_input_idx = self.connections.partition_point(|c| c.input < input);
        // from below `first_higher_input_idx` going down, seach for the same output
        for connection in self.connections[..first_higher_input_idx].iter().rev() {
            if connection.output == output {
                return false;
            }
        }

        // no same output, insert new connection before first_higher_input_idx
        let initial_weight = rng.next_f32() - 0.5;
        self.connections.insert(
            first_higher_input_idx,
            Connection {
                input,
                output,
                weight: initial_weight,
                enabled: true,
                innovation_id,
            },
        );
        true
    }

    pub fn add_node(&mut self, innovation_id_1: usize, innovation_id_2: usize, rng: &MyRandom) {
        let node_idx = self.num_hidden_nodes;
        self.num_hidden_nodes += 1;

        let connection_idx_to_split = rng.random_index(self.connections.len());
        self.connections[connection_idx_to_split].enabled = false;
        let connection = self.connections[connection_idx_to_split].clone();

        self.connections.push(Connection {
            input: connection.input,
            output: node_idx,
            weight: connection.weight,
            enabled: true,
            innovation_id: innovation_id_1,
        });
        self.connections.push(Connection {
            input: node_idx,
            output: connection.output,
            weight: connection.weight,
            enabled: true,
            innovation_id: innovation_id_2,
        });
    }

    pub fn mutate(&mut self, rng: &MyRandom) {
        for conn in &mut self.connections {
            let rand = rng.next_f32();
            if rand > WEIGHT_PERTUBATION_PROBABILITY {
                // do not perturb
                continue;
            } else if conn.weight < MINIMUM_CONNECTION_WEIGHT {
                // connection is too weak, disable
                conn.enabled = false;
            } else if rand > WEIGHT_PERTUBATION_PROBABILITY - WEIGHT_CUT_PROBABILITY {
                // try cutting the weight of this connection, to check if it is unnecessary
                conn.weight /= 4.0
            } else {
                const REMAINING_PROBABILITY: f32 = WEIGHT_PERTUBATION_PROBABILITY - WEIGHT_CUT_PROBABILITY;
                const REMAINING_PROBABILITY_HALF: f32 = REMAINING_PROBABILITY / 2.0;
                const FACTOR: f32 = WEIGHT_PERTUBATION_MAX / REMAINING_PROBABILITY_HALF;

                // [-WEIGHT_PERTUBATION_MAX, WEIGHT_PERTUBATION_MAX]
                let pertubation = (rand - REMAINING_PROBABILITY_HALF) * FACTOR;
                conn.weight += pertubation;
            }
        }
    }

    /// assumes that self has higher fitness than other
    pub fn cross_with(&mut self, other: &NeatNetwork, rng: &MyRandom) {
        for other_connection in &other.connections {
            // half of other's connections are not considered, whether they are present or not.
            // Given that non-matching connections are always dropped, this check here only reduces
            // the number of searches, without affecting correctness
            if rng.next_f32() < 0.5 {
                continue;
            }

            let index = self
                .connections
                .iter()
                .position(|c| c.innovation_id == other_connection.innovation_id);

            if let Some(index) = index {
                let new_connection = &mut self.connections[index];
                // if enabled in one parent, but disabled in the other
                if new_connection.enabled ^ other_connection.enabled
                    && rng.next_f32() < DISABLED_GENE_PROBABILITY
                {
                    new_connection.enabled = false
                }
                self.connections[index] = other_connection.clone();
            }
        }
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-value))
    }

    fn get_max(outputs: OutputTensor, allowed_options: BuildSet) -> BuildOptionId {
        let mut max_value = 0.0; // start at 0 to avoid picking an unactivated node
        let mut best_id = BuildOptionId::Invalid;
        for idx in 0..OUTPUT_SIZE {
            if !allowed_options.contains(OUTPUT_MAPPING[idx]) {
                continue;
            }

            let value = outputs[idx];
            if value > max_value {
                best_id = OUTPUT_MAPPING[idx];
                max_value = value;
            }
        }
        best_id
    }
}

pub struct Gene {
    id: usize,
    weight: f32,
}
