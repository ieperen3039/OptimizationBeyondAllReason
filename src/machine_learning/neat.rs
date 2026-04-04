use dfdx::prelude::Softmax;
use crate::data;
use crate::data::{BuildOptionId, BuildSet};
use crate::data::BuildOptionId::{
    AdvancedVehicleLab, ConstructionVehicleT1, ConstructionVehicleT2, VehicleLab,
};
use crate::machine_learning::common;
use crate::machine_learning::reward::Reward;
use crate::random::MyRandom;
use crate::search_handler::LocalState;

const INPUT_SIZE: usize = 16;
const OUTPUT_SIZE: usize = data::NUM_BUILD_OPTIONS;
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

impl NeatNetwork {
    pub fn forward(&self, inputs: &InputTensor) -> OutputTensor {
        let mut node_values = Vec::with_capacity(INPUT_SIZE + self.num_hidden_nodes + OUTPUT_SIZE);
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
        outputs.clone_from_slice(&node_values[INPUT_SIZE + self.num_hidden_nodes..]);
        outputs
    }

    pub fn add_connection(&self, innovation_id: usize, rng: &MyRandom) -> Option<NeatNetwork> {
        let input = rng.random_index(INPUT_SIZE + self.num_hidden_nodes);
        let output = rng.random_between(input + 1, self.connections.len() - 1);

        let first_higher_input_idx = self.connections.partition_point(|c| c.input < input);
        // from below `first_higher_input_idx` going down, seach for the same output
        for connection in self.connections[..first_higher_input_idx].iter().rev() {
            if connection.output == output {
                return None;
            }
        }

        // no same output, insert new connection before first_higher_input_idx
        let mut new_network = self.clone();
        let initial_weight = rng.next_f32() - 0.5;
        new_network.connections.insert(
            first_higher_input_idx,
            Connection {
                input,
                output,
                weight: initial_weight,
                enabled: true,
                innovation_id,
            },
        );
        Some(new_network)
    }

    pub fn add_node(
        &self,
        innovation_id_1: usize,
        innovation_id_2: usize,
        rng: &MyRandom,
    ) -> NeatNetwork {
        let mut new_network = self.clone();
        let node_idx = new_network.num_hidden_nodes;
        new_network.num_hidden_nodes += 1;

        let connection_idx_to_split = rng.random_index(new_network.connections.len());
        new_network.connections[connection_idx_to_split].enabled = false;
        let connection = new_network.connections[connection_idx_to_split].clone();

        new_network.connections.push(Connection {
            input: connection.input,
            output: node_idx,
            weight: connection.weight,
            enabled: true,
            innovation_id: innovation_id_1,
        });
        new_network.connections.push(Connection {
            input: node_idx,
            output: connection.output,
            weight: connection.weight,
            enabled: true,
            innovation_id: innovation_id_2,
        });
        new_network
    }

    pub fn mutate(&self, rng: &MyRandom) -> NeatNetwork {
        let mut new_network = self.clone();
        for conn in &mut new_network.connections {
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
        new_network
    }

    /// assumes that self has higher fitness than other
    pub fn cross_with(&self, other: &NeatNetwork, rng: &MyRandom) -> NeatNetwork {
        let mut new_network = self.clone();

        for other_connection in &other.connections {
            // half of other's connections are not considered, whether they are present or not.
            // Given that non-matching connections are always dropped, this only reduces the
            // number of searches, without affecting correctness
            if rng.next_f32() < 0.5 {
                continue;
            }

            let index = new_network
                .connections
                .iter()
                .position(|c| c.innovation_id == other_connection.innovation_id);

            if let Some(index) = index {
                let new_connection = &mut new_network.connections[index];
                // if enabled in one parent, but disabled in the other
                if new_connection.enabled ^ other_connection.enabled
                    && rng.next_f32() < DISABLED_GENE_PROBABILITY
                {
                    new_connection.enabled = false
                }
                new_network.connections[index] = other_connection.clone();
            }
        }

        new_network
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-value))
    }
}
