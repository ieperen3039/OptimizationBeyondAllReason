use std::hash::RandomState;
use crate::data;
use crate::data::BuildOptionId;
use crate::search_handler::LocalState;

trait Policy {
    fn get_next(&self, state: LocalState, sequence: Vec<BuildOptionId>) -> BuildOptionId;
}

/// not actually random because policies are stateless
struct PolicyRandom;

impl Policy for PolicyRandom {
    fn get_next(&self, state: LocalState, sequence: Vec<BuildOptionId>) -> BuildOptionId {
        let build_options = data::get_build_options(&state.has_built);
        let idx = sequence.len() % build_options.len();
        build_options.ids().nth(idx).unwrap()
    }
}
