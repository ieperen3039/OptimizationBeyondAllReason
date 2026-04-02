use crate::data;
use crate::data::BuildOptionId;
use crate::random::MyRandom;
use crate::search_handler::LocalState;

pub trait Policy {
    fn get_next(&self, state: &LocalState, sequence: &Vec<BuildOptionId>) -> BuildOptionId;
}

/// not actually random because policies are stateless
struct PolicyRandom {
    rng: MyRandom,
}

impl Policy for PolicyRandom {
    fn get_next(&self, state: &LocalState, _sequence: &Vec<BuildOptionId>) -> BuildOptionId {
        let build_options = data::get_build_options(&state.has_built);
        let idx = (self.rng.next_u32() as usize) % build_options.len();
        build_options.ids().nth(idx).unwrap()
    }
}
