use crate::build_option::BuildOption;
use crate::data::BuildOptionId::{ConstructionVehicleT2, EnergyConverter, SolarCollector, WindTurbine};
use crate::data::BuildSet;

mod data;
mod build_option;
mod search_handler;
mod brute_force_search;

fn main() {
    let result = search_handler::search();
    println!("Best sequence: {:?} in {} seconds", result.sequence.iter().map(|i| i.data()), result.time);
}

