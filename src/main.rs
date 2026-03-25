
mod data;
mod build_option;
mod breadth_first_search;

fn main() {
    let result = breadth_first_search::search();
    println!("Best sequence: {:?} in {} seconds", result.sequence, result.time);
}

