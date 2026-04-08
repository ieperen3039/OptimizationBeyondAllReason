use crate::search_handler::{LocalState, SearchResult};
use std::fmt::Display;
use std::sync::Arc;

pub trait Searcher {
    fn search(
        &mut self,
        initial_state: LocalState,
    ) -> SearchResult;

    fn new_progress_updater(&self) -> Arc<dyn Display + Send + Sync>;
}