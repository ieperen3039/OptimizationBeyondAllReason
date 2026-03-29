use std::sync::Arc;
use crate::search_handler::{LocalState, SearchResult, SharedState};

pub trait Searcher {
    fn search(
        &mut self,
        shared_state: &Arc<SharedState>,
        initial_state: LocalState,
    ) -> SearchResult;
}