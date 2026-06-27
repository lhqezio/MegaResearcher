//! Phase 1 smoke test: the research crate exists and its surface is reachable.
//! Real modules (orchestrator, worker, state, paper_chain, ...) arrive in
//! later phases; this test only asserts the crate compiles and links.
#[test]
fn research_crate_links() {
    // Force a reference to the crate so the linker exercises it.
    let _ = megaresearcher_research::CRATE_NAME;
}
