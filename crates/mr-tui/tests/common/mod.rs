// Shared test infra — copied verbatim from `crates/mr-cli/tests/common/`
// (GPL-3.0, same-repo test infra). Declared ahead of the tests that use them
// (T5+); allow dead_code so `cargo clippy --all-targets -D warnings` passes
// until the guided-session tests consume them.
#![allow(dead_code)]

pub mod fake_provider;
pub mod turns;
