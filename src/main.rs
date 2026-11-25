// The release binaries are static musl builds, and musl's allocator made `check` on a 7,000-file
// corpus about 3x slower (4.8 s vs 1.5 s). mimalloc brings it back to the glibc build's speed.
#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#[warn(unreachable_pub, clippy::pedantic)]
// CLI helper bounds/casts and flag structs legitimately trip pedantic noise lints.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::too_many_lines
)]
mod cli;

use std::io::IsTerminal;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env: Vec<(String, String)> = std::env::vars().collect();
    let stdin_is_tty = std::io::stdin().is_terminal();
    cli::run(&args, &env, stdin_is_tty)
}
