//! Modular CLI shell (`args` pure core + `docs` + `walk` + `commands`).

pub(crate) mod args;
pub(crate) mod commands;
pub(crate) mod docs;
pub(crate) mod error;
pub(crate) mod walk;

use std::io::Write;
use std::process::ExitCode;

use args::{parse_args, Action};
use commands::{run_bench, run_clones, run_recommend, run_search, run_triage, CommandOutcome};
use docs::{render_capabilities, render_help, render_robot_docs, render_schema, version_string};
use error::{CliError, ExitKind};

pub(crate) fn run(args: &[String], env: &[(String, String)], stdin_is_tty: bool) -> ExitCode {
    let parsed = match parse_args(args, env, stdin_is_tty) {
        Ok(p) => p,
        Err(err) => {
            let _ = std::io::stderr().write_all(err.render().as_bytes());
            return err.kind.exit_code();
        }
    };

    for note in &parsed.notes {
        let _ = writeln!(std::io::stderr(), "{note}");
    }

    let static_outcome = |stdout: String| {
        Ok(CommandOutcome {
            exit: ExitKind::Success,
            stdout,
            stderr_notes: Vec::new(),
        })
    };

    let res: Result<CommandOutcome, CliError> = match parsed.action {
        Action::Help(topic) => static_outcome(render_help(topic)),
        Action::Version => static_outcome(version_string()),
        Action::Capabilities(mode) => static_outcome(render_capabilities(mode)),
        Action::RobotDocs(topic) => static_outcome(render_robot_docs(topic)),
        Action::Schema { command, mode } => static_outcome(render_schema(command, mode)),
        Action::Triage(cmd) => run_triage(&cmd),
        Action::Clones(cmd) => run_clones(&cmd),
        Action::Search(cmd) => run_search(&cmd),
        Action::Recommend(cmd) => run_recommend(&cmd),
        Action::Bench(cmd) => run_bench(&cmd),
    };

    let outcome = match res {
        Ok(out) => out,
        Err(err) => {
            let _ = std::io::stderr().write_all(err.render().as_bytes());
            return err.kind.exit_code();
        }
    };

    for note in &outcome.stderr_notes {
        let _ = writeln!(std::io::stderr(), "{note}");
    }
    let out_bytes = outcome.stdout.as_bytes();
    let _ = std::io::stdout().write_all(out_bytes);
    if !outcome.stdout.ends_with('\n') {
        let _ = std::io::stdout().write_all(b"\n");
    }
    outcome.exit.exit_code()
}
