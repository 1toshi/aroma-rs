//! Modular CLI shell (`args` pure core + `docs` + `walk` + `commands`).

pub mod args;
pub mod commands;
pub mod docs;
pub mod error;
pub mod walk;

use std::io::Write;
use std::process::ExitCode;

use args::{parse_args, Action};
use commands::{run_bench, run_clones, run_recommend, run_search, run_triage, CommandOutcome};
use docs::{render_capabilities, render_help, render_robot_docs, render_schema, version_string};
use error::ExitKind;

pub fn run(args: &[String], env: &[(String, String)], stdin_is_tty: bool) -> ExitCode {
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

    let outcome: CommandOutcome = match parsed.action {
        Action::Help(topic) => CommandOutcome {
            exit: ExitKind::Success,
            stdout: render_help(topic),
            stderr_notes: Vec::new(),
        },
        Action::Version => CommandOutcome {
            exit: ExitKind::Success,
            stdout: version_string(),
            stderr_notes: Vec::new(),
        },
        Action::Capabilities(mode) => CommandOutcome {
            exit: ExitKind::Success,
            stdout: render_capabilities(mode),
            stderr_notes: Vec::new(),
        },
        Action::RobotDocs(topic) => CommandOutcome {
            exit: ExitKind::Success,
            stdout: render_robot_docs(topic),
            stderr_notes: Vec::new(),
        },
        Action::Schema { command, mode } => CommandOutcome {
            exit: ExitKind::Success,
            stdout: render_schema(command, mode),
            stderr_notes: Vec::new(),
        },
        Action::Triage(cmd) => match run_triage(&cmd) {
            Ok(out) => out,
            Err(err) => {
                let _ = std::io::stderr().write_all(err.render().as_bytes());
                return err.kind.exit_code();
            }
        },
        Action::Clones(cmd) => match run_clones(&cmd) {
            Ok(out) => out,
            Err(err) => {
                let _ = std::io::stderr().write_all(err.render().as_bytes());
                return err.kind.exit_code();
            }
        },
        Action::Search(cmd) => match run_search(&cmd) {
            Ok(out) => out,
            Err(err) => {
                let _ = std::io::stderr().write_all(err.render().as_bytes());
                return err.kind.exit_code();
            }
        },
        Action::Recommend(cmd) => match run_recommend(&cmd) {
            Ok(out) => out,
            Err(err) => {
                let _ = std::io::stderr().write_all(err.render().as_bytes());
                return err.kind.exit_code();
            }
        },
        Action::Bench(cmd) => match run_bench(&cmd) {
            Ok(out) => out,
            Err(err) => {
                let _ = std::io::stderr().write_all(err.render().as_bytes());
                return err.kind.exit_code();
            }
        },
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
