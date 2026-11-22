mod cli;

use std::io::IsTerminal;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env: Vec<(String, String)> = std::env::vars().collect();
    let stdin_is_tty = std::io::stdin().is_terminal();
    cli::run(&args, &env, stdin_is_tty)
}
