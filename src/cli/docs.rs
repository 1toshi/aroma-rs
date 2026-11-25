//! Help pages, `capabilities`, `robot-docs guide`, and JSON schemas.

use serde_json::{json, Value};

use crate::cli::args::{
    DocsTopic, HelpTopic, OutputMode, SchemaCommand, DEFAULT_MAX_FILE_BYTES, MAX_CLUSTER_LIMIT,
    MAX_FILE_BYTES_CAP, MAX_LINE_SPEC_COUNT, MAX_SEARCH_LIMIT, MAX_SNIPPET_BYTES, MAX_THREADS,
    MAX_TOP,
};
use crate::cli::error::ExitKind;
use crate::cli::walk::DEFAULT_IGNORES;

pub(crate) fn version_string() -> String {
    format!("aroma {} ({})", env!("CARGO_PKG_VERSION"), env!("CARGO_PKG_NAME"))
}

pub(crate) fn render_help(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Root => format!(
            "\
aroma {} — Structural code search, clone detection, and recommendation (Rust, Python, TS/JS)

USAGE:
  aroma <COMMAND> [OPTIONS] [PATH...]
  aroma --robot-triage [PATH...]

AGENT / AUTOMATION:
  --robot-triage [PATH...]     Mega-command: corpus stats + clone clusters + hotspots + next commands (JSON)
  capabilities --json          Machine-readable contract, commands, flags, limits, and exit codes
  robot-docs guide             Paste-ready agent handbook with canonical patterns and jq examples
  schema [--command <VERB>]    Export JSON schemas for command outputs

CORE COMMANDS:
  triage [PATH...]             Summarize duplicate clusters and file hotspots (alias for --robot-triage with --json)
  clones [PATH...]             Detect exact and near-duplicate code clones across a repository or paths
  search [PATH...]             Search for structurally similar methods via --snippet, --snippet-file, or --file/--lines
  recommend [PATH...]          Cluster structural matches into idiomatic code recommendations
  check (--diff | FILE[:LINES]...) [PATH...]
                               Lint changed code for existing functions that already do the same thing
  bench                        Run reproducible manifest/query benchmark suite (JSON output)
  help [COMMAND]               Show detailed help for a command

GLOBAL FLAGS:
  --json, --robot, -j          Emit structured deterministic JSON on stdout (no volatile timings)
  --format <json|human>        Select output format (default: human, or json when AROMA_ROBOT=1)
  --threads <N>, -t <N>        Worker thread count (1..={MAX_THREADS})
  --no-color                   Suppress color styling (aroma emits plain text without ANSI codes)
  -h, --help                   Show help
  -V, --version                Show version

FILES:
  Directories are walked skipping what git ignores (.gitignore files up to the repository root and
  .git/info/exclude; not the global gitignore), the default_ignores directory names in
  `capabilities --json`, and any --ignore names. A file passed explicitly is always read.

EXIT CODES:
  0  success                   Query/scan succeeded (including valid queries with 0 results)
  1  user-input-error          Unknown flag/subcommand, invalid value, or missing required query input
  2  safety-block              Reserved policy block (aroma is read-only)
  3  tool-environment-error    Missing/unreadable path or malformed benchmark JSON file
  4  findings                  `check` found changed functions with likely existing equivalents (unless --no-fail)

ENV VARS:
  AROMA_ROBOT=1                Default to --json output across read commands
  AROMA_THREADS=<N>            Default worker threads (1..={MAX_THREADS})
  AROMA_NO_ENV_WARN=1          Suppress warnings for unrecognized AROMA_* env vars

EXAMPLES:
  aroma --robot-triage ./src
  aroma clones ./src --lang rust --min-sim 0.80 --json
  aroma search ./src --snippet 'values.iter().sum()' --lang rust --json
  aroma recommend ./src --file src/lib.rs --lines 10-30 --json
  aroma check --diff .
",
            env!("CARGO_PKG_VERSION")
        ),
        HelpTopic::Triage => format!(
            "\
aroma triage — Structural clone & hotspot mega-command (alias: `aroma --robot-triage`)

USAGE:
  aroma triage [OPTIONS] [PATH...]
  aroma --robot-triage [OPTIONS] [PATH...]

OPTIONS:
  --json, --robot, -j          Emit structured JSON mega-envelope
  --top <N>, --limit <N>, -n   Max clusters and hotspot files returned (default: 10, max: {MAX_TOP})
  --min-sim <0.0..1.0>         Minimum structural Jaccard similarity (default: 0.75)
  --min-lines <N>              Minimum method line span to analyze (default: 7)
  --min-tokens <N>             Minimum simplified AST feature count (default: 30)
  --lang <rust|python|ts>      Restrict indexing to one language
  --include-tests              Include test/bench files alongside production code
  --ignore <DIR_NAME>          Extra directory name to skip (repeatable)
  --max-file-bytes <N>         Per-file byte cap (default: {DEFAULT_MAX_FILE_BYTES}, max: {MAX_FILE_BYTES_CAP})
  --threads <N>, -t <N>        Worker thread count (1..={MAX_THREADS})
"
        ),
        HelpTopic::Clones => format!(
            "\
aroma clones — Detect exact and near-duplicate functions/methods

USAGE:
  aroma clones [OPTIONS] [PATH...]

OPTIONS:
  --json, --robot, -j          Emit structured deterministic JSON
  --top <N>, --limit <N>, -n   Max clusters and file pairs returned (default: 25, max: {MAX_TOP})
  --min-sim <0.0..1.0>         Minimum mutual near-clone similarity (default: 0.75)
  --min-lines <N>              Minimum method line count (default: 7)
  --min-tokens <N>             Minimum structural size (default: 30)
  --lang <rust|python|ts>      Restrict indexing to one language
  --include-tests              Include test/bench files
  --include-recommendations    Attach Aroma recommendation previews per cluster
  --ignore <DIR_NAME>          Extra directory name to skip (repeatable)
  --max-file-bytes <N>         Per-file byte cap (default: {DEFAULT_MAX_FILE_BYTES}, max: {MAX_FILE_BYTES_CAP})
  --threads <N>, -t <N>        Worker thread count (1..={MAX_THREADS})
"
        ),
        HelpTopic::Search => format!(
            "\
aroma search — Search indexed methods structurally by snippet or file lines

USAGE:
  aroma search [OPTIONS] [PATH...] (--snippet <CODE> | --snippet-file <FILE> | --file <FILE> --lines <SPEC>)

QUERY INPUTS (choose one):
  --snippet <CODE>             Inline code snippet (pass '-' to read from stdin, max {MAX_SNIPPET_BYTES} bytes)
  --snippet-file <FILE>        Read query snippet from file (infers --lang from extension)
  --file <FILE> --lines <SPEC> Use indexed method overlapping line(s), e.g. 12 or 10-25,30

OPTIONS:
  --json, --robot, -j          Emit structured deterministic JSON
  --lang <rust|python|ts>      Query language (auto-inferred when single-language corpus)
  --top <N>, --limit <N>, -n   Max hits returned (default: 20, max: {MAX_TOP})
  --search-limit <N>           Phase II pruning candidate pool (default: 1000, max: {MAX_SEARCH_LIMIT})
  --min-sim <0.0..1.0>         Minimum similarity threshold (default: 0.0)
  --include-origin             Keep self-match when using --file/--lines
  --ignore <DIR_NAME>          Extra directory name to skip (repeatable)
  --max-file-bytes <N>         Per-file byte cap (default: {DEFAULT_MAX_FILE_BYTES}, max: {MAX_FILE_BYTES_CAP})
  --threads <N>, -t <N>        Worker thread count (1..={MAX_THREADS})
"
        ),
        HelpTopic::Recommend => format!(
            "\
aroma recommend — Cluster structural matches into idiomatic code recommendations

USAGE:
  aroma recommend [OPTIONS] [PATH...] (--snippet <CODE> | --snippet-file <FILE> | --file <FILE> --lines <SPEC>)

OPTIONS:
  --json, --robot, -j          Emit structured deterministic JSON
  --lang <rust|python|ts>      Query language (auto-inferred when single-language corpus)
  --top <N>, --limit <N>, -n   Max recommendations returned (default: 5, max: {MAX_TOP})
  --min-sim <0.0..1.0>         Minimum cluster member similarity (default: 0.5)
  --min-support <N>            Minimum cluster size for a recommendation (default: 2, max: {MAX_CLUSTER_LIMIT})
  --search-limit <N>           Phase II pruning pool (default: 1000, max: {MAX_SEARCH_LIMIT})
  --cluster-limit <N>          Clustering candidate cap (default: 100, max: {MAX_CLUSTER_LIMIT})
  --include-origin             Allow query origin method in cluster support
  --ignore <DIR_NAME>          Extra directory name to skip (repeatable)
  --max-file-bytes <N>         Per-file byte cap (default: {DEFAULT_MAX_FILE_BYTES}, max: {MAX_FILE_BYTES_CAP})
  --threads <N>, -t <N>        Worker thread count (1..={MAX_THREADS})
"
        ),
        HelpTopic::Check => format!(
            "\
aroma check — Lint changed code: report EXISTING functions that already do the same thing

USAGE:
  aroma check --diff [<BASE>] [OPTIONS] [PATH...]
  aroma check <FILE>[:<LINES>]... [OPTIONS] [PATH...]

Each changed function (a function overlapping a changed line) becomes a structural query against
ONE index over PATH... (default: .; same walking/ignore rules as `search`). Existing functions with
similarity >= --min-sim are reported so the change can reuse them instead of duplicating them.
Functions overlapping changed lines in any changed file are never reported as matches, so new
code never matches other new code in the same change.

CHANGED CODE (choose one):
  --diff [<BASE>]              Changed lines from `git diff --unified=0 <BASE>` (default BASE: HEAD, i.e.
                               uncommitted staged + unstaged edits; untracked files count as fully changed).
                               BASE may be any revision (e.g. origin/main); also --diff=<BASE>.
                               A first positional that is not an existing path is taken as BASE.
  --staged                     With --diff: only staged changes (`git diff --cached <BASE>`)
  <FILE>[:<LINES>]             Explicit changed lines, e.g. src/a.ts:12 or src/a.ts:10-25,30; a bare FILE
                               means every line. Files of unsupported languages are skipped with a warning.

OPTIONS:
  --json, --robot, -j          Emit structured deterministic JSON
  --top <N>, --limit <N>, -n   Max existing matches per changed function (default: 3, max: {MAX_TOP})
  --min-sim <0.0..1.0>         Minimum similarity for a finding (default: 0.71, the \"reuse with minimal changes\" threshold)
  --min-size <N>               Skip changed functions with fewer structural features (default: 30; tiny queries match unrelated code)
  --min-coverage <0.0..1.0>    Minimum share of the EXISTING function the changed code accounts for (default: 0.25).
                               --min-sim measures how much of the changed code appears in the existing function;
                               this drops matches where that overlap is a small part of a much larger function.
  --search-limit <N>           Phase II pruning candidate pool (default: 1000, max: {MAX_SEARCH_LIMIT})
  --include-tests              Also report matches inside test files (*.test.*, *.spec.*, __tests__/, tests/, ...)
  --no-fail                    Exit 0 even when findings are reported
  --lang <rust|python|ts>      Restrict indexing (and checked files) to one language
  --ignore <DIR_NAME>          Extra directory name to skip (repeatable)
  --max-file-bytes <N>         Per-file byte cap (default: {DEFAULT_MAX_FILE_BYTES}, max: {MAX_FILE_BYTES_CAP})
  --threads <N>, -t <N>        Worker thread count (1..={MAX_THREADS})

EXIT CODES:
  0  no changed function has a likely existing equivalent (or --no-fail)
  1  invalid flags/arguments or bad --diff base revision
  3  unreadable path, or git unavailable / not a git work tree
  4  findings: at least one changed function matched existing code at --min-sim

EXAMPLES:
  aroma check --diff .
  aroma check --diff origin/main src
  aroma check src/services/account.ts:1007-1028 src --json
"
        ),
        HelpTopic::Capabilities => "\
aroma capabilities — Emit machine-readable CLI contract, languages, commands, limits, and exit codes

USAGE:
  aroma capabilities [--json]
"
        .to_string(),
        HelpTopic::RobotDocs => "\
aroma robot-docs — Print self-contained agent integration handbook

USAGE:
  aroma robot-docs [guide]
"
        .to_string(),
        HelpTopic::Schema => "\
aroma schema — Export JSON Schemas for command stdout envelopes

USAGE:
  aroma schema [--command <triage|clones|search|recommend|check|capabilities|bench>] [--json]
"
        .to_string(),
        HelpTopic::Bench => format!(
            "\
aroma bench — Run reproducible corpus manifest & query latency benchmark suite

USAGE:
  aroma bench --manifest <corpus.json> --queries <queries.json> [--threads <N>]

OPTIONS:
  --manifest <FILE>            JSON array of {{ \"path\": ..., \"source\": ... }} objects
  --queries <FILE>             JSON array of QuerySpec objects
  --threads <N>, -t <N>        Worker thread count (1..={MAX_THREADS})
"
        ),
    }
}

pub(crate) fn capabilities_json() -> Value {
    let mut exit_codes = serde_json::Map::new();
    for kind in ExitKind::ALL {
        exit_codes.insert(
            kind.code().to_string(),
            json!({
                "meaning": kind.as_str(),
                "retryable": kind.retryable(),
                "notes": kind.notes(),
            }),
        );
    }

    json!({
        "ok": true,
        "tool_name": "aroma",
        "package_name": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "api_version": 1,
        "contract_version": "1",
        "features": [
            "rust_frontend",
            "python_lezer_frontend",
            "typescript_javascript_lezer_frontend",
            "json_output",
            "mega_command_robot_triage",
            "levenshtein_typo_hints",
            "deterministic_output",
            "read_only"
        ],
        "languages": {
            "rust": {
                "extensions": [".rs"],
                "frontend": "native zero-dependency byte-slice lexer + local binder + recursive-descent/Pratt parser"
            },
            "python": {
                "extensions": [".py"],
                "frontend": "Lezer LR/GLR runtime"
            },
            "typescript": {
                "extensions": [".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"],
                "frontend": "Lezer LR/GLR runtime"
            }
        },
        "commands": {
            "triage": {
                "description": "Mega-command returning corpus stats, top clone clusters, hotspot files, and follow-up commands",
                "aliases": [],
                "mutates": false,
                "json": true,
                "robot": "--robot-triage",
                "is_mega_command": true,
                "args": [{"name": "paths", "required": false, "variadic": true, "default": "."}],
                "flags": ["--json", "--top", "--min-sim", "--min-lines", "--min-tokens", "--lang", "--include-tests", "--ignore", "--max-file-bytes", "--threads"],
                "exit_codes": [0, 1, 3],
                "schema_uri": "aroma schema --command triage --json"
            },
            "clones": {
                "description": "Detect exact and near-duplicate functions/methods and file pairs",
                "aliases": ["scan", "dup", "dups", "duplicates"],
                "mutates": false,
                "json": true,
                "robot": "--json",
                "is_mega_command": false,
                "args": [{"name": "paths", "required": false, "variadic": true, "default": "."}],
                "flags": ["--json", "--top", "--min-sim", "--min-lines", "--min-tokens", "--lang", "--include-tests", "--include-recommendations", "--ignore", "--max-file-bytes", "--threads"],
                "exit_codes": [0, 1, 3],
                "schema_uri": "aroma schema --command clones --json"
            },
            "search": {
                "description": "Structural code search from --snippet, --snippet-file, or --file/--lines",
                "aliases": ["find"],
                "mutates": false,
                "json": true,
                "robot": "--json",
                "is_mega_command": false,
                "args": [{"name": "paths", "required": false, "variadic": true, "default": "."}],
                "flags": ["--snippet", "--snippet-file", "--file", "--lines", "--lang", "--top", "--search-limit", "--min-sim", "--include-origin", "--json", "--ignore", "--max-file-bytes", "--threads"],
                "exit_codes": [0, 1, 3],
                "schema_uri": "aroma schema --command search --json"
            },
            "recommend": {
                "description": "Cluster structural matches into idiomatic multi-line recommendations",
                "aliases": ["rec", "suggest"],
                "mutates": false,
                "json": true,
                "robot": "--json",
                "is_mega_command": false,
                "args": [{"name": "paths", "required": false, "variadic": true, "default": "."}],
                "flags": ["--snippet", "--snippet-file", "--file", "--lines", "--lang", "--top", "--min-sim", "--min-support", "--search-limit", "--cluster-limit", "--include-origin", "--json", "--ignore", "--max-file-bytes", "--threads"],
                "exit_codes": [0, 1, 3],
                "schema_uri": "aroma schema --command recommend --json"
            },
            "check": {
                "description": "Lint changed code (git diff or FILE[:LINES]) for EXISTING functions that already do the same thing; exits 4 on findings",
                "aliases": [],
                "mutates": false,
                "json": true,
                "robot": "--json",
                "is_mega_command": false,
                "args": [
                    {"name": "targets", "required": false, "variadic": true, "description": "FILE[:LINES] changed-code targets (without --diff); a bare FILE means every line"},
                    {"name": "base", "required": false, "description": "With --diff: first positional that is not an existing path is the git base revision", "default": "HEAD"},
                    {"name": "paths", "required": false, "variadic": true, "default": "."}
                ],
                "flags": ["--diff", "--staged", "--top", "--min-sim", "--min-size", "--min-coverage", "--search-limit", "--include-tests", "--no-fail", "--lang", "--json", "--ignore", "--max-file-bytes", "--threads"],
                "flag_defaults": {
                    "--diff": {"value": "<BASE>", "default": "HEAD", "description": "Changed lines from `git diff --unified=0 <BASE>`; untracked files count as fully changed when BASE is HEAD"},
                    "--staged": {"value": null, "default": false, "description": "With --diff: only staged changes (`git diff --cached`)"},
                    "--top": {"value": "<N>", "default": 3, "max": MAX_TOP, "description": "Max existing matches per changed function"},
                    "--min-sim": {"value": "<0.0..1.0>", "default": 0.71, "description": "Minimum similarity for a finding (reuse-with-minimal-changes threshold)"},
                    "--min-size": {"value": "<N>", "default": 30, "description": "Skip changed functions with fewer structural features"},
                    "--min-coverage": {"value": "<0.0..1.0>", "default": 0.25, "description": "Minimum share of the existing function's features the changed code accounts for"},
                    "--search-limit": {"value": "<N>", "default": 1000, "max": MAX_SEARCH_LIMIT, "description": "Phase II pruning candidate pool"},
                    "--include-tests": {"value": null, "default": false, "description": "Also report matches inside test files"},
                    "--no-fail": {"value": null, "default": false, "description": "Exit 0 even when findings are reported"},
                    "--lang": {"value": "<rust|python|typescript>", "default": null, "description": "Restrict indexing and checked files to one language"},
                    "--ignore": {"value": "<DIR_NAME>", "default": [], "description": "Extra directory name to skip (repeatable)"},
                    "--max-file-bytes": {"value": "<N>", "default": DEFAULT_MAX_FILE_BYTES, "max": MAX_FILE_BYTES_CAP, "description": "Per-file byte cap"}
                },
                "exit_codes": [0, 1, 3, 4],
                "schema_uri": "aroma schema --command check --json"
            },
            "capabilities": {
                "description": "Machine-readable capability contract, commands, exit codes, and limits",
                "aliases": ["caps", "info"],
                "mutates": false,
                "json": true,
                "robot": "--json",
                "is_mega_command": false,
                "args": [],
                "flags": ["--json"],
                "exit_codes": [0, 1],
                "schema_uri": "aroma schema --command capabilities --json"
            },
            "robot-docs": {
                "description": "Self-contained paste-ready handbook for AI agents",
                "aliases": ["guide", "docs"],
                "mutates": false,
                "json": false,
                "robot": "robot-docs guide",
                "is_mega_command": false,
                "args": [{"name": "topic", "required": false, "default": "guide"}],
                "flags": [],
                "exit_codes": [0, 1]
            },
            "schema": {
                "description": "Export JSON Schema for aroma command envelopes",
                "aliases": ["schemas"],
                "mutates": false,
                "json": true,
                "robot": "--json",
                "is_mega_command": false,
                "args": [],
                "flags": ["--command", "--json"],
                "exit_codes": [0, 1]
            },
            "bench": {
                "description": "Run manifest/query benchmark suite (timings + parity checksum JSON)",
                "aliases": [],
                "mutates": false,
                "json": true,
                "robot": "default JSON",
                "is_mega_command": false,
                "args": [],
                "flags": ["--manifest", "--queries", "--threads"],
                "exit_codes": [0, 1, 3],
                "schema_uri": "aroma schema --command bench --json"
            }
        },
        "flags": {
            "global": {
                "--robot-triage": {
                    "description": "Mega-command shortcut equivalent to `aroma triage --json`",
                    "alias_for": "triage --json"
                },
                "--json": {
                    "description": "Emit deterministic JSON on stdout (aliases: --robot, -j, --format=json)"
                },
                "--threads": {
                    "description": "Worker thread count (alias: -t)"
                },
                "--no-color": {
                    "description": "Explicitly disable ANSI escape codes (never emitted in stdout/stderr)"
                }
            }
        },
        "exit_codes": Value::Object(exit_codes),
        "env_vars": {
            "AROMA_ROBOT": {"description": "Set to 1/true to default to --json output", "default": "unset"},
            "AROMA_THREADS": {"description": "Default worker thread count (1..=256)", "default": "available_parallelism"},
            "AROMA_NO_ENV_WARN": {"description": "Set to 1/true to suppress warnings for unknown AROMA_* env vars", "default": "unset"},
            "NO_COLOR": {"description": "Honored cleanly; zero ANSI escape codes emitted", "default": "unset"},
            "SOURCE_DATE_EPOCH": {"description": "Determinism-safe: aroma read commands emit zero timestamps", "default": "unset"}
        },
        "limits": {
            "max_top": MAX_TOP,
            "max_search_limit": MAX_SEARCH_LIMIT,
            "max_cluster_limit": MAX_CLUSTER_LIMIT,
            "max_threads": MAX_THREADS,
            "default_max_file_bytes": DEFAULT_MAX_FILE_BYTES,
            "max_file_bytes_cap": MAX_FILE_BYTES_CAP,
            "max_snippet_bytes": MAX_SNIPPET_BYTES,
            "max_line_spec_count": MAX_LINE_SPEC_COUNT
        },
        "default_ignores": DEFAULT_IGNORES,
        "respects_gitignore": true,
        "schemas_uri": "aroma schema --json",
        "robot_docs_uri": "aroma robot-docs guide"
    })
}

pub(crate) fn render_capabilities(mode: OutputMode) -> String {
    match mode {
        OutputMode::Json => {
            serde_json::to_string_pretty(&capabilities_json()).unwrap_or_else(|_| "{}".to_string())
        }
        OutputMode::Human => format!(
            "\
aroma {} (package: {})
Supported languages:
  - rust (.rs): native zero-dependency lexer + local binder + parser
  - python (.py): Lezer LR/GLR front end
  - typescript (.ts/.tsx/.mts/.cts/.js/.jsx/.mjs/.cjs): Lezer LR/GLR front end

Commands:
  --robot-triage / triage   Mega-call: stats + clone clusters + hotspots + next commands
  clones                    Detect exact and near-duplicate functions across files
  search                    Structural search by --snippet, --snippet-file, or --file/--lines
  recommend                 Cluster matches into idiomatic code recommendations
  check                     Lint changed code for existing equivalents (exit 4 on findings)
  capabilities              Full JSON contract (limits, flags, exit codes, schemas)
  robot-docs guide          Agent automation handbook
  schema                    JSON Schema definitions
  bench                     Manifest/query benchmark runner

Machine contract: aroma capabilities --json
",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_NAME")
        ),
    }
}

pub(crate) fn render_robot_docs(_topic: DocsTopic) -> String {
    "\
# aroma Agent Handbook (`aroma robot-docs guide`)

## 1. One-Call Repository Triage (Recommended First Step)
Run the mega-command to get corpus stats, top clone clusters, file hotspots, and copy-paste follow-up commands in one deterministic JSON call:

  aroma --robot-triage ./src
  aroma --robot-triage ./src | jq '.quick_ref, .commands'

## 2. Canonical Agent Workflows
- **Find duplicated / copy-pasted functions across a codebase**:
  aroma clones ./src --min-sim 0.80 --top 20 --json
  aroma clones ./src --lang rust --include-recommendations --json

- **Search structurally for code matching a snippet** (variable names alpha-rename cleanly):
  aroma search ./src --snippet 'let mut acc = 0; for x in items { acc += x; }' --lang rust --json
  echo 'return a + b' | aroma search ./src --lang python --json

- **Search & recommend replacements for an existing method by file + line range**:
  aroma search ./src --file src/lib.rs --lines 25-45 --json
  aroma recommend ./src --file src/lib.rs --lines 25-45 --json

- **Inspect contract & limits dynamically**:
  aroma capabilities --json
  aroma schema --command triage --json

## 3. Reuse Check Before Committing (exit 4 = likely duplicate found)
Before finishing a change, ask whether the new/edited functions duplicate code that already exists:

  aroma check --diff .                        # uncommitted edits (staged + unstaged + untracked) vs HEAD
  aroma check --diff origin/main src          # everything on this branch
  aroma check src/lib.rs:10-40 src --json | jq '.findings[] | {changed: .changed.name, reuse: [.matches[].method.name]}'

Each finding names a changed function and up to --top (default 3) EXISTING functions with similarity >= --min-sim (default 0.71) and coverage >= --min-coverage (default 0.25). Similarity is the share of the changed code found in the existing function; coverage is the share of the existing function the changed code accounts for, so a short helper whose pieces are scattered through a much larger function is not reported. Consider calling or extending those instead of keeping the new copy. Changed functions below --min-size (default 30 features) are skipped; test files are excluded from matches unless --include-tests; functions touched by the change never match each other. Exit `4` means findings were reported (`--no-fail` forces `0`).

## 4. Stream, Exit-Code, and Determinism Guarantees
- **Stdout vs. Stderr**: `--json` / `--robot-triage` write pure JSON strictly to stdout. Notes, deprecation/alias hints, env warnings, and errors go strictly to stderr.
- **Determinism**: `triage`, `clones`, `search`, `recommend`, `check`, `capabilities`, and `schema` sort paths/ties deterministically and omit wall-clock timestamps and timings from JSON output (`data_hash` fingerprints the exact indexed inputs).
- **Empty results**: Valid queries with 0 matches return exit code `0` with an empty list (`[]`) and a non-empty `hint` field.
- **Exit codes**:
  - `0` (`success`): completed (including 0-match queries)
  - `1` (`user-input-error`): invalid flag/command/range/language or missing required query input
  - `2` (`safety-block`): reserved
  - `3` (`tool-environment-error`): missing/unreadable path, invalid JSON input, or git unavailable for `check --diff`
  - `4` (`findings`): `check` reported changed functions with likely existing equivalents (suppress with `--no-fail`)
"
    .to_string()
}

fn schema_for_one(cmd: SchemaCommand) -> Value {
    match cmd {
        SchemaCommand::Triage => json!({
            "command": "triage",
            "stdout_schema": {
                "type": "object",
                "required": ["ok", "tool_version", "contract_version", "data_hash", "corpus", "quick_ref", "clone_clusters", "hotspot_files", "commands", "warnings"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "tool_version": {"type": "string"},
                    "contract_version": {"type": "string"},
                    "data_hash": {"type": "string"},
                    "corpus": {"type": "object"},
                    "quick_ref": {"type": "object"},
                    "clone_clusters": {"type": "array"},
                    "hotspot_files": {"type": "array"},
                    "commands": {"type": "array"},
                    "warnings": {"type": "array"},
                    "hint": {"type": ["string", "null"]}
                }
            }
        }),
        SchemaCommand::Clones => json!({
            "command": "clones",
            "stdout_schema": {
                "type": "object",
                "required": ["ok", "tool_version", "contract_version", "data_hash", "corpus", "total_clusters", "clusters", "file_pairs", "commands", "warnings"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "tool_version": {"type": "string"},
                    "contract_version": {"type": "string"},
                    "data_hash": {"type": "string"},
                    "corpus": {"type": "object"},
                    "total_clusters": {"type": "integer"},
                    "clusters": {"type": "array"},
                    "file_pairs": {"type": "array"},
                    "commands": {"type": "array"},
                    "warnings": {"type": "array"},
                    "hint": {"type": ["string", "null"]}
                }
            }
        }),
        SchemaCommand::Search => json!({
            "command": "search",
            "stdout_schema": {
                "type": "object",
                "required": ["ok", "tool_version", "contract_version", "data_hash", "corpus", "query", "total", "hits", "commands", "warnings"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "tool_version": {"type": "string"},
                    "contract_version": {"type": "string"},
                    "data_hash": {"type": "string"},
                    "corpus": {"type": "object"},
                    "query": {"type": "object"},
                    "total": {"type": "integer"},
                    "hits": {"type": "array"},
                    "commands": {"type": "array"},
                    "warnings": {"type": "array"},
                    "hint": {"type": ["string", "null"]}
                }
            }
        }),
        SchemaCommand::Recommend => json!({
            "command": "recommend",
            "stdout_schema": {
                "type": "object",
                "required": ["ok", "tool_version", "contract_version", "data_hash", "corpus", "query", "total", "recommendations", "commands", "warnings"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "tool_version": {"type": "string"},
                    "contract_version": {"type": "string"},
                    "data_hash": {"type": "string"},
                    "corpus": {"type": "object"},
                    "query": {"type": "object"},
                    "total": {"type": "integer"},
                    "recommendations": {"type": "array"},
                    "commands": {"type": "array"},
                    "warnings": {"type": "array"},
                    "hint": {"type": ["string", "null"]}
                }
            }
        }),
        SchemaCommand::Check => {
            let method_ref = json!({
                "type": "object",
                "required": ["path", "name", "language", "start_line", "end_line"],
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string"},
                    "language": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                }
            });
            // Built separately: nesting it inline exceeds `json!`'s macro recursion limit.
            let match_item = json!({
                "type": "object",
                "required": ["method", "similarity", "coverage", "matched_lines"],
                "properties": {
                    "method": method_ref,
                    "similarity": {"type": "number", "description": "share of the changed code's features found in this function"},
                    "coverage": {"type": "number", "description": "share of this function's features the changed code accounts for"},
                    "matched_lines": {"type": "array", "items": {"type": "integer"}}
                }
            });
            json!({
                "command": "check",
                "exit_codes": {"0": "no findings (or --no-fail)", "4": "at least one finding at --min-sim"},
                "stdout_schema": {
                    "type": "object",
                    "required": ["ok", "tool_version", "contract_version", "data_hash", "corpus", "source", "base", "changed_files", "checked", "skipped_small", "total", "findings", "commands", "warnings"],
                    "properties": {
                        "ok": {"type": "boolean"},
                        "tool_version": {"type": "string"},
                        "contract_version": {"type": "string"},
                        "data_hash": {"type": "string"},
                        "corpus": {"type": "object"},
                        "source": {"enum": ["git_diff", "file_lines"]},
                        "base": {"type": ["string", "null"], "description": "git base revision when source is git_diff"},
                        "changed_files": {"type": "integer", "description": "changed supported files resolved for checking"},
                        "checked": {"type": "integer", "description": "changed functions searched (size >= --min-size)"},
                        "skipped_small": {"type": "integer", "description": "changed functions skipped below --min-size"},
                        "total": {"type": "integer", "description": "number of findings"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["changed", "changed_lines", "query_size", "matches"],
                                "properties": {
                                    "changed": method_ref,
                                    "changed_lines": {"type": "array", "items": {"type": "integer"}},
                                    "query_size": {"type": "integer"},
                                    "matches": {"type": "array", "items": match_item}
                                }
                            }
                        },
                        "commands": {"type": "array"},
                        "warnings": {"type": "array"},
                        "hint": {"type": ["string", "null"]}
                    }
                }
            })
        }
        SchemaCommand::Capabilities => json!({
            "command": "capabilities",
            "stdout_schema": {
                "type": "object",
                "required": ["ok", "tool_name", "package_name", "version", "contract_version", "languages", "commands", "exit_codes", "env_vars", "limits"]
            }
        }),
        SchemaCommand::Bench => json!({
            "command": "bench",
            "stdout_schema": {
                "type": "object",
                "required": ["impl", "threads", "stats", "index_ms", "query_build_total_ms", "search_total_ms", "recommend_total_ms", "work_checksum", "errors"]
            }
        }),
    }
}

pub(crate) fn render_schema(command: Option<SchemaCommand>, _mode: OutputMode) -> String {
    let val = match command {
        Some(c) => schema_for_one(c),
        None => json!({
            "ok": true,
            "contract_version": "1",
            "schemas": {
                SchemaCommand::Triage.as_str(): schema_for_one(SchemaCommand::Triage),
                SchemaCommand::Clones.as_str(): schema_for_one(SchemaCommand::Clones),
                SchemaCommand::Search.as_str(): schema_for_one(SchemaCommand::Search),
                SchemaCommand::Recommend.as_str(): schema_for_one(SchemaCommand::Recommend),
                SchemaCommand::Check.as_str(): schema_for_one(SchemaCommand::Check),
                SchemaCommand::Capabilities.as_str(): schema_for_one(SchemaCommand::Capabilities),
                SchemaCommand::Bench.as_str(): schema_for_one(SchemaCommand::Bench),
            }
        }),
    };
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "{}".to_string())
}
