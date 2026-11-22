//! Pure argv and environment parser with intent inference and allocation bounds.

use std::path::PathBuf;

use crate::cli::error::{closest_match, CliError};

pub const MAX_TOP: usize = 10_000;
pub const MAX_SEARCH_LIMIT: usize = 100_000;
pub const MAX_CLUSTER_LIMIT: usize = 10_000;
pub const MAX_THREADS: usize = 256;
pub const DEFAULT_MAX_FILE_BYTES: usize = 500_000;
pub const MAX_FILE_BYTES_CAP: usize = 100_000_000;
pub const MAX_SNIPPET_BYTES: usize = 10_000_000;
pub const MAX_LINE_SPEC_COUNT: usize = 100_000;

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "triage",
    "clones",
    "search",
    "recommend",
    "capabilities",
    "robot-docs",
    "schema",
    "bench",
    "help",
];

const KNOWN_ENV_VARS: &[&str] = &["AROMA_NO_ENV_WARN", "AROMA_ROBOT", "AROMA_THREADS"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Human,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestFilter {
    ExcludeTests,
    IncludeTests,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginFilter {
    ExcludeOrigin,
    IncludeOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecsMode {
    Omit,
    Include,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTopic {
    Root,
    Triage,
    Clones,
    Search,
    Recommend,
    Capabilities,
    RobotDocs,
    Schema,
    Bench,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocsTopic {
    Guide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaCommand {
    Triage,
    Clones,
    Search,
    Recommend,
    Capabilities,
    Bench,
}

impl SchemaCommand {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Triage => "triage",
            Self::Clones => "clones",
            Self::Search => "search",
            Self::Recommend => "recommend",
            Self::Capabilities => "capabilities",
            Self::Bench => "bench",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuerySource {
    Snippet(String),
    SnippetFile(PathBuf),
    FileLines { file: PathBuf, lines: Vec<u32> },
    Stdin,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TriageCmd {
    pub paths: Vec<PathBuf>,
    pub mode: OutputMode,
    pub threads: Option<usize>,
    pub top: usize,
    pub min_sim: f64,
    pub min_lines: u32,
    pub min_tokens: usize,
    pub lang: Option<String>,
    pub tests: TestFilter,
    pub ignores: Vec<String>,
    pub max_file_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClonesCmd {
    pub paths: Vec<PathBuf>,
    pub mode: OutputMode,
    pub threads: Option<usize>,
    pub top: usize,
    pub min_sim: f64,
    pub min_lines: u32,
    pub min_tokens: usize,
    pub lang: Option<String>,
    pub tests: TestFilter,
    pub recs: RecsMode,
    pub ignores: Vec<String>,
    pub max_file_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchCmd {
    pub paths: Vec<PathBuf>,
    pub query: QuerySource,
    pub mode: OutputMode,
    pub threads: Option<usize>,
    pub top: usize,
    pub search_limit: usize,
    pub min_sim: f64,
    pub origin: OriginFilter,
    pub lang: Option<String>,
    pub ignores: Vec<String>,
    pub max_file_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecommendCmd {
    pub paths: Vec<PathBuf>,
    pub query: QuerySource,
    pub mode: OutputMode,
    pub threads: Option<usize>,
    pub top: usize,
    pub min_sim: f64,
    pub min_support: usize,
    pub search_limit: usize,
    pub cluster_limit: usize,
    pub origin: OriginFilter,
    pub lang: Option<String>,
    pub ignores: Vec<String>,
    pub max_file_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchCmd {
    pub manifest: PathBuf,
    pub queries: PathBuf,
    pub threads: Option<usize>,
}

#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Help(HelpTopic),
    Version,
    Capabilities(OutputMode),
    RobotDocs(DocsTopic),
    Schema { command: Option<SchemaCommand>, mode: OutputMode },
    Triage(TriageCmd),
    Clones(ClonesCmd),
    Search(SearchCmd),
    Recommend(RecommendCmd),
    Bench(BenchCmd),
}

#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedRun {
    pub action: Action,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Subcmd {
    Triage,
    Clones,
    Search,
    Recommend,
    Capabilities,
    RobotDocs,
    Schema,
    Bench,
    Help,
}

impl Subcmd {
    const fn help_topic(self) -> HelpTopic {
        match self {
            Self::Triage => HelpTopic::Triage,
            Self::Clones => HelpTopic::Clones,
            Self::Search => HelpTopic::Search,
            Self::Recommend => HelpTopic::Recommend,
            Self::Capabilities => HelpTopic::Capabilities,
            Self::RobotDocs => HelpTopic::RobotDocs,
            Self::Schema => HelpTopic::Schema,
            Self::Bench => HelpTopic::Bench,
            Self::Help => HelpTopic::Root,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Triage => "triage",
            Self::Clones => "clones",
            Self::Search => "search",
            Self::Recommend => "recommend",
            Self::Capabilities => "capabilities",
            Self::RobotDocs => "robot-docs",
            Self::Schema => "schema",
            Self::Bench => "bench",
            Self::Help => "help",
        }
    }
}

fn is_truthy(val: &str) -> bool {
    matches!(val.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

fn resolve_subcommand(raw: &str, notes: &mut Vec<String>) -> Option<Subcmd> {
    match raw {
        "triage" => Some(Subcmd::Triage),
        "clones" => Some(Subcmd::Clones),
        "scan" | "dup" | "dups" | "duplicates" => {
            notes.push(format!("note: '{raw}' interpreted as 'clones' (alias)"));
            Some(Subcmd::Clones)
        }
        "search" => Some(Subcmd::Search),
        "find" => {
            notes.push("note: 'find' interpreted as 'search' (alias)".to_string());
            Some(Subcmd::Search)
        }
        "recommend" => Some(Subcmd::Recommend),
        "rec" | "suggest" => {
            notes.push(format!("note: '{raw}' interpreted as 'recommend' (alias)"));
            Some(Subcmd::Recommend)
        }
        "capabilities" => Some(Subcmd::Capabilities),
        "caps" | "info" => {
            notes.push(format!("note: '{raw}' interpreted as 'capabilities' (alias)"));
            Some(Subcmd::Capabilities)
        }
        "robot-docs" => Some(Subcmd::RobotDocs),
        "guide" | "docs" => {
            notes.push(format!("note: '{raw}' interpreted as 'robot-docs' (alias)"));
            Some(Subcmd::RobotDocs)
        }
        "schema" | "schemas" => Some(Subcmd::Schema),
        "bench" => Some(Subcmd::Bench),
        "help" => Some(Subcmd::Help),
        _ => None,
    }
}

pub fn normalize_language(raw: &str, notes: &mut Vec<String>) -> Result<String, CliError> {
    let lower = raw.to_ascii_lowercase();
    match lower.as_str() {
        "rust" | "python" | "typescript" => Ok(lower),
        "rs" => {
            notes.push("note: '--lang rs' interpreted as 'rust'".to_string());
            Ok("rust".to_string())
        }
        "py" => {
            notes.push("note: '--lang py' interpreted as 'python'".to_string());
            Ok("python".to_string())
        }
        "ts" | "tsx" | "js" | "jsx" | "mts" | "cts" | "mjs" | "cjs" | "javascript" => {
            notes.push(format!("note: '--lang {raw}' interpreted as 'typescript'"));
            Ok("typescript".to_string())
        }
        _ => {
            let try_lang = closest_match(&lower, &["rust", "python", "typescript"], 2)
                .unwrap_or("rust");
            Err(CliError::user_input(
                format!("unsupported language '{raw}'"),
                "Supported languages are 'rust', 'python', and 'typescript' (covering TS/JS/JSX/TSX)",
                format!("aroma search . --lang {try_lang} --snippet '<code...>' --json"),
                "aroma capabilities --json",
            ))
        }
    }
}

/// Parses 1-based line numbers, comma lists, and inclusive ranges (`10-25`) within `MAX_LINE_SPEC_COUNT`.
pub fn parse_line_spec(spec: &str) -> Result<Vec<u32>, CliError> {
    if spec.trim().is_empty() {
        return Err(CliError::user_input(
            "empty --lines specification",
            "Expected a line number or range such as '12', '10-25', or '10-15,20'",
            "aroma search . --file src/main.rs --lines 1-20 --json",
            "aroma search --help",
        ));
    }
    let mut lines = Vec::new();
    for part in spec.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(CliError::user_input(
                format!("invalid empty segment in --lines '{spec}'"),
                "Line lists must not contain trailing or consecutive commas",
                "aroma search . --file src/main.rs --lines 10-20,25 --json",
                "aroma search --help",
            ));
        }
        if let Some((a, b)) = trimmed.split_once('-') {
            let start: u32 = a.trim().parse().map_err(|_| {
                CliError::user_input(
                    format!("invalid line number '{a}' in --lines '{spec}'"),
                    "Line numbers must be positive 1-based integers",
                    "aroma search . --file src/main.rs --lines 10-25 --json",
                    "aroma search --help",
                )
            })?;
            let end: u32 = b.trim().parse().map_err(|_| {
                CliError::user_input(
                    format!("invalid line number '{b}' in --lines '{spec}'"),
                    "Line numbers must be positive 1-based integers",
                    "aroma search . --file src/main.rs --lines 10-25 --json",
                    "aroma search --help",
                )
            })?;
            if start == 0 || end < start {
                return Err(CliError::user_input(
                    format!("invalid line range '{trimmed}' in --lines '{spec}'"),
                    "Line ranges require 1 <= start <= end",
                    format!("aroma search . --file src/main.rs --lines {start}-{} --json", start.max(1) + 10),
                    "aroma search --help",
                ));
            }
            let span = (end - start) as usize + 1;
            if lines.len().saturating_add(span) > MAX_LINE_SPEC_COUNT {
                return Err(CliError::user_input(
                    format!(
                        "--lines range '{trimmed}' exceeds maximum cap of {MAX_LINE_SPEC_COUNT} lines"
                    ),
                    "Large bounds are rejected before allocating line vectors",
                    "aroma search . --file src/main.rs --lines 1-500 --json",
                    "aroma capabilities --json",
                ));
            }
            lines.extend(start..=end);
        } else {
            let single: u32 = trimmed.parse().map_err(|_| {
                CliError::user_input(
                    format!("invalid line number '{trimmed}' in --lines '{spec}'"),
                    "Line numbers must be positive 1-based integers",
                    "aroma search . --file src/main.rs --lines 15 --json",
                    "aroma search --help",
                )
            })?;
            if single == 0 {
                return Err(CliError::user_input(
                    "line number 0 is invalid in --lines",
                    "Line numbers are 1-based",
                    "aroma search . --file src/main.rs --lines 1-20 --json",
                    "aroma search --help",
                ));
            }
            if lines.len() >= MAX_LINE_SPEC_COUNT {
                return Err(CliError::user_input(
                    format!("--lines exceeds maximum cap of {MAX_LINE_SPEC_COUNT} lines"),
                    "Line lists are bounded before expanding",
                    "aroma search . --file src/main.rs --lines 1-500 --json",
                    "aroma capabilities --json",
                ));
            }
            lines.push(single);
        }
    }
    Ok(lines)
}

fn parse_usize_bounded(
    flag: &str,
    val: &str,
    min: usize,
    max: usize,
    subcmd: &str,
) -> Result<usize, CliError> {
    let n: usize = val.parse().map_err(|_| {
        CliError::user_input(
            format!("invalid integer '{val}' for {flag}"),
            format!("{flag} expects a whole number between {min} and {max}"),
            format!("aroma {subcmd} --help"),
            "aroma capabilities --json",
        )
    })?;
    if n < min || n > max {
        return Err(CliError::user_input(
            format!("{flag} value {n} is outside allowed range [{min}..={max}]"),
            "Numeric parameters are bounded to keep memory and runtime predictable",
            format!("aroma {subcmd} . {flag} {} --json", min.max(n.min(max))),
            "aroma capabilities --json",
        ));
    }
    Ok(n)
}

fn parse_u32_bounded(
    flag: &str,
    val: &str,
    min: u32,
    max: u32,
    subcmd: &str,
) -> Result<u32, CliError> {
    let n: u32 = val.parse().map_err(|_| {
        CliError::user_input(
            format!("invalid integer '{val}' for {flag}"),
            format!("{flag} expects a whole number between {min} and {max}"),
            format!("aroma {subcmd} --help"),
            "aroma capabilities --json",
        )
    })?;
    if n < min || n > max {
        return Err(CliError::user_input(
            format!("{flag} value {n} is outside allowed range [{min}..={max}]"),
            "Numeric parameters are bounded to prevent degenerate configurations",
            format!("aroma {subcmd} . {flag} {} --json", min.max(n.min(max))),
            "aroma capabilities --json",
        ));
    }
    Ok(n)
}

fn parse_unit_f64(flag: &str, val: &str, subcmd: &str) -> Result<f64, CliError> {
    let v: f64 = val.parse().map_err(|_| {
        CliError::user_input(
            format!("invalid number '{val}' for {flag}"),
            format!("{flag} expects a floating-point threshold in [0.0, 1.0]"),
            format!("aroma {subcmd} . {flag} 0.75 --json"),
            format!("aroma {subcmd} --help"),
        )
    })?;
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err(CliError::user_input(
            format!("{flag} value '{val}' must be within [0.0, 1.0]"),
            "Aroma Jaccard/overlap similarity ranges from 0.0 to 1.0",
            format!("aroma {subcmd} . {flag} 0.75 --json"),
            format!("aroma {subcmd} --help"),
        ));
    }
    Ok(v)
}

struct TokenStream<'a> {
    args: &'a [String],
    pos: usize,
    after_double_dash: bool,
}

impl<'a> TokenStream<'a> {
    fn new(args: &'a [String]) -> Self {
        Self { args, pos: 0, after_double_dash: false }
    }

    fn next_raw(&mut self) -> Option<&'a str> {
        let s = self.args.get(self.pos).map(String::as_str)?;
        self.pos += 1;
        Some(s)
    }

    fn take_val(&mut self, inline: Option<&str>, flag: &str, subcmd: &str) -> Result<String, CliError> {
        if let Some(v) = inline {
            return Ok(v.to_string());
        }
        if let Some(v) = self.next_raw() {
            return Ok(v.to_string());
        }
        Err(CliError::user_input(
            format!("flag '{flag}' requires a value"),
            format!("Provide the argument immediately after {flag} or with {flag}=<value>"),
            format!("aroma {subcmd} --help"),
            "aroma capabilities --json",
        ))
    }
}

fn split_flag(tok: &str) -> (&str, Option<&str>) {
    if tok.starts_with("--")
        && let Some((k, v)) = tok.split_once('=')
    {
        return (k, Some(v));
    }
    (tok, None)
}

fn unknown_flag_error(flag: &str, allowed: &[&str], subcmd: &str) -> CliError {
    if let Some(suggestion) = closest_match(flag, allowed, 2) {
        CliError::user_input(
            format!("unknown flag '{flag}' for '{subcmd}'"),
            format!("'{flag}' is not recognized; did you mean '{suggestion}'?"),
            format!("aroma {subcmd} {suggestion} --help"),
            format!("aroma {subcmd} --help (or 'aroma capabilities --json')"),
        )
    } else {
        CliError::user_input(
            format!("unknown flag '{flag}' for '{subcmd}'"),
            format!("'{flag}' is not a valid flag for '{subcmd}'"),
            format!("aroma {subcmd} --help"),
            "aroma capabilities --json",
        )
    }
}

fn check_env(env: &[(String, String)], notes: &mut Vec<String>) -> Result<(OutputMode, Option<usize>), CliError> {
    let suppress_warn = env
        .iter()
        .find(|(k, _)| k == "AROMA_NO_ENV_WARN")
        .is_some_and(|(_, v)| is_truthy(v));
    let mut default_mode = OutputMode::Human;
    let mut default_threads: Option<usize> = None;

    for (k, v) in env {
        if !k.starts_with("AROMA_") {
            continue;
        }
        match k.as_str() {
            "AROMA_ROBOT" => {
                if is_truthy(v) {
                    default_mode = OutputMode::Json;
                }
            }
            "AROMA_THREADS" => {
                if !v.trim().is_empty() {
                    let t = parse_usize_bounded("AROMA_THREADS", v.trim(), 1, MAX_THREADS, "triage")?;
                    default_threads = Some(t);
                }
            }
            "AROMA_NO_ENV_WARN" => {}
            _ => {
                if !suppress_warn {
                    if let Some(sug) = closest_match(k, KNOWN_ENV_VARS, 2) {
                        notes.push(format!(
                            "warning: unknown env var '{k}' — did you mean '{sug}'? (set AROMA_NO_ENV_WARN=1 to silence)"
                        ));
                    } else {
                        notes.push(format!(
                            "warning: unknown env var '{k}' (see 'aroma capabilities --json' for recognized env_vars)"
                        ));
                    }
                }
            }
        }
    }
    Ok((default_mode, default_threads))
}

#[derive(Default)]
struct CommonOpts {
    mode: Option<OutputMode>,
    threads: Option<usize>,
    top: Option<usize>,
    min_sim: Option<f64>,
    min_lines: Option<u32>,
    min_tokens: Option<usize>,
    search_limit: Option<usize>,
    cluster_limit: Option<usize>,
    min_support: Option<usize>,
    lang: Option<String>,
    tests: Option<TestFilter>,
    origin: Option<OriginFilter>,
    recs: Option<RecsMode>,
    ignores: Vec<String>,
    max_file_bytes: Option<usize>,
    snippet: Option<String>,
    snippet_file: Option<PathBuf>,
    file: Option<PathBuf>,
    lines: Option<Vec<u32>>,
    manifest: Option<PathBuf>,
    queries: Option<PathBuf>,
    positionals: Vec<String>,
}

fn parse_common_flag(
    flag: &str,
    inline: Option<&str>,
    ts: &mut TokenStream<'_>,
    subcmd: &str,
    opts: &mut CommonOpts,
    notes: &mut Vec<String>,
    allowed: &[&str],
) -> Result<Option<Action>, CliError> {
    match flag {
        "-h" | "--help" => Ok(Some(Action::Help(match subcmd {
            "triage" => HelpTopic::Triage,
            "clones" => HelpTopic::Clones,
            "search" => HelpTopic::Search,
            "recommend" => HelpTopic::Recommend,
            "capabilities" => HelpTopic::Capabilities,
            "robot-docs" => HelpTopic::RobotDocs,
            "schema" => HelpTopic::Schema,
            "bench" => HelpTopic::Bench,
            _ => HelpTopic::Root,
        }))),
        "-V" | "--version" => Ok(Some(Action::Version)),
        "--json" | "--robot" | "-j" => {
            opts.mode = Some(OutputMode::Json);
            Ok(None)
        }
        "--no-color" => Ok(None),
        "--format" => {
            let v = ts.take_val(inline, "--format", subcmd)?;
            match v.to_ascii_lowercase().as_str() {
                "json" => opts.mode = Some(OutputMode::Json),
                "human" | "text" => opts.mode = Some(OutputMode::Human),
                _ => {
                    return Err(CliError::user_input(
                        format!("unsupported --format value '{v}'"),
                        "Supported output formats are 'json' and 'human'",
                        format!("aroma {subcmd} --format json"),
                        format!("aroma {subcmd} --help"),
                    ));
                }
            }
            Ok(None)
        }
        "--threads" | "-t" => {
            let v = ts.take_val(inline, "--threads", subcmd)?;
            let n = parse_usize_bounded("--threads", &v, 1, MAX_THREADS, subcmd)?;
            opts.threads = Some(n);
            Ok(None)
        }
        "--top" | "--limit" | "-n" => {
            if !allowed.contains(&"--top") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            if flag != "--top" {
                notes.push(format!("note: '{flag}' interpreted as '--top' (alias)"));
            }
            let v = ts.take_val(inline, "--top", subcmd)?;
            opts.top = Some(parse_usize_bounded("--top", &v, 0, MAX_TOP, subcmd)?);
            Ok(None)
        }
        "--min-sim" => {
            if !allowed.contains(&"--min-sim") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--min-sim", subcmd)?;
            opts.min_sim = Some(parse_unit_f64("--min-sim", &v, subcmd)?);
            Ok(None)
        }
        "--min-lines" => {
            if !allowed.contains(&"--min-lines") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--min-lines", subcmd)?;
            opts.min_lines = Some(parse_u32_bounded("--min-lines", &v, 1, 10_000, subcmd)?);
            Ok(None)
        }
        "--min-tokens" => {
            if !allowed.contains(&"--min-tokens") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--min-tokens", subcmd)?;
            opts.min_tokens = Some(parse_usize_bounded("--min-tokens", &v, 1, 100_000, subcmd)?);
            Ok(None)
        }
        "--search-limit" => {
            if !allowed.contains(&"--search-limit") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--search-limit", subcmd)?;
            opts.search_limit = Some(parse_usize_bounded("--search-limit", &v, 1, MAX_SEARCH_LIMIT, subcmd)?);
            Ok(None)
        }
        "--cluster-limit" => {
            if !allowed.contains(&"--cluster-limit") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--cluster-limit", subcmd)?;
            opts.cluster_limit = Some(parse_usize_bounded("--cluster-limit", &v, 1, MAX_CLUSTER_LIMIT, subcmd)?);
            Ok(None)
        }
        "--min-support" => {
            if !allowed.contains(&"--min-support") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--min-support", subcmd)?;
            opts.min_support = Some(parse_usize_bounded("--min-support", &v, 1, MAX_CLUSTER_LIMIT, subcmd)?);
            Ok(None)
        }
        "--lang" | "--language" | "-l" => {
            if !allowed.contains(&"--lang") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            if flag != "--lang" {
                notes.push(format!("note: '{flag}' interpreted as '--lang' (alias)"));
            }
            let v = ts.take_val(inline, "--lang", subcmd)?;
            opts.lang = Some(normalize_language(&v, notes)?);
            Ok(None)
        }
        "--include-tests" => {
            if !allowed.contains(&"--include-tests") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            opts.tests = Some(TestFilter::IncludeTests);
            Ok(None)
        }
        "--include-origin" => {
            if !allowed.contains(&"--include-origin") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            opts.origin = Some(OriginFilter::IncludeOrigin);
            Ok(None)
        }
        "--include-recommendations" => {
            if !allowed.contains(&"--include-recommendations") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            opts.recs = Some(RecsMode::Include);
            Ok(None)
        }
        "--ignore" => {
            if !allowed.contains(&"--ignore") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--ignore", subcmd)?;
            opts.ignores.push(v);
            Ok(None)
        }
        "--max-file-bytes" => {
            if !allowed.contains(&"--max-file-bytes") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--max-file-bytes", subcmd)?;
            opts.max_file_bytes = Some(parse_usize_bounded(
                "--max-file-bytes",
                &v,
                1,
                MAX_FILE_BYTES_CAP,
                subcmd,
            )?);
            Ok(None)
        }
        "--snippet" => {
            if !allowed.contains(&"--snippet") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--snippet", subcmd)?;
            opts.snippet = Some(v);
            Ok(None)
        }
        "--snippet-file" => {
            if !allowed.contains(&"--snippet-file") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--snippet-file", subcmd)?;
            opts.snippet_file = Some(PathBuf::from(v));
            Ok(None)
        }
        "--file" => {
            if !allowed.contains(&"--file") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--file", subcmd)?;
            opts.file = Some(PathBuf::from(v));
            Ok(None)
        }
        "--lines" => {
            if !allowed.contains(&"--lines") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--lines", subcmd)?;
            opts.lines = Some(parse_line_spec(&v)?);
            Ok(None)
        }
        "--manifest" => {
            if !allowed.contains(&"--manifest") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--manifest", subcmd)?;
            opts.manifest = Some(PathBuf::from(v));
            Ok(None)
        }
        "--queries" => {
            if !allowed.contains(&"--queries") {
                return Err(unknown_flag_error(flag, allowed, subcmd));
            }
            let v = ts.take_val(inline, "--queries", subcmd)?;
            opts.queries = Some(PathBuf::from(v));
            Ok(None)
        }
        _ => Err(unknown_flag_error(flag, allowed, subcmd)),
    }
}

fn allowed_flags_for(subcmd: Subcmd) -> &'static [&'static str] {
    match subcmd {
        Subcmd::Triage => &[
            "--json", "--robot", "--format", "--no-color", "--threads", "--top", "--limit",
            "--min-sim", "--min-lines", "--min-tokens", "--lang", "--language", "--include-tests",
            "--ignore", "--max-file-bytes", "--help", "--version",
        ],
        Subcmd::Clones => &[
            "--json", "--robot", "--format", "--no-color", "--threads", "--top", "--limit",
            "--min-sim", "--min-lines", "--min-tokens", "--lang", "--language", "--include-tests",
            "--include-recommendations", "--ignore", "--max-file-bytes", "--help", "--version",
        ],
        Subcmd::Search => &[
            "--json", "--robot", "--format", "--no-color", "--threads", "--top", "--limit",
            "--search-limit", "--min-sim", "--include-origin", "--lang", "--language", "--snippet",
            "--snippet-file", "--file", "--lines", "--ignore", "--max-file-bytes", "--help",
            "--version",
        ],
        Subcmd::Recommend => &[
            "--json", "--robot", "--format", "--no-color", "--threads", "--top", "--limit",
            "--search-limit", "--cluster-limit", "--min-sim", "--min-support", "--include-origin",
            "--lang", "--language", "--snippet", "--snippet-file", "--file", "--lines", "--ignore",
            "--max-file-bytes", "--help", "--version",
        ],
        Subcmd::Capabilities => &["--json", "--robot", "--format", "--no-color", "--help", "--version"],
        Subcmd::RobotDocs => &["--no-color", "--help", "--version"],
        Subcmd::Schema => &["--command", "--json", "--robot", "--format", "--no-color", "--help", "--version"],
        Subcmd::Bench => &["--manifest", "--queries", "--threads", "--no-color", "--help", "--version"],
        Subcmd::Help => &["--help", "--version"],
    }
}

fn build_query_source(
    opts: &CommonOpts,
    subcmd: &str,
    stdin_is_tty: bool,
) -> Result<QuerySource, CliError> {
    let mut sources = 0usize;
    if opts.snippet.is_some() {
        sources += 1;
    }
    if opts.snippet_file.is_some() {
        sources += 1;
    }
    if opts.file.is_some() || opts.lines.is_some() {
        sources += 1;
    }
    if sources > 1 {
        return Err(CliError::user_input(
            "conflicting query inputs provided",
            "Specify exactly one of: --snippet <CODE>, --snippet-file <FILE>, or --file <FILE> --lines <SPEC>",
            format!("aroma {subcmd} . --snippet 'fn f(x: i32) -> i32 {{ x + 1 }}' --lang rust --json"),
            format!("aroma {subcmd} --help"),
        ));
    }
    if let Some(ref snip) = opts.snippet {
        if snip == "-" {
            return Ok(QuerySource::Stdin);
        }
        return Ok(QuerySource::Snippet(snip.clone()));
    }
    if let Some(ref sf) = opts.snippet_file {
        return Ok(QuerySource::SnippetFile(sf.clone()));
    }
    match (&opts.file, &opts.lines) {
        (Some(f), Some(l)) => Ok(QuerySource::FileLines { file: f.clone(), lines: l.clone() }),
        (Some(f), None) => Err(CliError::user_input(
            format!("'--file {}' requires '--lines <SPEC>'", f.display()),
            "Line-based search needs a 1-based line or range inside the indexed file",
            format!("aroma {subcmd} . --file {} --lines 1-25 --json", f.display()),
            format!("aroma {subcmd} --help"),
        )),
        (None, Some(_)) => Err(CliError::user_input(
            "'--lines' requires '--file <PATH>'",
            "Specify which indexed file contains those line numbers",
            format!("aroma {subcmd} . --file src/main.rs --lines 1-25 --json"),
            format!("aroma {subcmd} --help"),
        )),
        (None, None) => {
            if !stdin_is_tty {
                Ok(QuerySource::Stdin)
            } else {
                Err(CliError::user_input(
                    format!("'{subcmd}' requires a query input"),
                    "Provide --snippet <CODE>, --snippet-file <FILE>, or --file <FILE> --lines <SPEC>",
                    format!("aroma {subcmd} . --snippet 'return a + b' --lang python --json"),
                    format!("aroma {subcmd} --help"),
                ))
            }
        }
    }
}

/// Parses CLI arguments and relevant environment variables into an [`Action`].
pub fn parse_args(
    args: &[String],
    env: &[(String, String)],
    stdin_is_tty: bool,
) -> Result<ParsedRun, CliError> {
    let mut notes = Vec::new();
    let (env_mode, env_threads) = check_env(env, &mut notes)?;

    if args.is_empty() {
        return Ok(ParsedRun {
            action: Action::Help(HelpTopic::Root),
            notes,
        });
    }

    let mut ts = TokenStream::new(args);
    let mut pre_opts = CommonOpts::default();
    let mut chosen_sub: Option<Subcmd> = None;

    while let Some(raw) = ts.next_raw() {
        if raw == "--" {
            ts.after_double_dash = true;
            break;
        }
        let (flag, inline) = split_flag(raw);
        if flag == "--robot-triage" {
            pre_opts.mode = Some(OutputMode::Json);
            chosen_sub = Some(Subcmd::Triage);
            break;
        }
        if flag.starts_with('-') {
            match flag {
                "-h" | "--help" => {
                    return Ok(ParsedRun {
                        action: Action::Help(HelpTopic::Root),
                        notes,
                    });
                }
                "-V" | "--version" => {
                    return Ok(ParsedRun {
                        action: Action::Version,
                        notes,
                    });
                }
                "--json" | "--robot" | "-j" | "--no-color" | "--format" | "--threads" | "-t" => {
                    if let Some(act) = parse_common_flag(
                        flag,
                        inline,
                        &mut ts,
                        "triage",
                        &mut pre_opts,
                        &mut notes,
                        &[
                            "--json", "--robot", "--format", "--no-color", "--threads", "--help",
                            "--version",
                        ],
                    )? {
                        return Ok(ParsedRun { action: act, notes });
                    }
                }
                _ => {
                    let root_allowed = &[
                        "--robot-triage", "--json", "--robot", "--format", "--no-color",
                        "--threads", "--help", "--version",
                    ];
                    return Err(unknown_flag_error(flag, root_allowed, "triage"));
                }
            }
        } else {
            chosen_sub = match resolve_subcommand(raw, &mut notes) {
                Some(s) => Some(s),
                None => {
                    if let Some(sug) = closest_match(raw, KNOWN_SUBCOMMANDS, 2) {
                        return Err(CliError::user_input(
                            format!("unknown command '{raw}'"),
                            format!("'{raw}' is not an aroma subcommand; did you mean '{sug}'?"),
                            format!("aroma {sug} --help"),
                            "aroma --help (or 'aroma --robot-triage .')",
                        ));
                    }
                    return Err(CliError::user_input(
                        format!("unknown command '{raw}'"),
                        "Expected a subcommand (triage, clones, search, recommend, capabilities, robot-docs, schema, bench)",
                        "aroma --robot-triage .",
                        "aroma --help",
                    ));
                }
            };
            break;
        }
    }

    let Some(subcmd) = chosen_sub else {
        return Ok(ParsedRun {
            action: Action::Help(HelpTopic::Root),
            notes,
        });
    };

    if subcmd == Subcmd::Help {
        if let Some(topic_raw) = ts.next_raw() {
            if topic_raw == "--help" || topic_raw == "-h" {
                return Ok(ParsedRun {
                    action: Action::Help(HelpTopic::Root),
                    notes,
                });
            }
            if let Some(target) = resolve_subcommand(topic_raw, &mut notes) {
                return Ok(ParsedRun {
                    action: Action::Help(target.help_topic()),
                    notes,
                });
            }
            let try_sub = closest_match(topic_raw, KNOWN_SUBCOMMANDS, 2).unwrap_or("triage");
            return Err(CliError::user_input(
                format!("unknown help topic '{topic_raw}'"),
                "Provide a valid subcommand after 'help'",
                format!("aroma help {try_sub}"),
                "aroma --help",
            ));
        }
        return Ok(ParsedRun {
            action: Action::Help(HelpTopic::Root),
            notes,
        });
    }

    let allowed = allowed_flags_for(subcmd);
    let mut opts = pre_opts;
    let mut schema_cmd: Option<SchemaCommand> = None;

    while let Some(raw) = ts.next_raw() {
        if !ts.after_double_dash && raw == "--" {
            ts.after_double_dash = true;
            continue;
        }
        if !ts.after_double_dash && raw.starts_with('-') && raw != "-" {
            let (flag, inline) = split_flag(raw);
            if subcmd == Subcmd::Schema && flag == "--command" {
                let v = ts.take_val(inline, "--command", "schema")?;
                schema_cmd = Some(parse_schema_cmd(&v)?);
                continue;
            }
            if let Some(act) = parse_common_flag(
                flag,
                inline,
                &mut ts,
                subcmd.name(),
                &mut opts,
                &mut notes,
                allowed,
            )? {
                return Ok(ParsedRun { action: act, notes });
            }
        } else {
            opts.positionals.push(raw.to_string());
        }
    }

    let mode = opts.mode.unwrap_or(env_mode);
    let threads = opts.threads.or(env_threads);

    let action = match subcmd {
        Subcmd::Triage => Action::Triage(TriageCmd {
            paths: opts.positionals.into_iter().map(PathBuf::from).collect(),
            mode,
            threads,
            top: opts.top.unwrap_or(10),
            min_sim: opts.min_sim.unwrap_or(0.75),
            min_lines: opts.min_lines.unwrap_or(7),
            min_tokens: opts.min_tokens.unwrap_or(30),
            lang: opts.lang,
            tests: opts.tests.unwrap_or(TestFilter::ExcludeTests),
            ignores: opts.ignores,
            max_file_bytes: opts.max_file_bytes.unwrap_or(DEFAULT_MAX_FILE_BYTES),
        }),
        Subcmd::Clones => Action::Clones(ClonesCmd {
            paths: opts.positionals.into_iter().map(PathBuf::from).collect(),
            mode,
            threads,
            top: opts.top.unwrap_or(25),
            min_sim: opts.min_sim.unwrap_or(0.75),
            min_lines: opts.min_lines.unwrap_or(7),
            min_tokens: opts.min_tokens.unwrap_or(30),
            lang: opts.lang,
            tests: opts.tests.unwrap_or(TestFilter::ExcludeTests),
            recs: opts.recs.unwrap_or(RecsMode::Omit),
            ignores: opts.ignores,
            max_file_bytes: opts.max_file_bytes.unwrap_or(DEFAULT_MAX_FILE_BYTES),
        }),
        Subcmd::Search => {
            let query = build_query_source(&opts, "search", stdin_is_tty)?;
            Action::Search(SearchCmd {
                paths: opts.positionals.into_iter().map(PathBuf::from).collect(),
                query,
                mode,
                threads,
                top: opts.top.unwrap_or(20),
                search_limit: opts.search_limit.unwrap_or(1000),
                min_sim: opts.min_sim.unwrap_or(0.0),
                origin: opts.origin.unwrap_or(OriginFilter::ExcludeOrigin),
                lang: opts.lang,
                ignores: opts.ignores,
                max_file_bytes: opts.max_file_bytes.unwrap_or(DEFAULT_MAX_FILE_BYTES),
            })
        }
        Subcmd::Recommend => {
            let query = build_query_source(&opts, "recommend", stdin_is_tty)?;
            Action::Recommend(RecommendCmd {
                paths: opts.positionals.into_iter().map(PathBuf::from).collect(),
                query,
                mode,
                threads,
                top: opts.top.unwrap_or(5),
                min_sim: opts.min_sim.unwrap_or(0.5),
                min_support: opts.min_support.unwrap_or(2),
                search_limit: opts.search_limit.unwrap_or(1000),
                cluster_limit: opts.cluster_limit.unwrap_or(100),
                origin: opts.origin.unwrap_or(OriginFilter::ExcludeOrigin),
                lang: opts.lang,
                ignores: opts.ignores,
                max_file_bytes: opts.max_file_bytes.unwrap_or(DEFAULT_MAX_FILE_BYTES),
            })
        }
        Subcmd::Capabilities => Action::Capabilities(mode),
        Subcmd::RobotDocs => {
            if let Some(topic) = opts.positionals.first()
                && topic != "guide"
            {
                return Err(CliError::user_input(
                    format!("unknown robot-docs topic '{topic}'"),
                    "Currently only the 'guide' handbook topic is supported",
                    "aroma robot-docs guide",
                    "aroma capabilities --json",
                ));
            }
            Action::RobotDocs(DocsTopic::Guide)
        }
        Subcmd::Schema => {
            if schema_cmd.is_none()
                && let Some(pos) = opts.positionals.first()
            {
                schema_cmd = Some(parse_schema_cmd(pos)?);
            }
            Action::Schema { command: schema_cmd, mode }
        }
        Subcmd::Bench => {
            let Some(manifest) = opts.manifest else {
                return Err(CliError::user_input(
                    "'bench' requires '--manifest <FILE>'",
                    "Provide a JSON array of {path, source} objects via --manifest",
                    "aroma bench --manifest corpus.json --queries queries.json --threads 8",
                    "aroma bench --help",
                ));
            };
            let Some(queries) = opts.queries else {
                return Err(CliError::user_input(
                    "'bench' requires '--queries <FILE>'",
                    "Provide a benchmark query suite JSON file via --queries",
                    format!(
                        "aroma bench --manifest {} --queries queries.json",
                        manifest.display()
                    ),
                    "aroma bench --help",
                ));
            };
            Action::Bench(BenchCmd { manifest, queries, threads })
        }
        Subcmd::Help => Action::Help(HelpTopic::Root),
    };

    Ok(ParsedRun { action, notes })
}

fn parse_schema_cmd(raw: &str) -> Result<SchemaCommand, CliError> {
    match raw {
        "triage" => Ok(SchemaCommand::Triage),
        "clones" => Ok(SchemaCommand::Clones),
        "search" => Ok(SchemaCommand::Search),
        "recommend" => Ok(SchemaCommand::Recommend),
        "capabilities" => Ok(SchemaCommand::Capabilities),
        "bench" => Ok(SchemaCommand::Bench),
        _ => {
            let candidates = ["triage", "clones", "search", "recommend", "capabilities", "bench"];
            let try_cmd = closest_match(raw, &candidates, 2).unwrap_or("triage");
            Err(CliError::user_input(
                format!("unknown schema command '{raw}'"),
                "Expected one of: triage, clones, search, recommend, capabilities, bench",
                format!("aroma schema --command {try_cmd} --json"),
                "aroma capabilities --json",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn bare_invocation_shows_root_help() {
        let r = parse_args(&[], &[], true).expect("bare ok");
        assert_eq!(r.action, Action::Help(HelpTopic::Root));
        assert!(r.notes.is_empty());
    }

    #[test]
    fn robot_triage_maps_to_json_triage_anywhere() {
        let r = parse_args(&s(&["--robot-triage", "./src", "--top=5"]), &[], false).expect("ok");
        match r.action {
            Action::Triage(cmd) => {
                assert_eq!(cmd.mode, OutputMode::Json);
                assert_eq!(cmd.top, 5);
                assert_eq!(cmd.paths, vec![PathBuf::from("./src")]);
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn aliases_emit_notes_and_route_cleanly() {
        let r = parse_args(&s(&["find", ".", "--snippet", "return a", "-l", "py", "-j"]), &[], true)
            .expect("alias ok");
        assert!(r.notes.iter().any(|n| n.contains("'find' interpreted as 'search'")));
        assert!(r.notes.iter().any(|n| n.contains("'--lang py' interpreted as 'python'")));
        match r.action {
            Action::Search(cmd) => {
                assert_eq!(cmd.mode, OutputMode::Json);
                assert_eq!(cmd.lang.as_deref(), Some("python"));
                assert_eq!(cmd.query, QuerySource::Snippet("return a".to_string()));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn typo_flag_and_subcommand_suggest_correction() {
        let err = parse_args(&s(&["clones", ".", "--jsno"]), &[], true).unwrap_err();
        assert_eq!(err.kind, crate::cli::error::ExitKind::UserInput);
        assert!(err.why.contains("did you mean '--json'"));

        let err2 = parse_args(&s(&["serach", "."]), &[], true).unwrap_err();
        assert_eq!(err2.kind, crate::cli::error::ExitKind::UserInput);
        assert!(err2.why.contains("did you mean 'search'"));
    }

    #[test]
    fn line_spec_validates_and_caps_before_allocation() {
        assert_eq!(parse_line_spec("12,15-17,20").expect("lines"), vec![12, 15, 16, 17, 20]);
        let err = parse_line_spec("1-9999999").unwrap_err();
        assert!(err.what.contains("exceeds maximum cap"));
        assert!(parse_line_spec("0").is_err());
        assert!(parse_line_spec("20-10").is_err());
    }

    #[test]
    fn env_typo_warns_and_aroma_robot_sets_json() {
        let env = vec![
            ("AROMA_ROBOT".to_string(), "1".to_string()),
            ("AROMA_THREDS".to_string(), "4".to_string()),
        ];
        let r = parse_args(&s(&["capabilities"]), &env, false).expect("env ok");
        assert_eq!(r.action, Action::Capabilities(OutputMode::Json));
        assert!(r.notes.iter().any(|n| n.contains("AROMA_THREADS")));
    }

    #[test]
    fn allocation_caps_reject_excessive_limits() {
        let err = parse_args(&s(&["clones", ".", "--top", "999999"]), &[], true).unwrap_err();
        assert!(err.what.contains("outside allowed range"));
        let err2 = parse_args(&s(&["triage", ".", "--threads", "9999"]), &[], true).unwrap_err();
        assert!(err2.what.contains("outside allowed range"));
    }
}
