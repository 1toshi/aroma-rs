#![cfg(feature = "cli")]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

static DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let id = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("aroma_cli_test_{tag}_{pid}_{id}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn write(&self, rel: &str, contents: &str) -> PathBuf {
        let full = self.path.join(rel);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).expect("create parent dirs");
        }
        fs::write(&full, contents).expect("write fixture file");
        full
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn aroma_bin() -> &'static str {
    env!("CARGO_BIN_EXE_aroma")
}

fn run_aroma(args: &[&str], env: &[(&str, &str)], stdin_bytes: Option<&[u8]>) -> Output {
    let mut cmd = Command::new(aroma_bin());
    cmd.args(args)
        .env_remove("AROMA_ROBOT")
        .env_remove("AROMA_THREADS")
        .env_remove("AROMA_NO_ENV_WARN")
        .stdin(if stdin_bytes.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for &(k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn aroma");
    if let Some(bytes) = stdin_bytes {
        let mut stdin = child.stdin.take().expect("piped stdin");
        stdin.write_all(bytes).expect("write stdin");
    }
    child.wait_with_output().expect("wait aroma")
}

fn write_multi_lang_fixture(dir: &TempDir) {
    let rust_a = r#"
pub fn compute_total(items: &[i32]) -> i32 {
    let mut acc = 0;
    let mut count = 0;
    for &item in items {
        if item > 0 && item % 2 == 0 {
            acc += item * 3;
            count += 1;
        }
    }
    if count > 0 {
        acc / count
    } else {
        acc.max(0)
    }
}
"#;
    let rust_b = r#"
pub fn sum_scaled_positives(vals: &[i32]) -> i32 {
    let mut total = 0;
    let mut seen = 0;
    for &v in vals {
        if v > 0 && v % 2 == 0 {
            total += v * 3;
            seen += 1;
        }
    }
    if seen > 0 {
        total / seen
    } else {
        total.max(0)
    }
}
"#;
    let rust_test = r#"
#[test]
fn compute_total_check(vals: &[i32]) -> i32 {
    let mut total = 0;
    let mut seen = 0;
    for &v in vals {
        if v > 0 && v % 2 == 0 {
            total += v * 3;
            seen += 1;
        }
    }
    if seen > 0 {
        total / seen
    } else {
        total.max(0)
    }
}
"#;
    let py_file = r#"
def process_scores(scores):
    total = 0
    for s in scores:
        if s > 0:
            total += s * 2
    return total
"#;
    dir.write("src/a.rs", rust_a);
    dir.write("src/b.rs", rust_b);
    dir.write("tests/sample_test.rs", rust_test);
    dir.write("scripts/score.py", py_file);
}

#[test]
fn bare_help_and_version_exit_cleanly_without_ansi() {
    for args in [&[][..], &["--help"][..], &["-h"][..], &["help"][..], &["help", "search"][..], &["search", "--help"][..], &["--version"][..]] {
        let out = run_aroma(args, &[], None);
        assert_eq!(out.status.code(), Some(0), "failed for {args:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stdout.trim().is_empty(), "empty stdout for {args:?}");
        assert!(stderr.is_empty(), "unexpected stderr for {args:?}: {stderr}");
        assert!(!stdout.contains('\x1b'), "ANSI leaked in stdout for {args:?}");
    }
}

#[test]
fn capabilities_robot_docs_and_schema_contracts() {
    let caps = run_aroma(&["capabilities", "--json"], &[], None);
    assert_eq!(caps.status.code(), Some(0));
    assert!(caps.stderr.is_empty());
    let v: Value = serde_json::from_slice(&caps.stdout).expect("valid capabilities JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["tool_name"], "aroma");
    assert_eq!(v["package_name"], "aroma-rs");
    assert_eq!(v["contract_version"], "1");
    assert!(v["languages"]["rust"].is_object());
    assert!(v["languages"]["python"].is_object());
    assert!(v["languages"]["typescript"].is_object());
    for cmd in ["triage", "clones", "search", "recommend", "capabilities", "robot-docs", "schema", "bench"] {
        assert!(v["commands"][cmd].is_object(), "missing command {cmd}");
    }
    for ec in ["0", "1", "2", "3"] {
        assert!(v["exit_codes"][ec].is_object(), "missing exit code {ec}");
    }
    assert!(v["flags"]["global"]["--robot-triage"].is_object());
    assert!(v["limits"]["max_top"].as_u64().is_some());
    assert!(v["limits"]["max_line_spec_count"].as_u64().is_some());
    assert!(v["limits"]["max_snippet_bytes"].as_u64().is_some());

    let docs = run_aroma(&["robot-docs", "guide"], &[], None);
    assert_eq!(docs.status.code(), Some(0));
    let docs_txt = String::from_utf8_lossy(&docs.stdout);
    assert!(docs_txt.contains("--robot-triage"));

    let schema_all = run_aroma(&["schema", "--json"], &[], None);
    assert_eq!(schema_all.status.code(), Some(0));
    let sv: Value = serde_json::from_slice(&schema_all.stdout).expect("schema JSON");
    assert!(sv["schemas"]["triage"].is_object());

    let schema_one = run_aroma(&["schema", "--command", "triage", "--json"], &[], None);
    assert_eq!(schema_one.status.code(), Some(0));
    let sov: Value = serde_json::from_slice(&schema_one.stdout).expect("single schema JSON");
    assert_eq!(sov["command"], "triage");
}

#[test]
fn robot_triage_and_clones_are_deterministic_and_find_rust_clones() {
    let dir = TempDir::new("triage_clones");
    write_multi_lang_fixture(&dir);
    let root = dir.path.to_str().expect("utf8 path");

    let rt = run_aroma(&["--robot-triage", root], &[], None);
    let tj = run_aroma(&["triage", root, "--json"], &[], None);
    assert_eq!(rt.status.code(), Some(0));
    assert_eq!(tj.status.code(), Some(0));
    assert_eq!(rt.stdout, tj.stdout, "--robot-triage and triage --json must be byte-identical");

    let tv: Value = serde_json::from_slice(&rt.stdout).expect("triage json");
    assert_eq!(tv["ok"], true);
    assert!(tv["data_hash"].as_str().is_some_and(|h| h.len() == 16));
    assert!(tv["corpus"]["files_indexed"].as_u64().unwrap_or(0) >= 4);
    assert!(!tv["clone_clusters"].as_array().expect("clusters array").is_empty());
    assert!(!tv["hotspot_files"].as_array().expect("hotspots array").is_empty());
    assert!(!tv["commands"].as_array().expect("commands array").is_empty());

    // Without --include-tests: only src/a.rs and src/b.rs form the clone cluster
    let cl1 = run_aroma(&["clones", root, "--json"], &[], None);
    let cl2 = run_aroma(&["clones", root, "--json"], &[], None);
    assert_eq!(cl1.status.code(), Some(0));
    assert_eq!(cl1.stdout, cl2.stdout, "clones output must be deterministic");
    let cv: Value = serde_json::from_slice(&cl1.stdout).expect("clones json");
    let first_members = cv["clusters"][0]["members"].as_array().expect("members");
    assert_eq!(first_members.len(), 2);
    assert!(first_members[0]["language"] == "rust");
    assert!(cv["clusters"][0]["avg_sim"].as_f64().unwrap_or(0.0) >= 0.95);

    // With --include-tests: all 3 Rust methods (including tests/sample_test.rs) cluster together
    let cl_tests = run_aroma(&["clones", root, "--include-tests", "--json"], &[], None);
    assert_eq!(cl_tests.status.code(), Some(0));
    let ctv: Value = serde_json::from_slice(&cl_tests.stdout).expect("clones with tests json");
    assert_eq!(ctv["clusters"][0]["members"].as_array().expect("members").len(), 3);
}

#[test]
fn search_and_recommend_modes_and_empty_results() {
    let dir = TempDir::new("search_rec");
    write_multi_lang_fixture(&dir);
    let root = dir.path.to_str().expect("utf8 path");
    let rust_sub = dir.path.join("src");
    let rust_root = rust_sub.to_str().expect("utf8 path");

    // 1. Snippet search with --flag=value syntax and single-language auto-inference
    let sr = run_aroma(
        &[
            "search",
            rust_root,
            "--snippet=let mut sum = 0;\nlet mut cnt = 0;\nfor &n in nums {\n    if n > 0 && n % 2 == 0 {\n        sum += n * 3;\n        cnt += 1;\n    }\n}\nif cnt > 0 {\n    sum / cnt\n} else {\n    sum.max(0)\n}",
            "--json",
        ],
        &[],
        None,
    );
    assert_eq!(sr.status.code(), Some(0));
    let sv: Value = serde_json::from_slice(&sr.stdout).expect("search json");
    assert_eq!(sv["query"]["language"], "rust");
    assert!(sv["total"].as_u64().unwrap_or(0) >= 2);
    assert!(String::from_utf8_lossy(&sr.stderr).contains("inferred --lang rust"));

    // 2. --file + --lines search
    let a_path = dir.path.join("src/a.rs");
    let fl = run_aroma(
        &[
            "search",
            root,
            "--file",
            a_path.to_str().expect("utf8"),
            "--lines",
            "2-17",
            "--json",
        ],
        &[],
        None,
    );
    assert_eq!(fl.status.code(), Some(0));
    let flv: Value = serde_json::from_slice(&fl.stdout).expect("file lines json");
    assert_eq!(flv["query"]["origin"]["name"], "compute_total");
    assert!(flv["hits"].as_array().expect("hits").iter().any(|h| h["method"]["name"] == "sum_scaled_positives"));

    // 3. Piped stdin via --snippet -
    let st = run_aroma(
        &["search", root, "--snippet", "-", "--lang", "python", "--json"],
        &[],
        Some(b"def g(xs):\n    tot = 0\n    for x in xs:\n        if x > 0:\n            tot += x * 2\n    return tot\n"),
    );
    assert_eq!(st.status.code(), Some(0));
    let stv: Value = serde_json::from_slice(&st.stdout).expect("stdin search json");
    assert_eq!(stv["hits"][0]["method"]["name"], "process_scores");

    // 4. Recommend via snippet
    let rec = run_aroma(
        &[
            "recommend",
            root,
            "--snippet",
            "let mut s = 0;\nlet mut n = 0;\nfor &z in slice {\n    if z > 0 && z % 2 == 0 {\n        s += z * 3;\n    }\n}",
            "--lang",
            "rust",
            "--json",
        ],
        &[],
        None,
    );
    assert_eq!(rec.status.code(), Some(0));
    let rv: Value = serde_json::from_slice(&rec.stdout).expect("recommend json");
    assert!(rv["total"].as_u64().unwrap_or(0) >= 1);
    assert!(rv["recommendations"][0]["code"].as_str().is_some_and(|c| !c.is_empty()));

    // 4b. Recommend with --min-support 1 against single-method Python fixture must not panic
    let rec_single = run_aroma(
        &[
            "recommend",
            root,
            "--snippet",
            "total = 0",
            "--lang",
            "python",
            "--min-support",
            "1",
            "--json",
        ],
        &[],
        None,
    );
    assert_eq!(rec_single.status.code(), Some(0));
    let rsv: Value = serde_json::from_slice(&rec_single.stdout).expect("recommend single json");
    assert!(rsv["total"].as_u64().unwrap_or(0) >= 1);
    assert_eq!(rsv["recommendations"][0]["member_count"], 1);

    // 5. Zero-match query exits 0 with empty array and helpful hint
    let empty = run_aroma(
        &[
            "search",
            root,
            "--snippet",
            "match completely_unknown_token { Err(other) => other, Ok(v) => v }",
            "--lang",
            "rust",
            "--min-sim",
            "0.99",
            "--json",
        ],
        &[],
        None,
    );
    assert_eq!(empty.status.code(), Some(0));
    let ev: Value = serde_json::from_slice(&empty.stdout).expect("empty search json");
    assert_eq!(ev["total"], 0);
    assert!(ev["hits"].as_array().expect("hits").is_empty());
    assert!(ev["hint"].as_str().is_some_and(|h| !h.is_empty()));
}

#[test]
fn intent_inference_typo_errors_aliases_and_env_discipline() {
    let dir = TempDir::new("intent");
    write_multi_lang_fixture(&dir);
    let root = dir.path.to_str().expect("utf8");

    // Typo flag
    let bad_flag = run_aroma(&["clones", root, "--jsno"], &[], None);
    assert_eq!(bad_flag.status.code(), Some(1));
    assert!(bad_flag.stdout.is_empty());
    let bf_err = String::from_utf8_lossy(&bad_flag.stderr);
    assert!(bf_err.contains("did you mean '--json'"), "stderr: {bf_err}");
    assert!(bf_err.contains("try:"), "stderr: {bf_err}");

    // Typo subcommand
    let bad_sub = run_aroma(&["serach", root], &[], None);
    assert_eq!(bad_sub.status.code(), Some(1));
    let bs_err = String::from_utf8_lossy(&bad_sub.stderr);
    assert!(bs_err.contains("did you mean 'search'"), "stderr: {bs_err}");
    assert!(bs_err.contains("try:"), "stderr: {bs_err}");

    // Aliases + unknown AROMA_* env var warning stays in stderr while stdout stays valid JSON
    let alias_out = run_aroma(
        &["scan", root, "--format=json"],
        &[("AROMA_THREDS", "2")],
        None,
    );
    assert_eq!(alias_out.status.code(), Some(0));
    let _: Value = serde_json::from_slice(&alias_out.stdout).expect("pure JSON stdout");
    let al_err = String::from_utf8_lossy(&alias_out.stderr);
    assert!(al_err.contains("interpreted as 'clones'"));
    assert!(al_err.contains("AROMA_THREADS"));

    // Missing path -> exit 3 (tool-environment-error)
    let missing = Path::new("/nonexistent_aroma_test_path_xyz");
    let miss_out = run_aroma(&["clones", missing.to_str().expect("utf8")], &[], None);
    assert_eq!(miss_out.status.code(), Some(3));
    assert!(miss_out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&miss_out.stderr).contains("tool-environment-error"));

    // Oversized line range fails fast with exit 1
    let big_lines = run_aroma(
        &["search", root, "--file", "src/a.rs", "--lines", "1-999999"],
        &[],
        None,
    );
    assert_eq!(big_lines.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&big_lines.stderr).contains("exceeds maximum cap"));

    // Non-TTY / NO_COLOR / CI / closed stdin: zero ANSI codes, never hangs
    let non_tty = run_aroma(
        &["--robot-triage", root],
        &[("NO_COLOR", "1"), ("CI", "true"), ("TERM", "dumb")],
        None,
    );
    assert_eq!(non_tty.status.code(), Some(0));
    assert!(!String::from_utf8_lossy(&non_tty.stdout).contains('\x1b'));
    assert!(!String::from_utf8_lossy(&non_tty.stderr).contains('\x1b'));

    // Closed stdin with `search <dir>` (no --snippet) fails cleanly with user-input-error
    let closed_stdin = run_aroma(&["search", root], &[], None);
    assert_eq!(closed_stdin.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&closed_stdin.stderr).contains("empty query snippet"));
}

#[test]
fn bench_subcommand_preserves_json_output_contract() {
    let dir = TempDir::new("bench");
    let src = "pub fn add_two(a: i32, b: i32) -> i32 {\n    let s = a + b;\n    s\n}\n";
    let manifest_path = dir.write(
        "manifest.json",
        &serde_json::json!([{"path": "src/lib.rs", "source": src}]).to_string(),
    );
    let queries_path = dir.write(
        "queries.json",
        &serde_json::json!([{
            "id": 0,
            "language": "rust",
            "path": "src/lib.rs",
            "expected_name": "add_two",
            "start_line": 1,
            "end_line": 4,
            "changed_lines": [2],
            "snippet": "let sum = x + y;\nsum"
        }])
        .to_string(),
    );

    let out = run_aroma(
        &[
            "bench",
            "--manifest",
            manifest_path.to_str().expect("utf8"),
            "--queries",
            queries_path.to_str().expect("utf8"),
            "--threads",
            "1",
        ],
        &[],
        None,
    );
    assert_eq!(out.status.code(), Some(0));
    let v: Value = serde_json::from_slice(&out.stdout).expect("bench json");
    assert_eq!(v["impl"], "aroma-rs");
    assert_eq!(v["threads"], 1);
    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["errors"], 0);
    assert!(v["work_checksum"].as_u64().is_some());
}
