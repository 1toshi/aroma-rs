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
    for args in [&[][..], &["--help"][..], &["-h"][..], &["help"][..], &["help", "search"][..], &["search", "--help"][..], &["help", "check"][..], &["check", "--help"][..], &["--version"][..]] {
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
    for cmd in ["triage", "clones", "search", "recommend", "check", "capabilities", "robot-docs", "schema", "bench"] {
        assert!(v["commands"][cmd].is_object(), "missing command {cmd}");
    }
    for ec in ["0", "1", "2", "3", "4"] {
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

    let schema_check = run_aroma(&["schema", "--command", "check", "--json"], &[], None);
    assert_eq!(schema_check.status.code(), Some(0));
    let scv: Value = serde_json::from_slice(&schema_check.stdout).expect("check schema JSON");
    assert!(scv["stdout_schema"]["properties"]["findings"].is_object());
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

fn run_aroma_in(cwd: &Path, args: &[&str]) -> Output {
    Command::new(aroma_bin())
        .args(args)
        .current_dir(cwd)
        .env_remove("AROMA_ROBOT")
        .env_remove("AROMA_THREADS")
        .env_remove("AROMA_NO_ENV_WARN")
        .stdin(Stdio::null())
        .output()
        .expect("run aroma")
}

fn git(cwd: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(["-c", "user.name=aroma-test", "-c", "user.email=aroma@test.invalid", "-c", "commit.gpgsign=false"])
        .args(args)
        .current_dir(cwd)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("HOME", cwd)
        .output()
        .expect("spawn git");
    assert!(out.status.success(), "git {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
}

/// `sum_scaled_positives` from the multi-lang fixture, alpha-renamed under a new name.
const DUPLICATE_FN: &str = r#"
pub fn average_even_tripled(nums: &[i32]) -> i32 {
    let mut sum = 0;
    let mut n = 0;
    for &x in nums {
        if x > 0 && x % 2 == 0 {
            sum += x * 3;
            n += 1;
        }
    }
    if n > 0 {
        sum / n
    } else {
        sum.max(0)
    }
}
"#;

#[test]
fn walk_skips_files_git_ignores() {
    let dir = TempDir::new("gitignore");
    git(&dir.path, &["init", "-q"]);
    dir.write(".gitignore", "lib/\n");
    dir.write("pkg/.gitignore", "*.gen.rs\n");
    dir.write("pkg/src/real.rs", "pub fn one() -> i32 {\n    1\n}\n");
    // Build output and generated code that duplicate each other: indexing them would
    // report clones nobody can act on.
    dir.write("pkg/lib/out.rs", DUPLICATE_FN);
    dir.write("pkg/src/twin.gen.rs", DUPLICATE_FN);
    dir.write(".hidden/kept.rs", "pub fn two() -> i32 {\n    2\n}\n");

    for (cwd, path) in [(dir.path.clone(), "."), (dir.path.join("pkg"), "."), (dir.path.clone(), "pkg")] {
        let out = run_aroma_in(&cwd, &["clones", path, "--json"]);
        assert_eq!(out.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&out.stderr));
        let v: Value = serde_json::from_slice(&out.stdout).expect("clones json");
        let expected = if cwd == dir.path && path == "." { 2 } else { 1 };
        assert_eq!(v["corpus"]["files_indexed"], expected, "cwd={} path={path}: {v}", cwd.display());
        assert_eq!(v["total_clusters"], 0, "ignored duplicates were clustered: {v}");
    }
}

#[test]
fn check_file_lines_reports_existing_equivalent_and_gates_exit_code() {
    let dir = TempDir::new("check_lines");
    write_multi_lang_fixture(&dir);
    dir.write("src/small.rs", "pub fn one() -> i32 {\n    1\n}\n");
    let root = dir.path.to_str().expect("utf8");
    let a = dir.path.join("src/a.rs");
    let a_lines = format!("{}:2-17", a.to_str().expect("utf8"));

    // (a) human output names the existing near-identical function and exits 4.
    let human = run_aroma(&["check", &a_lines, root], &[], None);
    assert_eq!(human.status.code(), Some(4), "stderr: {}", String::from_utf8_lossy(&human.stderr));
    let txt = String::from_utf8_lossy(&human.stdout);
    assert!(txt.contains("compute_total"), "stdout: {txt}");
    assert!(txt.contains("consider reusing"), "stdout: {txt}");
    assert!(txt.contains("sum_scaled_positives"), "stdout: {txt}");
    assert!(txt.contains("1 changed function checked, 1 with likely existing equivalents"), "stdout: {txt}");

    // --no-fail keeps the report but exits 0.
    let no_fail = run_aroma(&["check", &a_lines, root, "--no-fail"], &[], None);
    assert_eq!(no_fail.status.code(), Some(0));
    assert_eq!(no_fail.stdout, human.stdout);

    // (d) JSON envelope with findings and matches at the default 0.71 threshold.
    let json = run_aroma(&["check", &a_lines, root, "--json"], &[], None);
    assert_eq!(json.status.code(), Some(4));
    let v: Value = serde_json::from_slice(&json.stdout).expect("check json");
    assert_eq!(v["ok"], true);
    assert_eq!(v["checked"], 1);
    assert_eq!(v["findings"][0]["changed"]["name"], "compute_total");
    assert_eq!(v["findings"][0]["matches"][0]["method"]["name"], "sum_scaled_positives");
    assert!(v["findings"][0]["matches"][0]["similarity"].as_f64().expect("sim") >= 0.71);

    // (e) matches inside test files are excluded unless --include-tests: here the only
    // equivalent of the changed function lives in tests/sample_test.rs.
    let tdir = TempDir::new("check_tests");
    let t_a = tdir.write("src/a.rs", &fs::read_to_string(&a).expect("read a.rs"));
    tdir.write("tests/sample_test.rs", &fs::read_to_string(dir.path.join("tests/sample_test.rs")).expect("read test"));
    let t_root = tdir.path.to_str().expect("utf8");
    let t_lines = format!("{}:2-17", t_a.to_str().expect("utf8"));
    let no_tests = run_aroma(&["check", &t_lines, t_root, "--json"], &[], None);
    assert_eq!(no_tests.status.code(), Some(0));
    let nv: Value = serde_json::from_slice(&no_tests.stdout).expect("check json without tests");
    assert_eq!(nv["checked"], 1);
    assert!(nv["findings"].as_array().expect("findings").is_empty(), "{}", nv["findings"]);
    let with_tests = run_aroma(&["check", &t_lines, t_root, "--include-tests", "--json"], &[], None);
    assert_eq!(with_tests.status.code(), Some(4));
    let wv: Value = serde_json::from_slice(&with_tests.stdout).expect("check json with tests");
    assert_eq!(wv["findings"][0]["matches"][0]["method"]["name"], "compute_total_check");
    assert_eq!(wv["findings"][0]["matches"][0]["method"]["path"].as_str().map(|p| p.ends_with("tests/sample_test.rs")), Some(true));

    // (b) a changed method below --min-size is skipped, not searched, and exits 0.
    let small = dir.path.join("src/small.rs");
    let small_out = run_aroma(&["check", small.to_str().expect("utf8"), root, "--json"], &[], None);
    assert_eq!(small_out.status.code(), Some(0));
    let sv: Value = serde_json::from_slice(&small_out.stdout).expect("small json");
    assert_eq!(sv["checked"], 0);
    assert_eq!(sv["skipped_small"], 1);
    assert!(sv["findings"].as_array().expect("findings").is_empty());
    let big_min = run_aroma(&["check", &a_lines, root, "--min-size", "100000"], &[], None);
    assert_eq!(big_min.status.code(), Some(0));
}

#[test]
fn check_diff_reports_preexisting_function_not_new_code() {
    let dir = TempDir::new("check_diff");
    write_multi_lang_fixture(&dir);
    git(&dir.path, &["init", "-q"]);
    git(&dir.path, &["add", "."]);
    git(&dir.path, &["commit", "-q", "-m", "corpus"]);

    // Uncommitted edit to a tracked file adds a duplicate; an untracked file adds another copy.
    let b = dir.path.join("src/b.rs");
    let mut b_src = fs::read_to_string(&b).expect("read b.rs");
    b_src.push_str(DUPLICATE_FN);
    fs::write(&b, b_src).expect("write b.rs");
    dir.write("src/fresh.rs", &DUPLICATE_FN.replace("average_even_tripled", "fresh_copy"));
    dir.write("notes.txt", "not code\n");

    let out = run_aroma_in(&dir.path, &["check", "--diff", ".", "--json"]);
    assert_eq!(out.status.code(), Some(4), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).expect("diff json");
    assert_eq!(v["source"], "git_diff");
    assert_eq!(v["base"], "HEAD");
    let findings = v["findings"].as_array().expect("findings");
    let changed: Vec<&str> = findings.iter().map(|f| f["changed"]["name"].as_str().expect("name")).collect();
    assert!(changed.contains(&"average_even_tripled"), "{changed:?}");
    assert!(changed.contains(&"fresh_copy"), "{changed:?}");
    // Unchanged pre-existing code in an edited file is not a changed function.
    assert!(!changed.contains(&"sum_scaled_positives"), "{changed:?}");
    for f in findings {
        let matched: Vec<&str> = f["matches"]
            .as_array()
            .expect("matches")
            .iter()
            .map(|m| m["method"]["name"].as_str().expect("name"))
            .collect();
        assert!(matched.contains(&"compute_total"), "{matched:?}");
        assert!(matched.contains(&"sum_scaled_positives"), "{matched:?}");
        assert!(!matched.contains(&"average_even_tripled"), "new code matched: {matched:?}");
        assert!(!matched.contains(&"fresh_copy"), "new code matched: {matched:?}");
    }
    assert!(
        v["warnings"].as_array().expect("warnings").iter().any(|w| w["code"] == "unsupported_changed_files"),
        "{}",
        v["warnings"]
    );

    // Staged-only mode sees nothing until the edit is staged.
    let staged = run_aroma_in(&dir.path, &["check", "--diff", "--staged", ".", "--json"]);
    assert_eq!(staged.status.code(), Some(0));
    let sv: Value = serde_json::from_slice(&staged.stdout).expect("staged json");
    assert_eq!(sv["changed_files"], 0);

    // Explicit base revision as a positional; an unknown revision is a user-input error.
    let head = run_aroma_in(&dir.path, &["check", "--diff", "HEAD", "."]);
    assert_eq!(head.status.code(), Some(4));
    let bad = run_aroma_in(&dir.path, &["check", "--diff=no-such-rev", "."]);
    assert_eq!(bad.status.code(), Some(1), "stderr: {}", String::from_utf8_lossy(&bad.stderr));
}
