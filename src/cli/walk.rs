//! Deterministic source tree walker and content fingerprinting.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use aroma::{builtin_languages, SourceFile};
use serde::Serialize;
use xxhash_rust::xxh3::Xxh3;

use crate::cli::error::CliError;

pub(crate) const DEFAULT_IGNORES: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "vendor",
    "vendored",
    "third_party",
    "dist",
    "build",
    ".next",
    "out",
    ".venv",
    "venv",
    "site-packages",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
];

const IGNORED_SUFFIXES: &[&str] = &[".min.js", ".min.mjs", ".bundle.js", ".d.ts"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WalkOptions {
    pub(crate) max_file_bytes: usize,
    pub(crate) extra_ignores: Vec<String>,
    pub(crate) lang_filter: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct WarningItem {
    pub(crate) code: String,
    pub(crate) message: String,
}

#[must_use]
#[derive(Debug, Clone)]
pub(crate) struct WalkResult {
    pub(crate) files: Vec<SourceFile>,
    pub(crate) fs_paths: Vec<PathBuf>,
    pub(crate) data_hash: String,
    pub(crate) language_counts: BTreeMap<String, usize>,
    pub(crate) total_bytes: usize,
    pub(crate) skipped_oversized: usize,
    pub(crate) skipped_non_utf8: usize,
    pub(crate) skipped_unsupported: usize,
    pub(crate) warnings: Vec<WarningItem>,
}

pub(crate) fn normalize_display_path(path: &Path) -> String {
    let raw = path.to_string_lossy().replace('\\', "/");
    let stripped = raw.strip_prefix("./").unwrap_or(&raw);
    if stripped.is_empty() {
        ".".to_string()
    } else {
        stripped.to_string()
    }
}

pub(crate) fn infer_language_for_path(path: &str) -> Option<&'static str> {
    let ext = Path::new(path).extension().and_then(|e| e.to_str())?;
    if ext.eq_ignore_ascii_case("rs") {
        Some("rust")
    } else if ext.eq_ignore_ascii_case("py") {
        Some("python")
    } else if ["ts", "tsx", "mts", "cts", "js", "jsx", "mjs", "cjs"]
        .iter()
        .any(|cand| ext.eq_ignore_ascii_case(cand))
    {
        Some("typescript")
    } else {
        None
    }
}

pub(crate) fn is_test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let filename = lower.rsplit('/').next().unwrap_or(&lower);
    lower.contains("/test/")
        || lower.contains("/tests/")
        || lower.contains("/__tests__/")
        || lower.contains("/benches/")
        || lower.contains("/fixtures/")
        || filename.starts_with("test_")
        || filename.ends_with("_test.py")
        || filename.ends_with("_test.rs")
        || filename.ends_with("_tests.rs")
        || filename.ends_with(".test.ts")
        || filename.ends_with(".test.tsx")
        || filename.ends_with(".test.js")
        || filename.ends_with(".test.jsx")
        || filename.ends_with(".spec.ts")
        || filename.ends_with(".spec.tsx")
        || filename.ends_with(".spec.js")
        || filename.ends_with(".spec.jsx")
}

fn is_ignored_name(name: &str, ignore_set: &HashSet<&str>) -> bool {
    ignore_set.contains(name)
}

fn collect_dir(
    dir: &Path,
    ignore_set: &HashSet<&str>,
    supported_exts: &[&str],
    opts: &WalkOptions,
    out: &mut Vec<(String, PathBuf)>,
    skipped_unsupported: &mut usize,
) -> Result<(), CliError> {
    let entries = fs::read_dir(dir).map_err(|e| {
        CliError::tool_env(
            format!("cannot read directory '{}': {e}", dir.display()),
            "Directory is missing, lacks read permissions, or was removed during scan",
            "ls -ld <path> && aroma triage . --json",
            "aroma capabilities --json",
        )
    })?;
    let mut children: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| {
            CliError::tool_env(
                format!("error iterating directory '{}': {e}", dir.display()),
                "Filesystem iteration failed while traversing directory entries",
                "aroma triage . --json",
                "aroma capabilities --json",
            )
        })?;
        children.push(entry.path());
    }
    children.sort();

    for path in children {
        let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if is_ignored_name(fname, ignore_set) {
            continue;
        }
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            collect_dir(
                &path,
                ignore_set,
                supported_exts,
                opts,
                out,
                skipped_unsupported,
            )?;
        } else if meta.is_file() {
            let disp = normalize_display_path(&path);
            let lower = disp.to_ascii_lowercase();
            if IGNORED_SUFFIXES.iter().any(|suf| lower.ends_with(suf)) {
                continue;
            }
            if !supported_exts.iter().any(|ext| lower.ends_with(ext)) {
                *skipped_unsupported += 1;
                continue;
            }
            if let Some(ref wanted) = opts.lang_filter
                && infer_language_for_path(&lower) != Some(wanted.as_str())
            {
                continue;
            }
            out.push((disp, path));
        }
    }
    Ok(())
}

/// Reads up to `cap` bytes safely using `Read::take(cap + 1)`. Returns `Ok(None)` if over cap.
pub(crate) fn read_file_bounded(path: &Path, cap: usize) -> Result<Option<Vec<u8>>, CliError> {
    let meta = fs::metadata(path).map_err(|e| {
        CliError::tool_env(
            format!("cannot stat file '{}': {e}", path.display()),
            "Target file does not exist or is unreadable",
            "ls -la <file> && aroma search --help",
            "aroma capabilities --json",
        )
    })?;
    if meta.len() > cap as u64 {
        return Ok(None);
    }
    let file = File::open(path).map_err(|e| {
        CliError::tool_env(
            format!("cannot open file '{}': {e}", path.display()),
            "Target file cannot be opened for reading",
            "ls -la <file> && aroma search --help",
            "aroma capabilities --json",
        )
    })?;
    let mut buf = Vec::new();
    let limit = (cap as u64).saturating_add(1);
    file.take(limit).read_to_end(&mut buf).map_err(|e| {
        CliError::tool_env(
            format!("cannot read file '{}': {e}", path.display()),
            "I/O failure while reading source file",
            "aroma triage . --json",
            "aroma capabilities --json",
        )
    })?;
    if buf.len() > cap {
        return Ok(None);
    }
    Ok(Some(buf))
}

/// Walks files/directories deterministically, reading bounded UTF-8 supported source files.
pub(crate) fn walk_paths(paths: &[PathBuf], opts: &WalkOptions) -> Result<WalkResult, CliError> {
    let default_paths = [PathBuf::from(".")];
    let roots: &[PathBuf] = if paths.is_empty() { &default_paths } else { paths };

    let mut ignore_set: HashSet<&str> = DEFAULT_IGNORES.iter().copied().collect();
    for extra in &opts.extra_ignores {
        ignore_set.insert(extra.as_str());
    }

    let langs = builtin_languages();
    let mut supported_exts: Vec<&str> = Vec::new();
    for l in &langs {
        for &ext in l.extensions() {
            supported_exts.push(ext);
        }
    }

    let mut candidates: Vec<(String, PathBuf)> = Vec::new();
    let mut skipped_unsupported = 0usize;

    for root in roots {
        let meta = fs::symlink_metadata(root).map_err(|e| {
            CliError::tool_env(
                format!("path '{}' cannot be accessed: {e}", root.display()),
                "Target path does not exist or lacks permissions",
                "ls -ld . && aroma --robot-triage .",
                "aroma capabilities --json",
            )
        })?;
        if meta.is_dir() {
            collect_dir(
                root,
                &ignore_set,
                &supported_exts,
                opts,
                &mut candidates,
                &mut skipped_unsupported,
            )?;
        } else if meta.is_file() {
            let disp = normalize_display_path(root);
            let lower = disp.to_ascii_lowercase();
            if IGNORED_SUFFIXES.iter().any(|suf| lower.ends_with(suf))
                || !supported_exts.iter().any(|ext| lower.ends_with(ext))
            {
                return Err(CliError::user_input(
                    format!("unsupported source file extension: '{}'", root.display()),
                    "Aroma supports Rust (.rs), Python (.py), and TypeScript/JavaScript (.ts/.tsx/.js/.jsx/.mts/.cts/.mjs/.cjs)",
                    format!("aroma triage {} --json", root.parent().unwrap_or_else(|| Path::new(".")).display()),
                    "aroma capabilities --json",
                ));
            }
            if let Some(ref wanted) = opts.lang_filter
                && infer_language_for_path(&lower) != Some(wanted.as_str())
            {
                continue;
            }
            candidates.push((disp, root.clone()));
        } else {
            return Err(CliError::tool_env(
                format!("path '{}' is neither a regular file nor directory", root.display()),
                "Only regular source files and directories can be indexed",
                "aroma --robot-triage .",
                "aroma capabilities --json",
            ));
        }
    }

    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    candidates.dedup_by(|a, b| a.0 == b.0);

    let mut files = Vec::new();
    let mut fs_paths = Vec::new();
    let mut language_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut skipped_oversized = 0usize;
    let mut skipped_non_utf8 = 0usize;
    let mut total_bytes = 0usize;
    let mut hasher = Xxh3::new();

    for (disp, full) in candidates {
        let Some(bytes) = read_file_bounded(&full, opts.max_file_bytes)? else {
            skipped_oversized += 1;
            continue;
        };
        let Ok(source) = String::from_utf8(bytes) else {
            skipped_non_utf8 += 1;
            continue;
        };
        if let Some(lang) = infer_language_for_path(&disp) {
            *language_counts.entry(lang.to_string()).or_insert(0) += 1;
        }
        total_bytes += source.len();
        hasher.update(disp.as_bytes());
        hasher.update(b"\0");
        hasher.update(source.as_bytes());
        hasher.update(b"\0");
        fs_paths.push(full);
        files.push(SourceFile { path: disp, source });
    }

    let mut warnings = Vec::new();
    if skipped_oversized > 0 {
        warnings.push(WarningItem {
            code: "oversized_files_skipped".to_string(),
            message: format!(
                "{skipped_oversized} file(s) exceeded --max-file-bytes ({}) and were skipped",
                opts.max_file_bytes
            ),
        });
    }
    if skipped_non_utf8 > 0 {
        warnings.push(WarningItem {
            code: "non_utf8_files_skipped".to_string(),
            message: format!("{skipped_non_utf8} non-UTF-8 file(s) were skipped"),
        });
    }

    Ok(WalkResult {
        files,
        fs_paths,
        data_hash: format!("{:016x}", hasher.digest()),
        language_counts,
        total_bytes,
        skipped_oversized,
        skipped_non_utf8,
        skipped_unsupported,
        warnings,
    })
}
