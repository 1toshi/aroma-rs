---
name: verify-aroma
description: Verify the aroma CLI and library end to end across triage, clone detection, search, recommendations, agent discovery contracts, exit codes, and offline crate packaging.
---

# verify-aroma

Project-local verification skill and deterministic feature runner for `aroma-rs`.
See [features/README.md](features/README.md) for the complete feature map and per-feature expectations.

## Launch

Build the release binary offline if it is missing or out of date:

```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh launch
```

This runs `cargo build --release --offline` from the repository root and confirms `./target/release/aroma --version` reports `aroma 0.1.0 (aroma-rs)`.

## Doctor

Check that the environment, release binary, JSON tooling, and writable scratch space are healthy before driving any feature suite:

```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh doctor
```

Doctor checks:
1. `./target/release/aroma` exists and is executable (or override via `AROMA_BIN`).
2. `aroma --version` outputs `aroma 0.1.0 (aroma-rs)`.
3. `jq` is installed on `PATH`.
4. `aroma capabilities --json` succeeds (exit `0`) with `.ok == true`.
5. `/tmp` is writable for isolated fixtures (`/tmp/aroma-verify-work/`) and proof receipts (`/tmp/aroma-verify-proofs/`).

## Drive

Run a single feature suite or the full matrix against an isolated multi-language scratch corpus (`rust`, `python`, `typescript`):

```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh drive all proof-run-1
```

Available feature selectors:
- `triage-and-clones` — validates `--robot-triage`, `clones --json`, and `--include-recommendations`.
- `search-and-recommend` — validates `search` (`--snippet`, `--file/--lines`) and `recommend` (`--min-support 1`).
- `agent-discovery-and-errors` — validates `capabilities --json`, `robot-docs guide`, `schema --command triage --json`, exit `1` typo hints, and exit `3` missing paths.
- `library-and-package` — validates `--no-default-features` unit/property tests and `cargo package --list --offline` `.claude/` exclusion.
- `all` — executes all four feature suites in sequence under one `run-id`.

## Evidence

Every command executed by `drive` stores immutable proof artifacts under `/tmp/aroma-verify-proofs/<run-id>/`:
- `<step>/command.txt` — exact command line invoked.
- `<step>/stdout.txt` — raw stdout stream captured separately from stderr.
- `<step>/stderr.txt` — raw stderr stream captured separately from stdout.
- `<step>/exit_code.txt` — observed process exit code.
- `summary.json` — machine-readable pass/fail ledger across all executed steps and features.

Proof directories are never deleted by `cleanup` so verification receipts remain inspectable after disposable fixtures are removed.

## Cleanup

Remove the temporary fixture workspace while preserving all proof receipts:

```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh cleanup proof-run-1
```

This deletes `/tmp/aroma-verify-work/<run-id>/` and verifies `/tmp/aroma-verify-proofs/<run-id>/summary.json` still exists on disk.

## Helpers

- Runner script: [`scripts/verify-aroma.sh`](scripts/verify-aroma.sh)
- Feature index: [`features/README.md`](features/README.md)
- Optional overrides: set `REPO_ROOT` to target another checkout or `AROMA_BIN` to test a specific binary path.
