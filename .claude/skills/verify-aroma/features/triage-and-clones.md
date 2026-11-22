# Feature: Triage and Clones

## Sub-features
- Repository-wide and directory-scoped duplicate clustering across Rust (`.rs`), Python (`.py`), and TypeScript/JavaScript (`.ts`/`.tsx`/`.js`/`.jsx`) files.
- Mega-command `triage` (`--robot-triage`) returning `corpus`, `quick_ref`, `clone_clusters`, `hotspot_files`, and follow-up `commands` in deterministic JSON.
- `clones` subcommand returning `total_clusters`, `clusters`, `file_pairs`, and optional `--include-recommendations`.
- Test filtering (`--include-tests` opt-in vs default exclusion) and configurable thresholds (`--min-sim`, `--min-lines`, `--min-tokens`, `--top`).

## How to get to it (user POV)
- Run `./target/release/aroma --robot-triage <path>` for an immediate single-call summary of repo hotspots and clone clusters.
- Run `./target/release/aroma clones <path> --json` (or `--include-recommendations --json`) to inspect every detected clone cluster and file pair.

## Driving it with verify-aroma.sh
```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh drive triage-and-clones proof-triage
```
The script builds an isolated multi-language scratch corpus containing paired Rust, Python, and TypeScript near-clones under `/tmp/aroma-verify-work/proof-triage/fixture/`, executes `--robot-triage`, `clones --json`, and `clones --include-recommendations --json`, asserts `.ok == true`, `files_indexed >= 4`, `total_clusters >= 1`, and clean stderr separation, and stores raw streams in `/tmp/aroma-verify-proofs/proof-triage/`.

## Gotchas
- Default clone thresholds skip methods shorter than `--min-lines 7` or smaller than `--min-tokens 30`; trivial 3-line helpers need lowered thresholds to cluster.
- Test files and `tests/` directories are excluded by default unless `--include-tests` is passed.
- Output `data_hash` and cluster ordering are deterministic across thread counts (`--threads`); timestamps are omitted intentionally so `SOURCE_DATE_EPOCH` runs match byte-for-byte.
