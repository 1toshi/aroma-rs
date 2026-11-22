# Feature: Search and Recommend

## Sub-features
- Structural code `search` from inline `--snippet`, `--snippet-file`, stdin (`--snippet -`), or `--file <FILE> --lines <SPEC>`.
- Phase II `prune` Algorithm 1 scoring (`search_limit`, `min_sim`, `top`) with `--include-origin` control when querying by `--file/--lines`.
- `recommend` structural clustering into reusable multi-line templates with configurable `--min-support` (including `min_support = 1` single-member edge case) and `--cluster-limit`.

## How to get to it (user POV)
- Search by inline snippet: `./target/release/aroma search ./src --snippet '<code fragment>' --lang rust --json`.
- Search by existing method span: `./target/release/aroma search ./src --file src/lib.rs --lines 1-40 --json`.
- Synthesize clone templates: `./target/release/aroma recommend ./src --snippet '<code fragment>' --lang rust --min-support 1 --json`.

## Driving it with verify-aroma.sh
```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh drive search-and-recommend proof-search
```
The driver queries the isolated fixture with `search --snippet`, `search --file .../alpha.rs --lines 1-14`, and `recommend --min-support 1 --min-sim 0.4`, asserting exit `0`, `.ok == true`, `.total >= 1` for searches, and `.recommendations | length >= 1` for recommendations without panicking.

## Gotchas
- Queries targeting `--snippet` require `--lang` whenever the indexed corpus mixes multiple languages (Rust, Python, TypeScript).
- Searching by `--file/--lines` excludes the query's own origin method unless `--include-origin` is specified.
- A valid `search` or `recommend` query returning `0` matches exits `0` with an empty list rather than treating empty results as an error.
