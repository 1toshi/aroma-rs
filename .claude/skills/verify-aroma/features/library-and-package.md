# Feature: Library and Package

## Sub-features
- Core library crate (`aroma-rs`) built with `--no-default-features` (zero CLI/walker/serde runtime dependencies, pure Rust index/search/prune/recommend API).
- Deterministic seeded property and metamorphic suite (`tests/properties.rs`) verifying alpha-renaming invariance across Rust/Python/TypeScript, thread-count determinism, similarity monotonicity, Algorithm 1 pruning score conservation, and `min_support = 1` bounds.
- Crate packaging hygiene via `Cargo.toml` exclusions (ensuring `.claude/` verification files stay tracked in git while remaining excluded from published `.crate` tarballs).

## How to get to it (user POV)
- Run `cargo test --no-default-features --lib --offline` and `cargo test --no-default-features --test properties --offline` to validate pure library behavior without default CLI features.
- Run `cargo package --list --offline` to verify the exact file list included in a `crates.io` release archive.

## Driving it with verify-aroma.sh
```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh drive library-and-package proof-pkg
```
The runner invokes `cargo test --no-default-features --lib --offline`, `cargo test --no-default-features --test properties --offline`, and `cargo package --list --offline`, asserting exit `0` across all three commands and verifying no `.claude/` path appears in the package manifest.

## Gotchas
- Disabling default features removes CLI dependencies (`serde`, `serde_json`, `walkdir`, `aroma` binary), so binary targets and CLI integration tests require default features.
- `cargo package --list --offline` inspects tracked git state unless `--allow-dirty` is passed on an uncommitted tree; always verify on a clean or committed checkout when checking release readiness.
- Never relax compiler/Clippy lint settings in `Cargo.toml` to push an unverified change through library builds.
