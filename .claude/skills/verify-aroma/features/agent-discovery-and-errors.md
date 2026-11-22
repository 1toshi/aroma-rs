# Feature: Agent Discovery and Errors

## Sub-features
- `capabilities --json` machine-readable contract listing supported languages, commands, flags, resource limits, environment variables, and exit code semantics.
- `robot-docs guide` self-contained agent handbook with canonical command recipes and jq filters.
- `schema [--command <VERB>] [--json]` JSON Schema exports for `triage`, `clones`, `search`, `recommend`, `capabilities`, and `bench`.
- Deterministic exit taxonomy: `0` success, `1` `user-input-error` (with Levenshtein "Did you mean ...?" hints), `2` `safety-block` (reserved), and `3` `tool-environment-error` (missing/unreadable path or malformed manifest JSON).

## How to get to it (user POV)
- Run `./target/release/aroma capabilities --json` to inspect commands, flags, limits, and exit codes before scripting calls.
- Run `./target/release/aroma schema --command triage --json` or `./target/release/aroma robot-docs guide` to inspect exact JSON envelope schemas.
- Run `./target/release/aroma triag` (typo) or `./target/release/aroma clones /nonexistent --json` to observe exit `1` suggestions and exit `3` environment errors.

## Driving it with verify-aroma.sh
```bash
./.claude/skills/verify-aroma/scripts/verify-aroma.sh drive agent-discovery-and-errors proof-discovery
```
The runner asserts `.ok == true` on `capabilities --json`, validates `robot-docs guide` non-empty stdout and `schema --command triage --json` fields, verifies `aroma triag` exits `1` with a `triage` hint on stderr, and verifies `aroma clones /tmp/nonexistent-aroma-verify-path-999999 --json` exits `3`.

## Gotchas
- `aroma capabilities --json` and `aroma schema --json` ignore workspace paths and inspect the compiled CLI contract directly.
- Unknown `AROMA_*` environment variables emit a diagnostic warning to stderr unless `AROMA_NO_ENV_WARN=1` is set; `--json` stdout remains clean JSON.
- Exit code `2` is documented in `capabilities --json` for policy parity, though the CLI itself performs zero filesystem mutations.
