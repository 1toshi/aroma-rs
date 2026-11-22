#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
BIN="${AROMA_BIN:-$REPO_ROOT/target/release/aroma}"
WORK_BASE="/tmp/aroma-verify-work"
PROOF_BASE="/tmp/aroma-verify-proofs"

usage() {
  cat <<'EOF'
Usage: verify-aroma.sh <subcommand> [args...]

Subcommands:
  launch                      Build ./target/release/aroma (--release --offline) and verify --version
  doctor                      Validate binary, jq, capabilities JSON contract, and writable /tmp
  drive <feature|all> [id]    Run verification suite(s) and capture proofs under /tmp/aroma-verify-proofs/<id>
  cleanup <id>                Remove scratch fixture /tmp/aroma-verify-work/<id> while preserving proofs

Features for drive:
  triage-and-clones
  search-and-recommend
  agent-discovery-and-errors
  library-and-package
  all
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

record_step() {
  local proof_dir="$1"
  local step_name="$2"
  local expected_ec="$3"
  shift 3

  local step_dir="$proof_dir/$step_name"
  mkdir -p "$step_dir"
  printf '%q ' "$@" > "$step_dir/command.txt"
  printf '\n' >> "$step_dir/command.txt"

  local actual_ec=0
  "$@" > "$step_dir/stdout.txt" 2> "$step_dir/stderr.txt" || actual_ec=$?
  printf '%s\n' "$actual_ec" > "$step_dir/exit_code.txt"

  if [ "$actual_ec" != "$expected_ec" ]; then
    echo "FAIL [$step_name]: expected exit $expected_ec, got $actual_ec" >&2
    cat "$step_dir/stderr.txt" >&2 || true
    exit 1
  fi
}

cmd_launch() {
  cargo build --release --offline --manifest-path "$REPO_ROOT/Cargo.toml"
  [ -x "$BIN" ] || die "binary not executable after release build: $BIN"
  local ver
  ver="$("$BIN" --version)"
  echo "LAUNCHED: $BIN ($ver)"
}

cmd_doctor() {
  [ -x "$BIN" ] || die "binary not found or not executable: $BIN (run 'verify-aroma.sh launch' first)"
  local ver
  ver="$("$BIN" --version)"
  [ "$ver" = "aroma 0.1.0 (aroma-rs)" ] || die "unexpected --version output: $ver"
  command -v jq >/dev/null 2>&1 || die "jq not found on PATH"
  local caps
  caps="$("$BIN" capabilities --json)"
  printf '%s\n' "$caps" | jq -e '.ok == true and .package_name == "aroma-rs"' >/dev/null \
    || die "capabilities JSON contract check failed"
  mkdir -p "$WORK_BASE" "$PROOF_BASE"
  local probe
  probe="$(mktemp "$WORK_BASE/.doctor.XXXXXX")"
  rm -f "$probe"
  echo "DOCTOR OK: bin=$BIN version=\"$ver\" jq=$(command -v jq)"
}

create_fixture() {
  local fixture_dir="$1"
  mkdir -p "$fixture_dir/src"

  cat > "$fixture_dir/src/alpha.rs" <<'EOF'
pub fn score_alpha_series(values: &[i32], scale: i32, bias: i32) -> i32 {
    let mut total = bias;
    let mut streak = 0;
    for &item in values {
        let adjusted = item * scale;
        if adjusted > 10 {
            total += adjusted + streak;
            streak += 1;
        } else {
            total -= scale;
            streak = 0;
        }
    }
    total + streak
}

pub fn normalize_alpha_bucket(values: &[i32], limit: i32) -> i32 {
    let mut sum = 0;
    for &entry in values {
        if entry > limit {
            sum += limit;
        } else {
            sum += entry;
        }
    }
    sum
}
EOF

  cat > "$fixture_dir/src/beta.rs" <<'EOF'
pub fn score_beta_series(samples: &[i32], factor: i32, offset: i32) -> i32 {
    let mut acc = offset;
    let mut run = 0;
    for &sample in samples {
        let scaled = sample * factor;
        if scaled > 10 {
            acc += scaled + run;
            run += 1;
        } else {
            acc -= factor;
            run = 0;
        }
    }
    acc + run
}

pub fn normalize_beta_bucket(samples: &[i32], cap: i32) -> i32 {
    let mut acc = 0;
    for &val in samples {
        if val > cap {
            acc += cap;
        } else {
            acc += val;
        }
    }
    acc
}
EOF

  cat > "$fixture_dir/src/py_clones.py" <<'EOF'
def compute_alpha_window(items, weight, offset):
    total = offset
    bonus = 0
    for item in items:
        scaled = item * weight
        if scaled > 10:
            total += scaled + bonus
            bonus += 1
        else:
            total -= weight
            bonus = 0
    return total + bonus


def compute_beta_window(entries, mult, bias):
    acc = bias
    streak = 0
    for entry in entries:
        adjusted = entry * mult
        if adjusted > 10:
            acc += adjusted + streak
            streak += 1
        else:
            acc -= mult
            streak = 0
    return acc + streak
EOF

  cat > "$fixture_dir/src/ts_clones.ts" <<'EOF'
export function evalAlphaStream(items: number[], weight: number, offset: number): number {
  let total = offset;
  let bonus = 0;
  for (const item of items) {
    const scaled = item * weight;
    if (scaled > 10) {
      total += scaled + bonus;
      bonus += 1;
    } else {
      total -= weight;
      bonus = 0;
    }
  }
  return total + bonus;
}

export function evalBetaStream(entries: number[], mult: number, bias: number): number {
  let acc = bias;
  let streak = 0;
  for (const entry of entries) {
    const adjusted = entry * mult;
    if (adjusted > 10) {
      acc += adjusted + streak;
      streak += 1;
    } else {
      acc -= mult;
      streak = 0;
    }
  }
  return acc + streak;
}
EOF
}

drive_triage_and_clones() {
  local work_dir="$1"
  local proof_dir="$2"
  local fixture="$work_dir/fixture"

  record_step "$proof_dir" "triage_robot" 0 "$BIN" --robot-triage "$fixture"
  jq -e '.ok == true and (.clone_clusters | length) >= 1 and .corpus.files_indexed >= 4' \
    "$proof_dir/triage_robot/stdout.txt" >/dev/null || die "triage_robot assertions failed"
  [ ! -s "$proof_dir/triage_robot/stderr.txt" ] || die "triage_robot wrote unexpectedly to stderr"

  record_step "$proof_dir" "clones_json" 0 "$BIN" clones "$fixture" --json
  jq -e '.ok == true and .total_clusters >= 1 and (.clusters | length) >= 1' \
    "$proof_dir/clones_json/stdout.txt" >/dev/null || die "clones_json assertions failed"

  record_step "$proof_dir" "clones_with_recs" 0 "$BIN" clones "$fixture" --include-recommendations --json
  jq -e '.ok == true and .total_clusters >= 1' \
    "$proof_dir/clones_with_recs/stdout.txt" >/dev/null || die "clones_with_recs assertions failed"
}

drive_search_and_recommend() {
  local work_dir="$1"
  local proof_dir="$2"
  local fixture="$work_dir/fixture"
  local snippet='let mut total = bias; let mut streak = 0; for &item in values { let adjusted = item * scale; if adjusted > 10 { total += adjusted + streak; } }'

  record_step "$proof_dir" "search_snippet" 0 \
    "$BIN" search "$fixture" --snippet "$snippet" --lang rust --json
  jq -e '.ok == true and .total >= 1 and (.hits | length) >= 1' \
    "$proof_dir/search_snippet/stdout.txt" >/dev/null || die "search_snippet assertions failed"

  record_step "$proof_dir" "search_file_lines" 0 \
    "$BIN" search "$fixture" --file "$fixture/src/alpha.rs" --lines 1-14 --json
  jq -e '.ok == true and .total >= 1 and (.hits | length) >= 1' \
    "$proof_dir/search_file_lines/stdout.txt" >/dev/null || die "search_file_lines assertions failed"

  record_step "$proof_dir" "recommend_min_support_1" 0 \
    "$BIN" recommend "$fixture" --snippet "$snippet" --lang rust --min-support 1 --min-sim 0.4 --json
  jq -e '.ok == true and (.recommendations | length) >= 1' \
    "$proof_dir/recommend_min_support_1/stdout.txt" >/dev/null || die "recommend_min_support_1 assertions failed"
}

drive_agent_discovery_and_errors() {
  local proof_dir="$1"

  record_step "$proof_dir" "capabilities_json" 0 "$BIN" capabilities --json
  jq -e '.ok == true and (.commands | has("triage")) and (.exit_codes | has("3"))' \
    "$proof_dir/capabilities_json/stdout.txt" >/dev/null || die "capabilities assertions failed"

  record_step "$proof_dir" "robot_docs_guide" 0 "$BIN" robot-docs guide
  [ -s "$proof_dir/robot_docs_guide/stdout.txt" ] || die "robot-docs guide output was empty"

  record_step "$proof_dir" "schema_triage" 0 "$BIN" schema --command triage --json
  jq -e '.command == "triage" and (.stdout_schema.properties | has("clone_clusters"))' \
    "$proof_dir/schema_triage/stdout.txt" >/dev/null || die "schema_triage assertions failed"

  record_step "$proof_dir" "typo_exit_1" 1 "$BIN" triag
  grep -qi 'triage' "$proof_dir/typo_exit_1/stderr.txt" || die "typo_exit_1 missing Levenshtein suggestion for triage"

  record_step "$proof_dir" "missing_path_exit_3" 3 \
    "$BIN" clones "/tmp/nonexistent-aroma-verify-path-999999" --json
  grep -q 'tool-environment-error' "$proof_dir/missing_path_exit_3/stderr.txt" \
    || jq -e '.ok == false' "$proof_dir/missing_path_exit_3/stdout.txt" >/dev/null \
    || die "missing_path_exit_3 did not report tool-environment-error"
}

drive_library_and_package() {
  local proof_dir="$1"

  record_step "$proof_dir" "lib_no_default_features" 0 \
    cargo test --no-default-features --lib --offline --manifest-path "$REPO_ROOT/Cargo.toml"

  record_step "$proof_dir" "properties_no_default_features" 0 \
    cargo test --no-default-features --test properties --offline --manifest-path "$REPO_ROOT/Cargo.toml"

  record_step "$proof_dir" "package_list" 0 \
    cargo package --list --offline --manifest-path "$REPO_ROOT/Cargo.toml"
  if grep -qE '^\.claude/' "$proof_dir/package_list/stdout.txt"; then
    die "package_list unexpectedly includes .claude/ files"
  fi
}

cmd_drive() {
  local feature="${1:-}"
  local run_id="${2:-run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
  [ -n "$feature" ] || die "drive requires <feature|all>"

  cmd_doctor >/dev/null

  local work_dir="$WORK_BASE/$run_id"
  local proof_dir="$PROOF_BASE/$run_id"
  mkdir -p "$work_dir" "$proof_dir"
  create_fixture "$work_dir/fixture"

  local features_ran=()
  case "$feature" in
    triage-and-clones)
      drive_triage_and_clones "$work_dir" "$proof_dir"
      features_ran+=("triage-and-clones")
      ;;
    search-and-recommend)
      drive_search_and_recommend "$work_dir" "$proof_dir"
      features_ran+=("search-and-recommend")
      ;;
    agent-discovery-and-errors)
      drive_agent_discovery_and_errors "$proof_dir"
      features_ran+=("agent-discovery-and-errors")
      ;;
    library-and-package)
      drive_library_and_package "$proof_dir"
      features_ran+=("library-and-package")
      ;;
    all)
      drive_triage_and_clones "$work_dir" "$proof_dir"
      drive_search_and_recommend "$work_dir" "$proof_dir"
      drive_agent_discovery_and_errors "$proof_dir"
      drive_library_and_package "$proof_dir"
      features_ran+=("triage-and-clones" "search-and-recommend" "agent-discovery-and-errors" "library-and-package")
      ;;
    *)
      die "unknown feature '$feature' (expected triage-and-clones|search-and-recommend|agent-discovery-and-errors|library-and-package|all)"
      ;;
  esac

  local features_json
  features_json="$(printf '%s\n' "${features_ran[@]}" | jq -R . | jq -s .)"
  local steps_count
  steps_count="$(find "$proof_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"

  jq -n \
    --arg run_id "$run_id" \
    --arg feature "$feature" \
    --arg work_dir "$work_dir" \
    --arg proof_dir "$proof_dir" \
    --arg bin "$BIN" \
    --argjson features "$features_json" \
    --argjson steps_count "$steps_count" \
    '{
      ok: true,
      run_id: $run_id,
      requested_feature: $feature,
      features_passed: $features,
      steps_recorded: $steps_count,
      work_dir: $work_dir,
      proof_dir: $proof_dir,
      binary: $bin
    }' > "$proof_dir/summary.json"

  echo "DRIVE OK: feature=$feature steps=$steps_count"
  echo "PROOF_DIR=$proof_dir"
  echo "WORK_DIR=$work_dir"
}

cmd_cleanup() {
  local run_id="${1:-}"
  [ -n "$run_id" ] || die "cleanup requires <run-id>"
  local work_dir="$WORK_BASE/$run_id"
  local proof_dir="$PROOF_BASE/$run_id"

  rm -rf "$work_dir"
  [ ! -e "$work_dir" ] || die "failed to remove scratch workspace: $work_dir"
  [ -d "$proof_dir" ] || die "expected proof directory missing: $proof_dir"
  [ -f "$proof_dir/summary.json" ] || die "expected proof summary missing: $proof_dir/summary.json"
  echo "CLEANUP OK: removed=$work_dir preserved_proofs=$proof_dir"
}

main() {
  local subcmd="${1:-}"
  case "$subcmd" in
    launch) shift; cmd_launch "$@" ;;
    doctor) shift; cmd_doctor "$@" ;;
    drive) shift; cmd_drive "$@" ;;
    cleanup) shift; cmd_cleanup "$@" ;;
    -h|--help|"") usage ;;
    *) die "unknown subcommand '$subcmd'" ;;
  esac
}

main "$@"
