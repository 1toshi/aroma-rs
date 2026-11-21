# Arena & Hillclimb Synthesis Report

Implementation and optimization record for `aroma-rs` using the **pstack** workflow (`arena` competing designs + `hillclimb` single-lever verification + `benchmark-checklist` alternating measurements).

## 1. Arena Round 1: Three Competing Rust Architectures

Three independent candidates were built against `SPEC.md` and `tests/conformance.rs`, verified at `15/15` conformance tests and `0` clippy warnings, and benchmarked in 5 alternating rounds (`--rounds 5 --include-ts`) on the frozen 2,794-file corpus (`corpus_manifest.json`, 29.14 MB, 100 queries):

| Implementation | `index_ms` (1T) | `index_ms` (15T) | `search_p50` (1T / 15T) | `rec_p50` (1T / 15T) | Peak RSS (1T / 15T) | `work_checksum` | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ts_bun` (Ref TS) | 5293.44 ms | — | 14648.3 us / — | 14481.6 us / — | 694.83 MB / — | `4173537680` | `methods=21866`, `features=756440` |
| `cand-1` | 666.31 ms | 86.38 ms | **555.2 / 280.3 us** | **64.3 / 74.1 us** | **193.98 / 397.78 MB** | **`4173537680`** | **Picked base**: exact Lezer port, 4KB Bloom + `qidx`, `run_greedy_sig`, `StructSpan`, `pool.broadcast` shrink |
| `cand-2` | **464.00 ms** | **59.16 ms** | 662.3 / 388.3 us | 65.0 / 74.7 us | 245.19 / 393.61 MB | `4234386981` | **Rejected as base**: custom scanner diverged on 80/2,794 files (`methods=21797`), swallowing 60 of 62 methods after f-string hex format specs in `windows_events.py` |
| `cand-3` | 634.62 ms | 81.18 ms | 635.6 / 314.3 us | 85.3 / 89.5 us | 232.77 / 408.39 MB | **`4173537680`** | Zero-`String` algebraic `mix` hashes, `token_end_delta` (1 byte + overflow), `[u64; 8]` 512-line stack bitset |

- **Cross-judge quota note**: Formal cross-vendor judging was skipped because all non-Gemini CLI model quotas (`claude-*` and `gpt-oss-120b`) were exhausted (`0%`) at the time of evaluation. Base selection used the pre-committed empirical rule: `cand-1` won best search/recommend/RSS with 100% bit-exact parity (`4173537680`) and zero structural parser discrepancies.

### Round 1 Synthesis (`25bb3af`, 5 alternating rounds)
Grafted four winning levers from `cand-3` and `cand-2` onto `cand-1`:
1. Algebraic zero-`String` 64-bit `mix` feature hashing (`src/features.rs`).
2. Compact 1-byte `token_end_delta` + `token_end_overflow` map (`src/index.rs`, `src/recommend.rs`).
3. `canonical_file_id` L1 origin filter (`src/index.rs`, `src/search.rs`).
4. 512-line `[u64; 8]` stack bitset `matched_lines_of` (`src/search.rs`).

Verified result vs `cand-1`: `index_ms` `668.34 -> 635.28 ms` (`1T`, -5.0%), `87.59 -> 83.96 ms` (`15T`, -4.1%), `search_p50` `539.9 -> 521.3 us` (`1T`), Peak RSS `193.98 -> 189.70 MB` (`1T`) / `392.42 -> 384.22 MB` (`15T`), `work_checksum = 4173537680`.

---

## 2. Stage Profile & Arena Round 2 Specialists

Before Round 2, stage instrumentation on `25bb3af` identified the exact limiters:
- **1T Index (`636.4 ms`)**: `parse = 442.8 ms` (`69.6%`: TS Lezer `203.5 ms` + walk `56.6 ms`; Python Lezer `142.4 ms` + walk `41.5 ms`), `featurize = 64.4 ms`, `intern = 57.5 ms`, `invert = 21.8 ms`, `loc = 5.5 ms`, `digest/seen/misc = ~44.5 ms`.
- **15T Index (`90.4 ms` instrumented)**: sequential `prescan = 7.12 ms`, `stage1_wall = 63.44 ms`, `stage2 = 2.30 ms`, `stage3a = 4.87 ms`, `stage3b = 2.17 ms`, `stage3c = 9.43 ms`.
- **Search (`78.2 ms` 1T / `28.6 ms` 15T)**: `phase1 = 8.49 ms (1T) / 9.48 ms (15T)`, **`phase2_prune = 70.54 ms (90.2% 1T) / 23.26 ms (81.3% 15T)`**.
- **Recommend (`14.11 ms` 1T / `15.85 ms` 15T)**: `rank = 6.88 ms`, `multiset = 4.96 ms` (`35.2%`), `cluster = 0.75 ms`, `render = 1.49 ms`; `15T` regressed (`15.85 ms`) due to Rayon wakeup overhead on tiny `rank_for_recommend` batches.

Two specialist Arena workers (`r2-index` and `r2-query`) ran independently in parallel and were verified via 5-round alternating benchmarks (`15/15` tests, `0` clippy warnings, `work_checksum = 4173537680`):

| Specialist | Key Kept Levers | Rejected In-Worker Experiments | Verified Delta vs `prev_aroma` |
| --- | --- | --- | --- |
| `r2-index` | Packed `FastState`, `SINK_BIT` DFA exit, inline `Stack.buffer` + `StackArena::collapse_to_root`, SIMD `memchr` tokenizers, `FrontScratch` + `NameSet`, algebraic `Body::digest`, `queries_from_lines` line-range pruning, `ShardedFeatureTable` H2 bit fix (`(h >> 32) & 31`), `utf16_len_exceeds`, contiguous `SourceStore`, reusable pools, `u16` `last_mid` + `MaybeUninit` Stage 3c | Lock-free `AtomicFeatureTable` (+18 ms 1T / +14 ms 15T from atomic contention); 3-batch bucketed Stage 3c (+6.8 ms from branch mispredicts); 4-wave indexing (+5 ms vs 3 waves) | `index_ms`: `627.5 -> 510.5 ms` (`1T`, -18.7%), `84.3 -> 68.3 ms` (`15T`, -19.1%); `qbuild`: `-33.9%`; RSS: `189.7 -> 167.8 MB` (`1T`), `381.4 -> 328.5 MB` (`15T`, -53 MB) |
| `r2-query` | Zero-greedy (`expected_score == Some(total_gain)`) + `max_gain == 1` fast paths, single-class & 64-bit inline `SigSlot` tags + 3/4 sorting networks, `SIG_THRESHOLD = 512`, branchless `cap <= 255` 4x scatter, 8-wide zero-skip, `select_nth_unstable` top-k, separate `PAR_PRUNE_REC_TOK_THRESHOLD`, lazy `whole` multisets + flat `SearchScratch` pools, `u128` jaccard, 512-byte stack `LineRole` | Open-addressed `QSlot` hash table replacing `qidx` (regressed `1T search` `72.9 -> 84.1 ms`); unconditional 2-pass `tok_hits` collection (only `0.8%` trigger rate, hurt cache locality); `.with_max_len(1)` Rayon split (`26.8 -> 29.8 ms`) | `search_total`: `79.3 -> 72.9 ms` (`1T`, -8.1%), `31.4 -> 27.4 ms` (`15T`, -12.6%); `recommend_total`: `14.9 -> 9.7 ms` (`1T`, -34.7%, `p50` `65.8 -> 46.8 us`), `16.5 -> 11.0 ms` (`15T`, -33.1%) |

---

## 3. Single-Lever Grafts & Hillclimb Decision Ledger (`.local/decision.tsv`)

Every change after Round 1 was grafted/tested one logical lever at a time with `cargo test --release --offline`, `cargo clippy -- -D warnings`, bit-exact `work_checksum == 4173537680`, and 5 alternating benchmark rounds against the prior committed binary:

| ID | Commit | Change | Before | After | Delta | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `R1_base` | `b2ce058` | Pick `cand-1` base | TS 5293.4ms idx / 14648us search / 14482us rec | 668.34ms(1T) 87.59ms(15T) idx; 539.9us search | 7.9x–60.4x vs TS | **KEEP** |
| `R1_grafts` | `25bb3af` | Algebraic `mix` + `end_delta` + `canonical_file_id` + stack bitset | 668.34ms(1T) 87.59ms(15T) idx; 194.0/392.4MB | 635.28ms(1T) 83.96ms(15T) idx; 189.7/384.2MB | -5.0% 1T idx, -4.1% 15T idx, -8.2MB RSS | **KEEP** |
| `R2G1_prune_search` | `df8f473` | `r2-query` `prune.rs` + `search.rs` | search 75.31ms(1T) 27.88ms(15T); rec 14.09/15.70ms | search 69.83ms(1T) 28.00ms(15T); rec 12.81/14.38ms | -7.3% 1T search (-8.7% p50), -9.1% 1T rec | **KEEP** |
| `R2G2_recommend` | `86b0455` | `r2-query` `recommend.rs` + flat `SearchScratch` pools | rec 12.62ms(1T) 14.39ms(15T), p95 491/566us | rec 9.20ms(1T) 10.80ms(15T), p95 394/435us | -27.1% 1T rec, -24.9% 15T rec, -19.7% p95 | **KEEP** |
| `R2G3_lezer_runtime` | `11634ff` | `r2-index` `src/lezer/*` runtime & GLR `collapse_to_root` | idx 628.50ms(1T) 83.44ms(15T); qbuild 23.42ms | idx 587.57ms(1T) 78.59ms(15T); qbuild 21.54ms | -6.5% 1T idx, -5.8% 15T idx, -5.8MB 15T RSS | **KEEP** |
| `R2G4_frontends_digest` | `a77d936` | `r2-index` front ends, `Body::digest`, `line_range` query pruning | idx 593.44ms(1T) 78.55ms(15T); qbuild 21.48ms | idx 564.47ms(1T) 71.75ms(15T); qbuild 15.54ms | -4.9% 1T idx, -8.7% 15T idx, -27.6% qbuild | **KEEP** |
| `R2G5_index_pipeline` | `cbb6a37` | `ShardedFeatureTable` H2 fix, `SourceStore`, wave pools, Stage 3c | idx 567.11ms(1T) 73.93ms(15T); RSS 185.2/371.8MB | idx 522.83ms(1T) 68.33ms(15T); RSS 167.4/329.1MB | -7.8% 1T idx, -7.6% 15T idx, -42.7MB 15T RSS | **KEEP** |
| `H1_target_cpu_native` | `56d6d7c` | `.cargo/config.toml` `-C target-cpu=native` | idx 534.46ms(1T) 66.97ms(15T); wall 661.9/157.6ms | idx 521.05ms(1T) 66.35ms(15T); wall 647.1/156.2ms | -2.5% 1T idx, -0.9% 15T idx/wall, -1.8% 15T search | **KEEP** |
| `H2a_jemalloc` | — | `tikv-jemallocator = 0.6.1` vs `mimalloc` | idx 532.53ms(1T) 67.64ms(15T); wall 661.4/157.3ms | idx 535.26ms(1T) 68.06ms(15T); RSS 150.4/304.2MB | +0.5%/+0.6% slower idx, +0.8%/+1.0% slower wall | **DISCARD** |
| `H2b_system_allocator` | `265a99f` | macOS system allocator vs `mimalloc` | idx 525.21ms(1T) 67.04ms(15T); RSS 167.4/327.5MB | idx 522.12ms(1T) 66.76ms(15T); RSS 158.9/292.2MB | -0.6%/-0.4% faster idx, -0.7% 15T wall, -35.3MB 15T RSS | **KEEP** |
| `H3_panic_abort` | `272ed32` | `panic = "abort"` in `[profile.release]` | 1T search 70.79ms; 1T wall 646.28ms | 1T search 69.74ms; 1T wall 642.33ms | -1.5% 1T search (zero range overlap), -0.6% wall | **KEEP** |
| `H4a_kept_buf_reuse` | `8e50826` | `prune_*_into` reusing `kept_buf` in `search()` | 1T search 69.79ms (491.3us p50), 15T 27.92ms | 1T search 68.53ms (471.8us p50), 15T 27.43ms | -1.8% 1T search (zero overlap), -4.0% 1T p50 | **KEEP** |
| `H4b_single_wave` | `2c12b02` | `num_waves = 1` LPT under system allocator | 15T idx 67.93ms (wall 154.65ms, RSS 293.98MB) | 15T idx 65.38ms (min 62.16ms, wall 151.36ms, 333.67MB) | -3.8% (-2.55ms) 15T idx, -2.1% wall | **KEEP** |

Final 7-round verification (`tests/extra.rs`, `README.md`, `ARENA_SYNTHESIS.md`): `aroma --threads 15` reached **63.02 ms** (`61.40..65.81 ms`, **84.4x** faster than TS/Bun's `5322.00 ms`), `aroma --threads 1` reached **525.66 ms** (**10.1x**), `search_median_us = 249.1 us` (**58.7x**), `recommend_median_us = 45.7 us` (**317.8x**), Peak RSS **158.94 MB** (`1T`) / **332.58 MB** (`15T`), with `21/21` passing tests (`15/15` conformance + `6/6` extra) and `0` clippy warnings.
