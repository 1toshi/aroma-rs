# aroma-rs

A high-performance Rust engine for **structural code search, clone detection, and code recommendation**, implementing **Aroma: Code Recommendation via Structural Code Search** (Luan et al., OOPSLA 2019).

Originally built with full behavioral and numerical parity with [L9Labs/aroma-typescript](https://github.com/L9Labs/aroma-typescript) for Python and TypeScript/JavaScript (all 15 ported conformance tests pass, Recall@1 = `1.0` = `50/50`, identical indexing statistics and bit-exact work checksum `4173537680` across 100 queries on the 2,794-file evaluation corpus while running **10.1x–84.4x faster at indexing**, **21.4x–55.3x faster at search** with **58.7x** lower median latency at 15 threads, **164.9x–194.4x faster at recommendation** with **317.8x** lower median latency at 1 thread, and **2.1x–4.4x lower Peak RSS**), `aroma-rs` **extends beyond the original implementation with a built-in native Rust (`.rs`) front end** (`src/lang_rs.rs`).

## Supported Languages

| Language | ID | Extensions | Front End | Parity / Status |
| --- | --- | --- | --- | --- |
| **Rust** | `"rust"` | `.rs` | Native zero-dependency byte-slice lexer + Pass-1 local binder + error-recovering recursive-descent/Pratt parser (`src/lang_rs.rs`) | **New in `aroma-rs`** (not in `aroma-typescript`) |
| **Python** | `"python"` | `.py` | Lezer LR/GLR runtime (`src/lang_py.rs`) | Full behavioral & numerical parity with `aroma-typescript` |
| **TypeScript / JavaScript** | `"typescript"` | `.ts`, `.tsx`, `.mts`, `.cts`, `.js`, `.jsx`, `.mjs`, `.cjs` | Lezer LR/GLR runtime (`src/lang_ts.rs`) | Full behavioral & numerical parity with `aroma-typescript` |

## Native Rust (`.rs`) Front End

- **Item extraction & qualified names**: Indexes free functions, inherent `impl` methods (`Type::method`), trait `impl` methods (`<Type as Trait>::method`), default trait methods (`Trait::method`), `mod`-qualified items (`mod::item`), and nested `fn`s, while skipping bodyless trait/`extern` declarations.
- **Scope-aware alpha-renaming & usage features**: Pass-1 local binder (`NameSet`) collects parameters, `let`, `for`, `if let`, `while let`, `match` arm, and closure bindings so local variables normalize to `#VAR`, `self` receivers match structurally, non-local paths/types/globals remain concrete `NAME` tokens, and receiver/method calls emit `#.#` usage (`U`) features.
- **Syntax coverage & error recovery**: Handles nested block comments, lifetime vs. character-literal disambiguation, raw (`r#"..."#`) / byte / raw-byte / C-string literals, raw identifiers (`r#ident`), turbofish, macros, `async`/`unsafe`/`const`/`extern` items, and partial or malformed snippets (`query_from_snippet` / `queries_from_lines`) without panicking.
- **No extra dependencies**: Compiles with the existing minimal dependency set in `Cargo.toml`.

## Architecture

- **Lezer runtime & front ends (`src/lezer/`, `src/lang_py.rs`, `src/lang_ts.rs`, `src/lang_rs.rs`, `src/simple.rs`, `src/text.rs`)**: Row-deduplicated LR action tables with 16-byte aligned `FastState` descriptors, `SINK_BIT` DFA early exit, inline `Stack.buffer` + `StackArena::collapse_to_root` GLR pruning, SIMD `memchr` literal/comment scanners, zero-copy ASCII `Src` wrapper, unified `FrontScratch` + zero-alloc `NameSet`, a zero-dependency byte-slice Rust lexer + recursive-descent/Pratt parser with local scope binding and `#.#` method-call extraction, and AST simplification into a flat `Body` arena with a 128-bit algebraic dedup digest.
- **Structural featurization (`src/features.rs`)**: Single-pass CSR emission of `T`, up-to-3 `P`, `S`, and consecutive-local `U` features using zero-`String` 64-bit algebraic `mix` hashes and 1-byte `token_end_delta` spans (`255` overflow map).
- **Indexing & storage (`src/index.rs`)**: Wait-free LPT chunk parsing -> 32-shard `ShardedFeatureTable` (bits `32..37` shard index preserving full SwissTable H2 entropy) -> parallel `MaybeUninit` feature remap -> parallel bucketed `(count << 24) | local_mid` inverted index -> single contiguous `SourceStore` blob. Single-threaded mode (`--threads 1`) skips all Rayon pools and intermediate chunk staging.
- **Phases I & II search (`src/search.rs`, `src/prune.rs`)**: Branchless 4x-unrolled Phase I scatter computing overlap and capped Phase II score bounds in one pass; `canonical_file_id` L1 origin filtering; `select_nth_unstable` top-k extraction; 4 KB L1 Bloom filter + direct `qidx[fid]` lookup; Algorithm 1 `prune` with zero-greedy & `max_gain == 1` fast paths, signature-equivalence-class greedy (`SIG_THRESHOLD = 512`) with inline 64-bit tags, `[u64; 8]` 512-line stack bitset line extraction, and reusable `kept_buf` scratch.
- **Clustering & recommendations (`src/recommend.rs`)**: Lazy on-demand candidate feature multisets backed by `SearchScratch` flat pools, `u128` popcount `jaccard` deduplication, precomputed `StructSpan` structure lines, and a 512-byte stack `LineRole` map.

## Build, Test & Benchmark

Requires Rust 2024 toolchain (`rustc 1.100.0` used during verification).

```bash
# Run all 27 conformance, Rust front-end, and stress/overflow/determinism tests
cargo test --release --offline

# Run clippy
cargo clippy --offline --lib --bin aroma -- -D warnings
cargo clippy --offline --test extra --test rust_frontend -- -D warnings

# Run CLI benchmark against a JSON corpus manifest and query suite
cargo run --release --offline -- bench \
  --manifest /path/to/corpus_manifest.json \
  --queries /path/to/queries.json \
  --threads 15
```

## Library Usage

`build_index` automatically dispatches `.rs`, `.py`, and `.ts`/`.tsx`/`.mts`/`.cts`/`.js`/`.jsx`/`.mjs`/`.cjs` files through `builtin_languages()`:

```rust
use aroma::{
    build_index, queries_from_lines, query_from_snippet, recommend, search,
    IndexOptions, RecommendOptions, SearchOptions, SnippetOptions, SourceFile,
};

let files = vec![
    SourceFile {
        path: "src/lib.rs".into(),
        source: "pub fn sum_even(xs: &[i32]) -> i32 { xs.iter().filter(|x| *x % 2 == 0).sum() }\n".into(),
    },
    SourceFile {
        path: "example.py".into(),
        source: "def add(x, y):\n    return x + y\n".into(),
    },
];

let index = build_index(&files, IndexOptions::default());

// Search & recommend by Rust snippet (or pass lines to `queries_from_lines` for clone lookup)
let query = query_from_snippet(
    &index,
    "values.iter().filter(|v| *v % 2 == 0).sum()",
    &SnippetOptions { language: Some("rust".into()), ..Default::default() },
)
.unwrap();

let hits = search(&index, &query, &SearchOptions::default());
let recs = recommend(&index, &query, &RecommendOptions::default());
let line_queries = queries_from_lines(&index, &files[0], &[1]);
```

## Verified Benchmarks

Measured on Apple M5 Pro (15 cores, 24 GB RAM), `-C target-cpu=native`, macOS system allocator, `panic = "abort"`, using alternating execution order (`compare_bench.py`) against upstream `L9Labs/aroma-typescript` under Bun v1.4.4.

### Python + TypeScript/JavaScript vs. `aroma-typescript`

#### Full Corpus (2,794 files: 1,447 Python + 1,347 TS/TSX/JS/JSX, 29.14 MB source, 100 queries — 7 alternating rounds)

All 3 targets produced identical outputs: `methods = 21,866`, `duplicate_methods = 7,112`, `features = 756,440`, `Recall@1 = 1.00 (50/50)`, `total_search_hits = 10,000`, `total_recommendations = 40`, `work_checksum = 4173537680`, `errors = 0`.

| Metric | TS / Bun (median [min..max]) | `aroma --threads 1` | Speedup (1T) | `aroma --threads 15` | Speedup (15T) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `index_ms` | 5322.00 ms [5288.8..5402.1] | **525.66 ms** [521.6..532.4] | **10.1x** | **63.02 ms** [61.4..65.8] | **84.4x** |
| `query_build_total_ms` | 216.42 ms [213.4..222.6] | **15.43 ms** [15.1..15.7] | **14.0x** | **16.03 ms** [15.6..16.4] | **13.5x** |
| `search_total_ms` | 1476.17 ms [1456.9..1489.9] | **68.83 ms** [68.1..69.8] | **21.4x** | **26.71 ms** [25.5..28.9] | **55.3x** |
| `search_median_us` | 14623.2 us [14207..14876] | **478.5 us** [464.3..500.2] | **30.6x** | **249.1 us** [238.9..287.8] | **58.7x** |
| `search_p95_us` | 22510.0 us [22279..23014] | **1419.5 us** [1393..1448] | **15.9x** | **390.7 us** [347.0..400.0] | **57.6x** |
| `recommend_total_ms` | 1859.87 ms [1827.7..1871.9] | **9.57 ms** [9.29..9.72] | **194.4x** | **11.28 ms** [10.89..11.48] | **164.9x** |
| `recommend_median_us` | 14523.6 us [14237..15205] | **45.7 us** [44.0..46.7] | **317.8x** | **54.5 us** [53.5..62.2] | **266.5x** |
| `recommend_p95_us` | 42368.6 us [41815..42761] | **413.6 us** [381.0..454.8] | **102.4x** | **449.0 us** [434.0..512.5] | **94.4x** |
| `wall_ms` | 8934.40 ms [8864.0..9029.8] | **649.80 ms** [647.4..657.4] | **13.7x** | **149.50 ms** [148.1..151.0] | **59.8x** |
| Peak RSS | 699.97 MB [682.3..725.8] | **158.94 MB** [158.9..159.0] | **4.40x smaller** | **332.58 MB** [328.5..335.4] | **2.10x smaller** |

#### Small-Corpus Sensitivity Check (300 files, 2.92 MB, 40 queries — 3 alternating rounds)

`methods = 2,692`, `duplicate_methods = 687`, `features = 109,103`, `Recall@1 = 1.00 (20/20)`, `work_checksum = 1152710901`, `errors = 0`.

| Metric | TS / Bun | `aroma --threads 1` | `aroma --threads 15` |
| --- | ---: | ---: | ---: |
| `index_ms` | 593.14 ms | **55.93 ms** (10.6x) | **11.21 ms** (52.9x) |
| `search_total_ms` (`p50`) | 177.53 ms (`4562.3 us`) | **13.18 ms** (`257.3 us`, 17.7x) | **7.62 ms** (`181.7 us`, 25.1x) |
| `recommend_total_ms` (`p50`) | 172.47 ms (`4615.9 us`) | **0.62 ms** (`10.8 us`, 427.4x) | **0.79 ms** (`14.3 us`, 322.8x) |
| `wall_ms` | 1006.60 ms | **79.71 ms** (12.6x) | **30.29 ms** (33.2x) |
| Peak RSS | 178.27 MB | **24.67 MB** (7.23x smaller) | **65.42 MB** (2.72x smaller) |

### Native Rust Front-End Throughput (Real-World Multi-Repo Stress Test)

Across a 72-repository multi-language stress corpus (`6,193` total files: `3,588` Python/TS/JS + `2,605` Rust), adding the native Rust front end indexed **`2,605` `.rs` files** (`36,149` canonical methods, `1,393` exact duplicate methods, `1,342,125` structural features) in **`606.7 ms` single-threaded** (`831.6 ms -> 1,438.3 ms` total 1T index build; **`~4,300` `.rs` files/s** and **`~59,600` canonical Rust methods/s** on 1 thread) with zero panics.

## Remaining Limiters ("Why Not Double?")

1. **Indexing (`525.7 ms` 1T / `63.0 ms` 15T on the Python+TS corpus)**: Exact Lezer LR/GLR parsing + AST simplification accounts for ~69% of single-thread indexing time to maintain 100% AST and malformed-syntax parity with `aroma-typescript`. At 15 threads (`8.34x` scaling over 1T on a heterogeneous 5 P-core + 10 E-core CPU), deterministic Stage 2 dedup/CSR merge + Stage 3 interning/inverted-index assembly (~15 ms post-parse) plus memory bandwidth form the remaining Amdahl floor.
2. **Search (`68.8 ms` 1T / `26.7 ms` 15T over 100 queries)**: Algorithm 1 exact greedy `prune` consumes ~90% of `1T` search time (`~81%` on `15T`). Queries with fewer than 768 candidate tokens stay single-threaded because Rayon synchronization exceeds the microsecond-scale compute.
3. **Recommend (`9.57 ms` 1T / `11.28 ms` 15T)**: Per-query clustering runs in ~46 us median (`1T`), below profitable Rayon fan-out granularity (`15T` pays minor L2/NUMA cache-coherency overhead from the index built across worker cores).

See [`ARENA_SYNTHESIS.md`](./ARENA_SYNTHESIS.md) for the full Arena Round 1/Round 2 architecture bakeoff and Hillclimb decision log.
