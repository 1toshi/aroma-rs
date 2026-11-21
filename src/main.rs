use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use aroma::{
    build_index, queries_from_lines, query_from_snippet, recommend, search, IndexOptions,
    RecommendOptions, SearchOptions, SnippetOptions, SourceFile,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct ManifestEntry {
    path: String,
    source: String,
}

#[derive(Deserialize)]
struct QuerySpec {
    #[allow(dead_code)]
    id: usize,
    language: String,
    path: String,
    #[allow(dead_code)]
    expected_name: String,
    start_line: u32,
    end_line: u32,
    changed_lines: Vec<u32>,
    snippet: String,
}

#[derive(Serialize)]
struct StatsOut {
    files: usize,
    methods: usize,
    duplicate_methods: usize,
    features: usize,
    unsupported_files: usize,
    oversized_files: usize,
}

#[derive(Serialize)]
struct BenchOutput {
    #[serde(rename = "impl")]
    impl_name: &'static str,
    threads: usize,
    stats: StatsOut,
    index_ms: f64,
    query_build_total_ms: f64,
    search_total_ms: f64,
    search_median_us: f64,
    search_p95_us: f64,
    recommend_total_ms: f64,
    recommend_median_us: f64,
    recommend_p95_us: f64,
    recall_at_1_snippet_hits: usize,
    recall_at_1_snippet_total: usize,
    recall_at_1_snippet: f64,
    total_search_hits: usize,
    total_recommendations: usize,
    work_checksum: u64,
    errors: usize,
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] != "bench" {
        eprintln!("Usage: aroma bench --manifest <corpus.json> --queries <queries.json> [--threads <N>]");
        std::process::exit(2);
    }
    let mut manifest_path = String::new();
    let mut queries_path = String::new();
    let mut threads: Option<usize> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                if i < args.len() {
                    manifest_path = args[i].clone();
                }
            }
            "--queries" => {
                i += 1;
                if i < args.len() {
                    queries_path = args[i].clone();
                }
            }
            "--threads" => {
                i += 1;
                if i < args.len() {
                    threads = Some(args[i].parse().expect("invalid --threads value"));
                }
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if manifest_path.is_empty() || queries_path.is_empty() {
        eprintln!("Usage: aroma bench --manifest <corpus.json> --queries <queries.json> [--threads <N>]");
        std::process::exit(2);
    }

    let threads_used = threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1);

    let files: Vec<SourceFile> = {
        let manifest_raw = std::fs::read_to_string(&manifest_path).expect("failed to read manifest");
        let entries: Vec<ManifestEntry> =
            serde_json::from_str(&manifest_raw).expect("failed to parse manifest JSON");
        entries
            .into_iter()
            .map(|e| SourceFile { path: e.path, source: e.source })
            .collect()
    };
    let file_map: HashMap<&str, &SourceFile> =
        files.iter().map(|f| (f.path.as_str(), f)).collect();

    let query_specs: Vec<QuerySpec> = {
        let queries_raw = std::fs::read_to_string(&queries_path).expect("failed to read queries");
        serde_json::from_str(&queries_raw).expect("failed to parse queries JSON")
    };

    // 1. Index build
    let t0 = Instant::now();
    let index = build_index(
        black_box(&files),
        IndexOptions { threads: Some(threads_used), ..Default::default() },
    );
    let index_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // 2. Query build
    let tq0 = Instant::now();
    let built: Vec<_> = query_specs
        .iter()
        .map(|spec| {
            let sq = query_from_snippet(
                &index,
                &spec.snippet,
                &SnippetOptions {
                    language: Some(spec.language.clone()),
                    path: Some(spec.path.clone()),
                    globals: None,
                },
            );
            let lq = file_map.get(spec.path.as_str()).and_then(|sf| {
                queries_from_lines(&index, sf, &spec.changed_lines)
                    .into_iter()
                    .find(|q| {
                        q.origin.as_ref().is_some_and(|o| {
                            o.path == spec.path
                                && o.start_line == spec.start_line
                                && o.end_line == spec.end_line
                        })
                    })
            });
            (spec, sq, lq)
        })
        .collect();
    let query_build_total_ms = tq0.elapsed().as_secs_f64() * 1000.0;

    // 3. Search & Recommend
    let search_opts = SearchOptions::default();
    let rec_opts = RecommendOptions::default();

    let mut search_us: Vec<f64> = Vec::with_capacity(built.len() * 2);
    let mut recommend_us: Vec<f64> = Vec::with_capacity(built.len() * 2);
    let mut recall_hits = 0usize;
    let mut recall_total = 0usize;
    let mut total_search_hits = 0usize;
    let mut total_recommendations = 0usize;
    let mut work_checksum = 0u64;
    let mut errors = 0usize;

    for (spec, sq, lq) in &built {
        if let Some(sq) = sq {
            recall_total += 1;
            let ts0 = Instant::now();
            let s_res = black_box(search(&index, black_box(sq), &search_opts));
            search_us.push(ts0.elapsed().as_secs_f64() * 1_000_000.0);
            total_search_hits += s_res.len();
            if s_res
                .first()
                .map(|r| r.method.path == spec.path && r.method.start_line == spec.start_line)
                == Some(true)
            {
                recall_hits += 1;
            }
            for r in &s_res {
                work_checksum = work_checksum
                    .wrapping_add((r.similarity * 1e6).round() as u64 + r.matched_lines.len() as u64);
            }

            let tr1 = Instant::now();
            let s_recs = black_box(recommend(&index, black_box(sq), &rec_opts));
            recommend_us.push(tr1.elapsed().as_secs_f64() * 1_000_000.0);
            total_recommendations += s_recs.len();
            for rec in &s_recs {
                work_checksum = work_checksum.wrapping_add(
                    (rec.similarity * 1e6).round() as u64
                        + rec.lines.len() as u64
                        + rec.members.len() as u64,
                );
            }
        } else {
            errors += 1;
        }

        if let Some(lq) = lq {
            let ts1 = Instant::now();
            let l_res = black_box(search(&index, black_box(lq), &search_opts));
            search_us.push(ts1.elapsed().as_secs_f64() * 1_000_000.0);
            total_search_hits += l_res.len();
            for r in &l_res {
                work_checksum = work_checksum
                    .wrapping_add((r.similarity * 1e6).round() as u64 + r.matched_lines.len() as u64);
            }

            let tr0 = Instant::now();
            let l_recs = black_box(recommend(&index, black_box(lq), &rec_opts));
            recommend_us.push(tr0.elapsed().as_secs_f64() * 1_000_000.0);
            total_recommendations += l_recs.len();
            for rec in &l_recs {
                work_checksum = work_checksum.wrapping_add(
                    (rec.similarity * 1e6).round() as u64
                        + rec.lines.len() as u64
                        + rec.members.len() as u64,
                );
            }
        } else {
            errors += 1;
        }
    }

    let search_total_ms = search_us.iter().sum::<f64>() / 1000.0;
    let recommend_total_ms = recommend_us.iter().sum::<f64>() / 1000.0;
    search_us.sort_by(|a, b| a.total_cmp(b));
    recommend_us.sort_by(|a, b| a.total_cmp(b));

    let st = index.stats();
    let out = BenchOutput {
        impl_name: "aroma-rs",
        threads: threads_used,
        stats: StatsOut {
            files: st.files,
            methods: st.methods,
            duplicate_methods: st.duplicate_methods,
            features: st.features,
            unsupported_files: st.unsupported_files,
            oversized_files: st.oversized_files,
        },
        index_ms: round3(index_ms),
        query_build_total_ms: round3(query_build_total_ms),
        search_total_ms: round3(search_total_ms),
        search_median_us: round1(percentile(&search_us, 0.5)),
        search_p95_us: round1(percentile(&search_us, 0.95)),
        recommend_total_ms: round3(recommend_total_ms),
        recommend_median_us: round1(percentile(&recommend_us, 0.5)),
        recommend_p95_us: round1(percentile(&recommend_us, 0.95)),
        recall_at_1_snippet_hits: recall_hits,
        recall_at_1_snippet_total: recall_total,
        recall_at_1_snippet: if recall_total == 0 {
            0.0
        } else {
            round3(recall_hits as f64 / recall_total as f64)
        },
        total_search_hits,
        total_recommendations,
        work_checksum: black_box(work_checksum),
        errors,
    };

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
