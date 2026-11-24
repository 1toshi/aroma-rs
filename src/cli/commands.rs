//! Command execution shell (`triage`, `clones`, `search`, `recommend`, `check`, `bench`).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use aroma::{
    build_index, queries_from_lines, query_from_snippet, recommend, search, AromaIndex,
    IndexOptions, LineRole, MethodRef, Query, RecommendOptions, SearchOptions, SnippetOptions,
    SourceFile,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::cli::args::{
    BenchCmd, CheckCmd, ClonesCmd, DiffSpec, OriginFilter, OutputMode, QuerySource, RecommendCmd,
    RecsMode, SearchCmd, TestFilter, TriageCmd, MAX_SNIPPET_BYTES,
};
use crate::cli::error::{closest_match, CliError, ExitKind};
use crate::cli::walk::{
    infer_language_for_path, is_test_path, normalize_display_path, read_file_bounded, walk_paths,
    WalkOptions, WalkResult, WarningItem,
};

#[must_use]
pub(crate) struct CommandOutcome {
    pub(crate) exit: ExitKind,
    pub(crate) stdout: String,
    pub(crate) stderr_notes: Vec<String>,
}

#[derive(Clone, Serialize)]
struct SuggestedCommand {
    action: String,
    command: String,
    destructive: bool,
}

#[derive(Clone, Serialize)]
struct CorpusSummary {
    files_indexed: usize,
    methods: usize,
    duplicate_methods: usize,
    features: usize,
    total_bytes: usize,
    languages: BTreeMap<String, usize>,
    skipped_oversized: usize,
    skipped_non_utf8: usize,
    skipped_unsupported: usize,
}

#[derive(Clone, Serialize)]
struct MemberOut {
    path: String,
    name: String,
    language: String,
    start_line: u32,
    end_line: u32,
    lines: u32,
    size: usize,
    is_test: bool,
    is_canonical: bool,
}

#[derive(Clone, Serialize)]
struct RecPreview {
    similarity: f64,
    member_count: usize,
    code_preview: String,
}

#[derive(Clone, Serialize)]
struct CloneClusterOut {
    cluster_id: usize,
    kind: String,
    language: String,
    score: f64,
    duplicated_lines: u32,
    avg_sim: f64,
    min_sim: f64,
    exact_edges: usize,
    near_edges: usize,
    distinct_files: usize,
    files: Vec<String>,
    members: Vec<MemberOut>,
    recommendations: Vec<RecPreview>,
}

#[derive(Clone, Serialize)]
struct FilePairOut {
    kind: String,
    path_a: String,
    path_b: String,
    matched_lines: u32,
    exact_edges: usize,
    near_edges: usize,
    mean_sim: f64,
    score: f64,
    sample_methods: Vec<String>,
}

#[derive(Clone, Serialize)]
struct FileHotspotOut {
    path: String,
    duplicated_lines: u32,
    cluster_count: usize,
    exact_clusters: usize,
}

struct MethodInfo {
    path: String,
    name: String,
    language: String,
    start_line: u32,
    end_line: u32,
    lines: u32,
    size: usize,
    is_test: bool,
    is_canonical: bool,
}

#[derive(Clone, Copy)]
struct CloneEdge {
    u: usize,
    v: usize,
    sim: f64,
    is_exact: bool,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn shell_quote(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_string();
    }
    let safe = arg.bytes().all(|b| {
        matches!(
            b,
            b'a'..=b'z'
                | b'A'..=b'Z'
                | b'0'..=b'9'
                | b'/'
                | b'.'
                | b'-'
                | b'_'
                | b':'
                | b','
        )
    });
    if safe {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', "'\"'\"'"))
    }
}

fn line_count_for(src: &str) -> u32 {
    if src.is_empty() {
        return 0;
    }
    let newlines = memchr::memchr_iter(b'\n', src.as_bytes()).count() as u32;
    if src.ends_with('\n') {
        newlines.max(1)
    } else {
        newlines + 1
    }
}

fn corpus_summary(walk: &WalkResult, index: &AromaIndex) -> CorpusSummary {
    let st = index.stats();
    CorpusSummary {
        files_indexed: st.files,
        methods: st.methods,
        duplicate_methods: st.duplicate_methods,
        features: st.features,
        total_bytes: walk.total_bytes,
        languages: walk.language_counts.clone(),
        skipped_oversized: walk.skipped_oversized,
        skipped_non_utf8: walk.skipped_non_utf8,
        skipped_unsupported: walk.skipped_unsupported,
    }
}

fn to_member_out(m: &MethodInfo) -> MemberOut {
    MemberOut {
        path: m.path.clone(),
        name: m.name.clone(),
        language: m.language.clone(),
        start_line: m.start_line,
        end_line: m.end_line,
        lines: m.lines,
        size: m.size,
        is_test: m.is_test,
        is_canonical: m.is_canonical,
    }
}

fn make_code_preview(lines: &[aroma::RecommendedLine]) -> String {
    let mut out = String::new();
    for (idx, line) in lines.iter().take(12).enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        let mark = match line.role {
            LineRole::Query => '+',
            LineRole::Added => ' ',
            LineRole::Structure => '.',
        };
        let _ = write!(out, "{:4} {} {}", line.line, mark, line.text);
    }
    if lines.len() > 12 {
        let _ = write!(out, "\n... ({} more lines)", lines.len() - 12);
    }
    out
}

struct CloneAnalysis {
    total_clusters: usize,
    clusters: Vec<CloneClusterOut>,
    file_pairs: Vec<FilePairOut>,
    hotspots: Vec<FileHotspotOut>,
}

struct CloneParams {
    min_sim: f64,
    min_lines: u32,
    min_tokens: usize,
    top: usize,
    tests: TestFilter,
    recs: RecsMode,
    threads: Option<usize>,
}

fn analyze_clones(walk: &WalkResult, index: &AromaIndex, params: &CloneParams) -> CloneAnalysis {
    struct RawMethod {
        path: String,
        name: String,
        language: String,
        start_line: u32,
        end_line: u32,
        lines: u32,
        size: usize,
        is_test: bool,
        is_canonical: bool,
        query: Query,
    }
    struct PairAccum {
        lines: u32,
        sim_sum: f64,
        exact_cnt: usize,
        near_cnt: usize,
        methods: BTreeSet<String>,
    }

    let canonical_set: FxHashSet<(String, u32, u32, String)> = index
        .method_refs()
        .into_iter()
        .map(|r| (r.path, r.start_line, r.end_line, r.name))
        .collect();

    let extract_one = |sf: &SourceFile| -> Vec<RawMethod> {
        let test_flag = is_test_path(&sf.path);
        if params.tests == TestFilter::ExcludeTests && test_flag {
            return Vec::new();
        }
        let lcnt = line_count_for(&sf.source);
        if lcnt == 0 {
            return Vec::new();
        }
        let line_nums: Vec<u32> = (1..=lcnt).collect();
        let qs = queries_from_lines(index, sf, &line_nums);
        let mut out = Vec::new();
        for q in qs {
            let Some(ref o) = q.origin else {
                continue;
            };
            let mlines = o.end_line.saturating_sub(o.start_line) + 1;
            let qsize = q.size();
            if mlines < params.min_lines || qsize < params.min_tokens {
                continue;
            }
            let is_can = canonical_set.contains(&(
                o.path.clone(),
                o.start_line,
                o.end_line,
                o.name.clone(),
            ));
            out.push(RawMethod {
                path: o.path.clone(),
                name: o.name.clone(),
                language: o.language.clone(),
                start_line: o.start_line,
                end_line: o.end_line,
                lines: mlines,
                size: qsize,
                is_test: test_flag,
                is_canonical: is_can,
                query: q,
            });
        }
        out
    };

    let per_file: Vec<Vec<RawMethod>> = if params.threads == Some(1) {
        walk.files.iter().map(extract_one).collect()
    } else {
        walk.files.par_iter().map(extract_one).collect()
    };

    let mut methods: Vec<MethodInfo> = Vec::new();
    let mut queries: Vec<Query> = Vec::new();
    let mut key_to_id: FxHashMap<(String, u32, u32, String), usize> = FxHashMap::default();

    for vec_m in per_file {
        for rm in vec_m {
            let id = methods.len();
            key_to_id
                .entry((rm.path.clone(), rm.start_line, rm.end_line, rm.name.clone()))
                .or_insert(id);
            methods.push(MethodInfo {
                path: rm.path,
                name: rm.name,
                language: rm.language,
                start_line: rm.start_line,
                end_line: rm.end_line,
                lines: rm.lines,
                size: rm.size,
                is_test: rm.is_test,
                is_canonical: rm.is_canonical,
            });
            queries.push(rm.query);
        }
    }

    let search_opts = SearchOptions {
        limit: 25,
        min_similarity: params.min_sim.min(0.72),
        ..Default::default()
    };

    let search_one = |u: usize| -> Vec<(usize, f64)> {
        let hits = search(index, &queries[u], &search_opts);
        let mut out = Vec::with_capacity(hits.len());
        for h in hits {
            let key = (
                h.method.path,
                h.method.start_line,
                h.method.end_line,
                h.method.name,
            );
            if let Some(&v) = key_to_id.get(&key)
                && v != u
            {
                out.push((v, h.similarity));
            }
        }
        out
    };

    let per_method_hits: Vec<Vec<(usize, f64)>> = if params.threads == Some(1) {
        (0..queries.len()).map(search_one).collect()
    } else {
        (0..queries.len()).into_par_iter().map(search_one).collect()
    };

    let mut dir_map: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for (u, hits) in per_method_hits.iter().enumerate() {
        for &(v, sim) in hits {
            dir_map
                .entry((u, v))
                .and_modify(|prev| {
                    if sim > *prev {
                        *prev = sim;
                    }
                })
                .or_insert(sim);
        }
    }

    let ratio_u = |a: usize, b: usize| -> f64 {
        let hi = a.max(b);
        if hi == 0 {
            1.0
        } else {
            a.min(b) as f64 / hi as f64
        }
    };

    let mut edge_map: FxHashMap<(usize, usize), (f64, bool)> = FxHashMap::default();
    for (&(u, v), &sim_uv) in &dir_map {
        let mu = &methods[u];
        let mv = &methods[v];
        let sr = ratio_u(mu.size, mv.size);
        let lr = ratio_u(mu.lines as usize, mv.lines as usize);
        let (a, b) = if u < v { (u, v) } else { (v, u) };

        if sim_uv >= 0.999 && sr >= 0.95 && lr >= 0.85 {
            edge_map
                .entry((a, b))
                .and_modify(|e| {
                    e.0 = e.0.max(sim_uv);
                    e.1 = true;
                })
                .or_insert((sim_uv, true));
        } else if u < v
            && let Some(&sim_vu) = dir_map.get(&(v, u))
        {
            let msim = sim_uv.min(sim_vu);
            if msim >= params.min_sim && sr >= 0.65 && lr >= 0.55 {
                edge_map
                    .entry((a, b))
                    .and_modify(|e| {
                        if !e.1 {
                            e.0 = e.0.max(msim);
                        }
                    })
                    .or_insert((msim, false));
            }
        }
    }

    let mut edges: Vec<CloneEdge> = edge_map
        .into_iter()
        .map(|((u, v), (sim, is_exact))| CloneEdge {
            u,
            v,
            sim,
            is_exact,
        })
        .collect();
    edges.sort_by_key(|e| (e.u, e.v));

    let mut uf = UnionFind::new(methods.len());
    for e in &edges {
        uf.union(e.u, e.v);
    }

    let mut comp_members: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for i in 0..methods.len() {
        let r = uf.find(i);
        comp_members.entry(r).or_default().push(i);
    }

    let mut comp_edges: FxHashMap<usize, Vec<CloneEdge>> = FxHashMap::default();
    for e in &edges {
        let r = uf.find(e.u);
        comp_edges.entry(r).or_default().push(*e);
    }

    let mut roots: Vec<usize> = comp_members.keys().copied().collect();
    roots.sort_unstable();

    let rec_opts = RecommendOptions {
        limit: 2,
        min_similarity: params.min_sim.min(0.65),
        min_support: 2,
        ..Default::default()
    };

    let mut clusters: Vec<CloneClusterOut> = Vec::new();
    let mut hotspot_acc: BTreeMap<String, (u32, usize, usize)> = BTreeMap::new();

    for r in roots {
        let Some(mut mems) = comp_members.remove(&r) else {
            continue;
        };
        if mems.len() < 2 {
            continue;
        }
        let first_path = &methods[mems[0]].path;
        let same_file = mems.iter().all(|&id| &methods[id].path == first_path);
        if same_file {
            mems.sort_by_key(|&id| (methods[id].start_line, methods[id].end_line));
            let mut all_overlap = true;
            let mut max_end = methods[mems[0]].end_line;
            for &id in &mems[1..] {
                if methods[id].start_line > max_end {
                    all_overlap = false;
                    break;
                }
                max_end = max_end.max(methods[id].end_line);
            }
            if all_overlap {
                continue;
            }
        }

        mems.sort_by(|&a, &b| {
            methods[b]
                .lines
                .cmp(&methods[a].lines)
                .then_with(|| methods[a].path.cmp(&methods[b].path))
                .then_with(|| methods[a].start_line.cmp(&methods[b].start_line))
                .then_with(|| methods[a].name.cmp(&methods[b].name))
        });

        let ces = comp_edges.remove(&r).unwrap_or_default();
        let exact_cnt = ces.iter().filter(|e| e.is_exact).count();
        let near_cnt = ces.len().saturating_sub(exact_cnt);
        let avg_sim = if ces.is_empty() {
            1.0
        } else {
            ces.iter().map(|e| e.sim).sum::<f64>() / ces.len() as f64
        };
        let min_sim = ces.iter().map(|e| e.sim).fold(1.0f64, f64::min);

        let sum_lines: u32 = mems.iter().map(|&id| methods[id].lines).sum();
        let max_lines: u32 = mems.iter().map(|&id| methods[id].lines).max().unwrap_or(0);
        let dup_lines = sum_lines.saturating_sub(max_lines);
        let files_set: BTreeSet<String> = mems.iter().map(|&id| methods[id].path.clone()).collect();
        let distinct_files = files_set.len();
        let kind = if near_cnt == 0 {
            "exact"
        } else if exact_cnt == 0 {
            "near"
        } else {
            "mixed"
        };

        let mut recs = Vec::new();
        if params.recs == RecsMode::Include {
            let seed = mems
                .iter()
                .copied()
                .find(|&id| methods[id].is_canonical)
                .unwrap_or(mems[0]);
            for rec in recommend(index, &queries[seed], &rec_opts) {
                recs.push(RecPreview {
                    similarity: round3(rec.similarity),
                    member_count: rec.members.len(),
                    code_preview: make_code_preview(&rec.lines),
                });
            }
        }

        let avg_r = round3(avg_sim);
        let min_r = round3(min_sim);
        let score = round3((dup_lines as f64) * avg_sim);
        let per_member_lines = dup_lines / (mems.len().saturating_sub(1).max(1) as u32);
        for fp in &files_set {
            let entry = hotspot_acc.entry(fp.clone()).or_insert((0, 0, 0));
            entry.0 = entry.0.saturating_add(per_member_lines.max(1));
            entry.1 += 1;
            if near_cnt == 0 {
                entry.2 += 1;
            }
        }

        clusters.push(CloneClusterOut {
            cluster_id: 0,
            kind: kind.to_string(),
            language: methods[mems[0]].language.clone(),
            score,
            duplicated_lines: dup_lines,
            avg_sim: avg_r,
            min_sim: min_r,
            exact_edges: exact_cnt,
            near_edges: near_cnt,
            distinct_files,
            files: files_set.into_iter().collect(),
            members: mems.iter().map(|&id| to_member_out(&methods[id])).collect(),
            recommendations: recs,
        });
    }

    clusters.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.duplicated_lines.cmp(&a.duplicated_lines))
            .then_with(|| a.files.cmp(&b.files))
    });
    for (idx, cl) in clusters.iter_mut().enumerate() {
        cl.cluster_id = idx + 1;
    }

    // Aggregate distinct file pairs
    let mut pair_acc: BTreeMap<(String, String), PairAccum> = BTreeMap::new();
    for e in &edges {
        let mu = &methods[e.u];
        let mv = &methods[e.v];
        if mu.path == mv.path {
            continue;
        }
        let (pa, pb, na, nb) = if mu.path <= mv.path {
            (mu.path.clone(), mv.path.clone(), &mu.name, &mv.name)
        } else {
            (mv.path.clone(), mu.path.clone(), &mv.name, &mu.name)
        };
        let slot = pair_acc.entry((pa, pb)).or_insert_with(|| PairAccum {
            lines: 0,
            sim_sum: 0.0,
            exact_cnt: 0,
            near_cnt: 0,
            methods: BTreeSet::new(),
        });
        slot.lines += mu.lines.min(mv.lines);
        slot.sim_sum += e.sim;
        if e.is_exact {
            slot.exact_cnt += 1;
        } else {
            slot.near_cnt += 1;
        }
        slot.methods.insert(format!("{na}<->{nb}"));
    }

    let mut file_pairs: Vec<FilePairOut> = pair_acc
        .into_iter()
        .map(|((path_a, path_b), acc)| {
            let cnt = (acc.exact_cnt + acc.near_cnt).max(1) as f64;
            let mean_sim = acc.sim_sum / cnt;
            let kind = if acc.near_cnt == 0 {
                "exact"
            } else if acc.exact_cnt == 0 {
                "near"
            } else {
                "mixed"
            };
            FilePairOut {
                kind: kind.to_string(),
                path_a,
                path_b,
                matched_lines: acc.lines,
                exact_edges: acc.exact_cnt,
                near_edges: acc.near_cnt,
                mean_sim: round3(mean_sim),
                score: round3((acc.lines as f64) * mean_sim),
                sample_methods: acc.methods.into_iter().take(6).collect(),
            }
        })
        .collect();
    file_pairs.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path_a.cmp(&b.path_a))
            .then_with(|| a.path_b.cmp(&b.path_b))
    });

    let mut hotspots: Vec<FileHotspotOut> = hotspot_acc
        .into_iter()
        .map(|(path, (duplicated_lines, cluster_count, exact_clusters))| FileHotspotOut {
            path,
            duplicated_lines,
            cluster_count,
            exact_clusters,
        })
        .collect();
    hotspots.sort_by(|a, b| {
        b.duplicated_lines
            .cmp(&a.duplicated_lines)
            .then_with(|| b.cluster_count.cmp(&a.cluster_count))
            .then_with(|| a.path.cmp(&b.path))
    });

    let total_clusters = clusters.len();
    clusters.truncate(params.top);
    file_pairs.truncate(params.top);
    hotspots.truncate(params.top);

    CloneAnalysis {
        total_clusters,
        clusters,
        file_pairs,
        hotspots,
    }
}

fn format_path_args(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        ".".to_string()
    } else {
        paths
            .iter()
            .map(|p| shell_quote(&p.to_string_lossy()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub(crate) fn run_triage(cmd: &TriageCmd) -> Result<CommandOutcome, CliError> {
    let walk = walk_paths(
        &cmd.paths,
        &WalkOptions {
            max_file_bytes: cmd.max_file_bytes,
            extra_ignores: cmd.ignores.clone(),
            lang_filter: cmd.lang.clone(),
        },
    )?;
    let index = build_index(
        &walk.files,
        IndexOptions {
            threads: cmd.threads,
            max_file_length: cmd.max_file_bytes,
            ..Default::default()
        },
    );
    let corpus = corpus_summary(&walk, &index);
    let analysis = analyze_clones(
        &walk,
        &index,
        &CloneParams {
            min_sim: cmd.min_sim,
            min_lines: cmd.min_lines,
            min_tokens: cmd.min_tokens,
            top: cmd.top,
            tests: cmd.tests,
            recs: RecsMode::Omit,
            threads: cmd.threads,
        },
    );

    let path_str = format_path_args(&cmd.paths);
    let mut commands = vec![SuggestedCommand {
        action: "clones_with_recommendations".to_string(),
        command: format!("aroma clones {path_str} --include-recommendations --json"),
        destructive: false,
    }];
    if let Some(top_c) = analysis.clusters.first()
        && let Some(m) = top_c.members.first()
    {
        commands.push(SuggestedCommand {
            action: "recommend_cluster".to_string(),
            command: format!(
                "aroma recommend {path_str} --file {} --lines {}-{} --json",
                shell_quote(&m.path),
                m.start_line,
                m.end_line
            ),
            destructive: false,
        });
        commands.push(SuggestedCommand {
            action: "search_similar".to_string(),
            command: format!(
                "aroma search {path_str} --file {} --lines {}-{} --json",
                shell_quote(&m.path),
                m.start_line,
                m.end_line
            ),
            destructive: false,
        });
    }

    let exact_total = analysis
        .clusters
        .iter()
        .filter(|c| c.kind == "exact")
        .count();
    let summary = format!(
        "{} files / {} methods indexed; {} clone clusters returned (top={}, exact={}, min_sim={:.2})",
        corpus.files_indexed,
        corpus.methods,
        analysis.total_clusters,
        analysis.clusters.len(),
        exact_total,
        cmd.min_sim
    );
    let hint = (analysis.total_clusters == 0).then(|| {
        format!(
            "No clones found at --min-sim {:.2}; try lowering threshold via `aroma triage {path_str} --min-sim 0.65 --include-tests --json`",
            cmd.min_sim
        )
    });

    let stdout = match cmd.mode {
        OutputMode::Json => {
            #[derive(Serialize)]
            struct QuickRef<'a> {
                summary: &'a str,
                top_clones: &'a [CloneClusterOut],
            }
            #[derive(Serialize)]
            struct TriageOut<'a> {
                ok: bool,
                tool_version: &'static str,
                contract_version: &'static str,
                data_hash: &'a str,
                corpus: &'a CorpusSummary,
                quick_ref: QuickRef<'a>,
                clone_clusters: &'a [CloneClusterOut],
                hotspot_files: &'a [FileHotspotOut],
                commands: &'a [SuggestedCommand],
                warnings: &'a [WarningItem],
                hint: Option<String>,
            }
            let out = TriageOut {
                ok: true,
                tool_version: env!("CARGO_PKG_VERSION"),
                contract_version: "1",
                data_hash: &walk.data_hash,
                corpus: &corpus,
                quick_ref: QuickRef {
                    summary: &summary,
                    top_clones: &analysis.clusters[..analysis.clusters.len().min(3)],
                },
                clone_clusters: &analysis.clusters,
                hotspot_files: &analysis.hotspots,
                commands: &commands,
                warnings: &walk.warnings,
                hint,
            };
            serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".to_string())
        }
        OutputMode::Human => {
            let mut txt = format!("Aroma Triage ({summary})\ndata_hash: {}\n", walk.data_hash);
            if analysis.clusters.is_empty() {
                if let Some(ref h) = hint {
                    let _ = writeln!(txt, "\nhint: {h}");
                }
            } else {
                txt.push_str("\nTop Clone Clusters:\n");
                for c in &analysis.clusters {
                    let members: Vec<String> = c
                        .members
                        .iter()
                        .take(3)
                        .map(|m| format!("{}:{}-{} `{}`", m.path, m.start_line, m.end_line, m.name))
                        .collect();
                    let _ = writeln!(
                        txt,
                        "  #{} [{} {}] score={:.1} dupL={} sim={:.3} | {}",
                        c.cluster_id,
                        c.language,
                        c.kind,
                        c.score,
                        c.duplicated_lines,
                        c.avg_sim,
                        members.join(" ; ")
                    );
                }
            }
            txt
        }
    };

    Ok(CommandOutcome {
        exit: ExitKind::Success,
        stdout,
        stderr_notes: Vec::new(),
    })
}

pub(crate) fn run_clones(cmd: &ClonesCmd) -> Result<CommandOutcome, CliError> {
    let walk = walk_paths(
        &cmd.paths,
        &WalkOptions {
            max_file_bytes: cmd.max_file_bytes,
            extra_ignores: cmd.ignores.clone(),
            lang_filter: cmd.lang.clone(),
        },
    )?;
    let index = build_index(
        &walk.files,
        IndexOptions {
            threads: cmd.threads,
            max_file_length: cmd.max_file_bytes,
            ..Default::default()
        },
    );
    let corpus = corpus_summary(&walk, &index);
    let analysis = analyze_clones(
        &walk,
        &index,
        &CloneParams {
            min_sim: cmd.min_sim,
            min_lines: cmd.min_lines,
            min_tokens: cmd.min_tokens,
            top: cmd.top,
            tests: cmd.tests,
            recs: cmd.recs,
            threads: cmd.threads,
        },
    );

    let path_str = format_path_args(&cmd.paths);
    let mut commands = Vec::new();
    if let Some(top_c) = analysis.clusters.first()
        && let Some(m) = top_c.members.first()
    {
        commands.push(SuggestedCommand {
            action: "recommend_cluster".to_string(),
            command: format!(
                "aroma recommend {path_str} --file {} --lines {}-{} --json",
                shell_quote(&m.path),
                m.start_line,
                m.end_line
            ),
            destructive: false,
        });
    }
    let hint = (analysis.total_clusters == 0).then(|| {
        format!(
            "No clones found at --min-sim {:.2}; try `aroma clones {path_str} --min-sim 0.65 --include-tests --json`",
            cmd.min_sim
        )
    });

    let stdout = match cmd.mode {
        OutputMode::Json => {
            #[derive(Serialize)]
            struct ClonesOut<'a> {
                ok: bool,
                tool_version: &'static str,
                contract_version: &'static str,
                data_hash: &'a str,
                corpus: &'a CorpusSummary,
                total_clusters: usize,
                clusters: &'a [CloneClusterOut],
                file_pairs: &'a [FilePairOut],
                commands: &'a [SuggestedCommand],
                warnings: &'a [WarningItem],
                hint: Option<String>,
            }
            let out = ClonesOut {
                ok: true,
                tool_version: env!("CARGO_PKG_VERSION"),
                contract_version: "1",
                data_hash: &walk.data_hash,
                corpus: &corpus,
                total_clusters: analysis.total_clusters,
                clusters: &analysis.clusters,
                file_pairs: &analysis.file_pairs,
                commands: &commands,
                warnings: &walk.warnings,
                hint,
            };
            serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".to_string())
        }
        OutputMode::Human => {
            let mut txt = format!(
                "aroma clones: {} clusters (showing {}), {} file pairs | files={} methods={}\n",
                analysis.total_clusters,
                analysis.clusters.len(),
                analysis.file_pairs.len(),
                corpus.files_indexed,
                corpus.methods,
            );
            if analysis.clusters.is_empty() {
                if let Some(ref h) = hint {
                    let _ = writeln!(txt, "hint: {h}");
                }
            } else {
                for c in &analysis.clusters {
                    let ms: Vec<String> = c
                        .members
                        .iter()
                        .take(4)
                        .map(|m| format!("{}:{}-{} `{}`", m.path, m.start_line, m.end_line, m.name))
                        .collect();
                    let _ = writeln!(
                        txt,
                        "#{} [{} {}] score={:.1} dupL={} sim={:.3} | {}",
                        c.cluster_id,
                        c.language,
                        c.kind,
                        c.score,
                        c.duplicated_lines,
                        c.avg_sim,
                        ms.join(" ; ")
                    );
                }
            }
            txt
        }
    };

    Ok(CommandOutcome {
        exit: ExitKind::Success,
        stdout,
        stderr_notes: Vec::new(),
    })
}

fn read_stdin_bounded() -> Result<String, CliError> {
    let mut buf = Vec::new();
    let limit = (MAX_SNIPPET_BYTES as u64).saturating_add(1);
    std::io::stdin()
        .lock()
        .take(limit)
        .read_to_end(&mut buf)
        .map_err(|e| {
            CliError::tool_env(
                format!("failed reading query snippet from stdin: {e}"),
                "I/O failure reading stdin",
                "aroma search . --snippet '<code...>' --lang rust --json",
                "aroma search --help",
            )
        })?;
    if buf.len() > MAX_SNIPPET_BYTES {
        return Err(CliError::user_input(
            format!("stdin query exceeds maximum size cap of {MAX_SNIPPET_BYTES} bytes"),
            "Query snippets are bounded before parsing",
            "aroma search . --file src/main.rs --lines 1-30 --json",
            "aroma capabilities --json",
        ));
    }
    String::from_utf8(buf).map_err(|_| {
        CliError::user_input(
            "stdin query is not valid UTF-8",
            "Source snippets must be valid UTF-8 text",
            "aroma search . --snippet '<code...>' --lang rust --json",
            "aroma search --help",
        )
    })
}

fn resolve_indexed_file_idx(walk: &WalkResult, target: &Path) -> Result<usize, CliError> {
    let norm = normalize_display_path(target);
    if let Some(pos) = walk.files.iter().position(|f| f.path == norm) {
        return Ok(pos);
    }
    if let Ok(target_canon) = fs::canonicalize(target) {
        for (i, p) in walk.fs_paths.iter().enumerate() {
            if let Ok(pc) = fs::canonicalize(p)
                && pc == target_canon
            {
                return Ok(i);
            }
        }
    }
    let suffix = format!("/{}", norm.trim_start_matches("./"));
    let suffix_matches: Vec<usize> = walk
        .files
        .iter()
        .enumerate()
        .filter(|(_, f)| f.path.ends_with(&suffix) || f.path == norm)
        .map(|(i, _)| i)
        .collect();
    if suffix_matches.len() == 1 {
        return Ok(suffix_matches[0]);
    }

    let cand_strs: Vec<&str> = walk.files.iter().map(|f| f.path.as_str()).collect();
    let hint_file = closest_match(&norm, &cand_strs, 6)
        .or_else(|| cand_strs.first().copied())
        .unwrap_or("src/main.rs");
    Err(CliError::user_input(
        format!("file '{}' was not found in the indexed corpus", target.display()),
        "The --file path must belong to the scanned target directory/paths",
        format!("aroma search . --file {} --lines 1-20 --json", shell_quote(hint_file)),
        "aroma search --help",
    ))
}

#[derive(Serialize)]
struct QueryMetaOut {
    source_kind: String,
    language: String,
    size: usize,
    origin: Option<MethodRefOut>,
}

#[derive(Serialize)]
struct MethodRefOut {
    path: String,
    name: String,
    language: String,
    start_line: u32,
    end_line: u32,
}

fn from_method_ref(m: &MethodRef) -> MethodRefOut {
    MethodRefOut {
        path: m.path.clone(),
        name: m.name.clone(),
        language: m.language.clone(),
        start_line: m.start_line,
        end_line: m.end_line,
    }
}

struct ResolvedQuery {
    query: Option<Query>,
    meta: QueryMetaOut,
    notes: Vec<String>,
    warnings: Vec<WarningItem>,
    empty_hint: Option<String>,
}

fn resolve_query(
    walk: &WalkResult,
    index: &AromaIndex,
    source: &QuerySource,
    explicit_lang: Option<&str>,
    subcmd: &str,
) -> Result<ResolvedQuery, CliError> {
    let mut notes = Vec::new();
    let mut warnings = walk.warnings.clone();

    match source {
        QuerySource::FileLines { file, lines } => {
            let idx = resolve_indexed_file_idx(walk, file)?;
            let sf = &walk.files[idx];
            let qs = queries_from_lines(index, sf, lines);
            let chosen = qs.into_iter().max_by_key(Query::size);
            let lang = infer_language_for_path(&sf.path)
                .unwrap_or("rust")
                .to_string();
            let empty_hint = chosen.is_none().then(|| {
                format!(
                    "No indexed method overlapped lines {:?} in '{}'; check line span or inspect via `aroma triage {} --json`",
                    lines,
                    sf.path,
                    shell_quote(&sf.path)
                )
            });
            let meta = QueryMetaOut {
                source_kind: "file_lines".to_string(),
                language: chosen
                    .as_ref()
                    .map_or(lang, |q| q.language().to_string()),
                size: chosen.as_ref().map_or(0, Query::size),
                origin: chosen
                    .as_ref()
                    .and_then(|q| q.origin.as_ref())
                    .map(from_method_ref),
            };
            Ok(ResolvedQuery {
                query: chosen,
                meta,
                notes,
                warnings,
                empty_hint,
            })
        }
        QuerySource::Snippet(text) => resolve_snippet_query(
            &SnippetQueryCtx {
                walk,
                index,
                explicit_lang,
                subcmd,
            },
            text,
            None,
            &mut notes,
            &mut warnings,
        ),
        QuerySource::SnippetFile(path) => {
            let bytes = read_file_bounded(path, MAX_SNIPPET_BYTES)?.ok_or_else(|| {
                CliError::user_input(
                    format!(
                        "--snippet-file '{}' exceeds maximum size cap of {MAX_SNIPPET_BYTES} bytes",
                        path.display()
                    ),
                    "Query snippet files are bounded before reading",
                    format!("aroma {subcmd} . --file {} --lines 1-40 --json", shell_quote(&path.to_string_lossy())),
                    "aroma capabilities --json",
                )
            })?;
            let text = String::from_utf8(bytes).map_err(|_| {
                CliError::user_input(
                    format!("--snippet-file '{}' is not valid UTF-8", path.display()),
                    "Query snippets must be UTF-8 text",
                    format!("aroma {subcmd} . --snippet '<code...>' --json"),
                    format!("aroma {subcmd} --help"),
                )
            })?;
            resolve_snippet_query(
                &SnippetQueryCtx {
                    walk,
                    index,
                    explicit_lang,
                    subcmd,
                },
                &text,
                Some(path.as_path()),
                &mut notes,
                &mut warnings,
            )
        }
        QuerySource::Stdin => {
            let text = read_stdin_bounded()?;
            resolve_snippet_query(
                &SnippetQueryCtx {
                    walk,
                    index,
                    explicit_lang,
                    subcmd,
                },
                &text,
                None,
                &mut notes,
                &mut warnings,
            )
        }
    }
}

struct SnippetQueryCtx<'a> {
    walk: &'a WalkResult,
    index: &'a AromaIndex,
    explicit_lang: Option<&'a str>,
    subcmd: &'a str,
}

fn resolve_snippet_query(
    ctx: &SnippetQueryCtx<'_>,
    text: &str,
    snippet_path: Option<&Path>,
    notes: &mut Vec<String>,
    warnings: &mut Vec<WarningItem>,
) -> Result<ResolvedQuery, CliError> {
    if text.trim().is_empty() {
        return Err(CliError::user_input(
            format!("empty query snippet for '{}'", ctx.subcmd),
            "Provide non-empty code via --snippet, --snippet-file, or stdin",
            format!("aroma {} . --snippet 'fn f(x: i32) -> i32 {{ x + 1 }}' --lang rust --json", ctx.subcmd),
            format!("aroma {} --help", ctx.subcmd),
        ));
    }

    let path_str = snippet_path.map(|p| p.to_string_lossy().to_string());
    let inferred = if let Some(l) = ctx.explicit_lang {
        l.to_string()
    } else if let Some(ref ps) = path_str {
        if let Some(l) = infer_language_for_path(ps) {
            notes.push(format!("note: inferred --lang {l} from --snippet-file '{ps}'"));
            l.to_string()
        } else if ctx.walk.language_counts.len() == 1 {
            let only = ctx.walk.language_counts.keys().next().cloned().unwrap_or_default();
            notes.push(format!("note: inferred --lang {only} from single-language corpus"));
            warnings.push(WarningItem {
                code: "inferred_language".to_string(),
                message: format!("Inferred --lang {only} from single-language indexed corpus"),
            });
            only
        } else {
            return Err(missing_lang_error(ctx.walk, ctx.subcmd));
        }
    } else if ctx.walk.language_counts.len() == 1 {
        let only = ctx.walk.language_counts.keys().next().cloned().unwrap_or_default();
        notes.push(format!("note: inferred --lang {only} from single-language corpus"));
        warnings.push(WarningItem {
            code: "inferred_language".to_string(),
            message: format!("Inferred --lang {only} from single-language indexed corpus"),
        });
        only
    } else if ctx.walk.files.is_empty() {
        return Ok(ResolvedQuery {
            query: None,
            meta: QueryMetaOut {
                source_kind: "snippet".to_string(),
                language: "unknown".to_string(),
                size: 0,
                origin: None,
            },
            notes: notes.clone(),
            warnings: warnings.clone(),
            empty_hint: Some("No supported source files were indexed under the given paths".to_string()),
        });
    } else {
        return Err(missing_lang_error(ctx.walk, ctx.subcmd));
    };

    let q = query_from_snippet(
        ctx.index,
        text,
        &SnippetOptions {
            language: Some(inferred.clone()),
            path: path_str,
            globals: None,
        },
    );
    let empty_hint = q.is_none().then(|| {
        format!(
            "Snippet did not match any indexed {inferred} structural features; try a longer code block or pass a different --lang"
        )
    });
    let meta = QueryMetaOut {
        source_kind: if snippet_path.is_some() {
            "snippet_file".to_string()
        } else {
            "snippet".to_string()
        },
        language: inferred,
        size: q.as_ref().map_or(0, Query::size),
        origin: None,
    };
    Ok(ResolvedQuery {
        query: q,
        meta,
        notes: notes.clone(),
        warnings: warnings.clone(),
        empty_hint,
    })
}

fn missing_lang_error(walk: &WalkResult, subcmd: &str) -> CliError {
    let langs: Vec<String> = walk.language_counts.keys().cloned().collect();
    let sample = langs.first().map_or("rust", String::as_str);
    CliError::user_input(
        format!("cannot infer snippet language across multi-language corpus ({})", langs.join(", ")),
        "Pass --lang when indexing multiple languages so Aroma parses the snippet with the matching grammar",
        format!("aroma {subcmd} . --snippet '<code...>' --lang {sample} --json"),
        format!("aroma {subcmd} --help"),
    )
}

struct QueryRunSpec<'a> {
    paths: &'a [PathBuf],
    max_file_bytes: usize,
    ignores: &'a [String],
    lang: &'a Option<String>,
    threads: Option<usize>,
    query: &'a QuerySource,
    subcmd: &'static str,
}

struct QueryRunCtx {
    walk: WalkResult,
    index: AromaIndex,
    corpus: CorpusSummary,
    resolved: ResolvedQuery,
    path_str: String,
}

fn prepare_query_run(spec: &QueryRunSpec<'_>) -> Result<QueryRunCtx, CliError> {
    let walk = walk_paths(
        spec.paths,
        &WalkOptions {
            max_file_bytes: spec.max_file_bytes,
            extra_ignores: spec.ignores.to_vec(),
            lang_filter: spec.lang.clone(),
        },
    )?;
    let index = build_index(
        &walk.files,
        IndexOptions {
            threads: spec.threads,
            max_file_length: spec.max_file_bytes,
            ..Default::default()
        },
    );
    let corpus = corpus_summary(&walk, &index);
    let resolved = resolve_query(&walk, &index, spec.query, spec.lang.as_deref(), spec.subcmd)?;
    let path_str = format_path_args(spec.paths);
    Ok(QueryRunCtx { walk, index, corpus, resolved, path_str })
}

fn is_excluded_origin(origin: OriginFilter, q: &Query, m: &MethodRef) -> bool {
    if origin == OriginFilter::IncludeOrigin {
        return false;
    }
    let Some(ref o) = q.origin else {
        return false;
    };
    m.path == o.path && m.start_line == o.start_line && m.end_line == o.end_line
}

fn render_query_json<P: Serialize>(
    ctx: &QueryRunCtx,
    total: usize,
    payload: P,
    commands: &[SuggestedCommand],
    hint: Option<String>,
) -> String {
    #[derive(Serialize)]
    struct QueryJsonOut<'a, P> {
        ok: bool,
        tool_version: &'static str,
        contract_version: &'static str,
        data_hash: &'a str,
        corpus: &'a CorpusSummary,
        query: &'a QueryMetaOut,
        total: usize,
        #[serde(flatten)]
        payload: P,
        commands: &'a [SuggestedCommand],
        warnings: &'a [WarningItem],
        hint: Option<String>,
    }
    let out = QueryJsonOut {
        ok: true,
        tool_version: env!("CARGO_PKG_VERSION"),
        contract_version: "1",
        data_hash: &ctx.walk.data_hash,
        corpus: &ctx.corpus,
        query: &ctx.resolved.meta,
        total,
        payload,
        commands,
        warnings: &ctx.resolved.warnings,
        hint,
    };
    serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".to_string())
}

pub(crate) fn run_search(cmd: &SearchCmd) -> Result<CommandOutcome, CliError> {
    #[derive(Serialize)]
    struct HitOut {
        method: MethodRefOut,
        similarity: f64,
        matched_lines: Vec<u32>,
    }
    #[derive(Serialize)]
    struct HitsPayload<'a> {
        hits: &'a [HitOut],
    }

    let mut ctx = prepare_query_run(&QueryRunSpec {
        paths: &cmd.paths,
        max_file_bytes: cmd.max_file_bytes,
        ignores: &cmd.ignores,
        lang: &cmd.lang,
        threads: cmd.threads,
        query: &cmd.query,
        subcmd: "search",
    })?;

    let mut hits_out: Vec<HitOut> = Vec::new();
    if let Some(ref q) = ctx.resolved.query {
        let exclude_origin = |m: &MethodRef| is_excluded_origin(cmd.origin, q, m);
        let opts = SearchOptions {
            search_limit: cmd.search_limit,
            limit: cmd.top,
            min_similarity: cmd.min_sim,
            exclude: Some(&exclude_origin),
        };
        for h in search(&ctx.index, q, &opts) {
            hits_out.push(HitOut {
                method: from_method_ref(&h.method),
                similarity: round3(h.similarity),
                matched_lines: h.matched_lines,
            });
        }
    }

    let mut commands = Vec::new();
    if let Some(first) = hits_out.first() {
        commands.push(SuggestedCommand {
            action: "recommend_top_hit".to_string(),
            command: format!(
                "aroma recommend {} --file {} --lines {}-{} --json",
                ctx.path_str,
                shell_quote(&first.method.path),
                first.method.start_line,
                first.method.end_line
            ),
            destructive: false,
        });
    }
    let hint = ctx.resolved.empty_hint.take().or_else(|| {
        hits_out.is_empty().then(|| {
            format!(
                "No matches found; try lowering threshold via `aroma search {} --lang {} --min-sim 0.0 --json`",
                ctx.path_str, ctx.resolved.meta.language
            )
        })
    });

    let stdout = match cmd.mode {
        OutputMode::Json => render_query_json(
            &ctx,
            hits_out.len(),
            HitsPayload { hits: &hits_out },
            &commands,
            hint,
        ),
        OutputMode::Human => {
            let mut txt = format!(
                "aroma search: {} hit(s) (lang={}, query_size={})\n",
                hits_out.len(),
                ctx.resolved.meta.language,
                ctx.resolved.meta.size
            );
            if let Some(ref h) = hint {
                let _ = writeln!(txt, "hint: {h}");
            }
            for (i, hit) in hits_out.iter().enumerate() {
                let _ = writeln!(
                    txt,
                    "  {}. sim={:.3} {}:{}-{} `{}` (matched_lines={:?})",
                    i + 1,
                    hit.similarity,
                    hit.method.path,
                    hit.method.start_line,
                    hit.method.end_line,
                    hit.method.name,
                    hit.matched_lines
                );
            }
            txt
        }
    };

    Ok(CommandOutcome {
        exit: ExitKind::Success,
        stdout,
        stderr_notes: ctx.resolved.notes,
    })
}

pub(crate) fn run_recommend(cmd: &RecommendCmd) -> Result<CommandOutcome, CliError> {
    #[derive(Serialize)]
    struct LineOut {
        line: u32,
        role: &'static str,
        text: String,
    }
    #[derive(Serialize)]
    struct RecOut {
        method: MethodRefOut,
        member_count: usize,
        members: Vec<MethodRefOut>,
        similarity: f64,
        lines: Vec<LineOut>,
        code: String,
    }
    #[derive(Serialize)]
    struct RecsPayload<'a> {
        recommendations: &'a [RecOut],
    }

    let mut ctx = prepare_query_run(&QueryRunSpec {
        paths: &cmd.paths,
        max_file_bytes: cmd.max_file_bytes,
        ignores: &cmd.ignores,
        lang: &cmd.lang,
        threads: cmd.threads,
        query: &cmd.query,
        subcmd: "recommend",
    })?;

    let mut recs_out: Vec<RecOut> = Vec::new();
    if let Some(ref q) = ctx.resolved.query {
        let exclude_origin = |m: &MethodRef| is_excluded_origin(cmd.origin, q, m);
        let opts = RecommendOptions {
            search_limit: cmd.search_limit,
            cluster_limit: cmd.cluster_limit,
            min_similarity: cmd.min_sim,
            limit: cmd.top,
            min_support: cmd.min_support,
            exclude: Some(&exclude_origin),
            ..Default::default()
        };
        for rec in recommend(&ctx.index, q, &opts) {
            let lines = rec
                .lines
                .into_iter()
                .map(|l| LineOut {
                    line: l.line,
                    role: match l.role {
                        LineRole::Query => "query",
                        LineRole::Added => "added",
                        LineRole::Structure => "structure",
                    },
                    text: l.text,
                })
                .collect();
            recs_out.push(RecOut {
                method: from_method_ref(&rec.method),
                member_count: rec.members.len(),
                members: rec.members.iter().map(from_method_ref).collect(),
                similarity: round3(rec.similarity),
                lines,
                code: rec.code,
            });
        }
    }

    let mut commands = Vec::new();
    if let Some(first) = recs_out.first() {
        commands.push(SuggestedCommand {
            action: "inspect_exemplar".to_string(),
            command: format!(
                "aroma search {} --file {} --lines {}-{} --json",
                ctx.path_str,
                shell_quote(&first.method.path),
                first.method.start_line,
                first.method.end_line
            ),
            destructive: false,
        });
    }
    let hint = ctx.resolved.empty_hint.take().or_else(|| {
        recs_out.is_empty().then(|| {
            format!(
                "No recommendations reached --min-support {} and --min-sim {:.2}; try `aroma search {} --lang {} --json` or lower --min-support",
                cmd.min_support, cmd.min_sim, ctx.path_str, ctx.resolved.meta.language
            )
        })
    });

    let stdout = match cmd.mode {
        OutputMode::Json => render_query_json(
            &ctx,
            recs_out.len(),
            RecsPayload { recommendations: &recs_out },
            &commands,
            hint,
        ),
        OutputMode::Human => {
            let mut txt = format!(
                "aroma recommend: {} cluster(s) (lang={}, query_size={})\n",
                recs_out.len(),
                ctx.resolved.meta.language,
                ctx.resolved.meta.size
            );
            if let Some(ref h) = hint {
                let _ = writeln!(txt, "hint: {h}");
            }
            for (i, r) in recs_out.iter().enumerate() {
                let _ = writeln!(
                    txt,
                    "\n--- Recommendation #{} (sim={:.3}, members={}) {}:{} `{}` ---\n{}",
                    i + 1,
                    r.similarity,
                    r.member_count,
                    r.method.path,
                    r.method.start_line,
                    r.method.name,
                    r.code
                );
            }
            txt
        }
    };

    Ok(CommandOutcome {
        exit: ExitKind::Success,
        stdout,
        stderr_notes: ctx.resolved.notes,
    })
}

/// One changed file for `aroma check`; `lines: None` means every line of the file.
struct CheckTarget {
    path: PathBuf,
    lines: Option<Vec<u32>>,
}

/// Where a resolved check target's source text lives.
enum TargetSource {
    Indexed(usize),
    External(SourceFile),
}

struct ResolvedTarget {
    source: TargetSource,
    lines: Option<Vec<u32>>,
}

impl ResolvedTarget {
    fn file<'a>(&'a self, walk: &'a WalkResult) -> &'a SourceFile {
        match &self.source {
            TargetSource::Indexed(i) => &walk.files[*i],
            TargetSource::External(sf) => sf,
        }
    }
}

#[derive(Serialize)]
struct CheckMatchOut {
    method: MethodRefOut,
    similarity: f64,
    matched_lines: Vec<u32>,
}

#[derive(Serialize)]
struct CheckFindingOut {
    changed: MethodRefOut,
    changed_lines: Vec<u32>,
    query_size: usize,
    matches: Vec<CheckMatchOut>,
}

/// Groups per-file skip reasons into one warning per code instead of one per file.
#[derive(Default)]
struct SkipLog {
    by_code: BTreeMap<&'static str, (&'static str, Vec<String>)>,
}

impl SkipLog {
    fn push(&mut self, code: &'static str, reason: &'static str, path: String) {
        self.by_code.entry(code).or_insert((reason, Vec::new())).1.push(path);
    }

    fn into_warnings(self) -> Vec<WarningItem> {
        self.by_code
            .into_iter()
            .map(|(code, (reason, paths))| {
                let shown: Vec<&str> = paths.iter().take(5).map(String::as_str).collect();
                let more = paths.len() - shown.len();
                let tail = if more > 0 { format!(", +{more} more") } else { String::new() };
                WarningItem {
                    code: code.to_string(),
                    message: format!("{} changed file(s) {reason}: {}{tail}", paths.len(), shown.join(", ")),
                }
            })
            .collect()
    }
}

fn run_git(dir: &Path, args: &[&str]) -> Result<std::process::Output, CliError> {
    Command::new("git").arg("-C").arg(dir).args(args).output().map_err(|e| {
        CliError::tool_env(
            format!("failed to run git: {e}"),
            "`aroma check --diff` shells out to git to discover changed lines; git must be on PATH",
            "aroma check <FILE>:<LINES> .",
            "aroma check --help",
        )
    })
}

fn git_stderr_line(out: &std::process::Output) -> String {
    let err = String::from_utf8_lossy(&out.stderr);
    err.lines().find(|l| !l.trim().is_empty()).unwrap_or("git exited with an error").trim().to_string()
}

/// Decodes a git C-quoted path (`"dir/caf\303\251.rs"`); unquoted input is returned as-is.
fn unquote_git_path(raw: &str) -> Option<String> {
    let Some(inner) = raw.strip_prefix('"').and_then(|r| r.strip_suffix('"')) else {
        return Some(raw.to_string());
    };
    let bytes = inner.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b != b'\\' {
            out.push(b);
            i += 1;
            continue;
        }
        let esc = *bytes.get(i + 1)?;
        if esc.is_ascii_digit() {
            let mut v: u32 = 0;
            for &d in bytes.get(i + 1..i + 4)? {
                if !(b'0'..=b'7').contains(&d) {
                    return None;
                }
                v = v * 8 + u32::from(d - b'0');
            }
            out.push(u8::try_from(v).ok()?);
            i += 4;
            continue;
        }
        out.push(match esc {
            b'n' => b'\n',
            b't' => b'\t',
            b'r' => b'\r',
            b'a' => 0x07,
            b'b' => 0x08,
            b'f' => 0x0c,
            b'v' => 0x0b,
            other => other,
        });
        i += 2;
    }
    String::from_utf8(out).ok()
}

/// Parses the new-side `+c[,d]` range of a `@@ -a,b +c,d @@` hunk header.
fn parse_hunk_new_range(line: &str) -> Option<(u32, u32)> {
    let rest = line.strip_prefix("@@ -")?;
    let new = &rest[rest.find(" +")? + 2..];
    let spec = &new[..new.find(' ').unwrap_or(new.len())];
    match spec.split_once(',') {
        Some((start, count)) => Some((start.parse().ok()?, count.parse().ok()?)),
        None => Some((spec.parse().ok()?, 1)),
    }
}

/// Extracts new-side changed lines per file from `git diff --unified=0` output. Paths are
/// repo-root relative; files whose hunks exceed `MAX_LINE_SPEC_COUNT` lines degrade to whole-file.
fn parse_unified_zero_diff(diff: &str) -> Vec<CheckTarget> {
    let mut out: Vec<CheckTarget> = Vec::new();
    let mut in_header = false;
    let mut current: Option<usize> = None;
    for line in diff.lines() {
        if line.starts_with("diff --git ") {
            in_header = true;
            current = None;
            continue;
        }
        if in_header && let Some(raw) = line.strip_prefix("+++ ") {
            let raw = raw.strip_suffix('\t').unwrap_or(raw);
            current = unquote_git_path(raw)
                .and_then(|p| p.strip_prefix("b/").map(str::to_string))
                .map(|p| {
                    out.push(CheckTarget { path: PathBuf::from(p), lines: Some(Vec::new()) });
                    out.len() - 1
                });
            continue;
        }
        if line.starts_with("@@ ") {
            in_header = false;
            let (Some(ci), Some((start, count))) = (current, parse_hunk_new_range(line)) else {
                continue;
            };
            let target = &mut out[ci].lines;
            if let Some(lines) = target {
                if lines.len().saturating_add(count as usize) > crate::cli::args::MAX_LINE_SPEC_COUNT {
                    *target = None;
                } else {
                    lines.extend(start..start.saturating_add(count));
                }
            }
        }
    }
    out.retain(|t| t.lines.as_ref().is_none_or(|l| !l.is_empty()));
    out
}

/// Runs git in the repository containing `hint_dir` and returns changed files as paths
/// relative to the current directory when possible (absolute otherwise).
fn discover_git_changes(diff: &DiffSpec, base: &str, hint_dir: &Path) -> Result<Vec<CheckTarget>, CliError> {
    let top_out = run_git(hint_dir, &["rev-parse", "--show-toplevel"])?;
    if !top_out.status.success() {
        return Err(CliError::tool_env(
            format!("'{}' is not inside a git work tree: {}", hint_dir.display(), git_stderr_line(&top_out)),
            "`aroma check --diff` reads changed lines from `git diff`; run it inside a git repository",
            "aroma check <FILE>:<LINES> .",
            "aroma check --help",
        ));
    }
    let toplevel = PathBuf::from(String::from_utf8_lossy(&top_out.stdout).trim_end_matches(['\n', '\r']));

    let mut diff_args = vec![
        "-c", "core.quotePath=false", "diff", "--unified=0", "--no-color", "--no-ext-diff",
        "--no-textconv", "--ignore-submodules", "--src-prefix=a/", "--dst-prefix=b/",
    ];
    if diff.staged {
        diff_args.push("--cached");
    }
    diff_args.push(base);
    diff_args.push("--");
    let diff_out = run_git(&toplevel, &diff_args)?;
    if !diff_out.status.success() {
        return Err(CliError::user_input(
            format!("git diff {base} failed: {}", git_stderr_line(&diff_out)),
            "The --diff base must be a valid git revision (default HEAD, e.g. origin/main)",
            "aroma check --diff=HEAD .",
            "aroma check --help",
        ));
    }
    let mut targets = parse_unified_zero_diff(&String::from_utf8_lossy(&diff_out.stdout));

    if !diff.staged && base == "HEAD" {
        let ls = run_git(&toplevel, &["ls-files", "--others", "--exclude-standard", "-z"])?;
        if !ls.status.success() {
            return Err(CliError::tool_env(
                format!("git ls-files failed: {}", git_stderr_line(&ls)),
                "Untracked files are listed with `git ls-files --others --exclude-standard`",
                "aroma check --diff --staged .",
                "aroma check --help",
            ));
        }
        for raw in ls.stdout.split(|&b| b == 0).filter(|s| !s.is_empty()) {
            let path = String::from_utf8_lossy(raw).into_owned();
            targets.push(CheckTarget { path: PathBuf::from(path), lines: None });
        }
    }

    let cwd = std::env::current_dir().and_then(fs::canonicalize).ok();
    let top_canon = fs::canonicalize(&toplevel).unwrap_or(toplevel);
    for t in &mut targets {
        let abs = top_canon.join(&t.path);
        t.path = match cwd.as_deref().and_then(|c| abs.strip_prefix(c).ok()) {
            Some(rel) => rel.to_path_buf(),
            None => abs,
        };
    }
    Ok(targets)
}

/// True when sorted `lines` contains any line in `start..=end`.
fn overlaps_sorted(lines: &[u32], start: u32, end: u32) -> bool {
    let i = lines.partition_point(|&l| l < start);
    i < lines.len() && lines[i] <= end
}

pub(crate) fn run_check(cmd: &CheckCmd) -> Result<CommandOutcome, CliError> {
    // Classify positionals: with --diff they are an optional leading base revision plus corpus
    // PATHs; otherwise FILE[:LINES] targets (bare regular files) plus corpus PATHs.
    let mut base = cmd.diff.as_ref().and_then(|d| d.base.clone());
    let mut corpus_paths: Vec<PathBuf> = Vec::new();
    let mut explicit: Vec<CheckTarget> = Vec::new();
    for (i, arg) in cmd.args.iter().enumerate() {
        let meta = fs::metadata(&arg.path);
        if cmd.diff.is_some() {
            if i == 0 && base.is_none() && meta.is_err() {
                base = Some(arg.path.to_string_lossy().into_owned());
            } else {
                corpus_paths.push(arg.path.clone());
            }
        } else if arg.lines.is_some() || meta.is_ok_and(|m| m.is_file()) {
            explicit.push(CheckTarget { path: arg.path.clone(), lines: arg.lines.clone() });
        } else {
            corpus_paths.push(arg.path.clone());
        }
    }
    let in_diff = cmd.diff.is_some();
    let base_ref = base.as_deref().unwrap_or("HEAD");

    let walk = walk_paths(
        &corpus_paths,
        &WalkOptions {
            max_file_bytes: cmd.max_file_bytes,
            extra_ignores: cmd.ignores.clone(),
            lang_filter: cmd.lang.clone(),
        },
    )?;

    let targets = match &cmd.diff {
        Some(d) => {
            let hint_dir = match corpus_paths.first() {
                Some(p) if p.is_dir() => p.clone(),
                Some(p) => p
                    .parent()
                    .filter(|d| !d.as_os_str().is_empty())
                    .map_or_else(|| PathBuf::from("."), Path::to_path_buf),
                None => PathBuf::from("."),
            };
            discover_git_changes(d, base_ref, &hint_dir)?
        }
        None => explicit,
    };

    // Resolve targets to corpus files (by display path, then canonical path), merging duplicates.
    let mut skips = SkipLog::default();
    let mut resolved: Vec<ResolvedTarget> = Vec::new();
    let mut by_path: FxHashMap<String, usize> = FxHashMap::default();
    let mut canon_map: Option<FxHashMap<PathBuf, usize>> = None;
    for t in targets {
        let disp = normalize_display_path(&t.path);
        let mut idx = walk.files.iter().position(|f| f.path == disp);
        if idx.is_none()
            && let Ok(tc) = fs::canonicalize(&t.path)
        {
            let map = canon_map.get_or_insert_with(|| {
                walk.fs_paths
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| fs::canonicalize(p).ok().map(|c| (c, i)))
                    .collect()
            });
            idx = map.get(&tc).copied();
        }
        let source = if let Some(i) = idx {
            TargetSource::Indexed(i)
        } else {
            let Some(lang) = infer_language_for_path(&disp) else {
                skips.push("unsupported_changed_files", "skipped (unsupported language)", disp);
                continue;
            };
            if cmd.lang.as_deref().is_some_and(|l| l != lang) {
                skips.push("lang_filtered_changed_files", "skipped (excluded by --lang)", disp);
                continue;
            }
            if in_diff {
                skips.push(
                    "changed_files_outside_corpus",
                    "skipped (outside the indexed PATHs, ignored, oversized, or non-UTF-8)",
                    disp,
                );
                continue;
            }
            let Some(bytes) = read_file_bounded(&t.path, cmd.max_file_bytes)? else {
                skips.push("oversized_changed_files", "skipped (exceeded --max-file-bytes)", disp);
                continue;
            };
            let Ok(source) = String::from_utf8(bytes) else {
                skips.push("non_utf8_changed_files", "skipped (not valid UTF-8)", disp);
                continue;
            };
            TargetSource::External(SourceFile { path: disp, source })
        };
        let key = match &source {
            TargetSource::Indexed(i) => walk.files[*i].path.clone(),
            TargetSource::External(sf) => sf.path.clone(),
        };
        if let Some(&ri) = by_path.get(&key) {
            match (&mut resolved[ri].lines, t.lines) {
                (Some(acc), Some(more)) => acc.extend(more),
                (slot, _) => *slot = None,
            }
        } else {
            by_path.insert(key, resolved.len());
            resolved.push(ResolvedTarget { source, lines: t.lines });
        }
    }

    let index = build_index(
        &walk.files,
        IndexOptions {
            threads: cmd.threads,
            max_file_length: cmd.max_file_bytes,
            ..Default::default()
        },
    );
    let corpus = corpus_summary(&walk, &index);

    // Sorted, deduplicated changed lines per target (whole-file targets expand here).
    let changed_lines: Vec<Vec<u32>> = resolved
        .iter_mut()
        .map(|r| {
            let mut lines = match r.lines.take() {
                Some(l) => l,
                None => (1..=line_count_for(&r.file(&walk).source)).collect(),
            };
            lines.sort_unstable();
            lines.dedup();
            lines
        })
        .collect();
    let changed_by_path: FxHashMap<&str, &[u32]> = resolved
        .iter()
        .zip(&changed_lines)
        .map(|(r, l)| (r.file(&walk).path.as_str(), l.as_slice()))
        .collect();

    let mut queries: Vec<Query> = Vec::new();
    for (r, lines) in resolved.iter().zip(&changed_lines) {
        queries.extend(queries_from_lines(&index, r.file(&walk), lines));
    }
    let total_changed = queries.len();
    queries.retain(|q| q.size() >= cmd.min_size);
    let checked = queries.len();
    let skipped_small = total_changed - checked;

    // New code must never match other new code: drop every method overlapping a changed line
    // in any changed file, plus test files unless --include-tests.
    let exclude_tests = cmd.tests == TestFilter::ExcludeTests;
    let exclude = |m: &MethodRef| -> bool {
        (exclude_tests && is_test_path(&m.path))
            || changed_by_path
                .get(m.path.as_str())
                .is_some_and(|lines| overlaps_sorted(lines, m.start_line, m.end_line))
    };
    let opts = SearchOptions {
        search_limit: cmd.search_limit,
        limit: cmd.top,
        min_similarity: cmd.min_sim,
        exclude: Some(&exclude),
    };
    let mut findings: Vec<CheckFindingOut> = Vec::new();
    for q in &queries {
        let Some(origin) = &q.origin else {
            continue;
        };
        let matches: Vec<CheckMatchOut> = search(&index, q, &opts)
            .into_iter()
            .map(|h| CheckMatchOut {
                method: from_method_ref(&h.method),
                similarity: round3(h.similarity),
                matched_lines: h.matched_lines,
            })
            .collect();
        if !matches.is_empty() {
            findings.push(CheckFindingOut {
                changed: from_method_ref(origin),
                changed_lines: q.lines.clone().unwrap_or_default(),
                query_size: q.size(),
                matches,
            });
        }
    }

    let path_str = format_path_args(&corpus_paths);
    let mut commands = Vec::new();
    if let Some(f) = findings.first() {
        commands.push(SuggestedCommand {
            action: "search_changed_function".to_string(),
            command: format!(
                "aroma search {path_str} --file {} --lines {}-{} --json",
                shell_quote(&f.changed.path),
                f.changed.start_line,
                f.changed.end_line
            ),
            destructive: false,
        });
    }
    let staged = cmd.diff.as_ref().is_some_and(|d| d.staged);
    let hint = if resolved.is_empty() {
        Some(if in_diff {
            format!(
                "No changed supported source files in `git diff {}{base_ref}`; nothing to check",
                if staged { "--cached " } else { "" }
            )
        } else {
            "No supported source files to check".to_string()
        })
    } else if total_changed == 0 {
        Some("No indexed function overlaps the changed lines (only top-level or non-code lines changed)".to_string())
    } else if checked == 0 {
        Some(format!(
            "All {skipped_small} changed function(s) are below --min-size {}; lower it to check small functions (tiny queries match unrelated code)",
            cmd.min_size
        ))
    } else if findings.is_empty() {
        Some(format!(
            "No existing function reached --min-sim {:.2}; the changed code does not duplicate the indexed corpus",
            cmd.min_sim
        ))
    } else {
        None
    };

    let skip_warnings = skips.into_warnings();
    let mut warnings = walk.warnings.clone();
    warnings.extend(skip_warnings.iter().cloned());

    let mut stderr_notes = Vec::new();
    let stdout = match cmd.mode {
        OutputMode::Json => {
            #[derive(Serialize)]
            struct CheckOut<'a> {
                ok: bool,
                tool_version: &'static str,
                contract_version: &'static str,
                data_hash: &'a str,
                corpus: &'a CorpusSummary,
                source: &'static str,
                base: Option<&'a str>,
                changed_files: usize,
                checked: usize,
                skipped_small: usize,
                total: usize,
                findings: &'a [CheckFindingOut],
                commands: &'a [SuggestedCommand],
                warnings: &'a [WarningItem],
                hint: Option<String>,
            }
            let out = CheckOut {
                ok: true,
                tool_version: env!("CARGO_PKG_VERSION"),
                contract_version: "1",
                data_hash: &walk.data_hash,
                corpus: &corpus,
                source: if in_diff { "git_diff" } else { "file_lines" },
                base: in_diff.then_some(base_ref),
                changed_files: resolved.len(),
                checked,
                skipped_small,
                total: findings.len(),
                findings: &findings,
                commands: &commands,
                warnings: &warnings,
                hint,
            };
            serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".to_string())
        }
        OutputMode::Human => {
            stderr_notes.extend(skip_warnings.iter().map(|w| format!("warning: {}", w.message)));
            let mut txt = String::new();
            for f in &findings {
                let _ = writeln!(
                    txt,
                    "{}:{}-{}  {}  likely duplicates existing code; consider reusing:",
                    f.changed.path, f.changed.start_line, f.changed.end_line, f.changed.name
                );
                for m in &f.matches {
                    let _ = writeln!(
                        txt,
                        "    {:.3}  {}  {}:{}-{}",
                        m.similarity, m.method.name, m.method.path, m.method.start_line, m.method.end_line
                    );
                }
                txt.push('\n');
            }
            if let Some(h) = &hint {
                let _ = writeln!(txt, "hint: {h}");
            }
            let _ = write!(
                txt,
                "{checked} changed function{} checked, {} with likely existing equivalents",
                if checked == 1 { "" } else { "s" },
                findings.len()
            );
            if skipped_small > 0 {
                let _ = write!(txt, " ({skipped_small} skipped below --min-size {})", cmd.min_size);
            }
            txt.push('\n');
            txt
        }
    };

    Ok(CommandOutcome {
        exit: if cmd.fail_on_findings && !findings.is_empty() { ExitKind::Findings } else { ExitKind::Success },
        stdout,
        stderr_notes,
    })
}

pub(crate) fn run_bench(cmd: &BenchCmd) -> Result<CommandOutcome, CliError> {
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

    const MAX_BENCH_JSON_BYTES: usize = 256 * 1024 * 1024;
    let manifest_bytes = read_file_bounded(&cmd.manifest, MAX_BENCH_JSON_BYTES)?.ok_or_else(|| {
        CliError::user_input(
            format!("manifest '{}' exceeds {MAX_BENCH_JSON_BYTES} byte cap", cmd.manifest.display()),
            "Benchmark manifests are bounded before loading",
            "aroma bench --manifest corpus.json --queries queries.json",
            "aroma bench --help",
        )
    })?;
    let queries_bytes = read_file_bounded(&cmd.queries, MAX_BENCH_JSON_BYTES)?.ok_or_else(|| {
        CliError::user_input(
            format!("queries '{}' exceeds {MAX_BENCH_JSON_BYTES} byte cap", cmd.queries.display()),
            "Benchmark query suites are bounded before loading",
            "aroma bench --manifest corpus.json --queries queries.json",
            "aroma bench --help",
        )
    })?;

    let entries: Vec<ManifestEntry> = serde_json::from_slice(&manifest_bytes).map_err(|e| {
        CliError::tool_env(
            format!("failed to parse manifest '{}': {e}", cmd.manifest.display()),
            "Expected JSON array of {path, source} objects",
            "aroma bench --manifest corpus.json --queries queries.json",
            "aroma schema --command bench --json",
        )
    })?;
    let files: Vec<SourceFile> = entries
        .into_iter()
        .map(|e| SourceFile {
            path: e.path,
            source: e.source,
        })
        .collect();
    let file_map: HashMap<&str, &SourceFile> =
        files.iter().map(|f| (f.path.as_str(), f)).collect();

    let query_specs: Vec<QuerySpec> = serde_json::from_slice(&queries_bytes).map_err(|e| {
        CliError::tool_env(
            format!("failed to parse queries '{}': {e}", cmd.queries.display()),
            "Expected JSON array of benchmark QuerySpec objects",
            "aroma bench --manifest corpus.json --queries queries.json",
            "aroma schema --command bench --json",
        )
    })?;

    let threads_used = cmd
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .max(1);

    let t0 = Instant::now();
    let index = build_index(
        black_box(&files),
        IndexOptions {
            threads: Some(threads_used),
            ..Default::default()
        },
    );
    let index_ms = t0.elapsed().as_secs_f64() * 1000.0;

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
                .is_some_and(|r| r.method.path == spec.path && r.method.start_line == spec.start_line)
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

    let percentile = |sorted: &[f64], p: f64| -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f64 * p).floor() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    let search_total_ms = search_us.iter().sum::<f64>() / 1000.0;
    let recommend_total_ms = recommend_us.iter().sum::<f64>() / 1000.0;
    search_us.sort_by(f64::total_cmp);
    recommend_us.sort_by(f64::total_cmp);

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

    let stdout = serde_json::to_string_pretty(&out).map_err(|e| {
        CliError::tool_env(
            format!("failed serializing bench output: {e}"),
            "Unexpected JSON serialization error",
            "aroma bench --help",
            "aroma capabilities --json",
        )
    })?;

    Ok(CommandOutcome {
        exit: ExitKind::Success,
        stdout,
        stderr_notes: Vec::new(),
    })
}
