use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;

use aroma::{
    build_index, prune, queries_from_lines, query_from_snippet, recommend, search, IndexOptions,
    RecommendOptions, SearchOptions, SnippetOptions, SourceFile, TokenTable,
};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn range(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

fn gen_variant_parts(rng: &mut Rng, renamed: bool) -> (String, String, String, String, String, usize, usize, usize, usize) {
    let tag = if renamed { "ren" } else { "orig" };
    let p0 = format!("arg_a_{tag}");
    let p1 = format!("arg_b_{tag}");
    let acc = format!("acc_{tag}");
    let step = format!("step_{tag}");
    let idx = format!("idx_{tag}");
    let c0 = rng.range(10) + 1;
    let c1 = rng.range(20) + 2;
    let c2 = rng.range(7) + 2;
    let c3 = rng.range(8) + 2;
    (p0, p1, acc, step, idx, c0, c1, c2, c3)
}

fn gen_rust_pair(rng: &mut Rng, seed: usize) -> (String, String) {
    let mut seed_copy = Rng(rng.0);
    let mode = seed % 3;
    let build = |r: &mut Rng, renamed: bool| {
        let (p0, p1, acc, step, idx, c0, c1, c2, c3) = gen_variant_parts(r, renamed);
        let fn_name = if renamed { format!("compute_ren_{seed}") } else { format!("compute_orig_{seed}") };
        let mut out = String::new();
        let _ = writeln!(out, "pub fn {fn_name}({p0}: i32, {p1}: i32) -> i32 {{");
        let _ = writeln!(out, "    let mut {acc} = {c0};");
        let _ = writeln!(out, "    let {step} = {p0} + {p1};");
        match mode {
            0 => {
                let _ = writeln!(out, "    if {step} > {c1} {{");
                let _ = writeln!(out, "        {acc} += {step} * {c2};");
                let _ = writeln!(out, "    }} else {{");
                let _ = writeln!(out, "        {acc} -= {p1};");
                let _ = writeln!(out, "    }}");
            }
            1 => {
                let _ = writeln!(out, "    let mut {idx} = 0;");
                let _ = writeln!(out, "    while {idx} < {c3} {{");
                let _ = writeln!(out, "        {acc} = helper_call({acc}, {step} + {idx});");
                let _ = writeln!(out, "        {idx} += 1;");
                let _ = writeln!(out, "    }}");
            }
            _ => {
                let _ = writeln!(out, "    for {idx} in 0..{c3} {{");
                let _ = writeln!(out, "        if ({idx} + {p0}) % 2 == 0 {{");
                let _ = writeln!(out, "            {acc} = helper_call({acc}, {step} * {c2});");
                let _ = writeln!(out, "        }}");
                let _ = writeln!(out, "    }}");
            }
        }
        let _ = writeln!(out, "    {acc}");
        out.push_str("}\n");
        out
    };
    let a = build(rng, false);
    let b = build(&mut seed_copy, true);
    (a, b)
}

fn gen_py_pair(rng: &mut Rng, seed: usize) -> (String, String) {
    let mut seed_copy = Rng(rng.0);
    let mode = seed % 3;
    let build = |r: &mut Rng, renamed: bool| {
        let (p0, p1, acc, step, idx, c0, c1, c2, c3) = gen_variant_parts(r, renamed);
        let fn_name = if renamed { format!("py_ren_{seed}") } else { format!("py_orig_{seed}") };
        let mut out = String::new();
        let _ = writeln!(out, "def {fn_name}({p0}, {p1}):");
        let _ = writeln!(out, "    {acc} = {c0}");
        let _ = writeln!(out, "    {step} = {p0} + {p1}");
        match mode {
            0 => {
                let _ = writeln!(out, "    if {step} > {c1}:");
                let _ = writeln!(out, "        {acc} += {step} * {c2}");
                let _ = writeln!(out, "    else:");
                let _ = writeln!(out, "        {acc} -= {p1}");
            }
            1 => {
                let _ = writeln!(out, "    {idx} = 0");
                let _ = writeln!(out, "    while {idx} < {c3}:");
                let _ = writeln!(out, "        {acc} = helper_call({acc}, {step} + {idx})");
                let _ = writeln!(out, "        {idx} += 1");
            }
            _ => {
                let _ = writeln!(out, "    for {idx} in range({c3}):");
                let _ = writeln!(out, "        if ({idx} + {p0}) % 2 == 0:");
                let _ = writeln!(out, "            {acc} = helper_call({acc}, {step} * {c2})");
            }
        }
        let _ = writeln!(out, "    return {acc}");
        out
    };
    let a = build(rng, false);
    let b = build(&mut seed_copy, true);
    (a, b)
}

fn gen_ts_pair(rng: &mut Rng, seed: usize) -> (String, String) {
    let mut seed_copy = Rng(rng.0);
    let mode = seed % 3;
    let build = |r: &mut Rng, renamed: bool| {
        let (p0, p1, acc, step, idx, c0, c1, c2, c3) = gen_variant_parts(r, renamed);
        let fn_name = if renamed { format!("ts_ren_{seed}") } else { format!("ts_orig_{seed}") };
        let mut out = String::new();
        let _ = writeln!(out, "function {fn_name}({p0}: number, {p1}: number): number {{");
        let _ = writeln!(out, "    let {acc} = {c0};");
        let _ = writeln!(out, "    const {step} = {p0} + {p1};");
        match mode {
            0 => {
                let _ = writeln!(out, "    if ({step} > {c1}) {{");
                let _ = writeln!(out, "        {acc} += {step} * {c2};");
                let _ = writeln!(out, "    }} else {{");
                let _ = writeln!(out, "        {acc} -= {p1};");
                let _ = writeln!(out, "    }}");
            }
            1 => {
                let _ = writeln!(out, "    let {idx} = 0;");
                let _ = writeln!(out, "    while ({idx} < {c3}) {{");
                let _ = writeln!(out, "        {acc} = helper_call({acc}, {step} + {idx});");
                let _ = writeln!(out, "        {idx} += 1;");
                let _ = writeln!(out, "    }}");
            }
            _ => {
                let _ = writeln!(out, "    for (let {idx} = 0; {idx} < {c3}; {idx}++) {{");
                let _ = writeln!(out, "        if (({idx} + {p0}) % 2 === 0) {{");
                let _ = writeln!(out, "            {acc} = helper_call({acc}, {step} * {c2});");
                let _ = writeln!(out, "        }}");
                let _ = writeln!(out, "    }}");
            }
        }
        let _ = writeln!(out, "    return {acc};");
        out.push_str("}\n");
        out
    };
    let a = build(rng, false);
    let b = build(&mut seed_copy, true);
    (a, b)
}

#[test]
fn prop_alpha_renaming_invariance_across_languages() {
    let mut rng = Rng(0x5A17_C0DE_2026_1122);
    for seed in 0..30 {
        let cases = [
            ("rust", "a.rs", gen_rust_pair(&mut rng, seed)),
            ("python", "a.py", gen_py_pair(&mut rng, seed)),
            ("typescript", "a.ts", gen_ts_pair(&mut rng, seed)),
        ];
        for (lang, path, (src_a, src_b)) in cases {
            let files = [
                SourceFile { path: path.into(), source: src_a },
                SourceFile { path: format!("ren_{path}"), source: src_b },
            ];
            let idx = build_index(&files, IndexOptions { threads: Some(1), ..Default::default() });

            let qa = queries_from_lines(&idx, &files[0], &[1, 2, 3, 4, 5, 6, 7, 8]);
            let qb = queries_from_lines(&idx, &files[1], &[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(qa.len(), 1, "{lang} seed {seed} orig query missing");
            assert_eq!(qb.len(), 1, "{lang} seed {seed} renamed query missing");
            assert_eq!(
                qa[0].size(),
                qb[0].size(),
                "{lang} seed {seed} feature count drifted across alpha-renaming"
            );

            let hits = search(&idx, &qb[0], &SearchOptions::default());
            assert_eq!(
                hits.len(),
                1,
                "{lang} seed {seed} expected 1 alpha-renamed hit (orig file), got {}",
                hits.len()
            );
            assert_eq!(hits[0].method.path, files[0].path);
            assert!(
                (hits[0].similarity - 1.0).abs() < 1e-12,
                "{lang} seed {seed} similarity={} != 1.0",
                hits[0].similarity
            );

            let rev_hits = search(&idx, &qa[0], &SearchOptions::default());
            assert_eq!(
                rev_hits.len(),
                1,
                "{lang} seed {seed} expected 1 reverse hit (renamed file), got {}",
                rev_hits.len()
            );
            assert_eq!(rev_hits[0].method.path, files[1].path);
            assert!(
                (rev_hits[0].similarity - 1.0).abs() < 1e-12,
                "{lang} seed {seed} reverse similarity={} != 1.0",
                rev_hits[0].similarity
            );
        }
    }
}

fn gen_corpus(rng: &mut Rng, num_groups: usize) -> Vec<SourceFile> {
    let mut files = Vec::new();
    for g in 0..num_groups {
        let (r_src, r_ren) = gen_rust_pair(rng, g);
        let (p_src, p_ren) = gen_py_pair(rng, g);
        let (t_src, t_ren) = gen_ts_pair(rng, g);
        files.push(SourceFile { path: format!("r_{g}_a.rs"), source: r_src });
        files.push(SourceFile { path: format!("r_{g}_b.rs"), source: r_ren });
        files.push(SourceFile { path: format!("p_{g}_a.py"), source: p_src });
        files.push(SourceFile { path: format!("p_{g}_b.py"), source: p_ren });
        files.push(SourceFile { path: format!("t_{g}_a.ts"), source: t_src });
        files.push(SourceFile { path: format!("t_{g}_b.ts"), source: t_ren });
    }
    files
}

#[test]
fn prop_thread_count_determinism() {
    let mut rng = Rng(0xD37E_A11C_0001_0002);
    for case_idx in 0..15 {
        let files = gen_corpus(&mut rng, 4);
        let idx1 = build_index(&files, IndexOptions { threads: Some(1), ..Default::default() });
        let idx4 = build_index(&files, IndexOptions { threads: Some(4), ..Default::default() });
        assert_eq!(idx1.stats(), idx4.stats(), "case {case_idx} stats mismatch");

        let rec_opts = RecommendOptions { min_support: 1, ..Default::default() };
        for f in &files {
            let q1 = queries_from_lines(&idx1, f, &[2, 3]);
            let q4 = queries_from_lines(&idx4, f, &[2, 3]);
            assert_eq!(q1.len(), q4.len());
            for (a, b) in q1.iter().zip(q4.iter()) {
                assert_eq!(search(&idx1, a, &SearchOptions::default()), search(&idx4, b, &SearchOptions::default()));
                assert_eq!(recommend(&idx1, a, &rec_opts), recommend(&idx4, b, &rec_opts));
            }
        }
    }
}

#[test]
fn prop_similarity_monotonicity_bounds_and_ordering() {
    let mut rng = Rng(0xA10E_070C_BEEF_9988);
    let files = gen_corpus(&mut rng, 6);
    let idx = build_index(&files, IndexOptions { threads: Some(2), ..Default::default() });
    let thresholds = [0.0, 0.25, 0.5, 0.75, 0.9];

    for f in files.iter().take(12) {
        let qs = queries_from_lines(&idx, f, &[2, 3, 4, 5]);
        for q in qs {
            let q_open = q.without_origin();
            let mut prev_set: Option<HashMap<(String, String, u32, u32), f64>> = None;
            for &thr in &thresholds {
                let hits = search(
                    &idx,
                    &q_open,
                    &SearchOptions {
                        limit: 1000,
                        search_limit: 2000,
                        min_similarity: thr,
                        min_coverage: 0.0,
                        exclude: None,
                    },
                );
                for h in &hits {
                    assert!(h.similarity > thr, "hit sim {} <= threshold {thr}", h.similarity);
                    assert!(
                        (0.0..=1.0 + 1e-12).contains(&h.similarity),
                        "out-of-bounds similarity {}",
                        h.similarity
                    );
                }
                for w in hits.windows(2) {
                    assert!(w[0].similarity >= w[1].similarity, "hits out of order");
                }
                let cur_map: HashMap<(String, String, u32, u32), f64> = hits
                    .iter()
                    .map(|h| {
                        (
                            (
                                h.method.path.clone(),
                                h.method.name.clone(),
                                h.method.start_line,
                                h.method.end_line,
                            ),
                            h.similarity,
                        )
                    })
                    .collect();
                if let Some(ref prev) = prev_set {
                    for (key, &sim) in &cur_map {
                        let prev_sim = prev.get(key).expect("higher threshold dropped match missing from lower threshold");
                        assert!((prev_sim - sim).abs() < 1e-12);
                    }
                }
                prev_set = Some(cur_map);
            }
        }
    }
}

#[test]
fn prop_prune_algorithm_1_invariants() {
    let mut rng = Rng(0x9A04_E123_4567_89AB);
    for _ in 0..50 {
        let num_tokens = rng.range(24) + 1;
        let mut features = Vec::new();
        let mut offsets = Vec::with_capacity(num_tokens + 1);
        offsets.push(0);
        for _ in 0..num_tokens {
            let n_feats = rng.range(7);
            for _ in 0..n_feats {
                features.push((rng.range(12) + 1) as i32);
            }
            offsets.push(features.len() as u32);
        }
        let table = TokenTable { features: features.clone(), offsets: offsets.clone() };

        let mut target: HashMap<i32, usize> = HashMap::new();
        let n_target = rng.range(8) + 1;
        for _ in 0..n_target {
            let feat = (rng.range(12) + 1) as i32;
            let cap = rng.range(4) + 1;
            target.insert(feat, cap);
        }
        let total_weight: usize = target.values().sum();

        let use_cand = rng.range(2) == 1;
        let cand_vec: Vec<usize> = if use_cand {
            let mut set = BTreeSet::new();
            let k = rng.range(num_tokens) + 1;
            for _ in 0..k {
                set.insert(rng.range(num_tokens));
            }
            set.into_iter().collect()
        } else {
            Vec::new()
        };
        let cand_opt = use_cand.then_some(cand_vec.as_slice());

        let pruned = prune(&target, &table, cand_opt);
        for w in pruned.tokens.windows(2) {
            assert!(w[0] < w[1], "pruned tokens not strictly increasing: {:?}", pruned.tokens);
        }
        for &tok in &pruned.tokens {
            assert!(tok < num_tokens, "token index {tok} >= {num_tokens}");
            if use_cand {
                assert!(cand_vec.contains(&tok), "token {tok} outside candidate subset");
            }
        }
        assert!(pruned.score <= total_weight);
        assert_eq!(pruned.tokens.is_empty(), pruned.score == 0);

        // Verify multiset score conservation on the returned tokens.
        let mut union_counts: HashMap<i32, usize> = HashMap::new();
        for &tok in &pruned.tokens {
            let s = offsets[tok] as usize;
            let e = offsets[tok + 1] as usize;
            for &f in &features[s..e] {
                *union_counts.entry(f).or_insert(0) += 1;
            }
        }
        let expected_score: usize = target
            .iter()
            .map(|(&f, &cap)| union_counts.get(&f).copied().unwrap_or(0).min(cap))
            .sum();
        assert_eq!(pruned.score, expected_score);
    }
}

#[test]
fn prop_recommend_min_support_invariants() {
    let mut rng = Rng(0x5EED_4EC0_0000_0001);
    for _ in 0..15 {
        let files = gen_corpus(&mut rng, 4);
        let idx = build_index(&files, IndexOptions { threads: Some(2), ..Default::default() });
        let q_snippets = [
            ("rust", "let mut acc = 0;\nlet step = a + b;"),
            ("python", "acc = 1\nstep = a + b"),
            ("typescript", "let acc = 1;\nconst step = a + b;"),
        ];
        for (lang, snip) in q_snippets {
            let Some(q) = query_from_snippet(
                &idx,
                snip,
                &SnippetOptions {
                    language: Some(lang.to_string()),
                    ..Default::default()
                },
            ) else {
                continue;
            };
            let recs_s2 = recommend(&idx, &q, &RecommendOptions { min_support: 2, ..Default::default() });
            for r in &recs_s2 {
                assert!(r.members.len() >= 2, "min_support=2 returned {} members", r.members.len());
                assert!((0.0..=1.0 + 1e-12).contains(&r.similarity));
            }
            let recs_s1 = recommend(&idx, &q, &RecommendOptions { min_support: 1, ..Default::default() });
            for r in &recs_s1 {
                assert!(!r.members.is_empty());
                assert!((0.0..=1.0 + 1e-12).contains(&r.similarity));
            }
        }
    }
}
