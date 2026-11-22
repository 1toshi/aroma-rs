use std::collections::HashSet;

use aroma::{
    build_index, queries_from_lines, query_from_snippet, recommend, search, IndexOptions,
    Language, MethodRef, Node, ParsedMethod, RecommendOptions, SearchOptions, SnippetOptions,
    SourceFile, Token,
};

#[test]
fn test_empty_unsupported_and_oversized_corpus() {
    let files = vec![
        SourceFile { path: "notes.md".into(), source: "# hello".into() },
        SourceFile { path: "main.go".into(), source: "package main\nfunc main() {}\n".into() },
        SourceFile {
            path: "big.py".into(),
            source: format!("def f(x):\n    return {}\n", "x + ".repeat(200)),
        },
        SourceFile {
            path: "ok.py".into(),
            source: "def ok(x):\n    return x + 1\n".into(),
        },
    ];

    let empty_idx = build_index(&[], IndexOptions { threads: Some(1), ..Default::default() });
    assert_eq!(empty_idx.method_count(), 0);
    assert_eq!(empty_idx.stats().files, 0);
    let q_none = query_from_snippet(
        &empty_idx,
        "return x + 1",
        &SnippetOptions { language: Some("python".into()), ..Default::default() },
    )
    .expect("snippet query should still parse on empty index");
    assert_eq!(q_none.size(), 8);
    assert!(search(&empty_idx, &q_none, &SearchOptions::default()).is_empty());

    let idx = build_index(
        &files,
        IndexOptions { max_file_length: 120, threads: Some(4), ..Default::default() },
    );
    assert_eq!(idx.stats().unsupported_files, 2);
    assert_eq!(idx.stats().oversized_files, 1);
    assert_eq!(idx.stats().files, 1);
    assert_eq!(idx.method_count(), 1);
    assert_eq!(idx.method_refs()[0].name, "ok");
}

#[test]
fn test_malformed_syntax_does_not_panic_or_abort() {
    let files = vec![
        SourceFile {
            path: "broken.py".into(),
            source: "def broken(:\n    x = f\"unterminated {a\n\ndef healthy(items):\n    return len(items)\n".into(),
        },
        SourceFile {
            path: "broken.tsx".into(),
            source: "function bad() { const x = <div attr=\"oops; }\n/* unclosed\nfunction good(arr: number[]) { return arr.length; }\n".into(),
        },
        SourceFile {
            path: "clean.ts".into(),
            source: "function measure(items: number[]) { return items.length; }\n".into(),
        },
    ];

    for threads in [1, 4] {
        let idx = build_index(
            &files,
            IndexOptions { threads: Some(threads), ..Default::default() },
        );
        assert_eq!(idx.stats().files, 3);
        let refs = idx.method_refs();
        assert!(refs.iter().any(|r| r.path == "clean.ts" && r.name == "measure"));

        let q = query_from_snippet(
            &idx,
            "return items.length",
            &SnippetOptions { language: Some("typescript".into()), ..Default::default() },
        )
        .unwrap();
        let hits = search(&idx, &q, &SearchOptions::default());
        assert!(hits.iter().any(|h| h.method.path == "clean.ts"));
    }
}

#[test]
fn test_python_class_fstring_hex_methods_preserved() {
    // Regression guard against parsers swallowing sibling methods after f-string format specs.
    let mut py = String::from("class OverlappedFuture:\n");
    py.push_str(
        "    def _repr_info(self):\n        info = [f'overlapped=<{self.state}, {self._ov:#x}>']\n        return info\n\n",
    );
    for i in 0..20 {
        py.push_str(&format!(
            "    def method_{i}(self, val):\n        res = self._check(val + {i})\n        return res\n\n"
        ));
    }

    let files = vec![SourceFile { path: "win_events.py".into(), source: py }];
    let idx = build_index(&files, IndexOptions { threads: Some(1), ..Default::default() });
    assert_eq!(idx.method_count(), 21);
    let names: HashSet<String> = idx.method_refs().into_iter().map(|m| m.name).collect();
    assert!(names.contains("OverlappedFuture._repr_info"));
    for i in 0..20 {
        assert!(
            names.contains(&format!("OverlappedFuture.method_{i}")),
            "missing method_{i}"
        );
    }
}

#[test]
fn test_single_vs_multi_thread_exact_determinism() {
    let mut files = Vec::new();
    for i in 0..16 {
        files.push(SourceFile {
            path: format!("pkg/mod_{i}.py"),
            source: format!(
                "def compute_{i}(rows, scale):\n    total = 0\n    for r in rows:\n        total += r * scale + {}\n    return total\n",
                i % 3
            ),
        });
        files.push(SourceFile {
            path: format!("web/comp_{i}.ts"),
            source: format!(
                "export function render{i}(rows: number[], scale: number) {{\n  let total = 0;\n  for (const r of rows) {{\n    total += r * scale + {};\n  }}\n  return total;\n}}\n",
                i % 3
            ),
        });
    }

    let idx1 = build_index(&files, IndexOptions { threads: Some(1), ..Default::default() });
    let idx15 = build_index(&files, IndexOptions { threads: Some(15), ..Default::default() });

    assert_eq!(idx1.stats(), idx15.stats());
    assert_eq!(idx1.method_refs(), idx15.method_refs());

    let lq1 = queries_from_lines(&idx1, &files[0], &[2, 3, 4]);
    let lq15 = queries_from_lines(&idx15, &files[0], &[2, 3, 4]);
    assert_eq!(lq1.len(), 1);
    assert_eq!(lq15.len(), 1);
    assert_eq!(lq1[0].size(), lq15[0].size());
    assert_eq!(lq1[0].origin(), lq15[0].origin());
    assert_eq!(lq1[0].lines(), lq15[0].lines());

    let s1 = search(&idx1, &lq1[0], &SearchOptions::default());
    let s15 = search(&idx15, &lq15[0], &SearchOptions::default());
    assert_eq!(s1, s15);

    let r1 = recommend(&idx1, &lq1[0], &RecommendOptions::default());
    let r15 = recommend(&idx15, &lq15[0], &RecommendOptions::default());
    assert_eq!(r1, r15);
}

#[test]
fn test_large_method_token_end_and_posting_overflows() {
    // 1. >512 line method (fallback path in matched_lines_of & render_recommendation)
    // 2. >255 line single token (token_end_overflow)
    // 3. >255 identical features in one method (posting_overflow)
    let mut big_py = String::from("def huge_fn(x):\n    msg = \"\"\"");
    for _ in 0..270 {
        big_py.push_str("line\n");
    }
    big_py.push_str("\"\"\"\n");
    for _ in 0..300 {
        big_py.push_str("    y = x + 1\n");
    }
    big_py.push_str("    return msg, y\n");

    let mut copy_py = String::from("def huge_copy(z):\n    msg = \"\"\"");
    for _ in 0..270 {
        copy_py.push_str("line\n");
    }
    copy_py.push_str("\"\"\"\n");
    for _ in 0..300 {
        copy_py.push_str("    y = z + 1\n");
    }
    copy_py.push_str("    extra = z + 2\n    return msg, y, extra\n");

    let files = vec![
        SourceFile { path: "huge1.py".into(), source: big_py },
        SourceFile { path: "huge2.py".into(), source: copy_py },
        // Pad >= 8 files to exercise parallel indexing path too
        SourceFile { path: "p1.py".into(), source: "def p1(a): return a".into() },
        SourceFile { path: "p2.py".into(), source: "def p2(a): return a + 2".into() },
        SourceFile { path: "p3.py".into(), source: "def p3(a): return a + 3".into() },
        SourceFile { path: "p4.py".into(), source: "def p4(a): return a + 4".into() },
        SourceFile { path: "p5.py".into(), source: "def p5(a): return a + 5".into() },
        SourceFile { path: "p6.py".into(), source: "def p6(a): return a + 6".into() },
    ];

    let q_src = format!("msg = \"\"\"{}\"\"\"\n{}", "line\n".repeat(270), "y = x + 1\n".repeat(300));
    let idx1 = build_index(&files, IndexOptions { threads: Some(1), ..Default::default() });
    let idx4 = build_index(&files, IndexOptions { threads: Some(4), ..Default::default() });

    let q1 = query_from_snippet(
        &idx1,
        &q_src,
        &SnippetOptions { language: Some("python".into()), ..Default::default() },
    )
    .unwrap();
    let q4 = query_from_snippet(
        &idx4,
        &q_src,
        &SnippetOptions { language: Some("python".into()), ..Default::default() },
    )
    .unwrap();

    let s1 = search(&idx1, &q1, &SearchOptions::default());
    let s4 = search(&idx4, &q4, &SearchOptions::default());
    assert_eq!(s1, s4);
    assert!(!s1.is_empty());
    assert!(s1[0].matched_lines.contains(&2));
    assert!(s1[0].matched_lines.contains(&272));
    assert!(s1[0].matched_lines.contains(&570));

    let r1 = recommend(
        &idx1,
        &q1,
        &RecommendOptions { min_expansion: 1.0, min_cohesion: 0.5, ..Default::default() },
    );
    let r4 = recommend(
        &idx4,
        &q4,
        &RecommendOptions { min_expansion: 1.0, min_cohesion: 0.5, ..Default::default() },
    );
    assert_eq!(r1, r4);
}

struct MiniLang;

impl Language for MiniLang {
    fn id(&self) -> &str {
        "mini"
    }

    fn extensions(&self) -> &[&str] {
        &[".mini"]
    }

    fn parse_methods(&self, source: &str, _path: &str) -> Vec<ParsedMethod> {
        let mut out = Vec::new();
        for (i, line) in source.lines().enumerate() {
            let ln = (i + 1) as u32;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kids: Vec<Node> = parts[1..]
                    .iter()
                    .enumerate()
                    .map(|(k, &tok)| {
                        Node::Token(Token {
                            text: tok.to_string(),
                            keyword: tok == "ret",
                            local: k == 1,
                            line: ln,
                            end_line: ln,
                        })
                    })
                    .collect();
                out.push(ParsedMethod {
                    name: parts[0].to_string(),
                    start_line: ln,
                    end_line: ln,
                    body: Node::Tree(kids),
                });
            }
        }
        out
    }

    fn parse_snippet(&self, source: &str, _options: &SnippetOptions) -> Option<Node> {
        let toks: Vec<Node> = source
            .split_whitespace()
            .enumerate()
            .map(|(k, tok)| {
                Node::Token(Token {
                    text: tok.to_string(),
                    keyword: tok == "ret",
                    local: k == 0,
                    line: 1,
                    end_line: 1,
                })
            })
            .collect();
        if toks.is_empty() { None } else { Some(Node::Tree(toks)) }
    }
}

#[test]
fn test_custom_language_origin_exclude_and_query_helpers() {
    let files = vec![
        SourceFile { path: "a.mini".into(), source: "m1 ret v1 alpha\nm2 ret v2 alpha\n".into() },
        SourceFile { path: "b.mini".into(), source: "m3 ret v3 alpha\n".into() },
    ];
    let idx = build_index(
        &files,
        IndexOptions { languages: Some(vec![Box::new(MiniLang)]), threads: Some(1), ..Default::default() },
    );
    let qs = queries_from_lines(&idx, &files[0], &[1]);
    assert_eq!(qs.len(), 1);
    assert_eq!(qs[0].language(), "mini");

    let with_orig = search(&idx, &qs[0], &SearchOptions::default());
    assert!(with_orig.iter().all(|r| !(r.method.path == "a.mini" && r.method.start_line == 1)));

    let no_orig_q = qs[0].without_origin();
    assert!(no_orig_q.origin().is_none());
    let without_orig = search(&idx, &no_orig_q, &SearchOptions::default());
    assert!(without_orig.iter().any(|r| r.method.path == "a.mini" && r.method.start_line == 1));

    let ex = |m: &MethodRef| m.path == "b.mini";
    let filtered = search(
        &idx,
        &qs[0],
        &SearchOptions { exclude: Some(&ex), ..Default::default() },
    );
    assert!(filtered.iter().all(|r| r.method.path != "b.mini"));

    let wrong_lang = no_orig_q.with_language("unknown");
    assert!(search(&idx, &wrong_lang, &SearchOptions::default()).is_empty());
}

#[test]
fn test_parallel_rayon_search_and_recommend_reentrancy() {
    use rayon::prelude::*;

    let files: Vec<SourceFile> = (0..12)
        .map(|i| SourceFile {
            path: format!("f_{i}.rs"),
            source: format!(
                r#"pub fn compute_batch_{i}(items: &[i32], factor: i32, limit: i32) -> i32 {{
    let mut total = 0;
    let mut valid = 0;
    let mut bonus = 0;
    for (idx, &val) in items.iter().enumerate() {{
        let adjusted = val * factor + (idx as i32);
        if adjusted > 0 && adjusted < limit {{
            if adjusted % 2 == 0 {{
                total += adjusted * 3 + 1;
                valid += 1;
            }} else {{
                total += adjusted * 2 - 1;
                bonus += adjusted / 4;
            }}
        }} else if adjusted == 0 {{
            bonus += factor + 5;
        }}
    }}
    let combined = total + bonus * 2;
    record_metric_{i}(combined);
    if valid > 0 {{
        (combined / valid).clamp(-1000, 1000)
    }} else {{
        combined.max(0)
    }}
}}
"#
            ),
        })
        .collect();

    let idx = build_index(
        &files,
        IndexOptions {
            threads: Some(4),
            ..Default::default()
        },
    );
    let all_lines: Vec<u32> = (2..=26).collect();
    let full_queries: Vec<_> = files
        .iter()
        .flat_map(|f| queries_from_lines(&idx, f, &all_lines))
        .collect();
    assert_eq!(full_queries.len(), 12);

    let partial_queries: Vec<_> = files
        .iter()
        .flat_map(|f| queries_from_lines(&idx, f, &[2, 3, 4]))
        .collect();
    assert_eq!(partial_queries.len(), 12);

    let search_opts = SearchOptions {
        limit: 25,
        min_similarity: 0.5,
        ..Default::default()
    };
    let rec_opts = RecommendOptions::default();

    let seq_search: Vec<_> = full_queries
        .iter()
        .map(|q| search(&idx, q, &search_opts))
        .collect();
    let seq_rec: Vec<_> = partial_queries
        .iter()
        .map(|q| recommend(&idx, q, &rec_opts))
        .collect();
    assert_eq!(seq_search[0].len(), 11);
    assert!(!seq_rec[0].is_empty());

    let par_search: Vec<_> = full_queries
        .par_iter()
        .map(|q| search(&idx, q, &search_opts))
        .collect();
    let par_rec: Vec<_> = partial_queries
        .par_iter()
        .map(|q| recommend(&idx, q, &rec_opts))
        .collect();
    assert_eq!(seq_search, par_search);
    assert_eq!(seq_rec, par_rec);

    // Same-thread re-entrant `search` invocation inside `exclude` must not panic on `RefCell`.
    let reentrant = |m: &MethodRef| -> bool {
        let nested = search(
            &idx,
            &full_queries[0],
            &SearchOptions {
                limit: 2,
                ..Default::default()
            },
        );
        !nested.is_empty() && m.path == "f_11.rs"
    };
    let filtered = search(
        &idx,
        &full_queries[1],
        &SearchOptions {
            exclude: Some(&reentrant),
            ..Default::default()
        },
    );
    assert!(!filtered.is_empty());
    assert!(filtered.iter().all(|r| r.method.path != "f_11.rs"));
}

#[test]
fn test_recommend_min_support_one_does_not_panic() {
    let files = vec![SourceFile {
        path: "single.rs".into(),
        source: concat!(
            "pub fn aggregate_positives(xs: &[i32], factor: i32) -> i32 {\n",
            "    let mut total = 0;\n",
            "    let mut count = 0;\n",
            "    for &x in xs {\n",
            "        if x > 0 {\n",
            "            total += x * factor;\n",
            "            count += 1;\n",
            "        }\n",
            "    }\n",
            "    if count > 0 { total / count } else { 0 }\n",
            "}\n",
        )
        .into(),
    }];
    let idx = build_index(&files, IndexOptions { threads: Some(1), ..Default::default() });
    let qs = queries_from_lines(&idx, &files[0], &[2, 3]);
    assert_eq!(qs.len(), 1);
    let q_no_orig = qs[0].without_origin();
    let recs = recommend(
        &idx,
        &q_no_orig,
        &RecommendOptions {
            min_support: 1,
            ..Default::default()
        },
    );
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].members.len(), 1);
    assert_eq!(recs[0].method.name, "aggregate_positives");
}
