use std::collections::HashSet;

use aroma::{
    build_index, featurize, queries_from_lines, query_from_snippet, recommend, rust_language,
    search, usage_feature, IndexOptions, LineRole, MethodRef, Node, RecommendOptions,
    SearchOptions, SnippetOptions, SourceFile, VAR,
};

#[test]
fn test_rust_items_naming_and_declarations() {
    let src = r#"
use std::fmt::Display;

const fn double_val(n: usize) -> usize {
    n * 2
}

extern "C" {
    fn external_only(x: i32) -> i32;
}

trait Summarizer {
    fn required_only(&self) -> usize;

    fn default_summary(&self) -> usize {
        self.required_only() + 1
    }
}

struct Container<T> {
    items: Vec<T>,
}

impl<T: Clone> Container<T> {
    pub fn push_twice(&mut self, item: T) {
        self.items.push(item.clone());
        self.items.push(item);
    }
}

impl<T: Display> Summarizer for Container<T> {
    fn required_only(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_works() {
        let x = 3 + 4;
        assert_eq!(x, 7);
    }
}
"#;

    let lang = rust_language();
    assert_eq!(lang.id(), "rust");
    assert_eq!(lang.extensions(), &[".rs"]);
    assert_eq!(lang.member_access_labels(), &["#.#"]);

    let methods = lang.parse_methods(src, "src/lib.rs");
    let names: Vec<&str> = methods.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "double_val",
            "Summarizer::default_summary",
            "Container::push_twice",
            "<Container as Summarizer>::required_only",
            "tests::helper_works",
        ]
    );
    assert!(!names.contains(&"external_only"));
    assert!(!names.contains(&"required_only"));

    let double = &methods[0];
    assert_eq!(double.start_line, 4);
    assert_eq!(double.end_line, 6);

    let push_twice = &methods[2];
    assert_eq!(push_twice.start_line, 25);
    assert_eq!(push_twice.end_line, 28);
}

#[test]
fn test_rust_alpha_renaming_and_self_parity() {
    let f1 = SourceFile {
        path: "a.rs".into(),
        source: r#"
pub struct Acc {
    vals: Vec<i32>,
}

impl Acc {
    pub fn accumulate(&mut self, rows: &[Option<i32>], opt: Result<i32, &str>) -> i32 {
        let mut total = 0;
        for entry in rows {
            if let Some(v) = entry {
                total += *v;
            }
        }
        match opt {
            Ok(delta) => {
                let f = |x: i32| x + delta;
                total = f(total);
            }
            Err(_) => total += 1,
        }
        self.vals.push(total);
        self.vals.sort();
        total
    }
}
"#
        .into(),
    };

    let f2 = SourceFile {
        path: "b.rs".into(),
        source: r#"
pub fn run_sum(state: &mut Acc, slices: &[Option<i32>], status: Result<i32, &str>) -> i32 {
    let mut sum = 0;
    for elem in slices {
        if let Some(num) = elem {
            sum += *num;
        }
    }
    match status {
        Ok(offset) => {
            let add = |y: i32| y + offset;
            sum = add(sum);
        }
        Err(_) => sum += 1,
    }
    state.vals.push(sum);
    state.vals.sort();
    sum
}
"#
        .into(),
    };

    let idx = build_index(&[f1.clone(), f2.clone()], IndexOptions { threads: Some(1), ..Default::default() });
    // Exact Body::digest distinguishes variable spellings (2 canonical methods), while search sees similarity == 1.0!
    assert_eq!(idx.stats().methods, 2);
    assert_eq!(idx.stats().duplicate_methods, 0);

    let qs = queries_from_lines(&idx, &f2, &(1..=20).collect::<Vec<u32>>());
    assert_eq!(qs.len(), 1);
    let hits = search(&idx, &qs[0], &SearchOptions::default());
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].method.path, "a.rs");
    assert_eq!(hits[0].method.name, "Acc::accumulate");
    assert!((hits[0].similarity - 1.0).abs() < 1e-12);
}

#[test]
fn test_rust_usage_features_and_non_locals() {
    let lang = rust_language();
    let methods = lang.parse_methods(
        r#"
fn process(buf: &mut Vec<u32>, flag: bool) -> Point {
    let local_cfg = Config { port: 8080, buf };
    if flag {
        buf.push(1);
        buf.sort();
        let data = std::fs::read("a.txt");
        println!("read {}", local_cfg.port);
    }
    Point { x: 1, y: 2 }
}
"#,
        "src/proc.rs",
    );
    assert_eq!(methods.len(), 1);

    fn walk_toks<'a>(n: &'a Node, out: &mut Vec<(&'a str, bool, bool)>) {
        match n {
            Node::Token(t) => out.push((t.text.as_str(), t.keyword, t.local)),
            Node::Tree(kids) => {
                for k in kids {
                    walk_toks(k, out);
                }
            }
        }
    }
    let mut toks = Vec::new();
    walk_toks(&methods[0].body, &mut toks);

    let find_tok = |name: &str| toks.iter().find(|&&(t, _, _)| t == name).copied();
    assert_eq!(find_tok("Config"), Some(("Config", false, false)));
    assert_eq!(find_tok("port"), Some(("port", false, false)));
    assert_eq!(find_tok("std"), Some(("std", false, false)));
    assert_eq!(find_tok("fs"), Some(("fs", false, false)));
    assert_eq!(find_tok("read"), Some(("read", false, false)));
    assert_eq!(find_tok("println"), Some(("println", false, false)));
    assert_eq!(find_tok("Point"), Some(("Point", false, false)));
    assert_eq!(find_tok("x"), Some(("x", false, false)));
    assert_eq!(find_tok("y"), Some(("y", false, false)));
    assert_eq!(find_tok("buf"), Some(("buf", false, true)));
    assert_eq!(find_tok("local_cfg"), Some(("local_cfg", false, true)));
    assert_eq!(find_tok("flag"), Some(("flag", false, true)));

    let feats = featurize(&methods[0].body, lang.member_access_labels(), None);
    let all_feats: HashSet<&str> = feats
        .iter()
        .flat_map(|tf| tf.features.iter().map(String::as_str))
        .collect();
    assert!(all_feats.contains(&usage_feature(".push", ".sort").as_str()));
    assert!(feats.iter().any(|tf| tf.token.text == "buf" && tf.token.local));
    assert!(feats.iter().all(|tf| !(tf.token.text == VAR && tf.token.local)));
}

#[test]
fn test_rust_lexer_and_parser_edge_cases() {
    let src = concat!(
        "#!/usr/bin/env rust\n",
        r##"/* outer /* nested comment fn fake_comment() {} */ still comment */
pub fn edge_torture<'a>(pair: (i32, (i32, i32)), xs: &'a [u8]) -> Vec<Option<Vec<u8>>> {
    let raw_s = r#"fn fake_in_raw() { let y = 1; }"#;
    let bytes_s = b"hello \" world";
    let c_s = c"nul";
    let raw_c = cr#"raw c"#;
    let ch1 = '\'';
    let ch2 = '\u{1f600}';
    let byte_ch = b'\n';
    let r#type = pair.1.0;
    let shifted = (r#type >> 2) << 1;
    let mut rng_sum = 0;
    for k in 0..=3 {
        rng_sum += (-1_i32).abs() + k + 0.abs();
    }
    let vec_val: Vec<Option<Vec<u8>>> = vec![Some(xs.to_vec())];
    if let Some(head) = vec_val.first() && head.is_some() {
        rng_sum += shifted + raw_s.len() as i32 + bytes_s.len() as i32 + c_s.to_bytes().len() as i32
            + raw_c.to_bytes().len() as i32 + (ch1 as i32) + (ch2 as i32) + (byte_ch as i32);
    }
    rng_sum >>= 1;
    let _ = rng_sum;
    vec_val
}
"##,
    );

    let methods = rust_language().parse_methods(src, "src/edge.rs");
    assert_eq!(methods.len(), 1);
    assert_eq!(methods[0].name, "edge_torture");
}

#[test]
fn test_rust_snippet_search_and_recommend() {
    let orig = SourceFile {
        path: "src/retry.rs".into(),
        source: r#"
pub fn fetch_with_retry(client: &HttpClient, url: &str, max_tries: usize) -> Result<String, Err> {
    let mut attempt = 0;
    let timeout = Duration::from_secs(5);
    while attempt < max_tries {
        if let Ok(resp) = client.get(url, timeout) {
            return Ok(resp);
        }
        attempt += 1;
    }
    Err(Err::Timeout)
}

fn unrelated(x: i32) -> i32 {
    x + 1
}
"#
        .into(),
    };
    let copy = SourceFile {
        path: "src/alt_retry.rs".into(),
        source: r#"
fn poll_endpoint(http: &HttpClient, endpoint: &str, limit: usize) -> Result<String, Err> {
    let mut tries = 0;
    let deadline = Duration::from_secs(5);
    let backoff = Duration::from_millis(50);
    while tries < limit {
        if let Ok(body) = http.get(endpoint, deadline) {
            log_hit(endpoint);
            return Ok(body);
        }
        sleep(backoff);
        tries += 1;
    }
    Err(Err::Timeout)
}
"#
        .into(),
    };

    let copy2 = SourceFile {
        path: "src/alt_retry_2.rs".into(),
        source: r#"
fn retry_http(client: &HttpClient, url: &str, max_tries: usize) -> Result<String, Err> {
    let mut count = 0;
    let max_wait = Duration::from_secs(5);
    let delay = Duration::from_millis(50);
    while count < max_tries {
        if let Ok(res) = client.get(url, max_wait) {
            log_hit(url);
            return Ok(res);
        }
        sleep(delay);
        count += 1;
    }
    Err(Err::Timeout)
}
"#
        .into(),
    };

    let idx = build_index(&[orig.clone(), copy.clone(), copy2], IndexOptions { threads: Some(1), ..Default::default() });
    assert_eq!(idx.method_count(), 4);

    // Snippet with free local variable `attempts`, uppercase global `Duration`, and callee `sleep`
    let mut globals = HashSet::new();
    globals.insert("custom_global".to_string());
    let snip_q = query_from_snippet(
        &idx,
        r#"
let mut attempt = 0;
let timeout = Duration::from_secs(5);
while attempt < custom_global {
    attempt += 1;
}
"#,
        &SnippetOptions {
            language: Some("rust".into()),
            globals: Some(globals),
            ..Default::default()
        },
    )
    .expect("rust snippet should produce query");

    let hits = search(&idx, &snip_q, &SearchOptions::default());
    assert!(!hits.is_empty());
    assert_eq!(hits[0].method.path, "src/retry.rs");

    // Line query + recommend Phase III structure/query/added classification
    let qs = queries_from_lines(&idx, &orig, &[3, 4, 5]);
    assert_eq!(
        qs.len(),
        1,
        "only fetch_with_retry covers lines 3..=5; unrelated should be pruned"
    );
    let q_origin = &qs[0];
    assert_eq!(
        q_origin.origin,
        Some(MethodRef {
            path: "src/retry.rs".into(),
            name: "fetch_with_retry".into(),
            language: "rust".into(),
            start_line: 2,
            end_line: 12,
        })
    );

    let rec_hits = search(&idx, q_origin, &SearchOptions::default());
    assert!(
        rec_hits[0].method.path == "src/alt_retry.rs"
            || rec_hits[0].method.path == "src/alt_retry_2.rs"
    );

    // `alt_retry` & `alt_retry_2` form a cohesive (>=0.9) cluster with Added lines (`log_hit`, `sleep`).
    let recs = recommend(&idx, q_origin, &RecommendOptions::default());
    assert!(!recs.is_empty());
    assert!(recs[0].lines.iter().any(|l| l.role == LineRole::Structure));
    assert!(recs[0].lines.iter().any(|l| l.role == LineRole::Query));
    assert!(recs[0].lines.iter().any(|l| l.role == LineRole::Added));
}

#[test]
fn test_rust_malformed_resilience_and_determinism() {
    let broken_raw = SourceFile {
        path: "broken_raw.rs".into(),
        source: "fn cut() { let s = r#\"unclosed raw\n".into(),
    };
    let broken_comment = SourceFile {
        path: "broken_comment.rs".into(),
        source: "fn a() { /* nested /* never closed".into(),
    };
    let broken_brace = SourceFile {
        path: "broken_brace.rs".into(),
        source: r#"
fn bad(x: i32 {
    if x > 0 { let y = (1 + 2;
}
fn healthy(items: &mut Vec<i32>, v: i32) {
    items.push(v);
    items.sort();
}
"#
        .into(),
    };
    let clean = SourceFile {
        path: "clean.rs".into(),
        source: r#"
pub fn add_sort(dst: &mut Vec<i32>, elem: i32) {
    dst.push(elem);
    dst.sort();
}
"#
        .into(),
    };

    let corpus: Vec<SourceFile> = (0..12)
        .map(|i| SourceFile {
            path: format!("pkg/m_{i}.rs"),
            source: format!(
                "pub fn calc_{i}(xs: &[i32], s: i32) -> i32 {{\n    let mut acc = {};\n    for &x in xs {{ acc += x * s; }}\n    acc\n}}\n",
                i % 3
            ),
        })
        .chain([broken_raw, broken_comment, broken_brace, clean])
        .collect();

    let idx1 = build_index(&corpus, IndexOptions { threads: Some(1), ..Default::default() });
    let idx4 = build_index(&corpus, IndexOptions { threads: Some(4), ..Default::default() });

    assert_eq!(idx1.stats(), idx4.stats());
    assert_eq!(idx1.method_refs(), idx4.method_refs());
    assert!(idx1.method_refs().iter().any(|m| m.path == "clean.rs" && m.name == "add_sort"));
    assert!(idx1.method_refs().iter().any(|m| m.path == "broken_brace.rs" && m.name == "healthy"));
}
