use std::collections::{HashMap, HashSet};

use aroma::{
    build_index, featurize, label, parent_feature, position_context, prune, python_language,
    queries_from_lines, query_from_snippet, recommend, search, sibling_feature, token_context,
    token_feature, typescript_language, typescript_language_with, usage_feature, IndexOptions,
    Language, LineRole, MethodRef, Node, ParsedMethod, RecommendOptions, SearchOptions,
    SnippetOptions, SourceFile, Token, TokenTable, VAR,
};

// ============================================================================
// Toy Language (ported directly from tests/toy.ts)
// ============================================================================

struct ToyLanguage;

const TOY_KEYWORDS: &[&str] = &[
    "if",
    "else",
    "for",
    "while",
    "int",
    "instanceof",
    "return",
    "new",
];

fn is_toy_name(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    if bytes.iter().all(|b| b.is_ascii_digit()) {
        return true;
    }
    let first = bytes[0];
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_')
}

fn toy_leaf(word: &str, line: u32) -> Token {
    let (local, text) = if let Some(rest) = word.strip_prefix('$') {
        (true, rest.to_string())
    } else {
        (false, word.to_string())
    };
    let keyword = !local && (TOY_KEYWORDS.contains(&text.as_str()) || !is_toy_name(&text));
    Token {
        text,
        keyword,
        local,
        line,
        end_line: line,
    }
}

fn toy_collapse(mut children: Vec<Node>) -> Node {
    if children.len() == 1 && matches!(children[0], Node::Tree(_)) {
        children.pop().unwrap()
    } else {
        Node::Tree(children)
    }
}

fn toy_parse(lines: &[&str], first_line: u32) -> Option<Node> {
    let mut stack: Vec<Vec<Node>> = vec![vec![]];
    for (offset, line) in lines.iter().enumerate() {
        let ln = first_line + offset as u32;
        for word in line.split_whitespace() {
            if word == "[" {
                stack.push(vec![]);
            } else if word == "]" {
                let children = stack.pop().expect("unmatched ]");
                let node = toy_collapse(children);
                stack.last_mut().unwrap().push(node);
            } else {
                stack.last_mut().unwrap().push(Node::Token(toy_leaf(word, ln)));
            }
        }
    }
    let root = stack.pop().unwrap();
    if root.is_empty() {
        None
    } else {
        Some(toy_collapse(root))
    }
}

impl Language for ToyLanguage {
    fn id(&self) -> &str {
        "toy"
    }

    fn extensions(&self) -> &[&str] {
        &[".toy"]
    }

    fn parse_methods(&self, source: &str, _path: &str) -> Vec<ParsedMethod> {
        let mut methods = Vec::new();
        let lines: Vec<&str> = source.split('\n').collect();
        let mut start: Option<usize> = None;
        let mut name = String::new();
        for (index, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("method ") {
                if !rest.is_empty() && !rest.contains(char::is_whitespace) {
                    start = Some(index + 1);
                    name = rest.to_string();
                    continue;
                }
            }
            if trimmed == "end" {
                if let Some(st) = start {
                    if let Some(body) = toy_parse(&lines[st..index], (st + 1) as u32) {
                        methods.push(ParsedMethod {
                            name: name.clone(),
                            start_line: (st + 1) as u32,
                            end_line: index as u32,
                            body,
                        });
                    }
                }
            }
        }
        methods
    }

    fn parse_snippet(&self, source: &str, _options: &SnippetOptions) -> Option<Node> {
        let lines: Vec<&str> = source.split('\n').collect();
        toy_parse(&lines, 1)
    }
}

fn toy_snippet(source: &str) -> Node {
    ToyLanguage
        .parse_snippet(source, &SnippetOptions::default())
        .unwrap()
}

fn features_of(source: &str) -> Vec<String> {
    let mut out: Vec<String> = featurize(&toy_snippet(source), &["#.#"], None)
        .into_iter()
        .flat_map(|e| e.features)
        .collect();
    out.sort();
    out
}

const LISTING_4_TOY: &str = "
[ if [ ( [ $view instanceof ViewGroup ] ) ]
  [ { [ for ( [ [ int [ $i = 0 ] ] ; [ $i < [ [ ( [ ( ViewGroup ) $view ] ) ] . [ getChildCount ( ) ] ] ] ; [ $i ++ ] ] )
        [ { [ [ View [ $innerView = [ [ ( [ ( ViewGroup ) $view ] ) ] . [ getChildAt ( $i ) ] ] ] ] ; ] } ] ] } ] ]
";

fn file(path: &str, lines: &[&str]) -> SourceFile {
    SourceFile {
        path: path.to_string(),
        source: lines.join("\n"),
    }
}

fn all_lines(sf: &SourceFile) -> Vec<u32> {
    (1..=sf.source.split('\n').count() as u32).collect()
}

fn collect_tokens(node: &Node, out: &mut Vec<Token>) {
    match node {
        Node::Token(t) => out.push(t.clone()),
        Node::Tree(cs) => {
            for c in cs {
                collect_tokens(c, out);
            }
        }
    }
}

fn collect_labels(node: &Node, out: &mut Vec<String>) {
    if let Node::Tree(cs) = node {
        out.push(label(cs));
        for c in cs {
            collect_labels(c, out);
        }
    }
}

fn names(node: &Node, local: bool) -> Vec<String> {
    let mut toks = Vec::new();
    collect_tokens(node, &mut toks);
    let mut set: Vec<String> = toks
        .into_iter()
        .filter(|t| !t.keyword && t.local == local)
        .map(|t| t.text)
        .collect();
    set.sort();
    set.dedup();
    set
}

// ============================================================================
// Tests from tests/aroma.test.ts
// ============================================================================

#[test]
fn test_simplified_parse_tree_labels() {
    let Node::Tree(cs1) = toy_snippet("[ x > [ y . f ] ]") else {
        panic!("expected tree");
    };
    assert_eq!(label(&cs1), "#>#");

    let Node::Tree(cs2) = toy_snippet(LISTING_4_TOY) else {
        panic!("expected tree");
    };
    assert_eq!(label(&cs2), "if##");
}

#[test]
fn test_featurize_table2_figure4() {
    let tokens = featurize(&toy_snippet(LISTING_4_TOY), &["#.#"], None);
    let texts: Vec<&str> = tokens.iter().map(|e| e.token.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            "view",
            "ViewGroup",
            "i",
            "0",
            "i",
            "ViewGroup",
            "view",
            "getChildCount",
            "i",
            "View",
            "innerView",
            "ViewGroup",
            "view",
            "getChildAt",
            "i",
        ]
    );

    let mut f3 = tokens[3].features.clone();
    f3.sort();
    let mut exp3 = vec![
        token_feature("0"),
        parent_feature("0", 2, "#=#"),
        parent_feature("0", 1, "int#"),
        parent_feature("0", 1, "#;#;#"),
        sibling_feature(VAR, "0"),
        sibling_feature("0", VAR),
    ];
    exp3.sort();
    assert_eq!(f3, exp3);

    let mut f6 = tokens[6].features.clone();
    f6.sort();
    let mut exp6 = vec![
        token_feature(VAR),
        parent_feature(VAR, 2, "(#)#"),
        parent_feature(VAR, 1, "(#)"),
        parent_feature(VAR, 1, "#.#"),
        sibling_feature("ViewGroup", VAR),
        sibling_feature(VAR, "getChildCount"),
        usage_feature(
            &position_context(1, "#instanceof#"),
            &position_context(2, "(#)#"),
        ),
        usage_feature(&position_context(2, "(#)#"), &position_context(2, "(#)#")),
    ];
    exp6.sort();
    assert_eq!(f6, exp6);
}

#[test]
fn test_featurize_consecutive_uses_and_member_access() {
    let source = "[ if [ ( [ $y < 0 ] ) ] [ [ $x = [ - $x ] ] ; ] ]";
    let pair = usage_feature(&position_context(1, "#=#"), &position_context(1, "-#"));
    assert_eq!(features_of(source).iter().filter(|f| *f == &pair).count(), 2);

    let mut y_feats = featurize(&toy_snippet(source), &["#.#"], None)[0]
        .features
        .clone();
    y_feats.sort();
    let mut exp_y = vec![
        token_feature(VAR),
        parent_feature(VAR, 1, "#<#"),
        parent_feature(VAR, 1, "(#)"),
        parent_feature(VAR, 1, "if##"),
        sibling_feature(VAR, "0"),
    ];
    exp_y.sort();
    assert_eq!(y_feats, exp_y);

    let member_feats = features_of("[ [ $x . [ foo ( ) ] ] ; ] [ [ $x . [ bar ( ) ] ] ; ]");
    assert!(member_feats.contains(&usage_feature(&token_context("foo"), &token_context("bar"))));
    assert!(!member_feats.contains(&usage_feature(
        &position_context(1, "#.#"),
        &position_context(1, "#.#")
    )));
}

#[test]
fn test_featurize_alpha_renaming_and_nesting_sensitivity() {
    let original = features_of("[ [ $a = [ load ( $b ) ] ] ; ] [ [ save ( $a ) ] ; ]");
    let renamed = features_of("[ [ $out = [ load ( $src ) ] ] ; ] [ [ save ( $out ) ] ; ]");
    let other_call = features_of("[ [ $a = [ fetch ( $b ) ] ] ; ] [ [ save ( $a ) ] ; ]");
    let other_flow = features_of("[ [ $a = [ load ( $b ) ] ] ; ] [ [ save ( $b ) ] ; ]");
    assert_eq!(renamed, original);
    assert_ne!(other_call, original);
    assert_ne!(other_flow, original);

    let first = features_of("[ if [ ( [ $x > 0 ] ) ] [ [ $z = 3 ] ; ] ]");
    let second = features_of("[ if [ ( [ $z > 3 ] ) ] [ [ $x = 0 ] ; ] ]");
    assert!(first.contains(&parent_feature("3", 2, "if##")));
    assert!(first.contains(&parent_feature("3", 2, "#=#")));
    assert!(!second.contains(&parent_feature("3", 2, "if##")));
    assert_ne!(second, first);
}

fn naive_prune(
    target: &HashMap<i32, usize>,
    table: &TokenTable,
    candidates: &[usize],
) -> aroma::Pruned {
    let score = |tokens: &[usize]| -> usize {
        let mut have: HashMap<i32, usize> = HashMap::new();
        for &tok in tokens {
            let s = table.offsets[tok] as usize;
            let e = table.offsets[tok + 1] as usize;
            for &f in &table.features[s..e] {
                *have.entry(f).or_insert(0) += 1;
            }
        }
        have.iter()
            .map(|(f, &cnt)| cnt.min(*target.get(f).unwrap_or(&0)))
            .sum()
    };

    let mut kept: Vec<usize> = Vec::new();
    loop {
        let mut best: Option<usize> = None;
        let mut best_score = score(&kept);
        for &tok in candidates {
            if !kept.contains(&tok) {
                let mut probe = kept.clone();
                probe.push(tok);
                let sc = score(&probe);
                if sc > best_score {
                    best = Some(tok);
                    best_score = sc;
                }
            }
        }
        if let Some(b) = best {
            kept.push(b);
        } else {
            kept.sort_unstable();
            return aroma::Pruned {
                tokens: kept,
                score: best_score,
            };
        }
    }
}

fn make_table(tokens: &[Vec<i32>]) -> TokenTable {
    let mut offsets = Vec::with_capacity(tokens.len() + 1);
    offsets.push(0u32);
    let mut features = Vec::new();
    for fs in tokens {
        features.extend_from_slice(fs);
        offsets.push(features.len() as u32);
    }
    TokenTable { features, offsets }
}

#[test]
fn test_prune_algorithm_1() {
    let tokens = make_table(&[vec![7], vec![7], vec![7, 8]]);
    let t1: HashMap<i32, usize> = [(7, 1), (8, 1)].into_iter().collect();
    let p1 = prune(&t1, &tokens, None);
    assert_eq!(p1.tokens, vec![2]);
    assert_eq!(p1.score, 2);

    let t2: HashMap<i32, usize> = [(7, 2), (8, 1)].into_iter().collect();
    let p2 = prune(&t2, &tokens, None);
    assert_eq!(p2.tokens, vec![0, 2]);
    assert_eq!(p2.score, 3);

    let tokens_self = make_table(&[vec![1, 2, 2], vec![3], vec![1, 4]]);
    let t_self: HashMap<i32, usize> = [(1, 2), (2, 2), (3, 1), (4, 1)].into_iter().collect();
    let p_self = prune(&t_self, &tokens_self, None);
    assert_eq!(p_self.tokens, vec![0, 1, 2]);
    assert_eq!(p_self.score, 6);

    let mut state: u32 = 20181201;
    let mut next = |bound: u32| -> u32 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        state % bound
    };

    for round in 0..300 {
        let alphabet = 1 + next(12);
        let ntoks = 1 + next(30);
        let mut toks = Vec::with_capacity(ntoks as usize);
        for _ in 0..ntoks {
            let nf = 1 + next(6);
            let mut fs = Vec::with_capacity(nf as usize);
            for _ in 0..nf {
                fs.push(next(alphabet) as i32);
            }
            toks.push(fs);
        }
        let mut target = HashMap::new();
        for f in 0..alphabet {
            let cnt = next(4);
            if cnt > 0 {
                target.insert(f as i32, cnt as usize);
            }
        }
        let all: Vec<usize> = (0..toks.len()).collect();
        let candidates: Vec<usize> = if round % 3 == 0 {
            all.iter().copied().filter(|_| next(2) == 0).collect()
        } else {
            all.clone()
        };
        let tbl = make_table(&toks);
        assert_eq!(
            prune(&target, &tbl, Some(&candidates)),
            naive_prune(&target, &tbl, &candidates)
        );
    }
}

fn toy_method(name: &str, statements: &[&str]) -> SourceFile {
    let mut lines = vec![format!("method {name}")];
    for s in statements {
        lines.push((*s).to_string());
    }
    lines.push("end".to_string());
    SourceFile {
        path: format!("{name}.toy"),
        source: lines.join("\n"),
    }
}

fn toy_call(name: &str) -> String {
    format!("[ [ {name} ( ) ] ; ]")
}

fn toy_calls(name: &str, names: &[&str]) -> SourceFile {
    let stmts: Vec<String> = names.iter().map(|n| toy_call(n)).collect();
    let refs: Vec<&str> = stmts.iter().map(|s| s.as_str()).collect();
    toy_method(name, &refs)
}

fn toy_index(files: &[SourceFile]) -> aroma::AromaIndex {
    build_index(
        files,
        IndexOptions {
            languages: Some(vec![Box::new(ToyLanguage)]),
            ..Default::default()
        },
    )
}

fn toy_query(index: &aroma::AromaIndex, statements: &[&str]) -> aroma::Query {
    query_from_snippet(
        index,
        &statements.join("\n"),
        &SnippetOptions {
            language: Some("toy".to_string()),
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn test_search_sections_3_3_1_and_3_3_2() {
    let idx1 = toy_index(&[toy_method("m2", &["[ [ y = 2 ] ; ]", "[ [ z = 3 ] ; ]"])]);
    let q1 = toy_query(&idx1, &["[ [ x = 1 ] ; ]", "[ [ y = 2 ] ; ]"]);
    let res1 = search(&idx1, &q1, &SearchOptions::default());
    assert_eq!(res1[0].matched_lines, vec![2]);

    let idx2 = toy_index(&[toy_method(
        "m2",
        &[
            "[ if [ ( [ z < 0 ] ) ]",
            "[ if [ ( [ y > 1 ] ) ]",
            "[ [ w = 4 ] ; ] ] ]",
            "[ [ v = 10 ] ; ]",
        ],
    )]);
    let q2 = toy_query(
        &idx2,
        &[
            "[ [ x = 1 ] ; ]",
            "[ if [ ( [ y > 1 ] ) ] [ if [ ( [ z < 0 ] ) ] [ [ w = 4 ] ; ] ] ]",
        ],
    );
    assert_eq!(
        search(&idx2, &q2, &SearchOptions::default())[0].matched_lines,
        vec![2, 3, 4]
    );

    let idx3 = toy_index(&[
        toy_calls("half", &["a", "d", "e"]),
        toy_calls("full", &["a", "b", "c"]),
    ]);
    let ca = toy_call("a");
    let cb = toy_call("b");
    let q3 = toy_query(&idx3, &[&ca, &cb]);
    let res3 = search(&idx3, &q3, &SearchOptions::default());
    let names3: Vec<&str> = res3.iter().map(|r| r.method.name.as_str()).collect();
    assert_eq!(names3, vec!["full", "half"]);
    assert!((res3[0].similarity - 0.8).abs() < 1e-10);
    assert!((res3[1].similarity - 0.3).abs() < 1e-10);
    let filtered = search(
        &idx3,
        &q3,
        &SearchOptions {
            min_similarity: 0.65,
            ..Default::default()
        },
    );
    assert_eq!(filtered, vec![res3[0].clone()]);

    let ten_a: Vec<&str> = vec!["a"; 10];
    let idx4 = toy_index(&[
        toy_calls("repeats", &ten_a),
        toy_calls("both", &["a", "b"]),
    ]);
    let q4 = toy_query(&idx4, &[&ca, &cb]);
    let res4 = search(
        &idx4,
        &q4,
        &SearchOptions {
            search_limit: 1,
            ..Default::default()
        },
    );
    assert_eq!(res4.iter().map(|r| r.method.name.as_str()).collect::<Vec<_>>(), vec!["both"]);

    let idx5 = toy_index(&[
        toy_calls("pair", &["a", "b"]),
        toy_calls("triple", &["a", "a", "a"]),
    ]);
    let q5 = toy_query(&idx5, &[&ca, &ca, &ca, &cb]);
    let res5 = search(&idx5, &q5, &SearchOptions::default());
    assert_eq!(
        res5.iter().map(|r| r.method.name.as_str()).collect::<Vec<_>>(),
        vec!["triple", "pair"]
    );
    assert!((res5[0].similarity * 22.0 - 13.0).abs() < 1e-10);
    assert!((res5[1].similarity * 22.0 - 8.0).abs() < 1e-10);
    let res5_lim = search(
        &idx5,
        &q5,
        &SearchOptions {
            search_limit: 1,
            ..Default::default()
        },
    );
    assert_eq!(
        res5_lim
            .iter()
            .map(|r| r.method.name.as_str())
            .collect::<Vec<_>>(),
        vec!["pair"]
    );

    let idx6 = toy_index(&[
        toy_calls("one", &["a", "b"]),
        toy_calls("two", &["a", "b", "c"]),
        SourceFile {
            path: "notes.txt".to_string(),
            source: "a b".to_string(),
        },
    ]);
    let q6 = toy_query(&idx6, &[&ca, &cb]);
    let excl = |m: &aroma::MethodRef| m.name == "one";
    let res6 = search(
        &idx6,
        &q6,
        &SearchOptions {
            exclude: Some(&excl),
            ..Default::default()
        },
    );
    assert_eq!(
        res6.iter().map(|r| r.method.name.as_str()).collect::<Vec<_>>(),
        vec!["two"]
    );
    assert!(search(&idx6, &q6.with_language("other"), &SearchOptions::default()).is_empty());
    assert_eq!(idx6.stats().files, 2);
    assert_eq!(idx6.stats().methods, 2);
    assert_eq!(idx6.stats().unsupported_files, 1);

    let idx7 = toy_index(&[
        toy_calls("first", &["a", "b"]),
        toy_calls("copy", &["a", "b"]),
        toy_calls("c", &["a"]),
    ]);
    let mnames: Vec<String> = idx7.method_refs().into_iter().map(|m| m.name).collect();
    assert_eq!(mnames, vec!["first", "c"]);
    assert_eq!(idx7.stats().methods, 2);
    assert_eq!(idx7.stats().duplicate_methods, 1);
}

#[test]
fn test_recommend_section_3_3_3() {
    let family = |prefix: &str, shared: &[&str]| -> Vec<SourceFile> {
        (1..=3)
            .map(|n| {
                let own = format!("{prefix}{n}_only");
                let mut all = vec!["open", "read"];
                all.extend_from_slice(shared);
                all.push(&own);
                toy_calls(&format!("{prefix}{n}"), &all)
            })
            .collect()
    };
    let xs = family("x", &["flush", "close"]);
    let ys = family("y", &["log", "retry"]);
    let families = toy_index(&[
        xs[0].clone(),
        ys[0].clone(),
        xs[1].clone(),
        ys[1].clone(),
        xs[2].clone(),
        ys[2].clone(),
    ]);
    let c_open = toy_call("open");
    let c_read = toy_call("read");
    let open_read = toy_query(&families, &[&c_open, &c_read]);

    let recs = recommend(&families, &open_read, &RecommendOptions::default());
    let cluster_names: Vec<Vec<&str>> = recs
        .iter()
        .map(|r| r.members.iter().map(|m| m.name.as_str()).collect())
        .collect();
    assert_eq!(
        cluster_names,
        vec![vec!["x1", "x2", "x3"], vec!["y1", "y2", "y3"]]
    );
    let first = &recs[0];
    let second = &recs[1];
    assert_eq!(first.method.path, "x1.toy");
    assert_eq!(first.method.name, "x1");
    assert_eq!(first.method.start_line, 2);
    assert_eq!(first.method.end_line, 6);
    assert!((first.similarity - 0.8).abs() < 1e-10);
    assert_eq!(
        first.lines,
        vec![
            aroma::RecommendedLine {
                line: 2,
                text: toy_call("open"),
                role: LineRole::Query
            },
            aroma::RecommendedLine {
                line: 3,
                text: toy_call("read"),
                role: LineRole::Query
            },
            aroma::RecommendedLine {
                line: 4,
                text: toy_call("flush"),
                role: LineRole::Added
            },
            aroma::RecommendedLine {
                line: 5,
                text: toy_call("close"),
                role: LineRole::Added
            },
        ]
    );
    assert_eq!(
        first.code,
        [
            toy_call("open"),
            toy_call("read"),
            toy_call("flush"),
            toy_call("close")
        ]
        .join("\n")
    );
    assert_eq!(
        second
            .lines
            .iter()
            .map(|l| l.text.clone())
            .collect::<Vec<_>>(),
        ["open", "read", "log", "retry"].map(toy_call).to_vec()
    );

    let recs_min1 = recommend(
        &families,
        &open_read,
        &RecommendOptions {
            min_support: 1,
            ..Default::default()
        },
    );
    let names_min1: Vec<Vec<&str>> = recs_min1
        .iter()
        .map(|r| r.members.iter().map(|m| m.name.as_str()).collect())
        .collect();
    assert_eq!(
        names_min1,
        vec![
            vec!["x1", "x2", "x3"],
            vec!["y1", "y2", "y3"],
            vec!["x3"],
            vec!["y3"]
        ]
    );
    let recs_lim1 = recommend(
        &families,
        &open_read,
        &RecommendOptions {
            limit: 1,
            ..Default::default()
        },
    );
    assert_eq!(
        recs_lim1
            .iter()
            .map(|r| r.members.iter().map(|m| m.name.as_str()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec!["x1", "x2", "x3"]]
    );

    let idx_only = toy_index(&[toy_calls("only", &["open", "read", "flush", "close"])]);
    let q_only = toy_query(&idx_only, &[&c_open, &c_read]);
    assert!(recommend(&idx_only, &q_only, &RecommendOptions::default()).is_empty());
    let recs_only = recommend(
        &idx_only,
        &q_only,
        &RecommendOptions {
            min_support: 1,
            ..Default::default()
        },
    );
    assert_eq!(recs_only.len(), 1);
    assert_eq!(
        recs_only[0]
            .members
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>(),
        vec!["only"]
    );
    assert_eq!(
        recs_only[0].lines.iter().map(|l| l.role).collect::<Vec<_>>(),
        vec![
            LineRole::Query,
            LineRole::Query,
            LineRole::Added,
            LineRole::Added
        ]
    );

    let idx_exp = toy_index(&[
        toy_calls("same", &["open", "read"]),
        toy_calls("one_more", &["open", "read", "x"]),
    ]);
    let q_exp = toy_query(&idx_exp, &[&c_open, &c_read]);
    assert!(recommend(
        &idx_exp,
        &q_exp,
        &RecommendOptions {
            min_support: 1,
            ..Default::default()
        }
    )
    .is_empty());
    let recs_exp = recommend(
        &idx_exp,
        &q_exp,
        &RecommendOptions {
            min_support: 1,
            min_expansion: 1.4,
            ..Default::default()
        },
    );
    assert_eq!(
        recs_exp
            .iter()
            .map(|r| r.members.iter().map(|m| m.name.as_str()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec!["one_more"]]
    );

    let idx_coh = toy_index(&[
        toy_calls("m1", &["a", "b", "c", "x", "y", "z"]),
        toy_calls("m2", &["b", "c", "d", "x", "y", "z"]),
    ]);
    let ca = toy_call("a");
    let cb = toy_call("b");
    let cc = toy_call("c");
    let cd = toy_call("d");
    let q_coh = toy_query(&idx_coh, &[&ca, &cb, &cc, &cd]);
    let r_coh1 = recommend(
        &idx_coh,
        &q_coh,
        &RecommendOptions {
            min_similarity: 0.3,
            min_support: 1,
            ..Default::default()
        },
    );
    assert_eq!(
        r_coh1
            .iter()
            .map(|r| r.members.iter().map(|m| m.name.as_str()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec!["m1"], vec!["m2"]]
    );
    assert!(recommend(
        &idx_coh,
        &q_coh,
        &RecommendOptions {
            min_similarity: 0.3,
            ..Default::default()
        }
    )
    .is_empty());
    let r_coh2 = recommend(
        &idx_coh,
        &q_coh,
        &RecommendOptions {
            min_similarity: 0.3,
            min_cohesion: 0.4,
            ..Default::default()
        },
    );
    assert_eq!(
        r_coh2
            .iter()
            .map(|r| r.members.iter().map(|m| m.name.as_str()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec!["m1", "m2"]]
    );

    let idx_keep = toy_index(&[
        toy_calls("whole", &["a", "b", "c", "x", "y", "z", "w"]),
        toy_calls("partial", &["a", "b", "x", "y", "z", "w"]),
    ]);
    let q_keep = toy_query(&idx_keep, &[&ca, &cb, &cc]);
    let r_keep = recommend(
        &idx_keep,
        &q_keep,
        &RecommendOptions {
            min_similarity: 0.4,
            min_cohesion: 0.4,
            ..Default::default()
        },
    );
    assert_eq!(
        r_keep[0]
            .members
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>(),
        vec!["whole", "partial"]
    );
    assert_eq!(
        r_keep[0]
            .lines
            .iter()
            .map(|l| (l.text.clone(), l.role))
            .collect::<Vec<_>>(),
        vec![
            (toy_call("a"), LineRole::Query),
            (toy_call("b"), LineRole::Query),
            (toy_call("c"), LineRole::Query),
            (toy_call("x"), LineRole::Added),
            (toy_call("y"), LineRole::Added),
            (toy_call("z"), LineRole::Added),
            (toy_call("w"), LineRole::Added),
        ]
    );

    let idx_far = toy_index(&[toy_calls("far", &["a", "x", "y", "z", "w"])]);
    assert!(recommend(
        &idx_far,
        &toy_query(&idx_far, &[&ca, &cb]),
        &RecommendOptions {
            min_support: 1,
            ..Default::default()
        }
    )
    .is_empty());
}

// ============================================================================
// Tests from tests/languages.test.ts
// ============================================================================

fn python_module() -> SourceFile {
    file(
        "store.py",
        &[
            "import json",
            "",
            "def top(path, limit=10):",
            "    \"\"\"Load rows.\"\"\"",
            "    rows = []",
            "    def keep(row):",
            "        return row[\"n\"] < limit",
            "    with open(path) as handle:",
            "        for line in handle:  # one row per line",
            "            rows.append(json.loads(line))",
            "    return [row for row in rows if keep(row)]",
            "",
            "class Store:",
            "    size = 0",
            "    async def put(self, key, value):",
            "        self.items[key] = value if value else None",
            "    @staticmethod",
            "    def make():",
            "        return Store()",
            "squares = lambda n: n * n",
        ],
    )
}

fn typescript_module() -> SourceFile {
    file(
        "store.ts",
        &[
            "import { parse } from \"./parse.ts\";",
            "",
            "export function plain(input: string, limit = 10): number[] {",
            "  // Parse, then keep the small ones.",
            "  const values = parse(input);",
            "  return values.filter((value) => value < limit);",
            "}",
            "",
            "export const useThing = async (id: string) => {",
            "  const response = await fetch(`/things/${id}`);",
            "  return response?.ok ? response.json() : null;",
            "};",
            "",
            "class Cache<T> {",
            "  private items = new Map<string, T>();",
            "  get(key: string): T | undefined {",
            "    return this.items.get(key);",
            "  }",
            "  clear = () => {",
            "    this.items.clear();",
            "  };",
            "}",
            "",
            "const handlers = {",
            "  onError(error: Error) {",
            "    console.error(error instanceof TypeError, error.message);",
            "  },",
            "};",
        ],
    )
}

#[test]
fn test_python_front_end() {
    let py = python_language();
    let src = python_module();
    let methods = py.parse_methods(&src.source, &src.path);
    let sigs: Vec<(&str, u32, u32)> = methods
        .iter()
        .map(|m| (m.name.as_str(), m.start_line, m.end_line))
        .collect();
    assert_eq!(
        sigs,
        vec![("top", 3, 11), ("Store.put", 15, 16), ("Store.make", 18, 19)]
    );

    let mut exp_local = vec!["handle", "keep", "limit", "line", "path", "row", "rows"];
    exp_local.sort();
    assert_eq!(names(&methods[0].body, true), exp_local);

    let mut exp_nonlocal = vec!["\"n\"", "append", "json", "loads", "open"];
    exp_nonlocal.sort();
    assert_eq!(names(&methods[0].body, false), exp_nonlocal);

    let mut toks = Vec::new();
    collect_tokens(&methods[0].body, &mut toks);
    let joined: String = toks
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!joined.contains("Load rows"));
    assert!(!joined.contains("one row per line"));
    assert_eq!(names(&methods[1].body, false), vec!["None", "items"]);

    let mut labs = Vec::new();
    collect_labels(&methods[1].body, &mut labs);
    for expected in ["#=#", "#.#", "#[#]", "#if#else#"] {
        assert!(labs.iter().any(|l| l == expected), "missing label {expected} in {labs:?}");
    }
    let loads = toks.iter().find(|t| t.text == "loads").unwrap();
    assert_eq!(loads.line, 10);
    assert_eq!(loads.end_line, 10);
    assert!(!loads.keyword);
}

#[test]
fn test_typescript_front_end() {
    let ts = typescript_language();
    let src = typescript_module();
    let methods = ts.parse_methods(&src.source, &src.path);
    let sigs: Vec<(&str, u32, u32)> = methods
        .iter()
        .map(|m| (m.name.as_str(), m.start_line, m.end_line))
        .collect();
    assert_eq!(
        sigs,
        vec![
            ("plain", 3, 7),
            ("useThing", 9, 12),
            ("Cache.get", 16, 18),
            ("Cache.clear", 19, 21),
            ("handlers.onError", 25, 27),
        ]
    );

    assert_eq!(
        names(&methods[0].body, true),
        vec!["input", "limit", "value", "values"]
    );
    assert_eq!(names(&methods[0].body, false), vec!["filter", "parse"]);

    assert_eq!(names(&methods[4].body, true), vec!["error"]);
    let mut exp_nl4 = vec!["TypeError", "console", "error", "message"];
    exp_nl4.sort();
    assert_eq!(names(&methods[4].body, false), exp_nl4);

    let mut toks = Vec::new();
    collect_tokens(&methods[0].body, &mut toks);
    let joined: String = toks
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!joined.contains("keep the small ones"));

    let mut labs1 = Vec::new();
    collect_labels(&methods[1].body, &mut labs1);
    for expected in ["#?.#", "#?#:#", "await#"] {
        assert!(labs1.iter().any(|l| l == expected), "missing {expected} in {labs1:?}");
    }
    let mut labs4 = Vec::new();
    collect_labels(&methods[4].body, &mut labs4);
    assert!(labs4.iter().any(|l| l == "#instanceof#"));

    let component = file(
        "Banner.tsx",
        &[
            "export function Banner({ title }: Props) {",
            "  const [open, setOpen] = useState(true);",
            "  return open ? <Alert onClose={() => setOpen(false)}>{title}</Alert> : null;",
            "}",
        ],
    );
    let b_methods = ts.parse_methods(&component.source, &component.path);
    assert_eq!(b_methods[0].name, "Banner");
    assert_eq!(b_methods[0].start_line, 1);
    assert_eq!(b_methods[0].end_line, 4);
    assert_eq!(
        names(&b_methods[0].body, true),
        vec!["open", "setOpen", "title"]
    );
    let b_nonlocal = names(&b_methods[0].body, false);
    for expected in ["useState", "Alert", "onClose", "true", "false", "null"] {
        assert!(b_nonlocal.iter().any(|n| n == expected), "missing {expected} in {b_nonlocal:?}");
    }
}

/// A service factory: its methods are what callers use, so each is a method of its own.
fn account_service() -> SourceFile {
    file(
        "account.ts",
        &[
            "export const AccountService = ({ db }: Deps) => {",
            "  const load = async (sid: string) => {",
            "    const account = await db.account.findUniqueOrThrow({ where: { sid } });",
            "    return account;",
            "  };",
            "  audit(db);",
            "  return {",
            "    load,",
            "    rename: async function (sid: string, name: string) {",
            "      const account = await load(sid);",
            "      return db.account.update({ where: { id: account.id }, data: { name } });",
            "    },",
            "    count() {",
            "      return [1, 2].map((n) => n * 2).length;",
            "    },",
            "  };",
            "};",
        ],
    )
}

#[test]
fn test_typescript_splits_large_functions_into_named_functions() {
    let src = account_service();
    let methods = typescript_language_with(10).parse_methods(&src.source, &src.path);
    let sigs: Vec<(&str, u32, u32)> = methods
        .iter()
        .map(|m| (m.name.as_str(), m.start_line, m.end_line))
        .collect();
    assert_eq!(
        sigs,
        vec![
            ("AccountService", 1, 17),
            ("AccountService.load", 2, 5),
            ("AccountService.rename", 9, 12),
            ("AccountService.count", 13, 15),
        ]
    );
    // The inner bodies are gone from the outer method; the anonymous `map` callback stays in `count`.
    let mut outer = Vec::new();
    collect_tokens(&methods[0].body, &mut outer);
    assert!(!outer.iter().any(|t| t.text == "findUniqueOrThrow"));
    let mut count = Vec::new();
    collect_tokens(&methods[3].body, &mut count);
    assert!(count.iter().any(|t| t.text == "map"));
    // `db` is the factory's parameter, so it keeps its name in the inner method.
    // (`sid` in `{ where: { sid } }` is shorthand, which Lezer parses as a property name.)
    assert!(names(&methods[1].body, false).iter().any(|n| n == "db"));
    assert_eq!(names(&methods[1].body, true), vec!["account"]);

    // Below the split size, and with the default of 80 lines, the factory stays whole.
    let whole = typescript_language().parse_methods(&src.source, &src.path);
    assert_eq!(whole.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), vec!["AccountService"]);
}

#[test]
fn test_typescript_split_search_finds_inner_method() {
    let src = account_service();
    let index = build_index(
        &[src],
        IndexOptions {
            languages: Some(vec![typescript_language_with(10)]),
            ..Default::default()
        },
    );
    let query = query_from_snippet(
        &index,
        "const account = await db.account.findUniqueOrThrow({ where: { sid } });",
        &SnippetOptions {
            language: Some("typescript".into()),
            globals: Some(["db".to_string()].into_iter().collect()),
            ..Default::default()
        },
    )
    .unwrap();
    let hits = search(&index, &query, &SearchOptions::default());
    assert_eq!(hits[0].method.name, "AccountService.load");
    assert!((hits[0].similarity - 1.0).abs() < 1e-12);
}

const PY_FETCH: &[&str] = &[
    "def fetch_rows(client, table, timeout):",
    "    attempts = 0",
    "    while attempts < 3:",
    "        try:",
    "            with client.connect(timeout=timeout) as conn:",
    "                return [dict(row) for row in conn.query(table)]",
    "        except ConnectionError as err:",
    "            attempts += 1",
    "            log.warning(\"retry %s\", err)",
    "    raise RuntimeError(table)",
];

const PY_FETCH_RENAMED: &[&str] = &[
    "def read_all(db, name, wait):",
    "    tries = 0",
    "    while tries < 3:",
    "        try:",
    "            # The keyword argument keeps its name; only the value is local.",
    "            with db.connect(timeout=wait) as link:",
    "                return [dict(item) for item in link.query(name)]",
    "        except ConnectionError as exc:",
    "            tries += 1",
    "            log.warning(\"retry %s\", exc)",
    "    raise RuntimeError(name)",
];

const TS_LOAD: &[&str] = &[
    "export async function loadUser(api: Api, id: string): Promise<User | undefined> {",
    "  const { data, error } = await api.get(`/users/${id}`, { retries: 2 });",
    "  if (error) {",
    "    logger.warn(\"load failed\", error);",
    "    return undefined;",
    "  }",
    "  const roles = data.roles.map((role) => role.name);",
    "  for (const [index, name] of roles.entries()) {",
    "    if (name === \"admin\") {",
    "      return { ...data, primary: index };",
    "    }",
    "  }",
    "  try {",
    "    return parseUser(data);",
    "  } catch (cause) {",
    "    throw new LoadError(id, cause);",
    "  }",
    "}",
];

const TS_LOAD_RENAMED: &[&str] = &[
    "export async function readAccount(client: Api, key: string): Promise<User | undefined> {",
    "  const { data, error } = await client.get(`/users/${key}`, { retries: 2 });",
    "  if (error) {",
    "    logger.warn(\"load failed\", error);",
    "    return undefined;",
    "  }",
    "  const labels = data.roles.map((entry) => entry.name);",
    "  for (const [position, label] of labels.entries()) {",
    "    if (label === \"admin\") {",
    "      return { ...data, primary: position };",
    "    }",
    "  }",
    "  try {",
    "    return parseUser(data);",
    "  } catch (reason) {",
    "    throw new LoadError(key, reason);",
    "  }",
    "}",
];

#[test]
fn test_local_variables_renamed_identical() {
    let cases = [
        (
            file("a.py", PY_FETCH),
            file("b.py", PY_FETCH_RENAMED),
            SourceFile {
                path: "c.py".to_string(),
                source: PY_FETCH
                    .iter()
                    .map(|l| l.replace("client.connect", "client.open_session"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            },
        ),
        (
            file("a.ts", TS_LOAD),
            file("b.ts", TS_LOAD_RENAMED),
            SourceFile {
                path: "c.ts".to_string(),
                source: TS_LOAD
                    .iter()
                    .map(|l| l.replace("api.get", "api.post"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            },
        ),
    ];

    for (orig, renamed, other) in cases {
        let index = build_index(&[orig.clone(), renamed.clone(), other.clone()], IndexOptions::default());
        assert_eq!(index.stats().methods, 3);
        assert_eq!(index.stats().duplicate_methods, 0);
        let qs = queries_from_lines(&index, &orig, &all_lines(&orig));
        assert_eq!(qs.len(), 1);
        let res = search(&index, &qs[0], &SearchOptions::default());
        assert_eq!(res[0].method.path, renamed.path);
        assert_eq!(res[1].method.path, other.path);
        assert!((res[0].similarity - 1.0).abs() < 1e-12);
        assert!(res[1].similarity < 1.0);
        assert!(res[1].similarity > 0.65);
    }
}

#[test]
fn test_python_nonlocal_and_global() {
    let counter = |name: &str, total: &str| {
        file(
            &format!("{name}.py"),
            &[
                &format!("def {name}(items):"),
                &format!("    {total} = 0"),
                "    def visit(item):",
                &format!("        nonlocal {total}"),
                "        global VISITS",
                &format!("        {total} += item.size"),
                "        VISITS += 1",
                "    for item in items:",
                "        visit(item)",
                &format!("    return {total}"),
            ],
        )
    };
    let original = counter("count_sizes", "count");
    let py = python_language();
    let methods = py.parse_methods(&original.source, &original.path);
    assert_eq!(
        names(&methods[0].body, true),
        vec!["count", "item", "items", "visit"]
    );
    let nl = names(&methods[0].body, false);
    assert!(nl.contains(&"VISITS".to_string()));
    assert!(nl.contains(&"size".to_string()));

    let index = build_index(
        &[original.clone(), counter("sum_sizes", "total")],
        IndexOptions::default(),
    );
    let qs = queries_from_lines(&index, &original, &all_lines(&original));
    let res = search(&index, &qs[0], &SearchOptions::default());
    assert_eq!(res[0].method.name, "sum_sizes");
    assert!((res[0].similarity - 1.0).abs() < 1e-12);
}

#[test]
fn test_indexing_source_files_edge_cases() {
    let index = build_index(
        &[
            file("global.ts", &["function readGlobal() {", "  return value;", "}"]),
            file(
                "identity.ts",
                &["function identity(value: number) {", "  return value;", "}"],
            ),
            file(
                "inside.py",
                &["def inside(a):", "    if a:", "        log(a)", "        done(a)"],
            ),
            file(
                "after.py",
                &["def after(a):", "    if a:", "        log(a)", "    done(a)"],
            ),
        ],
        IndexOptions::default(),
    );
    let mnames: Vec<String> = index.method_refs().into_iter().map(|m| m.name).collect();
    assert_eq!(mnames, vec!["readGlobal", "identity", "inside", "after"]);
    assert_eq!(index.stats().methods, 4);
    assert_eq!(index.stats().duplicate_methods, 0);

    // Multiline token lines
    let report = |name: &str, table: &str| {
        file(
            &format!("{name}.py"),
            &[
                &format!("def {name}(db, {table}):"),
                "    statement = \"\"\"",
                "        SELECT id, total",
                "        FROM invoices",
                "    \"\"\"",
                &format!("    return db.run(statement, {table})"),
            ],
        )
    };
    let changed = report("load_invoices", "name");
    let idx_ml = build_index(
        &[changed.clone(), report("read_invoices", "table")],
        IndexOptions::default(),
    );
    let qs_ml = queries_from_lines(&idx_ml, &changed, &[3]);
    assert_eq!(qs_ml[0].lines.as_deref(), Some(&[3][..]));
    let res_ml = search(&idx_ml, &qs_ml[0], &SearchOptions::default());
    assert_eq!(res_ml[0].method.name, "read_invoices");
    assert_eq!(res_ml[0].matched_lines, vec![2, 3, 4, 5]);

    // Multiple functions on one line
    let callbacks = file(
        "jobs.ts",
        &[
            "register(() => one(), () => {",
            "  const rows = load(\"jobs\");",
            "  audit(rows);",
            "  metrics.count(\"loaded\", rows.length);",
            "  return rows.length;",
            "});",
            "const first = () => start(), second = () => {",
            "  stop();",
            "};",
        ],
    );
    let ts = typescript_language();
    let cb_methods = ts.parse_methods(&callbacks.source, &callbacks.path);
    assert_eq!(
        cb_methods
            .iter()
            .map(|m| (m.name.as_str(), m.start_line))
            .collect::<Vec<_>>(),
        vec![
            ("<anonymous>", 1),
            ("<anonymous>", 1),
            ("first", 7),
            ("second", 7),
        ]
    );
    let idx_cb = build_index(&[callbacks], IndexOptions::default());
    let q_cb = query_from_snippet(
        &idx_cb,
        "const rows = load(\"jobs\");\naudit(rows);",
        &SnippetOptions {
            language: Some("typescript".to_string()),
            ..Default::default()
        },
    )
    .unwrap();
    let rec_cb = recommend(
        &idx_cb,
        &q_cb,
        &RecommendOptions {
            min_support: 1,
            ..Default::default()
        },
    );
    assert_eq!(rec_cb[0].method.name, "<anonymous>");
    assert_eq!(rec_cb[0].method.start_line, 1);
    assert_eq!(rec_cb[0].method.end_line, 6);
    assert_eq!(
        rec_cb[0]
            .lines
            .iter()
            .map(|l| (l.line, l.role))
            .collect::<Vec<_>>(),
        vec![
            (1, LineRole::Structure),
            (2, LineRole::Query),
            (3, LineRole::Query),
            (4, LineRole::Added),
            (5, LineRole::Added),
            (6, LineRole::Structure),
        ]
    );

    // Deduplicate identical comments/layout diffs
    let mut b_py_lines = vec!["# A copy."];
    for l in PY_FETCH {
        b_py_lines.push(l);
        b_py_lines.push("");
    }
    let b_ts_lines: Vec<String> = TS_LOAD.iter().map(|l| format!("{l} // same")).collect();
    let b_ts_refs: Vec<&str> = b_ts_lines.iter().map(|s| s.as_str()).collect();
    let idx_dup = build_index(
        &[
            file("a.py", PY_FETCH),
            file("b.py", &b_py_lines),
            file("a.ts", TS_LOAD),
            file("b.ts", &b_ts_refs),
        ],
        IndexOptions::default(),
    );
    assert_eq!(idx_dup.stats().files, 4);
    assert_eq!(idx_dup.stats().methods, 2);
    assert_eq!(idx_dup.stats().duplicate_methods, 2);
    assert_eq!(
        idx_dup
            .method_refs()
            .into_iter()
            .map(|m| m.path)
            .collect::<Vec<_>>(),
        vec!["a.py", "a.ts"]
    );
}

#[test]
fn test_pasted_function_and_broken_syntax() {
    let original = file("lib/fetch.py", PY_FETCH);
    let mut pasted_lines = vec!["import log", ""];
    pasted_lines.extend_from_slice(&PY_FETCH[0..2]);
    pasted_lines.push("    # Pasted from lib.");
    pasted_lines.extend_from_slice(&PY_FETCH[2..]);
    let pasted = file("api/handlers.py", &pasted_lines);

    let orig_ref = MethodRef {
        path: "lib/fetch.py".to_string(),
        name: "fetch_rows".to_string(),
        language: "python".to_string(),
        start_line: 1,
        end_line: 10,
    };
    let pasted_ref = MethodRef {
        path: "api/handlers.py".to_string(),
        name: "fetch_rows".to_string(),
        language: "python".to_string(),
        start_line: 3,
        end_line: 13,
    };

    for files in [
        vec![pasted.clone(), original.clone()],
        vec![original.clone(), pasted.clone()],
    ] {
        let idx = build_index(&files, IndexOptions::default());
        assert_eq!(idx.stats().methods, 1);
        assert_eq!(idx.stats().duplicate_methods, 1);
        let q_from_pasted = &queries_from_lines(&idx, &pasted, &all_lines(&pasted))[0];
        let s1 = search(&idx, q_from_pasted, &SearchOptions::default());
        assert_eq!(s1[0].method, orig_ref);
        assert!((s1[0].similarity - 1.0).abs() < 1e-12);
        assert_eq!(s1[0].matched_lines, vec![2, 3, 5, 6, 7, 8, 9, 10]);

        let q_from_orig = &queries_from_lines(&idx, &original, &all_lines(&original))[0];
        let s2 = search(&idx, q_from_orig, &SearchOptions::default());
        assert_eq!(s2[0].method, pasted_ref);
        assert!((s2[0].similarity - 1.0).abs() < 1e-12);
        assert_eq!(s2[0].matched_lines, vec![4, 6, 8, 9, 10, 11, 12, 13]);
    }

    let idx_ex = build_index(&[pasted.clone(), original.clone()], IndexOptions::default());
    let q_ex = &queries_from_lines(&idx_ex, &original, &[2, 3])[0];
    let anywhere = q_ex.without_origin();
    let excl_pasted = |m: &MethodRef| m.path == pasted.path;
    let s_any = search(
        &idx_ex,
        &anywhere,
        &SearchOptions {
            exclude: Some(&excl_pasted),
            ..Default::default()
        },
    );
    assert_eq!(s_any[0].method, orig_ref);
    assert!(search(
        &idx_ex,
        q_ex,
        &SearchOptions {
            exclude: Some(&excl_pasted),
            ..Default::default()
        }
    )
    .is_empty());

    let idx_rec = build_index(&[original.clone(), pasted.clone()], IndexOptions::default());
    let q_rec = &queries_from_lines(&idx_rec, &original, &[2, 3])[0];
    let recs = recommend(
        &idx_rec,
        q_rec,
        &RecommendOptions {
            min_support: 1,
            ..Default::default()
        },
    );
    assert_eq!(recs[0].method, pasted_ref);
    assert_eq!(
        recs[0].lines.iter().map(|l| l.line).collect::<Vec<_>>(),
        vec![3, 4, 6, 7, 8, 9, 10, 11, 12, 13]
    );
    assert_eq!(
        &recs[0].lines[0..3],
        &[
            aroma::RecommendedLine {
                line: 3,
                text: "def fetch_rows(client, table, timeout):".to_string(),
                role: LineRole::Structure
            },
            aroma::RecommendedLine {
                line: 4,
                text: "    attempts = 0".to_string(),
                role: LineRole::Query
            },
            aroma::RecommendedLine {
                line: 6,
                text: "    while attempts < 3:".to_string(),
                role: LineRole::Query
            },
        ]
    );

    let idx_broken = build_index(
        &[
            file("types.d.ts", &["export declare function f(x: number): void;"]),
            file("notes.md", &["# Notes"]),
            file("big.ts", TS_LOAD),
            file(
                "broken.py",
                &[
                    "def ok(a):",
                    "    return a + 1",
                    "",
                    "def broken(:",
                    "    return (",
                ],
            ),
        ],
        IndexOptions {
            max_file_length: 400,
            ..Default::default()
        },
    );
    assert_eq!(idx_broken.stats().files, 2);
    assert_eq!(idx_broken.stats().unsupported_files, 1);
    assert_eq!(idx_broken.stats().oversized_files, 1);
    assert!(idx_broken
        .method_refs()
        .iter()
        .any(|m| m.name == "ok"));
}

#[test]
fn test_queries_from_changed_lines_and_snippets() {
    let changed = file(
        "billing.py",
        &[
            "LIMIT = 3",
            "",
            "def charge(client, account):",
            "    invoice = client.invoices.create(account=account.id)",
            "    client.invoices.finalize(invoice.id)",
            "    return invoice",
            "",
            "def refund(client, charge_id):",
            "    return client.refunds.create(charge=charge_id)",
        ],
    );
    let twin = file(
        "legacy.py",
        &[
            "def bill(api, customer):",
            "    draft = api.invoices.create(account=customer.id)",
            "    api.invoices.finalize(draft.id)",
            "    audit.record(\"billed\", customer.id)",
            "    return draft",
        ],
    );
    let idx = build_index(&[changed.clone(), twin.clone()], IndexOptions::default());
    let qs = queries_from_lines(&idx, &changed, &[1, 2, 4, 5]);
    assert_eq!(qs.len(), 1);
    assert_eq!(
        qs[0].origin.as_ref().unwrap(),
        &MethodRef {
            path: "billing.py".to_string(),
            name: "charge".to_string(),
            language: "python".to_string(),
            start_line: 3,
            end_line: 6,
        }
    );
    assert_eq!(qs[0].lines.as_deref(), Some(&[4, 5][..]));
    assert!(queries_from_lines(&idx, &file("notes.md", &["x"]), &[1]).is_empty());

    let qs2 = queries_from_lines(&idx, &changed, &[4, 5]);
    let res = search(&idx, &qs2[0], &SearchOptions::default());
    assert_eq!(res[0].method.path, "legacy.py");
    assert_eq!(res[0].method.name, "bill");
    assert_eq!(res[0].matched_lines, vec![2, 3]);
    assert!(!res.iter().any(|r| r.method.name == "charge"));

    let res_no_orig = search(&idx, &qs2[0].without_origin(), &SearchOptions::default());
    assert_eq!(res_no_orig[0].method.name, "charge");
    assert!((res_no_orig[0].similarity - 1.0).abs() < 1e-12);
    assert_eq!(res_no_orig[0].matched_lines, vec![4, 5]);

    // Snippet globals guessing
    let idx_snip = build_index(
        &[file(
            "load.py",
            &[
                "import json",
                "",
                "def load(raw):",
                "    data = json.loads(raw)",
                "    return data[\"items\"]",
            ],
        )],
        IndexOptions::default(),
    );
    let source = "data = json.loads(text)";
    let guessed = query_from_snippet(
        &idx_snip,
        source,
        &SnippetOptions {
            language: Some("python".to_string()),
            ..Default::default()
        },
    )
    .unwrap();
    let told = query_from_snippet(
        &idx_snip,
        source,
        &SnippetOptions {
            language: Some("python".to_string()),
            globals: Some(["json".to_string()].into_iter().collect()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        search(&idx_snip, &told, &SearchOptions::default())[0].similarity
            > search(&idx_snip, &guessed, &SearchOptions::default())[0].similarity
    );
    assert!(query_from_snippet(
        &idx_snip,
        "# nothing here",
        &SnippetOptions {
            language: Some("python".to_string()),
            ..Default::default()
        }
    )
    .is_none());
    assert!(query_from_snippet(
        &idx_snip,
        source,
        &SnippetOptions {
            language: Some("cobol".to_string()),
            ..Default::default()
        }
    )
    .is_none());
    assert_eq!(
        query_from_snippet(
            &idx_snip,
            "const a = b;",
            &SnippetOptions {
                path: Some("x.ts".to_string()),
                ..Default::default()
            }
        )
        .unwrap()
        .language(),
        "typescript"
    );

    let ts = typescript_language();
    let script = ts
        .parse_snippet(
            &[
                "import { cache, type Row } from \"./cache.ts\";",
                "const rows: Row[] = await load(source, Limits.max);",
                "cache.set(key, new Table(rows));",
            ]
            .join("\n"),
            &SnippetOptions::default(),
        )
        .unwrap();
    assert_eq!(names(&script, true), vec!["key", "rows", "source"]);
    for g in ["cache", "load", "Limits", "Table"] {
        assert!(names(&script, false).contains(&g.to_string()));
    }

    let py = python_language();
    let pymod = py
        .parse_snippet(
            &[
                "    import os",
                "    target = os.path.join(root, name)",
                "    return Path(target)",
            ]
            .join("\n"),
            &SnippetOptions::default(),
        )
        .unwrap();
    assert_eq!(names(&pymod, true), vec!["name", "root", "target"]);
    for g in ["os", "path", "join", "Path"] {
        assert!(names(&pymod, false).contains(&g.to_string()));
    }
}

#[test]
fn test_recommendations_from_real_source_files() {
    let listing_5 = file(
        "arcbit.ts",
        &[
            "export function setupUIToHideKeyBoardOnTouch(view: View): void {",
            "  if (!(view instanceof EditText)) {",
            "    view.setOnTouchListener({",
            "      onTouch(v: View, event: MotionEvent): boolean {",
            "        hideKeyBoard();",
            "        return false;",
            "      },",
            "    });",
            "  }",
            "  if (view instanceof ViewGroup) {",
            "    for (let i = 0; i < view.getChildCount(); i++) {",
            "      const innerView = view.getChildAt(i);",
            "      setupUIToHideKeyBoardOnTouch(innerView);",
            "    }",
            "  }",
            "}",
        ],
    );
    let listing_6 = file(
        "applozic.ts",
        &[
            "export function setupUI(view: View): void {",
            "  if (!(view instanceof EditText)) {",
            "    view.setOnTouchListener({",
            "      onTouch(v: View, event: MotionEvent): boolean {",
            "        Utils.toggleSoftKeyBoard(activity, true);",
            "        return false;",
            "      },",
            "    });",
            "  }",
            "  if (view instanceof ViewGroup) {",
            "    for (let i = 0; i < view.getChildCount(); i++) {",
            "      const innerView = view.getChildAt(i);",
            "      setupUI(innerView);",
            "    }",
            "  }",
            "}",
        ],
    );
    let listing_4 = [
        "if (view instanceof ViewGroup) {",
        "  for (let i = 0; i < view.getChildCount(); i++) {",
        "    const innerView = view.getChildAt(i);",
        "  }",
        "}",
    ]
    .join("\n");

    let idx = build_index(
        &[listing_5.clone(), listing_6, typescript_module()],
        IndexOptions::default(),
    );
    let q = query_from_snippet(
        &idx,
        &listing_4,
        &SnippetOptions {
            language: Some("typescript".to_string()),
            ..Default::default()
        },
    )
    .unwrap();
    let recs = recommend(&idx, &q, &RecommendOptions::default());
    assert_eq!(recs.len(), 1);
    let rec = &recs[0];
    assert_eq!(
        rec.members
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>(),
        vec!["setupUIToHideKeyBoardOnTouch", "setupUI"]
    );
    assert!(rec.similarity > 0.9);
    let src_lines: Vec<&str> = listing_5.source.split('\n').collect();
    let expected = |ln: u32, role: LineRole| aroma::RecommendedLine {
        line: ln,
        text: src_lines[(ln - 1) as usize].to_string(),
        role,
    };
    assert_eq!(
        rec.lines,
        vec![
            expected(1, LineRole::Structure),
            expected(2, LineRole::Added),
            expected(3, LineRole::Added),
            expected(4, LineRole::Added),
            expected(6, LineRole::Added),
            expected(7, LineRole::Structure),
            expected(8, LineRole::Structure),
            expected(9, LineRole::Structure),
            expected(10, LineRole::Query),
            expected(11, LineRole::Query),
            expected(12, LineRole::Query),
            expected(13, LineRole::Added),
            expected(14, LineRole::Structure),
            expected(15, LineRole::Structure),
            expected(16, LineRole::Structure),
        ]
    );
    assert_eq!(
        rec.code.split('\n').nth(1).unwrap(),
        "  if (!(view instanceof EditText)) {"
    );

    // Python error handling recommendation
    let fetcher = |name: &str, what: &str, tail: &[&str]| {
        let mut lines = vec![
            format!("def {name}(session, url):"),
            "    try:".to_string(),
            "        response = session.get(url, timeout=30)".to_string(),
            "        response.raise_for_status()".to_string(),
            "        payload = response.json()".to_string(),
            "    except requests.RequestException as error:".to_string(),
            format!("        logger.warning(\"{what} fetch failed: %s\", error)"),
            "        return None".to_string(),
        ];
        for t in tail {
            lines.push((*t).to_string());
        }
        SourceFile {
            path: format!("{what}.py"),
            source: lines.join("\n"),
        }
    };
    let idx_py = build_index(
        &[
            fetcher(
                "fetch_profile",
                "profile",
                &[
                    "    metrics.increment(\"profile.fetched\")",
                    "    return Profile.from_dict(payload)",
                ],
            ),
            fetcher(
                "fetch_invoices",
                "invoice",
                &[
                    "    cache.put(url, payload)",
                    "    return payload[\"invoices\"]",
                ],
            ),
            fetcher(
                "fetch_teams",
                "team",
                &["    return [Team(**item) for item in payload[\"teams\"]]"],
            ),
            file(
                "unrelated.py",
                &[
                    "def total(rows):",
                    "    amount = 0",
                    "    for row in rows:",
                    "        amount += row.price * row.quantity",
                    "    return amount",
                ],
            ),
        ],
        IndexOptions::default(),
    );
    let q_py = query_from_snippet(
        &idx_py,
        "response = session.get(url, timeout=30)\npayload = response.json()",
        &SnippetOptions {
            language: Some("python".to_string()),
            ..Default::default()
        },
    )
    .unwrap();
    let recs_py = recommend(&idx_py, &q_py, &RecommendOptions::default());
    assert_eq!(recs_py[0].method.path, "profile.py");
    assert_eq!(recs_py[0].method.name, "fetch_profile");
    let mut mem_names: Vec<&str> = recs_py[0]
        .members
        .iter()
        .map(|m| m.name.as_str())
        .collect();
    mem_names.sort();
    assert_eq!(
        mem_names,
        vec!["fetch_invoices", "fetch_profile", "fetch_teams"]
    );
    let roles: HashMap<u32, LineRole> = recs_py[0]
        .lines
        .iter()
        .map(|l| (l.line, l.role))
        .collect();
    assert_eq!(
        [2, 3, 4, 5, 6, 7, 8]
            .iter()
            .map(|l| roles.get(l).copied())
            .collect::<Vec<_>>(),
        vec![
            Some(LineRole::Structure),
            Some(LineRole::Query),
            Some(LineRole::Added),
            Some(LineRole::Query),
            Some(LineRole::Added),
            Some(LineRole::Added),
            Some(LineRole::Added),
        ]
    );
    assert!(!roles.contains_key(&9));
    assert!(recs_py[0].code.contains("response.raise_for_status()"));
    assert!(!recs_py[0].code.contains("metrics.increment"));
}
