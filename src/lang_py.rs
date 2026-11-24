//! Python language front-end, matching `ref-ts/src/languages/python.ts`.
//! Uses a shared thread-local `FrontScratch` (reused across Python & TS to halve
//! peak worker RSS), single-byte `kind` dispatch table, and zero-alloc `NameSet`.

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::lezer::{self, Grammar, SyntaxTree, NONE};
use crate::simple::{keep_all, Body, Cfg, Simplifier};
use crate::text::{dedent_source, NameSet, Src, SrcBuf};
use crate::{Language, Node, ParsedMethod, SnippetOptions};

/// Shared per-worker scratch for both Python and TypeScript front-ends.
#[derive(Default)]
pub(crate) struct FrontScratch {
    pub tree: SyntaxTree,
    pub src_buf: SrcBuf,
    pub set_a: NameSet,
    pub set_b: NameSet,
    pub classes: Vec<(u32, u32)>,
}

impl FrontScratch {
    pub fn release(&mut self) {
        self.tree.nodes.clear();
        self.tree.nodes.shrink_to_fit();
        self.src_buf.release();
        self.set_a.release();
        self.set_b.release();
        self.classes.clear();
        self.classes.shrink_to_fit();
    }
}

thread_local! {
    pub(crate) static TL_FRONT: RefCell<FrontScratch> = RefCell::new(FrontScratch::default());
}

pub(crate) fn shrink_tl_tree() {
    TL_FRONT.with(|cell| cell.borrow_mut().release());
}

// Single-byte node kinds for O(1) switch dispatch (0 = irrelevant)
const K_NONE: u8 = 0;
const K_VAR: u8 = 1;
const K_PATTERN: u8 = 2;
const K_LOOP: u8 = 3;
const K_PARAM_LIST: u8 = 4;
const K_ASSIGN_STMT: u8 = 5;
const K_ASSIGN_OP: u8 = 6;
const K_UPDATE_OR_NAMED: u8 = 7;
const K_WITH_OR_TRY: u8 = 8;
const K_FUNC_DEF: u8 = 9;
const K_CLASS_DEF: u8 = 10;
const K_LAMBDA: u8 = 11;
const K_BODY: u8 = 12;
const K_CAPTURE_PAT: u8 = 13;
const K_AS_PAT: u8 = 14;
const K_SCOPE_STMT: u8 = 15;
const K_IMPORT_STMT: u8 = 16;
const K_ARG_LIST: u8 = 17;
const K_CALL_EXPR: u8 = 18;
const K_KW_FOR: u8 = 19;
const K_KW_IN: u8 = 20;
const K_KW_AS: u8 = 21;
const K_KW_DEF: u8 = 22;
const K_KW_CLASS: u8 = 23;
const K_KW_NONLOCAL: u8 = 24;

struct PyTables {
    cfg: Cfg,
    kind: Vec<u8>,
}

fn tables() -> &'static PyTables {
    static T: OnceLock<PyTables> = OnceLock::new();
    T.get_or_init(|| {
        let nn = lezer::PYTHON.node_names;
        let kind = nn
            .iter()
            .map(|&n| match n {
                "VariableName" => K_VAR,
                "TupleExpression" | "ArrayExpression" | "ParenthesizedExpression" => K_PATTERN,
                "ForStatement"
                | "ComprehensionExpression"
                | "ArrayComprehensionExpression"
                | "DictionaryComprehensionExpression"
                | "SetComprehensionExpression" => K_LOOP,
                "ParamList" => K_PARAM_LIST,
                "AssignStatement" => K_ASSIGN_STMT,
                "AssignOp" => K_ASSIGN_OP,
                "UpdateStatement" | "NamedExpression" => K_UPDATE_OR_NAMED,
                "WithStatement" | "TryStatement" => K_WITH_OR_TRY,
                "FunctionDefinition" => K_FUNC_DEF,
                "ClassDefinition" => K_CLASS_DEF,
                "LambdaExpression" => K_LAMBDA,
                "Body" => K_BODY,
                "CapturePattern" => K_CAPTURE_PAT,
                "AsPattern" => K_AS_PAT,
                "ScopeStatement" => K_SCOPE_STMT,
                "ImportStatement" => K_IMPORT_STMT,
                "ArgList" => K_ARG_LIST,
                "CallExpression" => K_CALL_EXPR,
                "for" => K_KW_FOR,
                "in" => K_KW_IN,
                "as" => K_KW_AS,
                "def" => K_KW_DEF,
                "class" => K_KW_CLASS,
                "nonlocal" => K_KW_NONLOCAL,
                _ => K_NONE,
            })
            .collect();
        PyTables {
            cfg: Cfg::new(
                nn,
                &[
                    "VariableName",
                    "PropertyName",
                    "Number",
                    "Boolean",
                    "None",
                    "Ellipsis",
                    "String",
                    "ContinuedString",
                ],
                &["FormatString"],
                &["Comment"],
                true,
            ),
            kind,
        }
    })
}

#[inline(always)]
fn kind_of(t: &PyTables, tree: &SyntaxTree, n: u32) -> u8 {
    unsafe { *t.kind.get_unchecked(tree.ty(n) as usize) }
}

#[inline(always)]
fn is_keyword_arg(t: &PyTables, tree: &SyntaxTree, n: u32) -> bool {
    let p = tree.parent(n);
    if p == NONE || kind_of(t, tree, p) != K_ARG_LIST {
        return false;
    }
    let next = tree.next_sibling(n);
    next != NONE && kind_of(t, tree, next) == K_ASSIGN_OP
}

#[inline(always)]
fn is_callee(t: &PyTables, tree: &SyntaxTree, n: u32) -> bool {
    let p = tree.parent(n);
    p != NONE && kind_of(t, tree, p) == K_CALL_EXPR && tree.first_child(p) == n
}

fn bind_target(
    t: &PyTables,
    src: &Src<'_>,
    tree: &SyntaxTree,
    target: u32,
    bound: &mut NameSet,
) {
    if target == NONE {
        return;
    }
    match kind_of(t, tree, target) {
        K_VAR => bound.insert(src, tree.from(target), tree.to(target)),
        K_PATTERN => {
            let mut c = tree.first_child(target);
            while c != NONE {
                bind_target(t, src, tree, c, bound);
                c = tree.next_sibling(c);
            }
        }
        _ => {}
    }
}

fn names_under(
    t: &PyTables,
    src: &Src<'_>,
    tree: &SyntaxTree,
    root: u32,
    bound: &mut NameSet,
    outer: &mut NameSet,
) {
    bound.clear();
    outer.clear();
    let first = tree.first_child(root);
    if first == NONE {
        return;
    }
    let mut cur = first;
    let mut depth = 1u32;
    loop {
        match kind_of(t, tree, cur) {
            K_PARAM_LIST => {
                let mut prev_assign = false;
                let mut c = tree.first_child(cur);
                while c != NONE {
                    if !prev_assign {
                        bind_target(t, src, tree, c, bound);
                    }
                    prev_assign = kind_of(t, tree, c) == K_ASSIGN_OP;
                    c = tree.next_sibling(c);
                }
            }
            K_ASSIGN_STMT => {
                let mut last_op = NONE;
                let mut c = tree.first_child(cur);
                while c != NONE {
                    if kind_of(t, tree, c) == K_ASSIGN_OP {
                        last_op = c;
                    }
                    c = tree.next_sibling(c);
                }
                if last_op != NONE {
                    let mut c = tree.first_child(cur);
                    while c != NONE && c != last_op {
                        bind_target(t, src, tree, c, bound);
                        c = tree.next_sibling(c);
                    }
                }
            }
            K_UPDATE_OR_NAMED | K_CAPTURE_PAT => {
                bind_target(t, src, tree, tree.first_child(cur), bound);
            }
            K_LOOP => {
                let mut binding = false;
                let mut c = tree.first_child(cur);
                while c != NONE {
                    match kind_of(t, tree, c) {
                        K_KW_FOR => binding = true,
                        K_KW_IN => binding = false,
                        _ if binding => bind_target(t, src, tree, c, bound),
                        _ => {}
                    }
                    c = tree.next_sibling(c);
                }
            }
            K_WITH_OR_TRY => {
                let mut prev_as = false;
                let mut c = tree.first_child(cur);
                while c != NONE {
                    if prev_as {
                        bind_target(t, src, tree, c, bound);
                    }
                    prev_as = kind_of(t, tree, c) == K_KW_AS;
                    c = tree.next_sibling(c);
                }
            }
            K_FUNC_DEF => {
                let mut prev_def = false;
                let mut c = tree.first_child(cur);
                while c != NONE {
                    if prev_def {
                        bind_target(t, src, tree, c, bound);
                    }
                    prev_def = kind_of(t, tree, c) == K_KW_DEF;
                    c = tree.next_sibling(c);
                }
            }
            K_CLASS_DEF => {
                let mut prev_cls = false;
                let mut c = tree.first_child(cur);
                while c != NONE {
                    if prev_cls {
                        bind_target(t, src, tree, c, bound);
                    }
                    prev_cls = kind_of(t, tree, c) == K_KW_CLASS;
                    c = tree.next_sibling(c);
                }
            }
            K_AS_PAT => {
                bind_target(t, src, tree, tree.last_child(cur), bound);
            }
            K_SCOPE_STMT => {
                let fc = tree.first_child(cur);
                if fc != NONE && kind_of(t, tree, fc) == K_KW_NONLOCAL {
                    let mut c = fc;
                    while c != NONE {
                        bind_target(t, src, tree, c, bound);
                        c = tree.next_sibling(c);
                    }
                } else {
                    let mut c = fc;
                    while c != NONE {
                        if kind_of(t, tree, c) == K_VAR {
                            outer.insert(src, tree.from(c), tree.to(c));
                        }
                        c = tree.next_sibling(c);
                    }
                }
            }
            K_IMPORT_STMT => {
                let mut c = tree.first_child(cur);
                while c != NONE {
                    if kind_of(t, tree, c) == K_VAR {
                        outer.insert(src, tree.from(c), tree.to(c));
                    }
                    c = tree.next_sibling(c);
                }
            }
            _ => {}
        }

        let fc = tree.first_child(cur);
        if fc != NONE {
            cur = fc;
            depth += 1;
            continue;
        }
        loop {
            let ns = tree.next_sibling(cur);
            if ns != NONE {
                cur = ns;
                break;
            }
            cur = tree.parent(cur);
            depth -= 1;
            if depth == 0 {
                return;
            }
        }
    }
}

#[inline]
fn span_after(t: &PyTables, tree: &SyntaxTree, parent: u32, kw: u8) -> (u32, u32) {
    let mut prev_match = false;
    let mut c = tree.first_child(parent);
    while c != NONE {
        if prev_match {
            return (tree.from(c), tree.to(c));
        }
        prev_match = kind_of(t, tree, c) == kw;
        c = tree.next_sibling(c);
    }
    (0, 0)
}

pub(crate) struct FastMethod {
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub body_root: u32,
}

pub(crate) fn collect_parsed_methods(
    source: &str,
    parse: impl FnOnce(&mut Body, &mut Vec<FastMethod>),
) -> Vec<ParsedMethod> {
    let mut body = Body::default();
    let mut raw = Vec::new();
    parse(&mut body, &mut raw);
    let base = source.as_bytes();
    raw.into_iter()
        .map(|m| ParsedMethod {
            name: m.name,
            start_line: m.start_line,
            end_line: m.end_line,
            body: body.to_node(base, m.body_root),
        })
        .collect()
}

/// Parses a Python file directly into a reusable `Body` arena, returning
/// each method's metadata and root index inside `body`.
/// When `line_range` is `Some((min_ln, max_ln))`, methods outside that line range
/// are skipped BEFORE variable analysis and AST simplification.
pub(crate) fn parse_methods_into(
    source: &str,
    line_range: Option<(u32, u32)>,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    let t = tables();
    TL_FRONT.with(|cell| {
        let mut sc = cell.borrow_mut();
        let FrontScratch { tree, src_buf, set_a, set_b, classes } = &mut *sc;
        let src = Src::build(source, src_buf);
        src.parse_into(&lezer::PYTHON, tree);
        classes.clear();

        #[allow(clippy::too_many_arguments)]
        fn walk(
            t: &PyTables,
            src: &Src<'_>,
            tree: &SyntaxTree,
            n: u32,
            line_range: Option<(u32, u32)>,
            classes: &mut Vec<(u32, u32)>,
            bound: &mut NameSet,
            outer: &mut NameSet,
            body: &mut Body,
            out: &mut Vec<FastMethod>,
        ) {
            match kind_of(t, tree, n) {
                K_FUNC_DEF => {
                    let b = tree.last_child(n);
                    if b == NONE || kind_of(t, tree, b) != K_BODY {
                        return;
                    }
                    let (start_line, end_line) = if let Some((min_ln, max_ln)) = line_range {
                        let (sl, el) = src.line_span(tree.from(n), tree.to(n));
                        if el < min_ln || sl > max_ln {
                            return;
                        }
                        (sl, el)
                    } else {
                        (0, 0)
                    };
                    names_under(t, src, tree, n, bound, outer);
                    let simp = Simplifier {
                        src,
                        tree,
                        cfg: &t.cfg,
                        is_local: |node| {
                            !bound.is_empty()
                                && kind_of(t, tree, node) == K_VAR
                                && bound.contains(src, tree.from(node), tree.to(node))
                                && !outer.contains(src, tree.from(node), tree.to(node))
                                && !is_keyword_arg(t, tree, node)
                        },
                        skip: keep_all,
                    };
                    if let Some(root) = simp.run(body, b) {
                        let fn_span = span_after(t, tree, n, K_KW_DEF);
                        let name = if classes.is_empty() {
                            match src.str_range(fn_span.0, fn_span.1) {
                                Some(s) => s.to_string(),
                                None => src.string(fn_span.0, fn_span.1),
                            }
                        } else {
                            let mut s = String::new();
                            for &cs in classes.iter() {
                                src.push_span(cs, &mut s);
                                s.push('.');
                            }
                            src.push_span(fn_span, &mut s);
                            s
                        };
                        let (sl, el) = if line_range.is_some() {
                            (start_line, end_line)
                        } else {
                            src.line_span(tree.from(n), tree.to(n))
                        };
                        out.push(FastMethod { name, start_line: sl, end_line: el, body_root: root });
                    }
                    return;
                }
                K_LAMBDA => return,
                K_CLASS_DEF => {
                    classes.push(span_after(t, tree, n, K_KW_CLASS));
                    let mut c = tree.first_child(n);
                    while c != NONE {
                        walk(t, src, tree, c, line_range, classes, bound, outer, body, out);
                        c = tree.next_sibling(c);
                    }
                    classes.pop();
                    return;
                }
                _ => {}
            }
            let mut c = tree.first_child(n);
            while c != NONE {
                walk(t, src, tree, c, line_range, classes, bound, outer, body, out);
                c = tree.next_sibling(c);
            }
        }

        walk(t, &src, tree, 0, line_range, classes, set_a, set_b, body, out);
    });
}

pub(crate) fn simplify_lezer_snippet(
    source: &str,
    grammar: &'static Grammar,
    cfg: &Cfg,
    body: &mut Body,
    collect_names: impl FnOnce(&Src<'_>, &SyntaxTree, &mut NameSet, &mut NameSet),
    is_local: impl Fn(&Src<'_>, &SyntaxTree, &NameSet, &NameSet, u32) -> bool,
) -> Option<u32> {
    TL_FRONT.with(|cell| {
        let mut sc = cell.borrow_mut();
        let FrontScratch { tree, src_buf, set_a, set_b, .. } = &mut *sc;
        let src = Src::build(source, src_buf);
        src.parse_into(grammar, tree);
        collect_names(&src, tree, set_a, set_b);
        let a = &*set_a;
        let b = &*set_b;
        let simp = Simplifier {
            src: &src,
            tree,
            cfg,
            is_local: |node| is_local(&src, tree, a, b, node),
            skip: keep_all,
        };
        simp.run(body, 0)
    })
}

/// Parses a Python snippet into `body`. Returns `(root, dedented_source)` so
/// token spans referencing `dedented_source` remain valid.
pub(crate) fn parse_snippet_into(
    source: &str,
    options: &SnippetOptions,
    body: &mut Body,
) -> Option<(u32, String)> {
    let t = tables();
    let dedented = dedent_source(source);
    let globals = options.globals.as_ref();
    let root = simplify_lezer_snippet(
        &dedented,
        &lezer::PYTHON,
        &t.cfg,
        body,
        |src, tree, bound, outer| names_under(t, src, tree, 0, bound, outer),
        |src, tree, bound, outer, node| {
            if kind_of(t, tree, node) != K_VAR {
                return false;
            }
            let (a, b) = (tree.from(node), tree.to(node));
            if outer.contains(src, a, b) {
                return false;
            }
            if bound.contains(src, a, b) {
                return !is_keyword_arg(t, tree, node);
            }
            !(src.looks_global(a, b, globals)
                || is_keyword_arg(t, tree, node)
                || is_callee(t, tree, node))
        },
    )?;
    Some((root, dedented))
}

pub struct PythonLanguage;

impl Language for PythonLanguage {
    fn id(&self) -> &str {
        "python"
    }

    fn extensions(&self) -> &[&str] {
        &[".py"]
    }

    fn parse_methods(&self, source: &str, _path: &str) -> Vec<ParsedMethod> {
        collect_parsed_methods(source, |body, raw| {
            parse_methods_into(source, None, body, raw);
        })
    }

    fn parse_snippet(&self, source: &str, options: &SnippetOptions) -> Option<Node> {
        let mut body = Body::default();
        let (root, dedented) = parse_snippet_into(source, options, &mut body)?;
        Some(body.to_node(dedented.as_bytes(), root))
    }

    fn builtin_kind(&self) -> u8 {
        1
    }
}
