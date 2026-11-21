//! Python language front-end, matching `ref-ts/src/languages/python.ts`.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::OnceLock;

use rustc_hash::FxHashSet;

use crate::lezer::{self, SyntaxTree, NONE};
use crate::simple::{Body, Cfg, Simplifier};
use crate::text::{dedent_source, js_ws, Src};
use crate::{Language, Node, ParsedMethod, SnippetOptions};

thread_local! {
    static TL_PY_TREE: RefCell<SyntaxTree> = RefCell::new(SyntaxTree::default());
}

pub(crate) fn shrink_tl_tree() {
    TL_PY_TREE.with(|cell| {
        let mut t = cell.borrow_mut();
        t.nodes.clear();
        t.nodes.shrink_to_fit();
    });
}

struct PyTables {
    cfg: Cfg,
    is_var_name: Vec<bool>,
    is_pattern: Vec<bool>,
    is_loop: Vec<bool>,
    is_param_list: Vec<bool>,
    is_assign_stmt: Vec<bool>,
    is_assign_op: Vec<bool>,
    is_update_or_named: Vec<bool>,
    is_with_or_try: Vec<bool>,
    is_func_def: Vec<bool>,
    is_class_def: Vec<bool>,
    is_lambda: Vec<bool>,
    is_body: Vec<bool>,
    is_capture_pat: Vec<bool>,
    is_as_pat: Vec<bool>,
    is_scope_stmt: Vec<bool>,
    is_import_stmt: Vec<bool>,
    is_arg_list: Vec<bool>,
    is_call_expr: Vec<bool>,
    is_kw_for: Vec<bool>,
    is_kw_in: Vec<bool>,
    is_kw_as: Vec<bool>,
    is_kw_def: Vec<bool>,
    is_kw_class: Vec<bool>,
    is_kw_nonlocal: Vec<bool>,
}

fn tables() -> &'static PyTables {
    static T: OnceLock<PyTables> = OnceLock::new();
    T.get_or_init(|| {
        let nn = lezer::PYTHON.node_names;
        let flags = |set: &[&str]| nn.iter().map(|n| set.contains(n)).collect::<Vec<bool>>();
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
            is_var_name: flags(&["VariableName"]),
            is_pattern: flags(&["TupleExpression", "ArrayExpression", "ParenthesizedExpression"]),
            is_loop: flags(&[
                "ForStatement",
                "ComprehensionExpression",
                "ArrayComprehensionExpression",
                "DictionaryComprehensionExpression",
                "SetComprehensionExpression",
            ]),
            is_param_list: flags(&["ParamList"]),
            is_assign_stmt: flags(&["AssignStatement"]),
            is_assign_op: flags(&["AssignOp"]),
            is_update_or_named: flags(&["UpdateStatement", "NamedExpression"]),
            is_with_or_try: flags(&["WithStatement", "TryStatement"]),
            is_func_def: flags(&["FunctionDefinition"]),
            is_class_def: flags(&["ClassDefinition"]),
            is_lambda: flags(&["LambdaExpression"]),
            is_body: flags(&["Body"]),
            is_capture_pat: flags(&["CapturePattern"]),
            is_as_pat: flags(&["AsPattern"]),
            is_scope_stmt: flags(&["ScopeStatement"]),
            is_import_stmt: flags(&["ImportStatement"]),
            is_arg_list: flags(&["ArgList"]),
            is_call_expr: flags(&["CallExpression"]),
            is_kw_for: flags(&["for"]),
            is_kw_in: flags(&["in"]),
            is_kw_as: flags(&["as"]),
            is_kw_def: flags(&["def"]),
            is_kw_class: flags(&["class"]),
            is_kw_nonlocal: flags(&["nonlocal"]),
        }
    })
}

#[inline]
fn node_text<'a>(src: &Src<'a>, tree: &SyntaxTree, n: u32) -> Cow<'a, str> {
    let (a, b) = (tree.from(n), tree.to(n));
    match src.str_range(a, b) {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(src.string(a, b)),
    }
}

fn line_span(src: &Src<'_>, from: u32, to: u32) -> (u32, u32) {
    let mut end = to;
    while end > from + 1 && js_ws(src.unit(end - 1)) {
        end -= 1;
    }
    (src.line_of(from), src.line_of(end.saturating_sub(1)))
}

#[inline]
fn looks_global(word: &str, globals: Option<&HashSet<String>>) -> bool {
    word.as_bytes().first().is_some_and(|b| b.is_ascii_uppercase())
        || globals.is_some_and(|g| g.contains(word))
}

#[inline]
fn is_keyword_arg(t: &PyTables, tree: &SyntaxTree, n: u32) -> bool {
    let p = tree.parent(n);
    if p == NONE || !t.is_arg_list[tree.ty(p) as usize] {
        return false;
    }
    let next = tree.next_sibling(n);
    next != NONE && t.is_assign_op[tree.ty(next) as usize]
}

#[inline]
fn is_callee(t: &PyTables, tree: &SyntaxTree, n: u32) -> bool {
    let p = tree.parent(n);
    p != NONE && t.is_call_expr[tree.ty(p) as usize] && tree.first_child(p) == n
}

fn bind_target<'a>(
    t: &PyTables,
    src: &Src<'a>,
    tree: &SyntaxTree,
    target: u32,
    bound: &mut FxHashSet<Cow<'a, str>>,
) {
    if target == NONE {
        return;
    }
    let ty = tree.ty(target) as usize;
    if t.is_var_name[ty] {
        bound.insert(node_text(src, tree, target));
    } else if t.is_pattern[ty] {
        let mut c = tree.first_child(target);
        while c != NONE {
            bind_target(t, src, tree, c, bound);
            c = tree.next_sibling(c);
        }
    }
}

fn names_under<'a>(
    t: &PyTables,
    src: &Src<'a>,
    tree: &SyntaxTree,
    root: u32,
    bound: &mut FxHashSet<Cow<'a, str>>,
    outer: &mut FxHashSet<Cow<'a, str>>,
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
        let ty = tree.ty(cur) as usize;
        if t.is_param_list[ty] {
            let mut prev_assign = false;
            let mut c = tree.first_child(cur);
            while c != NONE {
                if !prev_assign {
                    bind_target(t, src, tree, c, bound);
                }
                prev_assign = t.is_assign_op[tree.ty(c) as usize];
                c = tree.next_sibling(c);
            }
        } else if t.is_assign_stmt[ty] {
            // Find last AssignOp child
            let mut last_op = NONE;
            let mut c = tree.first_child(cur);
            while c != NONE {
                if t.is_assign_op[tree.ty(c) as usize] {
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
        } else if t.is_update_or_named[ty] {
            bind_target(t, src, tree, tree.first_child(cur), bound);
        } else if t.is_loop[ty] {
            let mut binding = false;
            let mut c = tree.first_child(cur);
            while c != NONE {
                let cty = tree.ty(c) as usize;
                if t.is_kw_for[cty] {
                    binding = true;
                } else if t.is_kw_in[cty] {
                    binding = false;
                } else if binding {
                    bind_target(t, src, tree, c, bound);
                }
                c = tree.next_sibling(c);
            }
        } else if t.is_with_or_try[ty] {
            let mut prev_as = false;
            let mut c = tree.first_child(cur);
            while c != NONE {
                if prev_as {
                    bind_target(t, src, tree, c, bound);
                }
                prev_as = t.is_kw_as[tree.ty(c) as usize];
                c = tree.next_sibling(c);
            }
        } else if t.is_func_def[ty] {
            let mut prev_def = false;
            let mut c = tree.first_child(cur);
            while c != NONE {
                if prev_def {
                    bind_target(t, src, tree, c, bound);
                }
                prev_def = t.is_kw_def[tree.ty(c) as usize];
                c = tree.next_sibling(c);
            }
        } else if t.is_class_def[ty] {
            let mut prev_cls = false;
            let mut c = tree.first_child(cur);
            while c != NONE {
                if prev_cls {
                    bind_target(t, src, tree, c, bound);
                }
                prev_cls = t.is_kw_class[tree.ty(c) as usize];
                c = tree.next_sibling(c);
            }
        } else if t.is_capture_pat[ty] {
            bind_target(t, src, tree, tree.first_child(cur), bound);
        } else if t.is_as_pat[ty] {
            bind_target(t, src, tree, tree.last_child(cur), bound);
        } else if t.is_scope_stmt[ty] {
            let fc = tree.first_child(cur);
            if fc != NONE && t.is_kw_nonlocal[tree.ty(fc) as usize] {
                let mut c = fc;
                while c != NONE {
                    bind_target(t, src, tree, c, bound);
                    c = tree.next_sibling(c);
                }
            } else {
                let mut c = fc;
                while c != NONE {
                    if t.is_var_name[tree.ty(c) as usize] {
                        outer.insert(node_text(src, tree, c));
                    }
                    c = tree.next_sibling(c);
                }
            }
        } else if t.is_import_stmt[ty] {
            let mut c = tree.first_child(cur);
            while c != NONE {
                if t.is_var_name[tree.ty(c) as usize] {
                    outer.insert(node_text(src, tree, c));
                }
                c = tree.next_sibling(c);
            }
        }

        // Advance DFS over strict descendants of `root`
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
                for name in outer.iter() {
                    bound.remove(name);
                }
                return;
            }
        }
    }
}

fn name_after<'a>(src: &Src<'a>, tree: &SyntaxTree, parent: u32, kw_flags: &[bool]) -> Cow<'a, str> {
    let mut prev_match = false;
    let mut c = tree.first_child(parent);
    while c != NONE {
        if prev_match {
            return node_text(src, tree, c);
        }
        prev_match = kw_flags[tree.ty(c) as usize];
        c = tree.next_sibling(c);
    }
    Cow::Borrowed("")
}

pub(crate) struct FastMethod {
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub body_root: u32,
}

/// Parses a Python file directly into a reusable `Body` arena, returning
/// each method's metadata and root index inside `body`.
pub(crate) fn parse_methods_into(
    source: &str,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    let t = tables();
    let src = Src::new(source);
    TL_PY_TREE.with(|cell| {
        let mut tree = cell.borrow_mut();
        src.parse_into(&lezer::PYTHON, &mut tree);
        let mut classes: Vec<Cow<'_, str>> = Vec::new();
        let mut bound: FxHashSet<Cow<'_, str>> = FxHashSet::default();
        let mut outer: FxHashSet<Cow<'_, str>> = FxHashSet::default();

        #[allow(clippy::too_many_arguments)]
        fn walk<'a>(
            t: &PyTables,
            src: &Src<'a>,
            tree: &SyntaxTree,
            n: u32,
            classes: &mut Vec<Cow<'a, str>>,
            bound: &mut FxHashSet<Cow<'a, str>>,
            outer: &mut FxHashSet<Cow<'a, str>>,
            body: &mut Body,
            out: &mut Vec<FastMethod>,
        ) {
            let ty = tree.ty(n) as usize;
            if t.is_func_def[ty] {
                let b = tree.last_child(n);
                if b == NONE || !t.is_body[tree.ty(b) as usize] {
                    return;
                }
                names_under(t, src, tree, n, bound, outer);
                let simp = Simplifier {
                    src,
                    tree,
                    cfg: &t.cfg,
                    is_local: |node| {
                        !bound.is_empty()
                            && t.is_var_name[tree.ty(node) as usize]
                            && bound.contains(&node_text(src, tree, node))
                            && !is_keyword_arg(t, tree, node)
                    },
                };
                if let Some(root) = simp.run(body, b) {
                    let fn_name = name_after(src, tree, n, &t.is_kw_def);
                    let name = if classes.is_empty() {
                        fn_name.into_owned()
                    } else {
                        let mut s = String::new();
                        for c in classes.iter() {
                            s.push_str(c);
                            s.push('.');
                        }
                        s.push_str(&fn_name);
                        s
                    };
                    let (start_line, end_line) = line_span(src, tree.from(n), tree.to(n));
                    out.push(FastMethod { name, start_line, end_line, body_root: root });
                }
                return;
            }
            if t.is_lambda[ty] {
                return;
            }
            let is_class = t.is_class_def[ty];
            if is_class {
                classes.push(name_after(src, tree, n, &t.is_kw_class));
            }
            let mut c = tree.first_child(n);
            while c != NONE {
                walk(t, src, tree, c, classes, bound, outer, body, out);
                c = tree.next_sibling(c);
            }
            if is_class {
                classes.pop();
            }
        }

        walk(t, &src, &tree, 0, &mut classes, &mut bound, &mut outer, body, out);
        if tree.nodes.capacity() > 32768 {
            tree.nodes.clear();
            tree.nodes.shrink_to(8192);
        }
    });
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
    let src = Src::new(&dedented);
    let root = TL_PY_TREE.with(|cell| {
        let mut tree = cell.borrow_mut();
        src.parse_into(&lezer::PYTHON, &mut tree);
        let mut bound: FxHashSet<Cow<'_, str>> = FxHashSet::default();
        let mut outer: FxHashSet<Cow<'_, str>> = FxHashSet::default();
        names_under(t, &src, &tree, 0, &mut bound, &mut outer);
        let globals = options.globals.as_ref();
        let simp = Simplifier {
            src: &src,
            tree: &tree,
            cfg: &t.cfg,
            is_local: |node| {
                if !t.is_var_name[tree.ty(node) as usize] {
                    return false;
                }
                let word = node_text(&src, &tree, node);
                if bound.contains(&word) {
                    return !is_keyword_arg(t, &tree, node);
                }
                !(outer.contains(&word)
                    || looks_global(&word, globals)
                    || is_keyword_arg(t, &tree, node)
                    || is_callee(t, &tree, node))
            },
        };
        simp.run(body, 0)
    })?;
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
        let mut body = Body::default();
        let mut raw = Vec::new();
        parse_methods_into(source, &mut body, &mut raw);
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

    fn parse_snippet(&self, source: &str, options: &SnippetOptions) -> Option<Node> {
        let mut body = Body::default();
        let (root, dedented) = parse_snippet_into(source, options, &mut body)?;
        Some(body.to_node(dedented.as_bytes(), root))
    }

    fn builtin_kind(&self) -> u8 {
        1
    }
}
