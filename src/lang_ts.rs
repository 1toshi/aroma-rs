//! TypeScript / JavaScript / JSX / TSX language front-end, matching
//! `ref-ts/src/languages/typescript.ts`.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::OnceLock;

use rustc_hash::FxHashSet;

use crate::lang_py::FastMethod;
use crate::lezer::{self, Grammar, SyntaxTree, NONE};
use crate::simple::{Body, Cfg, Simplifier};
use crate::text::{js_ws, Src};
use crate::{Language, Node, ParsedMethod, SnippetOptions};

thread_local! {
    static TL_TS_TREE: RefCell<SyntaxTree> = RefCell::new(SyntaxTree::default());
}

pub(crate) fn shrink_tl_tree() {
    TL_TS_TREE.with(|cell| {
        let mut t = cell.borrow_mut();
        t.nodes.clear();
        t.nodes.shrink_to_fit();
    });
}

const NAMED_NONE: u8 = 0;
const NAMED_VAR_DEF: u8 = 1; // FunctionDeclaration, FunctionExpression, ClassDeclaration, ClassExpression, VariableDeclaration
const NAMED_PROP_OR_PRIV: u8 = 2; // MethodDeclaration, PropertyDeclaration
const NAMED_PROP_ONLY: u8 = 3; // Property

struct TsTables {
    cfg: Cfg,
    is_var: Vec<bool>,
    is_var_def: Vec<bool>,
    is_prop_def: Vec<bool>,
    is_priv_prop_def: Vec<bool>,
    is_function: Vec<bool>,
    is_property: Vec<bool>,
    is_param_list: Vec<bool>,
    is_block: Vec<bool>,
    is_arrow: Vec<bool>,
    is_import_decl: Vec<bool>,
    is_call_expr: Vec<bool>,
    is_new_expr: Vec<bool>,
    is_kw_new: Vec<bool>,
    named_kind: Vec<u8>,
}

fn tables() -> &'static TsTables {
    static T: OnceLock<TsTables> = OnceLock::new();
    T.get_or_init(|| {
        let nn = lezer::TS.node_names;
        let flags = |set: &[&str]| nn.iter().map(|n| set.contains(n)).collect::<Vec<bool>>();
        let named_kind = nn
            .iter()
            .map(|&n| match n {
                "FunctionDeclaration"
                | "FunctionExpression"
                | "ClassDeclaration"
                | "ClassExpression"
                | "VariableDeclaration" => NAMED_VAR_DEF,
                "MethodDeclaration" | "PropertyDeclaration" => NAMED_PROP_OR_PRIV,
                "Property" => NAMED_PROP_ONLY,
                _ => NAMED_NONE,
            })
            .collect();
        TsTables {
            cfg: Cfg::new(
                nn,
                &[
                    "VariableName",
                    "VariableDefinition",
                    "PropertyName",
                    "PropertyDefinition",
                    "PrivatePropertyName",
                    "PrivatePropertyDefinition",
                    "TypeName",
                    "TypeDefinition",
                    "Label",
                    "Number",
                    "String",
                    "RegExp",
                    "BooleanLiteral",
                    "null",
                    "JSXIdentifier",
                    "JSXText",
                    "JSXAttributeValue",
                ],
                &["TemplateString", "TemplateType"],
                &["LineComment", "BlockComment", "Hashbang"],
                false,
            ),
            is_var: flags(&["VariableName", "VariableDefinition"]),
            is_var_def: flags(&["VariableDefinition"]),
            is_prop_def: flags(&["PropertyDefinition"]),
            is_priv_prop_def: flags(&["PrivatePropertyDefinition"]),
            is_function: flags(&[
                "FunctionDeclaration",
                "FunctionExpression",
                "ArrowFunction",
                "MethodDeclaration",
            ]),
            is_property: flags(&["Property"]),
            is_param_list: flags(&["ParamList"]),
            is_block: flags(&["Block"]),
            is_arrow: flags(&["ArrowFunction"]),
            is_import_decl: flags(&["ImportDeclaration"]),
            is_call_expr: flags(&["CallExpression"]),
            is_new_expr: flags(&["NewExpression"]),
            is_kw_new: flags(&["new"]),
            named_kind,
        }
    })
}

#[inline]
fn grammar_for(path: Option<&str>) -> &'static Grammar {
    match path {
        None => &lezer::TS,
        Some(p) if p.ends_with(".ts") || p.ends_with(".mts") || p.ends_with(".cts") => &lezer::TS,
        Some(p) if p.ends_with(".tsx") => &lezer::TSX,
        Some(_) => &lezer::JSX,
    }
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
fn is_function(t: &TsTables, tree: &SyntaxTree, n: u32) -> bool {
    let ty = tree.ty(n) as usize;
    if t.is_function[ty] {
        return true;
    }
    if t.is_property[ty] {
        let mut c = tree.first_child(n);
        while c != NONE {
            if t.is_param_list[tree.ty(c) as usize] {
                return true;
            }
            c = tree.next_sibling(c);
        }
    }
    false
}

#[inline]
fn is_callee(t: &TsTables, tree: &SyntaxTree, n: u32) -> bool {
    let p = tree.parent(n);
    if p == NONE {
        return false;
    }
    let pty = tree.ty(p) as usize;
    if t.is_call_expr[pty] && tree.first_child(p) == n {
        return true;
    }
    if t.is_new_expr[pty] {
        let mut prev = NONE;
        let mut c = tree.first_child(p);
        while c != NONE && c != n {
            prev = c;
            c = tree.next_sibling(c);
        }
        return prev != NONE && t.is_kw_new[tree.ty(prev) as usize];
    }
    false
}

fn names_under<'a>(
    t: &TsTables,
    src: &Src<'a>,
    tree: &SyntaxTree,
    root: u32,
    declared: &mut FxHashSet<Cow<'a, str>>,
    imported: &mut FxHashSet<Cow<'a, str>>,
) {
    declared.clear();
    imported.clear();
    let first = tree.first_child(root);
    if first == NONE {
        return;
    }
    let mut cur = first;
    let mut depth = 1u32;
    loop {
        if t.is_var_def[tree.ty(cur) as usize] {
            let p = tree.parent(cur);
            let is_import = if p != NONE {
                if t.is_import_decl[tree.ty(p) as usize] {
                    true
                } else {
                    let gp = tree.parent(p);
                    gp != NONE && t.is_import_decl[tree.ty(gp) as usize]
                }
            } else {
                false
            };
            let w = node_text(src, tree, cur);
            if is_import {
                imported.insert(w);
            } else {
                declared.insert(w);
            }
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

fn name_of(t: &TsTables, src: &Src<'_>, tree: &SyntaxTree, node: u32) -> String {
    let mut parts: Vec<Cow<'_, str>> = Vec::new();
    let mut inner = NONE;
    let mut at = node;
    while at != NONE {
        let nk = t.named_kind[tree.ty(at) as usize];
        if nk != NAMED_NONE {
            let from = if inner == NONE { u32::MAX } else { tree.from(inner) };
            // Find last matching child with child.to <= from
            let mut last_match = NONE;
            let mut c = tree.first_child(at);
            while c != NONE {
                if tree.to(c) > from {
                    break;
                }
                let cty = tree.ty(c) as usize;
                let matches = match nk {
                    NAMED_VAR_DEF => t.is_var_def[cty],
                    NAMED_PROP_OR_PRIV => t.is_prop_def[cty] || t.is_priv_prop_def[cty],
                    NAMED_PROP_ONLY => t.is_prop_def[cty],
                    _ => false,
                };
                if matches {
                    last_match = c;
                }
                c = tree.next_sibling(c);
            }
            if last_match != NONE {
                parts.push(node_text(src, tree, last_match));
            }
        }
        inner = at;
        at = tree.parent(at);
    }
    if parts.is_empty() {
        return "<anonymous>".to_string();
    }
    if parts.len() == 1 {
        return parts.pop().unwrap().into_owned();
    }
    let mut out = String::new();
    for (i, p) in parts.iter().rev().enumerate() {
        if i > 0 {
            out.push('.');
        }
        out.push_str(p);
    }
    out
}

pub(crate) fn parse_methods_into(
    source: &str,
    path: &str,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    let t = tables();
    let src = Src::new(source);
    TL_TS_TREE.with(|cell| {
        let mut tree = cell.borrow_mut();
        src.parse_into(grammar_for(Some(path)), &mut tree);
        let mut declared: FxHashSet<Cow<'_, str>> = FxHashSet::default();
        let mut imported: FxHashSet<Cow<'_, str>> = FxHashSet::default();

        #[allow(clippy::too_many_arguments)]
        fn walk<'a>(
            t: &TsTables,
            src: &Src<'a>,
            tree: &SyntaxTree,
            n: u32,
            declared: &mut FxHashSet<Cow<'a, str>>,
            imported: &mut FxHashSet<Cow<'a, str>>,
            body: &mut Body,
            out: &mut Vec<FastMethod>,
        ) {
            if is_function(t, tree, n) {
                let b = tree.last_child(n);
                if b == NONE || (!t.is_block[tree.ty(b) as usize] && !t.is_arrow[tree.ty(n) as usize]) {
                    return;
                }
                names_under(t, src, tree, n, declared, imported);
                let simp = Simplifier {
                    src,
                    tree,
                    cfg: &t.cfg,
                    is_local: |node| {
                        !declared.is_empty()
                            && t.is_var[tree.ty(node) as usize]
                            && declared.contains(&node_text(src, tree, node))
                    },
                };
                if let Some(root) = simp.run(body, b) {
                    let name = name_of(t, src, tree, n);
                    let (start_line, end_line) = line_span(src, tree.from(n), tree.to(n));
                    out.push(FastMethod { name, start_line, end_line, body_root: root });
                }
                return;
            }
            let mut c = tree.first_child(n);
            while c != NONE {
                walk(t, src, tree, c, declared, imported, body, out);
                c = tree.next_sibling(c);
            }
        }

        walk(t, &src, &tree, 0, &mut declared, &mut imported, body, out);
        if tree.nodes.capacity() > 32768 {
            tree.nodes.clear();
            tree.nodes.shrink_to(8192);
        }
    });
}

pub(crate) fn parse_snippet_into(
    source: &str,
    options: &SnippetOptions,
    body: &mut Body,
) -> Option<u32> {
    let t = tables();
    let src = Src::new(source);
    TL_TS_TREE.with(|cell| {
        let mut tree = cell.borrow_mut();
        src.parse_into(grammar_for(options.path.as_deref()), &mut tree);
        let mut declared: FxHashSet<Cow<'_, str>> = FxHashSet::default();
        let mut imported: FxHashSet<Cow<'_, str>> = FxHashSet::default();
        names_under(t, &src, &tree, 0, &mut declared, &mut imported);
        let globals = options.globals.as_ref();
        let simp = Simplifier {
            src: &src,
            tree: &tree,
            cfg: &t.cfg,
            is_local: |node| {
                if !t.is_var[tree.ty(node) as usize] {
                    return false;
                }
                let word = node_text(&src, &tree, node);
                if declared.contains(&word) {
                    return true;
                }
                !(imported.contains(&word) || looks_global(&word, globals) || is_callee(t, &tree, node))
            },
        };
        simp.run(body, 0)
    })
}

pub struct TypeScriptLanguage;

impl Language for TypeScriptLanguage {
    fn id(&self) -> &str {
        "typescript"
    }

    fn extensions(&self) -> &[&str] {
        &[".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"]
    }

    fn member_access_labels(&self) -> &[&str] {
        &["#.#", "#?.#"]
    }

    fn parse_methods(&self, source: &str, path: &str) -> Vec<ParsedMethod> {
        let mut body = Body::default();
        let mut raw = Vec::new();
        parse_methods_into(source, path, &mut body, &mut raw);
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
        let root = parse_snippet_into(source, options, &mut body)?;
        Some(body.to_node(source.as_bytes(), root))
    }

    fn builtin_kind(&self) -> u8 {
        2
    }
}
