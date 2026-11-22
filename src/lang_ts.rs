//! TypeScript / JavaScript / JSX / TSX language front-end, matching
//! `ref-ts/src/languages/typescript.ts`.
//! Shares `TL_FRONT` scratch with Python front-end, uses packed single-byte
//! `kind` + `named_kind` tables, and zero-alloc `NameSet`.

use std::sync::OnceLock;

use crate::lang_py::{collect_parsed_methods, simplify_lezer_snippet, FastMethod, FrontScratch, TL_FRONT};
use crate::lezer::{self, Grammar, SyntaxTree, NONE};
use crate::simple::{Body, Cfg, Simplifier};
use crate::text::{NameSet, Src};
use crate::{Language, Node, ParsedMethod, SnippetOptions};

pub(crate) fn shrink_tl_tree() {
    crate::lang_py::shrink_tl_tree();
}

// Single-byte node kinds (0 = irrelevant)
const K_NONE: u8 = 0;
const K_VAR_NAME: u8 = 1;
const K_VAR_DEF: u8 = 2;
const K_PROP_DEF: u8 = 3;
const K_PRIV_PROP_DEF: u8 = 4;
const K_FUNC_NON_ARROW: u8 = 5; // FunctionDeclaration, FunctionExpression, MethodDeclaration
const K_ARROW: u8 = 6;
const K_PROPERTY: u8 = 7;
const K_PARAM_LIST: u8 = 8;
const K_BLOCK: u8 = 9;
const K_IMPORT_DECL: u8 = 10;
const K_CALL_EXPR: u8 = 11;
const K_NEW_EXPR: u8 = 12;
const K_KW_NEW: u8 = 13;

const NAMED_NONE: u8 = 0;
const NAMED_VAR_DEF: u8 = 1;
const NAMED_PROP_OR_PRIV: u8 = 2;
const NAMED_PROP_ONLY: u8 = 3;

struct TsTables {
    cfg: Cfg,
    kind: Vec<u8>,
    named_kind: Vec<u8>,
}

fn tables() -> &'static TsTables {
    static T: OnceLock<TsTables> = OnceLock::new();
    T.get_or_init(|| {
        let nn = lezer::TS.node_names;
        let kind = nn
            .iter()
            .map(|&n| match n {
                "VariableName" => K_VAR_NAME,
                "VariableDefinition" => K_VAR_DEF,
                "PropertyDefinition" => K_PROP_DEF,
                "PrivatePropertyDefinition" => K_PRIV_PROP_DEF,
                "FunctionDeclaration" | "FunctionExpression" | "MethodDeclaration" => K_FUNC_NON_ARROW,
                "ArrowFunction" => K_ARROW,
                "Property" => K_PROPERTY,
                "ParamList" => K_PARAM_LIST,
                "Block" => K_BLOCK,
                "ImportDeclaration" => K_IMPORT_DECL,
                "CallExpression" => K_CALL_EXPR,
                "NewExpression" => K_NEW_EXPR,
                "new" => K_KW_NEW,
                _ => K_NONE,
            })
            .collect();
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
            kind,
            named_kind,
        }
    })
}

#[inline(always)]
fn kind_of(t: &TsTables, tree: &SyntaxTree, n: u32) -> u8 {
    unsafe { *t.kind.get_unchecked(tree.ty(n) as usize) }
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

#[inline(always)]
fn is_function(t: &TsTables, tree: &SyntaxTree, n: u32) -> (bool, bool) {
    let k = kind_of(t, tree, n);
    if k == K_FUNC_NON_ARROW {
        return (true, false);
    }
    if k == K_ARROW {
        return (true, true);
    }
    if k == K_PROPERTY {
        let mut c = tree.first_child(n);
        while c != NONE {
            if kind_of(t, tree, c) == K_PARAM_LIST {
                return (true, false);
            }
            c = tree.next_sibling(c);
        }
    }
    (false, false)
}

#[inline]
fn is_callee(t: &TsTables, tree: &SyntaxTree, n: u32) -> bool {
    let p = tree.parent(n);
    if p == NONE {
        return false;
    }
    match kind_of(t, tree, p) {
        K_CALL_EXPR => tree.first_child(p) == n,
        K_NEW_EXPR => {
            let mut prev = NONE;
            let mut c = tree.first_child(p);
            while c != NONE && c != n {
                prev = c;
                c = tree.next_sibling(c);
            }
            prev != NONE && kind_of(t, tree, prev) == K_KW_NEW
        }
        _ => false,
    }
}

fn names_under(
    t: &TsTables,
    src: &Src<'_>,
    tree: &SyntaxTree,
    root: u32,
    declared: &mut NameSet,
    imported: &mut NameSet,
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
        if kind_of(t, tree, cur) == K_VAR_DEF {
            let p = tree.parent(cur);
            let is_import = if p != NONE {
                if kind_of(t, tree, p) == K_IMPORT_DECL {
                    true
                } else {
                    let gp = tree.parent(p);
                    gp != NONE && kind_of(t, tree, gp) == K_IMPORT_DECL
                }
            } else {
                false
            };
            if is_import {
                imported.insert(src, tree.from(cur), tree.to(cur));
            } else {
                declared.insert(src, tree.from(cur), tree.to(cur));
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

fn name_of(
    t: &TsTables,
    src: &Src<'_>,
    tree: &SyntaxTree,
    node: u32,
    parts: &mut Vec<(u32, u32)>,
) -> String {
    parts.clear();
    let mut inner = NONE;
    let mut at = node;
    while at != NONE {
        let nk = unsafe { *t.named_kind.get_unchecked(tree.ty(at) as usize) };
        if nk != NAMED_NONE {
            let from = if inner == NONE { u32::MAX } else { tree.from(inner) };
            let mut last_match = NONE;
            let mut c = tree.first_child(at);
            while c != NONE {
                if tree.to(c) > from {
                    break;
                }
                let ck = kind_of(t, tree, c);
                let matches = match nk {
                    NAMED_VAR_DEF => ck == K_VAR_DEF,
                    NAMED_PROP_OR_PRIV => ck == K_PROP_DEF || ck == K_PRIV_PROP_DEF,
                    NAMED_PROP_ONLY => ck == K_PROP_DEF,
                    _ => false,
                };
                if matches {
                    last_match = c;
                }
                c = tree.next_sibling(c);
            }
            if last_match != NONE {
                parts.push((tree.from(last_match), tree.to(last_match)));
            }
        }
        inner = at;
        at = tree.parent(at);
    }
    if parts.is_empty() {
        return "<anonymous>".to_string();
    }
    if parts.len() == 1 {
        let p = parts[0];
        return match src.str_range(p.0, p.1) {
            Some(s) => s.to_string(),
            None => src.string(p.0, p.1),
        };
    }
    let mut out = String::new();
    for (i, &p) in parts.iter().rev().enumerate() {
        if i > 0 {
            out.push('.');
        }
        src.push_span(p, &mut out);
    }
    out
}

pub(crate) fn parse_methods_into(
    source: &str,
    path: &str,
    line_range: Option<(u32, u32)>,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    let t = tables();
    TL_FRONT.with(|cell| {
        let mut sc = cell.borrow_mut();
        let FrontScratch { tree, src_buf, set_a, set_b, classes } = &mut *sc;
        let src = Src::build(source, src_buf);
        src.parse_into(grammar_for(Some(path)), tree);

        #[allow(clippy::too_many_arguments)]
        fn walk(
            t: &TsTables,
            src: &Src<'_>,
            tree: &SyntaxTree,
            n: u32,
            line_range: Option<(u32, u32)>,
            parts: &mut Vec<(u32, u32)>,
            declared: &mut NameSet,
            imported: &mut NameSet,
            body: &mut Body,
            out: &mut Vec<FastMethod>,
        ) {
            let (is_fn, is_arrow) = is_function(t, tree, n);
            if is_fn {
                let b = tree.last_child(n);
                if b == NONE || (!is_arrow && kind_of(t, tree, b) != K_BLOCK) {
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
                names_under(t, src, tree, n, declared, imported);
                let simp = Simplifier {
                    src,
                    tree,
                    cfg: &t.cfg,
                    is_local: |node| {
                        if declared.is_empty() {
                            return false;
                        }
                        let k = kind_of(t, tree, node);
                        (k == K_VAR_NAME || k == K_VAR_DEF)
                            && declared.contains(src, tree.from(node), tree.to(node))
                    },
                };
                if let Some(root) = simp.run(body, b) {
                    let name = name_of(t, src, tree, n, parts);
                    let (sl, el) = if line_range.is_some() {
                        (start_line, end_line)
                    } else {
                        src.line_span(tree.from(n), tree.to(n))
                    };
                    out.push(FastMethod { name, start_line: sl, end_line: el, body_root: root });
                }
                return;
            }
            let mut c = tree.first_child(n);
            while c != NONE {
                walk(t, src, tree, c, line_range, parts, declared, imported, body, out);
                c = tree.next_sibling(c);
            }
        }

        walk(t, &src, tree, 0, line_range, classes, set_a, set_b, body, out);
    });
}

pub(crate) fn parse_snippet_into(
    source: &str,
    options: &SnippetOptions,
    body: &mut Body,
) -> Option<u32> {
    let t = tables();
    let globals = options.globals.as_ref();
    simplify_lezer_snippet(
        source,
        grammar_for(options.path.as_deref()),
        &t.cfg,
        body,
        |src, tree, declared, imported| names_under(t, src, tree, 0, declared, imported),
        |src, tree, declared, imported, node| {
            let k = kind_of(t, tree, node);
            if k != K_VAR_NAME && k != K_VAR_DEF {
                return false;
            }
            let (a, b) = (tree.from(node), tree.to(node));
            if declared.contains(src, a, b) {
                return true;
            }
            !(imported.contains(src, a, b)
                || src.looks_global(a, b, globals)
                || is_callee(t, tree, node))
        },
    )
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
        collect_parsed_methods(source, |body, raw| {
            parse_methods_into(source, path, None, body, raw);
        })
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
