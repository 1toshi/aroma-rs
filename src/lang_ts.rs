//! TypeScript / JavaScript / JSX / TSX front end on oxc (`oxc_parser` + `oxc_semantic`).
//!
//! A simplified parse tree (Section 3.1) has one subtree per AST node. Its
//! children are the node's AST children, in source order, and the keyword and
//! punctuation tokens of the source text between them; names and literals are
//! leaves. Gap text is split into words and punctuators, so formatting does not
//! change the tree.
//!
//! Local variables come from oxc's scope analysis: a name is local when it
//! resolves to a value binding, other than an import, declared inside the
//! method. For a method split out of a large function (see
//! [`DEFAULT_SPLIT_LINES`]), bindings of the enclosing function, such as a
//! factory's `db` parameter, are not local, since every call of the factory
//! shares them. A query snippet has no method around it, so a name it never
//! declares is guessed to be a local variable of the code above it unless it
//! is imported, called, capitalized, or listed in `SnippetOptions::globals`.

use std::cell::RefCell;
use std::collections::HashSet;

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    ArrowFunctionBody, ArrowFunctionExpression, BindingIdentifier, Function, FunctionBody,
    FunctionType, IdentifierReference, Program, PropertyKey, PropertyKind, TSTemplateLiteralType,
    TemplateLiteral,
};
use oxc_ast::{AstKind, AstType};
use oxc_ast_visit::{walk, Visit};
use oxc_parser::{ParseOptions, Parser};
use oxc_semantic::{Scoping, SemanticBuilder};
use oxc_span::{GetSpan, SourceType, Span};
use oxc_syntax::scope::{ScopeFlags, ScopeId};
use oxc_syntax::symbol::{SymbolFlags, SymbolId};

use crate::lang_py::{collect_parsed_methods, FastMethod};
use crate::simple::{Body, KW, NAME};
use crate::text::{is_ascii_ws, js_ws_char};
use crate::{Language, Node, ParsedMethod, SnippetOptions};

/// Functions at least this many lines long that define named functions are
/// split into one method per named function plus the rest of their body.
pub const DEFAULT_SPLIT_LINES: u32 = 80;

/// Punctuators that are one token in gap text, longest first.
const PUNCTUATORS: &[&str] = &[
    ">>>=", "...", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=", "??=", "=>", "==", "!=", "<=", ">=",
    "&&", "||", "??", "?.", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>", "${",
    "</", "/>",
];

thread_local! {
    static TL_ALLOC: RefCell<Allocator> = RefCell::new(Allocator::default());
}

pub(crate) fn shrink_tl_tree() {
    TL_ALLOC.with(|cell| *cell.borrow_mut() = Allocator::default());
}

fn source_type_for(path: Option<&str>) -> SourceType {
    match path.and_then(|p| p.rsplit_once('.')).map(|(_, ext)| ext) {
        Some("tsx") => SourceType::tsx(),
        Some("js" | "jsx" | "mjs" | "cjs") => SourceType::jsx(),
        _ => SourceType::ts(),
    }
}

/// Byte offsets of line starts.
struct Lines(Vec<u32>);

impl Lines {
    fn new(source: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(memchr::memchr_iter(b'\n', source.as_bytes()).map(|i| i as u32 + 1));
        Lines(starts)
    }

    /// 1-based line of a byte offset.
    fn line_of(&self, offset: u32) -> u32 {
        self.0.partition_point(|&s| s <= offset) as u32
    }

    /// Lines a span covers, not counting whitespace before its end.
    fn span(&self, source: &[u8], span: Span) -> (u32, u32) {
        let mut end = span.end;
        while end > span.start + 1 && is_ascii_ws(source[end as usize - 1]) {
            end -= 1;
        }
        (self.line_of(span.start), self.line_of(end.saturating_sub(1)))
    }
}

/// Nodes a simplified tree keeps as one name or literal.
fn is_leaf(ty: AstType) -> bool {
    matches!(
        ty,
        AstType::IdentifierName
            | AstType::IdentifierReference
            | AstType::BindingIdentifier
            | AstType::LabelIdentifier
            | AstType::PrivateIdentifier
            | AstType::StringLiteral
            | AstType::NumericLiteral
            | AstType::BigIntLiteral
            | AstType::RegExpLiteral
            | AstType::BooleanLiteral
            | AstType::NullLiteral
            | AstType::TemplateElement
            | AstType::JSXIdentifier
            | AstType::JSXText
    )
}

fn is_function_node(ty: AstType) -> bool {
    matches!(ty, AstType::Function | AstType::ArrowFunctionExpression)
}

/// An open node of the tree being built.
struct Frame {
    /// Scratch position of the node's first child.
    mark: usize,
    /// End of the source text the node's children have covered so far.
    at: u32,
    /// Whether an AST child has been added, as opposed to gap tokens only.
    has_child: bool,
}

/// Builds the simplified tree of the AST node a walk starts at.
struct Builder<'s, 'b> {
    source: &'s str,
    /// Where the input ends in `source`: before the closers a snippet was parsed with.
    limit: u32,
    lines: &'s Lines,
    /// Comment spans in source order.
    comments: &'s [Span],
    scoping: &'s Scoping,
    /// Whether a binding declared in a scope is local to the method.
    local_scope: &'s dyn Fn(ScopeId) -> bool,
    /// Functions left out of this method, because they are methods of their own.
    skip: &'s [Span],
    /// For a snippet: guess unresolved names, keeping these as globals.
    snippet: Option<Option<&'s HashSet<String>>>,
    callees: Vec<Span>,
    body: &'b mut Body,
    frames: Vec<Frame>,
    /// The last leaf emitted, so that a shorthand `{ a }` is one token.
    last_leaf: Option<(Span, u32)>,
    /// Depth inside a subtree whose descendants are not visited.
    muted: u32,
}

impl Builder<'_, '_> {
    fn token(&mut self, from: u32, to: u32, keyword: bool, local: bool) -> u32 {
        let line = self.lines.line_of(from);
        let end_line = self.lines.line_of(from.max(to.saturating_sub(1)));
        let id = self.body.push_tok(if keyword { KW } else { NAME }, local, line, end_line, from, to);
        self.body.push_child(id);
        id
    }

    /// Emits the keyword and punctuation tokens of `from..to`, skipping comments.
    fn gap(&mut self, from: u32, to: u32) {
        let (mut from, to) = (from, to.min(self.limit));
        let mut ci = self.comments.partition_point(|c| c.end <= from);
        while from < to {
            match self.comments.get(ci) {
                Some(c) if c.start < to => {
                    self.words(from, c.start.max(from));
                    from = c.end.max(from);
                    ci += 1;
                }
                _ => {
                    self.words(from, to);
                    break;
                }
            }
        }
    }

    fn words(&mut self, from: u32, to: u32) {
        let source = self.source;
        let bytes = source.as_bytes();
        let end = to as usize;
        let mut i = from as usize;
        while i < end {
            let b = bytes[i];
            if b < 0x80 {
                if is_ascii_ws(b) {
                    i += 1;
                    continue;
                }
            } else if source[i..].chars().next().is_some_and(js_ws_char) {
                i += char_len(b);
                continue;
            }
            let start = i;
            if is_word_byte(b) {
                while i < end && is_word_byte(bytes[i]) && !(bytes[i] >= 0x80 && is_unicode_space(source, i)) {
                    i += char_len(bytes[i]);
                }
            } else {
                let rest = &bytes[i..end];
                i += PUNCTUATORS.iter().find(|p| rest.starts_with(p.as_bytes())).map_or(1, |p| p.len());
                if b == b'\\' && i - start == 1 {
                    continue;
                }
            }
            self.token(start as u32, i as u32, true, false);
        }
    }

    fn is_local_symbol(&self, symbol: SymbolId) -> bool {
        let flags = self.scoping.symbol_flags(symbol);
        !flags.is_import()
            && (flags.is_value() || flags.contains(SymbolFlags::CatchVariable))
            && (self.local_scope)(self.scoping.symbol_scope_id(symbol))
    }

    fn reference_is_local(&self, r: &IdentifierReference<'_>) -> bool {
        let symbol = r.reference_id.get().and_then(|id| self.scoping.get_reference(id).symbol_id());
        match (symbol, self.snippet) {
            (Some(s), _) => self.is_local_symbol(s),
            (None, Some(globals)) => {
                let name = r.name.as_str();
                !(name.starts_with(|c: char| c.is_ascii_uppercase())
                    || globals.is_some_and(|g| g.contains(name))
                    || self.callees.contains(&r.span))
            }
            (None, None) => false,
        }
    }

    fn binding_is_local(&self, b: &BindingIdentifier<'_>) -> bool {
        b.symbol_id.get().is_some_and(|s| self.is_local_symbol(s))
    }

    fn leaf(&mut self, kind: &AstKind<'_>) {
        let span = kind.span();
        let (from, to) = trim(self.source, span.start, span.end.min(self.limit));
        if from >= to {
            return;
        }
        let local = match kind {
            AstKind::IdentifierReference(r) => self.reference_is_local(r),
            AstKind::BindingIdentifier(b) => self.binding_is_local(b),
            _ => false,
        };
        // A shorthand property or specifier, `{ a }`, is a key and a value with one span.
        if let Some((last, id)) = self.last_leaf
            && last == span
        {
            let item = &mut self.body.items[id as usize];
            item.local |= local;
            return;
        }
        let id = self.token(from, to, false, local);
        self.last_leaf = Some((span, id));
    }

    /// Emits the gap before a child that starts at `start`.
    fn before_child(&mut self, start: u32) {
        if let Some(top) = self.frames.last() {
            let at = top.at;
            if at < start {
                self.gap(at, start);
            }
        }
    }

    /// Advances the open node past a child that ends at `end`.
    fn after_child(&mut self, end: u32) {
        if let Some(top) = self.frames.last_mut() {
            top.at = top.at.max(end);
            top.has_child = true;
        }
    }

    fn push_frame(&mut self, at: u32) {
        let mark = self.body.mark();
        self.frames.push(Frame { mark, at, has_child: false });
    }

    /// Closes the open node and returns whether it produced anything.
    fn pop_frame(&mut self, end: u32) -> Option<u32> {
        let at = self.frames.last().map_or(end, |f| f.at);
        if at < end {
            self.gap(at, end);
        }
        let frame = self.frames.pop()?;
        // A node of keywords only, such as `this`, is that keyword.
        if !frame.has_child && self.body.mark() == frame.mark + 1 {
            return None;
        }
        self.body.close_node(frame.mark)
    }
}

impl<'a> Visit<'a> for Builder<'_, '_> {
    fn enter_node(&mut self, kind: AstKind<'a>) {
        if self.muted > 0 {
            self.muted += 1;
            return;
        }
        let span = kind.span();
        let ty = kind.ty();
        if span.start == span.end || span.start >= self.limit {
            // Synthesized by the parser, or added by snippet repair.
            self.muted = 1;
            return;
        }
        if is_function_node(ty) && self.skip.contains(&span) {
            self.before_child(span.start);
            self.after_child(span.end);
            self.muted = 1;
            return;
        }
        if self.snippet.is_some() {
            match &kind {
                AstKind::CallExpression(c) => self.callees.push(c.callee.span()),
                AstKind::NewExpression(n) => self.callees.push(n.callee.span()),
                _ => {}
            }
        }
        self.before_child(span.start);
        if is_leaf(ty) {
            self.leaf(&kind);
            self.after_child(span.end);
            self.muted = 1;
            return;
        }
        self.push_frame(span.start);
    }

    fn leave_node(&mut self, kind: AstKind<'a>) {
        if self.muted > 0 {
            self.muted -= 1;
            return;
        }
        let end = kind.span().end;
        if let Some(id) = self.pop_frame(end) {
            self.body.push_child(id);
        }
        self.after_child(end);
    }

    // The default walks visit every quasi before every expression; source order is needed.
    fn visit_template_literal(&mut self, it: &TemplateLiteral<'a>) {
        let kind = AstKind::TemplateLiteral(self.alloc(it));
        self.enter_node(kind);
        for (i, quasi) in it.quasis.iter().enumerate() {
            self.visit_template_element(quasi);
            if let Some(expression) = it.expressions.get(i) {
                self.visit_expression(expression);
            }
        }
        self.leave_node(kind);
    }

    fn visit_ts_template_literal_type(&mut self, it: &TSTemplateLiteralType<'a>) {
        let kind = AstKind::TSTemplateLiteralType(self.alloc(it));
        self.enter_node(kind);
        for (i, quasi) in it.quasis.iter().enumerate() {
            self.visit_template_element(quasi);
            if let Some(ty) = it.types.get(i) {
                self.visit_ts_type(ty);
            }
        }
        self.leave_node(kind);
    }
}

#[inline]
fn char_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$' || b >= 0x80
}

fn is_unicode_space(source: &str, i: usize) -> bool {
    source[i..].chars().next().is_some_and(js_ws_char)
}

fn trim(source: &str, mut from: u32, mut to: u32) -> (u32, u32) {
    let bytes = source.as_bytes();
    while from < to && is_ascii_ws(bytes[from as usize]) {
        from += 1;
    }
    while to > from && is_ascii_ws(bytes[to as usize - 1]) {
        to -= 1;
    }
    (from, to)
}

/// A method's body: a block, or the expression an arrow function returns.
#[derive(Clone, Copy)]
enum MethodBody<'a> {
    Block(&'a FunctionBody<'a>),
    Arrow(&'a ArrowFunctionBody<'a>),
}

/// A function with a body that the finder saw.
struct FoundFn<'a> {
    /// The function node, which a method split out of it skips.
    span: Span,
    /// What its lines are reported as: the method definition for `get() {}`.
    lines: Span,
    body: MethodBody<'a>,
    scope: ScopeId,
    /// Names of the declarations and properties around it, outermost first.
    name: Vec<Span>,
    /// Bound to a name of its own, as opposed to an anonymous callback.
    named: bool,
    /// The function it is directly inside.
    parent: Option<usize>,
}

/// Collects every function with a body, its enclosing function, and its name.
struct Finder<'s, 'a> {
    source: &'s str,
    found: Vec<FoundFn<'a>>,
    /// Open functions, as indices into `found`.
    open: Vec<usize>,
    /// Open nodes: type, span, and whether it is a method shorthand or accessor.
    parents: Vec<(AstType, Span, bool)>,
    /// Names of open declarations and properties, with the depth of their node.
    names: Vec<(Span, usize)>,
}

impl<'a> Finder<'_, 'a> {
    /// The nearest open node that is not a parenthesis or a type assertion.
    fn binder(&self) -> Option<(AstType, Span, bool)> {
        self.parents.iter().rev().copied().find(|(ty, _, _)| {
            !matches!(
                ty,
                AstType::ParenthesizedExpression
                    | AstType::TSAsExpression
                    | AstType::TSSatisfiesExpression
                    | AstType::TSNonNullExpression
                    | AstType::TSTypeAssertion
            )
        })
    }

    fn add(&mut self, span: Span, body: MethodBody<'a>, scope: Option<ScopeId>, own_name: Option<Span>) {
        let Some(scope) = scope else { return };
        let binder = self.binder();
        let bound = binder.is_some_and(|(ty, _, _)| {
            matches!(
                ty,
                AstType::VariableDeclarator
                    | AstType::ObjectProperty
                    | AstType::MethodDefinition
                    | AstType::PropertyDefinition
            )
        });
        let lines = match binder {
            Some((AstType::MethodDefinition, s, _) | (AstType::ObjectProperty, s, true)) => s,
            _ => span,
        };
        let mut name: Vec<Span> = self.names.iter().map(|(s, _)| *s).collect();
        name.extend(own_name);
        self.found.push(FoundFn {
            span,
            lines,
            body,
            scope,
            name,
            named: own_name.is_some() || bound,
            parent: self.open.last().copied(),
        });
    }

    fn name_text(&self, spans: &[Span]) -> String {
        if spans.is_empty() {
            return "<anonymous>".to_string();
        }
        let mut out = String::new();
        for (i, s) in spans.iter().enumerate() {
            if i > 0 {
                out.push('.');
            }
            out.push_str(&self.source[s.start as usize..s.end as usize]);
        }
        out
    }
}

impl<'a> Visit<'a> for Finder<'_, 'a> {
    fn enter_node(&mut self, kind: AstKind<'a>) {
        // A computed key, `[k]: v`, has no static name; `static_name` returns `None` for it.
        let key_name = |key: &PropertyKey<'_>| key.static_name().map(|_| key.span());
        let name = match &kind {
            AstKind::VariableDeclarator(d) => d.id.get_binding_identifier().map(|b| b.span),
            AstKind::Class(c) => c.id.as_ref().map(|b| b.span),
            AstKind::ObjectProperty(p) => key_name(&p.key),
            AstKind::MethodDefinition(m) => key_name(&m.key),
            AstKind::PropertyDefinition(p) => key_name(&p.key),
            _ => None,
        };
        let method = matches!(&kind, AstKind::ObjectProperty(p) if p.method || p.kind != PropertyKind::Init);
        self.parents.push((kind.ty(), kind.span(), method));
        if let Some(span) = name {
            self.names.push((span, self.parents.len()));
        }
    }

    fn leave_node(&mut self, _kind: AstKind<'a>) {
        let depth = self.parents.len();
        while self.names.last().is_some_and(|(_, d)| *d == depth) {
            self.names.pop();
        }
        self.parents.pop();
    }

    fn visit_function(&mut self, it: &Function<'a>, flags: ScopeFlags) {
        let it: &'a Function<'a> = self.alloc(it);
        let declared = matches!(it.r#type, FunctionType::TSDeclareFunction | FunctionType::TSEmptyBodyFunctionExpression);
        let Some(body) = it.body.as_deref().filter(|_| !declared) else {
            walk::walk_function(self, it, flags);
            return;
        };
        let own = it.id.as_ref().map(|b| b.span);
        let before = self.found.len();
        self.add(it.span, MethodBody::Block(body), it.scope_id.get(), own);
        let added = self.found.len() > before;
        if added {
            self.open.push(before);
        }
        // Inside the function, its own name is part of every nested function's name.
        if let Some(span) = own {
            self.names.push((span, self.parents.len() + 1));
        }
        walk::walk_function(self, it, flags);
        if added {
            self.open.pop();
        }
    }

    fn visit_arrow_function_expression(&mut self, it: &ArrowFunctionExpression<'a>) {
        let it: &'a ArrowFunctionExpression<'a> = self.alloc(it);
        let before = self.found.len();
        self.add(it.span, MethodBody::Arrow(&it.body), it.scope_id.get(), None);
        let added = self.found.len() > before;
        if added {
            self.open.push(before);
        }
        walk::walk_arrow_function_expression(self, it);
        if added {
            self.open.pop();
        }
    }
}

/// Parses `source` and hands its program, scoping, and sorted comment spans to `f`.
fn with_program<R>(
    source: &str,
    path: Option<&str>,
    options: ParseOptions,
    f: impl FnOnce(&Program<'_>, &Scoping, &[Span], bool) -> R,
) -> R {
    TL_ALLOC.with(|cell| {
        let mut alloc = cell.borrow_mut();
        alloc.reset();
        let parsed = Parser::new(&alloc, source, source_type_for(path)).with_options(options).parse();
        let fatal = parsed.fatal_error;
        let program = parsed.program;
        let scoping = SemanticBuilder::new().build(&program).semantic.into_scoping();
        let mut comments: Vec<Span> = program.comments.iter().map(|c| c.span).collect();
        comments.sort_unstable_by_key(|s| s.start);
        f(&program, &scoping, &comments, fatal)
    })
}

/// The tree a visit of the node at `start` builds, as the root id. The node
/// becomes the one child of a frame that `close_node` then collapses to it.
fn build<'s, 'b>(builder: &mut Builder<'s, 'b>, start: u32, visit: impl FnOnce(&mut Builder<'s, 'b>)) -> Option<u32> {
    builder.push_frame(start);
    visit(builder);
    let frame = builder.frames.pop()?;
    builder.body.close_node(frame.mark)
}

pub(crate) fn parse_methods_into(
    source: &str,
    path: &str,
    line_range: Option<(u32, u32)>,
    split_lines: u32,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    let lines = Lines::new(source);
    let bytes = source.as_bytes();
    with_program(source, Some(path), ParseOptions::default(), |program, scoping, comments, _| {
        let mut finder = Finder { source, found: Vec::new(), open: Vec::new(), parents: Vec::new(), names: Vec::new() };
        finder.visit_program(program);
        let found = &finder.found;

        // Methods are the outermost functions and, in a method of at least
        // `split_lines` lines, the named functions directly inside it.
        let mut is_method = vec![false; found.len()];
        let mut split = vec![false; found.len()];
        for (i, f) in found.iter().enumerate() {
            is_method[i] = f.parent.is_none_or(|p| split[p] && f.named);
            if is_method[i] {
                let (start, end) = lines.span(bytes, f.lines);
                split[i] = end + 1 - start >= split_lines;
            }
        }

        // The methods split out of each method, so it can skip them.
        let mut inner: Vec<Vec<usize>> = vec![Vec::new(); found.len()];
        for (j, f) in found.iter().enumerate() {
            if let Some(p) = f.parent
                && is_method[j]
            {
                inner[p].push(j);
            }
        }

        for (i, f) in found.iter().enumerate() {
            if !is_method[i] {
                continue;
            }
            let (start_line, end_line) = lines.span(bytes, f.lines);
            if let Some((min_ln, max_ln)) = line_range
                && (end_line < min_ln || start_line > max_ln)
            {
                continue;
            }
            let skip: Vec<Span> = inner[i].iter().map(|&j| found[j].span).collect();
            let inner_scopes: Vec<ScopeId> = inner[i].iter().map(|&j| found[j].scope).collect();
            let own = f.scope;
            let local_scope = |mut scope: ScopeId| loop {
                if scope == own {
                    return true;
                }
                if inner_scopes.contains(&scope) {
                    return false;
                }
                match scoping.scope_parent_id(scope) {
                    Some(parent) => scope = parent,
                    None => return false,
                }
            };
            let mut builder = Builder {
                source,
                limit: source.len() as u32,
                lines: &lines,
                comments,
                scoping,
                local_scope: &local_scope,
                skip: &skip,
                snippet: None,
                callees: Vec::new(),
                body,
                frames: Vec::new(),
                last_leaf: None,
                muted: 0,
            };
            let root = match f.body {
                MethodBody::Block(block) => build(&mut builder, block.span.start, |b| b.visit_function_body(block)),
                MethodBody::Arrow(arrow) => {
                    build(&mut builder, arrow.span().start, |b| b.visit_arrow_function_body(arrow))
                }
            };
            if let Some(root) = root {
                out.push(FastMethod { name: finder.name_text(&f.name), start_line, end_line, body_root: root });
            }
        }
    });
}

/// Closing brackets, and closing tags for JSX elements left open, that make an
/// unfinished snippet, such as the first lines of an `if` block, parse.
fn closers(source: &str, jsx: bool) -> String {
    let mut stack: Vec<Vec<u8>> = Vec::new();
    let mut quote: Option<u8> = None;
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        match quote {
            Some(q) => {
                if b == b'\\' {
                    i += 1;
                } else if b == q || (b == b'\n' && q != b'`') {
                    quote = None;
                }
            }
            None => match b {
                b'"' | b'\'' | b'`' => quote = Some(b),
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if bytes.get(i + 1) == Some(&b'*') => {
                    match source[i + 2..].find("*/") {
                        Some(end) => i += end + 3,
                        None => return format!("*/{}", closers(&source[..i], jsx)),
                    }
                }
                b'(' => stack.push(b")".to_vec()),
                b'[' => stack.push(b"]".to_vec()),
                b'{' => stack.push(b"}".to_vec()),
                b')' | b']' | b'}' => {
                    stack.pop();
                }
                b'<' if jsx && bytes.get(i + 1).is_some_and(|c| c.is_ascii_alphabetic()) => {
                    let name_end = bytes[i + 1..]
                        .iter()
                        .position(|c| !(c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-')))
                        .map_or(bytes.len(), |p| i + 1 + p);
                    let name = &bytes[i + 1..name_end];
                    // Self-closing `<X ... />` closes itself; others need `</X>`.
                    let close = source[name_end..].find(['>']).map(|p| name_end + p);
                    let self_closing = close.is_some_and(|p| p > 0 && bytes[p - 1] == b'/');
                    if !self_closing {
                        let mut tag = b"</".to_vec();
                        tag.extend_from_slice(name);
                        tag.push(b'>');
                        if close.is_none() {
                            tag.splice(0..0, b">".iter().copied());
                        }
                        stack.push(tag);
                    }
                    i = name_end;
                    continue;
                }
                b'<' if jsx && bytes.get(i + 1) == Some(&b'/') => {
                    stack.pop();
                }
                _ => {}
            },
        }
        i += 1;
    }
    let mut out = String::from("\n");
    for closer in stack.iter().rev() {
        out.push_str(std::str::from_utf8(closer).unwrap_or(""));
    }
    out
}

/// A query snippet, parsed as-is or with closing brackets added. A snippet no
/// repair makes parse falls back to [`lex_snippet`]. Wrapping it in a
/// function, an object, or `<>` also makes signature and JSX fragments parse,
/// but on our corpus those trees ranked the snippet's own method worse than
/// the flat fallback did (Section 5, first 5 body lines parsed alone: 91.3%
/// Recall@1 for parsed snippets, and fewer parsed with every wrapping tried).
pub(crate) fn parse_snippet_into(source: &str, options: &SnippetOptions, body: &mut Body) -> Option<u32> {
    let parse_options = ParseOptions { allow_return_outside_function: true, ..ParseOptions::default() };
    let path = options.path.as_deref();
    let globals = options.globals.as_ref();
    let jsx = source_type_for(path).is_jsx();
    // Only the snippet's own bytes become tokens; the closers only let it parse.
    let limit = source.len() as u32;
    for suffix in [String::new(), closers(source, jsx)] {
        let text = format!("{source}{suffix}");
        let lines = Lines::new(&text);
        let root = with_program(&text, path, parse_options, |program, scoping, comments, fatal| {
            if fatal || program.body.is_empty() {
                return None;
            }
            let all = |_: ScopeId| true;
            let mut builder = Builder {
                source: &text,
                limit,
                lines: &lines,
                comments,
                scoping,
                local_scope: &all,
                skip: &[],
                snippet: Some(globals),
                callees: Vec::new(),
                body: &mut *body,
                frames: Vec::new(),
                last_leaf: None,
                muted: 0,
            };
            build(&mut builder, 0, |b| b.visit_program(program))
        });
        if root.is_some() {
            return root;
        }
    }
    lex_snippet(source, globals, body)
}

/// Reserved words and contextual keywords, which a flat snippet keeps as keyword tokens.
const KEYWORDS: &[&str] = &[
    "abstract", "as", "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger",
    "declare", "default", "delete", "do", "else", "enum", "export", "extends", "finally", "for", "from",
    "function", "get", "if", "implements", "import", "in", "instanceof", "interface", "keyof", "let", "new",
    "of", "private", "protected", "public", "readonly", "return", "satisfies", "set", "static", "super",
    "switch", "this", "throw", "try", "type", "typeof", "var", "void", "while", "with", "yield",
];

/// A snippet no repair makes parse, such as lines cut from the middle of a
/// signature, as one flat tree of its tokens. Like an error-recovering parse,
/// it still has token and sibling features to search with. Names guess local
/// as in a parsed snippet: not after `.`, not called, not capitalized, not global.
fn lex_snippet(source: &str, globals: Option<&HashSet<String>>, body: &mut Body) -> Option<u32> {
    let lines = Lines::new(source);
    let bytes = source.as_bytes();
    let mark = body.mark();
    let mut names = 0usize;
    let mut push = |body: &mut Body, from: usize, to: usize, keyword: bool, local: bool| {
        let (line, end_line) = (lines.line_of(from as u32), lines.line_of((to - 1).max(from) as u32));
        let id = body.push_tok(if keyword { KW } else { NAME }, local, line, end_line, from as u32, to as u32);
        body.push_child(id);
        names += usize::from(!keyword);
    };
    let mut i = 0;
    let mut after_dot = false;
    while i < bytes.len() {
        let b = bytes[i];
        let start = i;
        if b < 0x80 && is_ascii_ws(b) || b >= 0x80 && is_unicode_space(source, i) {
            i += char_len(b);
            continue;
        }
        if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
            i = source[i..].find('\n').map_or(bytes.len(), |p| i + p);
            continue;
        }
        if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            i = source[i + 2..].find("*/").map_or(bytes.len(), |p| i + p + 4);
            continue;
        }
        if matches!(b, b'"' | b'\'' | b'`') {
            i += 1;
            while i < bytes.len() && bytes[i] != b {
                i += if bytes[i] == b'\\' { 2 } else { 1 };
            }
            i = (i + 1).min(bytes.len());
            push(body, start, i, false, false);
            after_dot = false;
            continue;
        }
        if b.is_ascii_digit() {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'.' | b'_')) {
                i += 1;
            }
            push(body, start, i, false, false);
            after_dot = false;
            continue;
        }
        if is_word_byte(b) {
            while i < bytes.len() && is_word_byte(bytes[i]) && !(bytes[i] >= 0x80 && is_unicode_space(source, i)) {
                i += char_len(bytes[i]);
            }
            let word = &source[start..i];
            if KEYWORDS.contains(&word) && !after_dot {
                push(body, start, i, true, false);
            } else {
                let called = source[i..].trim_start().starts_with('(');
                let local = !after_dot
                    && !called
                    && !matches!(word, "true" | "false" | "null" | "undefined")
                    && !word.starts_with(|c: char| c.is_ascii_uppercase())
                    && !globals.is_some_and(|g| g.contains(word));
                push(body, start, i, false, local);
            }
            after_dot = false;
            continue;
        }
        let rest = &bytes[i..];
        i += PUNCTUATORS.iter().find(|p| rest.starts_with(p.as_bytes())).map_or(1, |p| p.len());
        after_dot = matches!(&source[start..i], "." | "?.");
        push(body, start, i, true, false);
    }
    // Keywords alone say nothing about a snippet.
    let root = body.close_node(mark);
    if names == 0 { None } else { root }
}

/// TypeScript and JavaScript, with JSX in `.tsx`, `.jsx`, and `.js` files.
pub struct TypeScriptLanguage {
    /// See [`DEFAULT_SPLIT_LINES`]; `u32::MAX` keeps every outermost function whole.
    pub split_lines: u32,
}

impl Default for TypeScriptLanguage {
    fn default() -> Self {
        Self { split_lines: DEFAULT_SPLIT_LINES }
    }
}

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
            parse_methods_into(source, path, None, self.split_lines, body, raw);
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

    fn split_lines(&self) -> u32 {
        self.split_lines
    }
}
