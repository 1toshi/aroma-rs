//! Fast structural code search, clone detection, and code recommendation
//! implementing **Aroma: Code Recommendation via Structural Code Search**
//! (Luan, Yang, Barnaby, Sen, Chandra; OOPSLA 2019).
//!
//! `aroma` includes built-in front ends for:
//! - **Rust** (`.rs`): zero-dependency byte-slice lexer, Pass-1 local binder, and
//!   error-recovering recursive-descent/Pratt parser.
//! - **Python** (`.py`): Lezer LR/GLR front end.
//! - **TypeScript / JavaScript** (`.ts`, `.tsx`, `.mts`, `.cts`, `.js`, `.jsx`, `.mjs`, `.cjs`):
//!   Lezer LR/GLR front end.
//!
//! # Quick Example
//!
//! ```rust
//! use aroma::{
//!     build_index, queries_from_lines, query_from_snippet, recommend, search,
//!     IndexOptions, RecommendOptions, SearchOptions, SnippetOptions, SourceFile,
//! };
//!
//! let files = vec![
//!     SourceFile {
//!         path: "src/math.rs".into(),
//!         source: concat!(
//!             "pub fn sum_even(xs: &[i32]) -> i32 {\n",
//!             "    let mut total = 0;\n",
//!             "    for &x in xs {\n",
//!             "        if x % 2 == 0 {\n",
//!             "            total += x;\n",
//!             "        }\n",
//!             "    }\n",
//!             "    total\n",
//!             "}\n",
//!         ).into(),
//!     },
//!     SourceFile {
//!         path: "src/add.py".into(),
//!         source: "def add(x, y):\n    return x + y\n".into(),
//!     },
//! ];
//!
//! let index = build_index(&files, IndexOptions::default());
//! assert_eq!(index.stats().files, 2);
//!
//! // Alpha-renamed Rust snippet matches `sum_even` structurally.
//! let query = query_from_snippet(
//!     &index,
//!     "let mut acc = 0;\nfor &v in vals {\n    if v % 2 == 0 {\n        acc += v;\n    }\n}\nacc",
//!     &SnippetOptions { language: Some("rust".into()), ..Default::default() },
//! )
//! .expect("rust snippet parses");
//!
//! let hits = search(&index, &query, &SearchOptions::default());
//! assert!(!hits.is_empty());
//! assert_eq!(hits[0].method.name, "sum_even");
//!
//! let _recs = recommend(&index, &query, &RecommendOptions::default());
//! let line_queries = queries_from_lines(&index, &files[0], &[2, 3, 4]);
//! assert_eq!(line_queries.len(), 1);
//! ```

use std::collections::HashSet;

pub(crate) mod features;
pub(crate) mod index;
pub(crate) mod lang_py;
pub(crate) mod lang_rs;
pub(crate) mod lang_ts;
pub(crate) mod lezer;
pub(crate) mod prune;
pub(crate) mod recommend;
pub(crate) mod search;
pub(crate) mod simple;
pub(crate) mod text;

/// An in-memory source file passed to [`build_index`] or [`queries_from_lines`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub path: String,
    pub source: String,
}

/// Identifies a method or function indexed in an [`AromaIndex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodRef {
    pub path: String,
    pub name: String,
    pub language: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// A leaf token in a simplified Aroma parse tree ([`Node`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub keyword: bool,
    pub local: bool,
    pub line: u32,
    pub end_line: u32,
}

/// A simplified parse tree node (either a leaf [`Token`] or an interior subtree).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Token(Token),
    Tree(Vec<Node>),
}

/// A parsed method extracted by a [`Language`] front end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedMethod {
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub body: Node,
}

/// Options controlling how a query snippet is parsed by [`query_from_snippet`].
#[derive(Debug, Clone, Default)]
pub struct SnippetOptions {
    pub language: Option<String>,
    pub path: Option<String>,
    pub globals: Option<HashSet<String>>,
}

/// Pluggable language front end for extracting methods and parsing query snippets.
pub trait Language: Send + Sync {
    fn id(&self) -> &str;
    fn extensions(&self) -> &[&str];
    fn member_access_labels(&self) -> &[&str] {
        &["#.#"]
    }
    fn parse_methods(&self, source: &str, path: &str) -> Vec<ParsedMethod>;
    fn parse_snippet(&self, source: &str, options: &SnippetOptions) -> Option<Node>;
    #[doc(hidden)]
    fn builtin_kind(&self) -> u8 {
        0
    }
    /// For the built-in TypeScript front end: the split size of large functions.
    #[doc(hidden)]
    fn split_lines(&self) -> u32 {
        u32::MAX
    }
}

/// Returns the built-in Python (`.py`) language front end.
pub fn python_language() -> Box<dyn Language> {
    Box::new(lang_py::PythonLanguage)
}

/// Returns the built-in TypeScript/JavaScript (`.ts`, `.tsx`, `.js`, `.jsx`, ...) front end.
/// Functions of at least [`DEFAULT_SPLIT_LINES`] lines that define named functions are
/// indexed as one method per named function; see [`typescript_language_with`].
pub fn typescript_language() -> Box<dyn Language> {
    Box::new(lang_ts::TypeScriptLanguage::default())
}

/// The TypeScript front end with a split size of its own; `u32::MAX` keeps every
/// outermost function whole, as the paper's reference implementation does for Java.
pub fn typescript_language_with(split_lines: u32) -> Box<dyn Language> {
    Box::new(lang_ts::TypeScriptLanguage { split_lines })
}

pub use lang_ts::DEFAULT_SPLIT_LINES;

/// Returns the built-in native Rust (`.rs`) language front end.
pub fn rust_language() -> Box<dyn Language> {
    Box::new(lang_rs::RustLanguage)
}

/// Returns all built-in language front ends (Python, TypeScript/JavaScript, and Rust).
pub fn builtin_languages() -> Vec<Box<dyn Language>> {
    vec![python_language(), typescript_language(), rust_language()]
}

pub use features::{
    featurize, label, parent_feature, position_context, sibling_feature, token_context,
    token_feature, usage_feature, TokenFeatures, VAR,
};
pub use index::{
    build_index, queries_from_lines, query_from_snippet, AromaIndex, IndexOptions, IndexStats,
    Query,
};
pub use prune::{prune, Pruned, TokenTable};
pub use recommend::{
    recommend, LineRole, RecommendOptions, Recommendation, RecommendedLine,
};
pub use search::{search, SearchOptions, SearchResult};

#[cfg(test)]
mod api_contracts {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_sync() {
        assert_send_sync::<SourceFile>();
        assert_send_sync::<MethodRef>();
        assert_send_sync::<ParsedMethod>();
        assert_send_sync::<AromaIndex>();
        assert_send_sync::<Query>();
        assert_send_sync::<SearchResult>();
        assert_send_sync::<Recommendation>();
    }
}
