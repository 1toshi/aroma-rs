//! Aroma: Code Recommendation via Structural Code Search
//! (Luan, Yang, Barnaby, Sen, Chandra; OOPSLA 2019).

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub path: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodRef {
    pub path: String,
    pub name: String,
    pub language: String,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub keyword: bool,
    pub local: bool,
    pub line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Token(Token),
    Tree(Vec<Node>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedMethod {
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub body: Node,
}

#[derive(Debug, Clone, Default)]
pub struct SnippetOptions {
    pub language: Option<String>,
    pub path: Option<String>,
    pub globals: Option<HashSet<String>>,
}

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
}

pub fn python_language() -> Box<dyn Language> {
    Box::new(lang_py::PythonLanguage)
}

pub fn typescript_language() -> Box<dyn Language> {
    Box::new(lang_ts::TypeScriptLanguage)
}

pub fn rust_language() -> Box<dyn Language> {
    Box::new(lang_rs::RustLanguage)
}

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
