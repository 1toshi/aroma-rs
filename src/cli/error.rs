//! Typed exit-code dictionary and 3-part teaching errors for the `aroma` CLI.

use std::process::ExitCode;

/// Documented CLI exit codes returned by `aroma`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ExitKind {
    Success = 0,
    UserInput = 1,
    SafetyBlock = 2,
    ToolEnv = 3,
    Findings = 4,
}

impl ExitKind {
    pub(crate) const ALL: [Self; 5] = [
        Self::Success,
        Self::UserInput,
        Self::SafetyBlock,
        Self::ToolEnv,
        Self::Findings,
    ];

    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::Success => 0,
            Self::UserInput => 1,
            Self::SafetyBlock => 2,
            Self::ToolEnv => 3,
            Self::Findings => 4,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::UserInput => "user-input-error",
            Self::SafetyBlock => "safety-block",
            Self::ToolEnv => "tool-environment-error",
            Self::Findings => "findings",
        }
    }

    pub(crate) const fn retryable(self) -> bool {
        match self {
            Self::Success | Self::UserInput | Self::SafetyBlock | Self::Findings => false,
            Self::ToolEnv => true,
        }
    }

    pub(crate) const fn notes(self) -> &'static str {
        match self {
            Self::Success => "Includes valid searches or scans with 0 matches",
            Self::UserInput => "Unknown flag/command, missing argument, or out-of-range limit",
            Self::SafetyBlock => "Reserved safety gate; aroma CLI is read-only",
            Self::ToolEnv => "Unreadable/missing target path or malformed JSON file",
            Self::Findings => "`aroma check` found changed functions with likely existing equivalents (suppress with --no-fail)",
        }
    }

    pub(crate) fn exit_code(self) -> ExitCode {
        ExitCode::from(self.code())
    }
}

/// Actionable CLI error printed strictly to stderr (never stdout, never raw stack trace).
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CliError {
    pub(crate) kind: ExitKind,
    pub(crate) what: String,
    pub(crate) why: String,
    pub(crate) try_cmd: String,
    pub(crate) see: String,
}

impl CliError {
    fn new(
        kind: ExitKind,
        what: impl Into<String>,
        why: impl Into<String>,
        try_cmd: impl Into<String>,
        see: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            what: what.into(),
            why: why.into(),
            try_cmd: try_cmd.into(),
            see: see.into(),
        }
    }

    pub(crate) fn user_input(
        what: impl Into<String>,
        why: impl Into<String>,
        try_cmd: impl Into<String>,
        see: impl Into<String>,
    ) -> Self {
        Self::new(ExitKind::UserInput, what, why, try_cmd, see)
    }

    pub(crate) fn tool_env(
        what: impl Into<String>,
        why: impl Into<String>,
        try_cmd: impl Into<String>,
        see: impl Into<String>,
    ) -> Self {
        Self::new(ExitKind::ToolEnv, what, why, try_cmd, see)
    }

    pub(crate) fn render(&self) -> String {
        format!(
            "error: {}\n  why: {}\n  try: {}\n  see: {}\n\nexit {} ({})\n",
            self.what,
            self.why,
            self.try_cmd,
            self.see,
            self.kind.code(),
            self.kind.as_str(),
        )
    }
}

/// Computes bounded Levenshtein edit distance between short identifiers/flags.
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    if ac.len().abs_diff(bc.len()) > 8 || ac.len() > 64 || bc.len() > 64 {
        return ac.len().abs_diff(bc.len()).max(9);
    }
    let mut prev: Vec<usize> = (0..=bc.len()).collect();
    let mut curr = vec![0usize; bc.len() + 1];
    for (i, &ca) in ac.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in bc.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        prev.copy_from_slice(&curr);
    }
    prev[bc.len()]
}

/// Returns the closest candidate within `max_dist` (ties broken lexicographically).
pub(crate) fn closest_match<'a>(
    input: &str,
    candidates: &[&'a str],
    max_dist: usize,
) -> Option<&'a str> {
    let lower = input.to_ascii_lowercase();
    let mut best: Option<(usize, &'a str)> = None;
    for &cand in candidates {
        let d = levenshtein(&lower, &cand.to_ascii_lowercase());
        if d <= max_dist {
            match best {
                None => best = Some((d, cand)),
                Some((bd, bc)) => {
                    if d < bd || (d == bd && cand < bc) {
                        best = Some((d, cand));
                    }
                }
            }
        }
    }
    best.map(|(_, c)| c)
}
