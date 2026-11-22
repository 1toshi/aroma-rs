# verify-aroma Feature Map

Project-local verification coverage for `aroma-rs`. Each feature document follows the four-section verification contract (`Sub-features`, `How to get to it (user POV)`, `Driving it with verify-aroma.sh`, `Gotchas`).

| Feature | CLI / API Surface | Script Selector | Doc |
| --- | --- | --- | --- |
| Triage and Clones | `--robot-triage`, `triage`, `clones`, `--include-recommendations` | `triage-and-clones` | [triage-and-clones.md](triage-and-clones.md) |
| Search and Recommend | `search` (`--snippet`, `--file`/`--lines`), `recommend` (`--min-support`) | `search-and-recommend` | [search-and-recommend.md](search-and-recommend.md) |
| Agent Discovery and Errors | `capabilities`, `robot-docs`, `schema`, exit codes `0`/`1`/`3`, Levenshtein typo hints | `agent-discovery-and-errors` | [agent-discovery-and-errors.md](agent-discovery-and-errors.md) |
| Library and Package | Core library with `--no-default-features`, property suite, `.crate` archive contents | `library-and-package` | [library-and-package.md](library-and-package.md) |
