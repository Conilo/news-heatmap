---
name: commit-style
description: Writes git commits for the news-analyzer repository following its established subject-line and body conventions, and applies the project's commit workflow (explicit staging, asking before bundling unrelated changes, never pushing without explicit request). Use whenever the user asks to commit, stage, or amend changes in this repo, or asks for help drafting a commit message.
---

# news-analyzer commit style

## Subject line

- Imperative mood, capitalized first word: `Fix`, `Add`, `Remove`, `Improve`, `Simplify`, `Replace`, `Refactor`.
- Plain prose. No Conventional Commits prefix (no `fix(...):`, no `feat:`, no scopes in parentheses at the start).
- Target 50-72 characters. Hard ceiling: 72.
- For multi-purpose commits, comma-join two clauses: `Fix X, add Y`.
- For commits scoped to one area with several sub-changes, use `Area: a, b, c`.
- A trailing `(file.py)` is acceptable when the change is essentially one file.
- No emojis. No issue/PR refs unless the user supplies them. No `Co-Authored-By` or AI/tool attribution lines.

### Subject examples (verbatim from this repo)

```
Fix NaN event_id bug in clustering, add disturbio crime_type
Simplify fetch: single OR-query driven by config, add test suite
Improve article card UI: overviews, unassociated stats, no description
Remove extra sidebar buttons and their backing functions
Add event deduplication layer (cluster.py)
Fix Desconocido state: better prompt, chart filtering, re-process button
Fix group normalization and disable map scroll-zoom
Dashboard v2: state detail panel, full map, trend charts, group aliases
```

## Body

Add a body when the change touches more than one file, has non-obvious motivation, or carries measurable impact worth recording. Skip the body for trivial single-file edits whose subject already explains the why.

Conventions:

- Blank line between subject and body.
- Wrap at ~72 characters.
- Explain *why* and the user-visible effect, not a line-by-line rehash of the diff.
- Use bullets with `-` (not `*`).
- For multi-file commits, prefer per-file sections introduced with `path/to/file.py — short purpose`, then a bullet list under each. See commit `0c9d84e` for the canonical pattern.
- Single-purpose commits can use one or two prose paragraphs instead of bullets. See commits `640b0d8` and `243f52a`.
- Include concrete numbers when relevant (`95 articles → 56 events (avg 1.7 art/event)`).
- Keep code identifiers and Spanish category names verbatim and unquoted in prose: `crime_type`, `disturbio`, `Desconocido`, `narcobloqueos`.

### Body shape: per-file sections

```
src/extract.py — SYSTEM_PROMPT overhaul:
- Distinguishes "Internacional" (event outside Mexico) from "Desconocido"
- Adds a comprehensive city→state inference table covering all 32 states
- Adds two-shot examples (one domestic, one international)

src/store.py — add update_rows() to overwrite specific rows by URL in-place

src/dashboard.py:
- _render_trend_charts() now uses geo_df for top-states charts
- Stats row shows Geolocalizados + Internacional counts separately
- New sidebar button "Re-process unknown states" triggers it
```

### Body shape: prose paragraphs

```
cluster.py guarded the truthy-NaN trap that mapped every Stage-1 key to
NaN when articles.csv held empty event_id cells (groupby then dropped
all groups, producing 0 events from N articles). store.py reads CSVs
with keep_default_na=False so empty cells stay as "" instead of NaN.
extract.py adds disturbio to the crime_type vocabulary with a usage
guideline and worked example for narcobloqueos / quema de vehículos.
```

## Trailers

Do not write trailers manually. The IDE auto-appends `Made-with: Cursor` to commits made through it; leave that alone if it appears, do not duplicate it, and do not add `Co-Authored-By` or other attribution.

## Commit workflow

Follow these every time the user asks for a commit in this repo:

1. Run `git status` and `git diff` first. Read the staged and unstaged diffs end-to-end before drafting a message.
2. Stage explicit file paths: `git add path/a path/b`. Never `git add .` and never `git add -A`.
3. **Triage unrelated changes.** If the working tree contains edits that aren't part of the requested change (e.g. a tweak to `config.py` while fixing a bug in `src/cluster.py`), ask the user before bundling them. Default options to offer:
   - include in this commit
   - separate commit
   - leave unstaged
   - revert
4. Never commit files that look like secrets (`.env`, `credentials*`, `*.pem`, anything matching `**/secrets/**`). Surface them and stop if encountered.
5. Pass multi-line messages via HEREDOC so subject + body formatting survives the shell:
   ```bash
   git commit -m "$(cat <<'EOF'
   Subject line in imperative mood

   Body paragraph with the why and the user-visible effect, wrapped
   at about 72 characters.
   EOF
   )"
   ```
6. After committing, run `git status` and `git log --oneline -3` to confirm the commit landed and report the short SHA back to the user.
7. **Never push** unless the user explicitly asks. `git commit` is the terminal step of the default workflow.
8. Do not use `git commit --amend`, `git rebase`, `git reset --hard`, or `git push --force` unless the user explicitly requests them. If a pre-commit hook rejects the commit, fix the issue and create a new commit rather than amending.

## Anti-patterns

- `feat(auth): ...`, `fix: ...`, or any other Conventional Commits prefix.
- Subject lines that just name files (`Update cluster.py`).
- Bodies that paraphrase the diff line-by-line.
- Emojis in subject or body.
- Manually adding `Co-Authored-By`, `Signed-off-by`, or AI attribution.
- Bundling unrelated working-tree edits into the commit without asking.
- Pushing to `origin` without an explicit ask.
