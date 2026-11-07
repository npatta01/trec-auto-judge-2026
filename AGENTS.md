# Repository Agent Instructions

## TREC AutoJudge skills

- The organizer-maintained skill sources live under `.claude/skills/`.
- Codex discovery aliases live under `.agents/skills/` and symlink to those source directories. Do not duplicate or independently edit the aliased files.
- In Codex, invoke them as `$autojudge-setup`, `$autojudge-develop`, and `$autojudge-submit`.
- Read the selected skill's complete `SKILL.md` before following it. Each skill defers to the canonical TREC AutoJudge Participant HowTo and requires the data-handling policy to be read before touching evaluation data.
