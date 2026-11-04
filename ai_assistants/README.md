# hermes-rheo for AI coding assistants

Ready-made files that teach an AI assistant to use hermes-rheo instead of
writing its own rheology parsing, FFT, shifting or fitting code. They are
plain Markdown and not tied to any one tool: use them with GitHub Copilot,
Claude, Cursor, Codex, Databricks Genie Code, chat assistants, or anything
that accepts instruction or skill files.

| File | Kind | Loaded |
|---|---|---|
| [`skills/hermes-rheo/SKILL.md`](skills/hermes-rheo/SKILL.md) | **Skill**: the hermes-rheo playbook (data model, readers, task-to-tool table, tested recipes, exports, plot helpers, pitfalls, environment notes). Follows the open [Agent Skills](https://agentskills.io) format. | On demand, when a task involves rheology data |
| [`project_instructions.md`](project_instructions.md) | **Project / team instructions** template: approved libraries, install pattern, data locations, conventions, guardrails. | In every conversation, for everyone in the project |
| [`user_instructions.md`](user_instructions.md) | **Personal instructions** template: your role, experience, data paths, plot and response style. | In every conversation you start |

Put long API detail in the skill and short rules in the instructions.
Fill in the `<angle brackets>` in the templates before using them.

## Where to put the files

Locations used by common tools at the time of writing — check your tool's
documentation if it has changed.

| Tool | Skill | Project instructions | Personal instructions |
|---|---|---|---|
| GitHub Copilot (VS Code, CLI) | `.github/skills/hermes-rheo/SKILL.md` in the repo, or `~/.copilot/skills/hermes-rheo/SKILL.md` | `.github/copilot-instructions.md` or `AGENTS.md` | VS Code user instructions / user `*.instructions.md` |
| Claude Code | `.claude/skills/hermes-rheo/SKILL.md`, or `~/.claude/skills/hermes-rheo/SKILL.md` | `CLAUDE.md` in the project | `~/.claude/CLAUDE.md` |
| Cursor | add `SKILL.md` as a project rule (`.cursor/rules/`) | `AGENTS.md` or a project rule | *User Rules* in settings |
| Codex and other `AGENTS.md` tools | tool's skills folder, or reference `SKILL.md` from `AGENTS.md` | `AGENTS.md` | the tool's global `AGENTS.md` |
| Databricks Genie Code | `/Users/<email>/.assistant/skills/hermes-rheo/SKILL.md` (personal) or `/Workspace/.assistant/skills/hermes-rheo/SKILL.md` (admin, everyone) | `/Workspace/.assistant_workspace_instructions.md` (admin) | `/Users/<email>/.assistant_instructions.md` (Genie Code pane → gear → *User instructions*) |
| Chat assistants (ChatGPT, Claude.ai, Copilot Chat, ...) | upload `SKILL.md` as a project/knowledge file or skill | project instructions field | custom / personal instructions field |

Notes:

- Keep the skill's folder name `hermes-rheo` and its `name` /
  `description` frontmatter: assistants use the description to decide
  when to load it. In most tools you can also call it explicitly (e.g.
  `@hermes-rheo` or `/hermes-rheo`).
- Most tools pick up new or edited files only in a **new** conversation.
- Some tools limit instruction length (Databricks Genie Code: 20,000
  characters). Both templates are about 2–4 K characters.
- To keep the skill up to date, point the tool at this folder in a clone
  of hermes-rheo (or a Git folder in Databricks) and `git pull`.
