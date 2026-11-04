# hermes-rheo for Databricks Genie Code

Example files that teach the Databricks AI assistant (Genie Code) to use
hermes-rheo instead of writing its own rheology code. Copy them into your
workspace, edit the `<angle brackets>`, and start a **new** chat; Genie
Code picks up changes only in new conversations.

| File | What it is | Where it goes in Databricks |
|---|---|---|
| [`skills/hermes-rheo/SKILL.md`](skills/hermes-rheo/SKILL.md) | **Skill**: the hermes-rheo playbook (API, recipes, task-to-tool table, Databricks notes). Loaded only when a task involves rheology. | Just for you: `/Users/<your-email>/.assistant/skills/hermes-rheo/SKILL.md`<br>For everyone (admin): `/Workspace/.assistant/skills/hermes-rheo/SKILL.md` |
| [`.assistant_instructions.md`](.assistant_instructions.md) | **User instructions**: your role, preferences, data paths and response style. Read in every conversation you start. | `/Users/<your-email>/.assistant_instructions.md` |
| [`.assistant_workspace_instructions.md`](.assistant_workspace_instructions.md) | **Workspace instructions**: team-wide rules (approved libraries, install pattern, data locations, guardrails). Read in every conversation for everyone; takes priority over user instructions. | `/Workspace/.assistant_workspace_instructions.md` (workspace admins only) |

## Setup

1. **Skill**
   - In the workspace browser, create the folder `.assistant/skills/hermes-rheo`
     in your home folder (or in `/Workspace` as an admin).
   - Import `SKILL.md` into it. Keep the file name and the
     `name`/`description` frontmatter.
   - In chat, Genie Code loads it automatically for rheology tasks. You can
     also call it explicitly with `@hermes-rheo`.
2. **User instructions**
   - Open the Genie Code pane, then the gear icon, then **Instructions**,
     then **User instructions**.
   - Paste the edited content of `.assistant_instructions.md`.
3. **Workspace instructions** (admins)
   - Open the same settings and choose **Workspace instructions**.
   - Paste the edited content of `.assistant_workspace_instructions.md`.

Instruction files are limited to 20,000 characters; anything beyond is
ignored. Both templates are well below that. The skill has no such limit.
Keep long API detail in the skill and short rules in the instructions.

Alternatively, if you clone hermes-rheo as a Git folder, you can register
`databricks_genie_code/skills` as an extra skills folder, so the skill
stays up to date with `git pull`.

See the Databricks documentation for
[skills](https://docs.databricks.com/aws/en/genie-code/skills) and
[instructions](https://docs.databricks.com/aws/en/genie-code/instructions).

The same `SKILL.md` follows the open Agent Skills format. It also works
with other assistants (e.g. GitHub Copilot or Claude Code) when placed in
`.github/skills/hermes-rheo/` or `.claude/skills/hermes-rheo/`.
