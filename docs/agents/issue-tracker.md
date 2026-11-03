# Issue tracker: GitHub

Issues and specs live in GitHub Issues for this repository. Use the `gh` CLI; it infers the repository from the Git remote.

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`; also fetch labels.
- List: `gh issue list --state open`, with appropriate label and state filters.
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`

Use a heredoc for multiline issue bodies.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill operations

- Publish to the issue tracker: create a GitHub issue.
- Fetch the relevant ticket: run `gh issue view <number> --comments`.
