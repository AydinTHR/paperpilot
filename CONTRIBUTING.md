# Contributing to PaperPilot

Thanks for taking the time to contribute.

## Getting set up

```bash
git clone https://github.com/AydinTHR/paperpilot.git
cd paperpilot

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pre-commit install        # installs both the pre-commit and commit-msg hooks
```

## Workflow

1. Create a short-lived branch off `main`, named for the change
   (for example `feat/atr-position-sizing` or `fix/journal-timezone`).
2. Make focused, atomic commits. Keep each commit to one logical change.
3. Open a pull request. CI runs format check, lint, and tests, and must pass.
4. Self-review your own diff before asking for a merge. Squash and merge when green.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org). The type prefix
drives the changelog and the version bump.

- `feat:` a new feature (minor version bump)
- `fix:` a bug fix (patch version bump)
- `docs:`, `refactor:`, `test:`, `chore:`, `ci:`, `perf:`, `build:` for the rest
- A `!` after the type, or a `BREAKING CHANGE:` footer, marks a breaking change (major bump)

Subjects are short and in the imperative mood, for example
`feat(auth): add token refresh`.

## House rules

- No em dashes anywhere in the project, the README most of all. A local check fails
  the commit if one slips in. Use a comma, a colon, parentheses, or two sentences.
- Commit messages carry no AI or tool attribution. A local hook strips it as a backstop.

## Code style

Formatting and linting are enforced by pre-commit, so you do not need to format by
hand. Run `pre-commit run --all-files` to check everything at once.
