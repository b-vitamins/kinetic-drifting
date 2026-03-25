# Contributing

## Commit style

Use Conventional Commits:

- `feat(...)`: new functionality
- `fix(...)`: correctness fix
- `refactor(...)`: structural change without intended behavior change
- `test(...)`: test-only change
- `docs(...)`: documentation-only change
- `chore(...)`: tooling or repository maintenance

Examples:

- `feat(models): port MAE encoder and decoder`
- `test(loss): add drift loss parity coverage`
- `chore(repo): add pre-commit and strict pyright gate`

## Local checks

Run the full gate before committing:

```bash
ruff check .
ruff format --check .
pyright
pytest
```

Warnings are treated as failures.
