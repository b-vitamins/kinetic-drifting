# AGENTS

## Purpose

This repository is a faithful PyTorch port of the upstream Drift codebase in
`/home/b/projects/drifting`.

## Non-negotiables

- Preserve the original algorithmic behavior unless a change is explicitly
  called out and justified.
- Prefer readable, typed, testable code over framework trickery.
- Keep public behavior stable while improving structure and maintainability.
- Do not silently "simplify" math, RNG flow, checkpoint semantics, or eval
  logic.

## Working rules

- Use conventional commits: `type(scope): summary`.
- Keep commits narrow enough to review in isolation.
- Every commit must leave `ruff`, `pyright`, and `pytest` green.
- Pytest warnings are errors.
- Add or update tests whenever ported behavior becomes observable.

## Parity rules

- Match tensor shapes, dtype intent, and reduction semantics.
- Preserve config field names when practical.
- Prefer compatibility layers over breaking changes during the port.
- When exact framework mechanics differ, preserve the algorithm and document the
  difference.
