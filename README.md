# kdrifting

`kdrifting` is a PyTorch reimplementation of the Drift release at
`/home/b/projects/drifting`.

The project goal is not a loose "inspired by" rewrite. The goal is a faithful
port of the original algorithms, math, and data flow while replacing the JAX
and Flax execution model with modern, idiomatic PyTorch code and stronger
engineering conventions.

## Porting contract

- Preserve algorithmic and mathematical behavior.
- Prefer clear, typed interfaces over framework-specific cleverness.
- Treat parity as a testable property, not an aspiration.
- Make behavior changes only when they are explicit, justified, and covered.

## Development

Install the project in editable mode with the development tools:

```bash
python -m pip install -e '.[dev,hf]'
pre-commit install
```

The repository gate is intentionally strict:

```bash
ruff check .
ruff format --check .
pyright
pytest
```

Warnings are treated as test failures.

## CLI

The package exposes a single `kdrifting` command:

```bash
kdrifting train-mae --config path/to/mae.yaml --workdir runs/mae
kdrifting train-gen --config path/to/gen.yaml --workdir runs/gen
kdrifting infer --init-from runs/gen/params_ema --workdir runs/infer
```
