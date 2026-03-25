# Single-Machine Certification Report

Generated at UTC: `2026-03-25T19:45:33+00:00`
Git commit: `7fd085d878a85a1cabe635eb475a9b1ee441f7e4`
Git branch: `master`
Python: `3.11.14` via `/home/b/projects/kdrifting/.venv/bin/python`
Torch: `2.10.0`
CUDA available: `True`
CUDA device count: `1`

## Devices
- GPU 0: `NVIDIA GeForce RTX 3060` with `11.62` GiB VRAM

## Suite

Overall passed: `True`
Total duration (s): `200.53`

### artifact-parity

Model rebuild and checkpoint import parity against upstream JAX.

- Passed: `True`
- Duration (s): `46.11`
- Command: `.venv/bin/python -m pytest -q tests/unit/test_jax_artifacts.py tests/unit/test_checkpointing.py tests/unit/test_export.py -o cache_dir=/tmp/kdrifting-pytest-cache`

```text
............                                                             [100%]
12 passed in 42.27s
```

### math-parity

Loss and metric math parity against upstream helpers.

- Passed: `True`
- Duration (s): `7.75`
- Command: `.venv/bin/python -m pytest -q tests/unit/test_losses.py tests/unit/test_eval.py -o cache_dir=/tmp/kdrifting-pytest-cache`

```text
............                                                             [100%]
12 passed in 5.01s
```

### training-parity

Toy and real-model resumed training traces against upstream JAX.

- Passed: `True`
- Duration (s): `131.25`
- Command: `.venv/bin/python -m pytest -q tests/unit/test_training_steps.py -o cache_dir=/tmp/kdrifting-pytest-cache`

```text
......                                                                   [100%]
6 passed in 125.11s (0:02:05)
```

### runtime-parity

Resume, runner, inference, and public evaluation-path parity.

- Passed: `True`
- Duration (s): `15.41`
- Command: `.venv/bin/python -m pytest -q tests/unit/test_runners.py tests/unit/test_inference.py -o cache_dir=/tmp/kdrifting-pytest-cache`

```text
........                                                                 [100%]
8 passed in 12.85s
```

