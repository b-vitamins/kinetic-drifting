# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows Keep a Changelog and the project uses Conventional
Commits for commit subjects.

## [Unreleased]

- Initialized repository conventions, strict tooling, and development gates.
- Ported the ImageNet pixel and latent data pipeline, VAE encode/decode helpers,
  and the config-driven model builder.
- Added the ConvNeXt V2 feature backbone and combined MAE/ConvNeXt activation
  builder used by drift training.
- Added local/`hf://` torch artifact loaders for MAE and generator models,
  plus metadata-driven feature model construction helpers.
- Added high-level MAE and generator runners, shared runner utilities, and
  checkpoint/artifact save wiring for the top-level training loops.
- Added generator inference helpers and top-level CLI subcommands for training
  and sampling.
- Added a torch-native release-metric stack for FID, Inception Score, and
  precision/recall using the FID-specific Inception V3 weights.
- Wired generator evaluation to the source-compatible sanity/full metric flow
  instead of the earlier preview-only placeholder.
- Added JAX artifact import and conversion for MAE and generator bundles,
  including numerical parity tests against the upstream JAX modules.
- Added a source-compatible `eval-fid` CLI path on top of the torch metric
  stack, alongside the existing preview-style sampling command.
- Hardened resumable generator training by checkpointing memory-bank state,
  restoring it on resume, and making checkpoint/artifact writes rank-safe.
- Extended JAX import parity to resumable checkpoint directories, including
  numerical parity coverage for loading from run roots and `checkpoints/`.
- Added torch checkpoint-directory and direct checkpoint-file restore support
  for local MAE and generator loading, using EMA weights plus sibling metadata.
- Added direct upstream JAX parity coverage for the drift loss, including
  value and gradient checks on fixed feature tensors.
- Fixed MAE and generator train steps to apply the configured learning-rate
  schedule to the optimizer on every step instead of only at initialization.
- Added upstream JAX train-step parity tests for deterministic toy MAE and
  generator models, covering schedule-driven parameter and EMA updates.
- Added external torch and JAX checkpoint restore for training state bootstraps,
  including conversion of upstream Optax AdamW moments into torch AdamW state.
- Made the generator positional embedding trainable to match upstream JAX
  semantics and extended parity coverage to resumed real-model MAE and
  generator training traces driven from imported JAX checkpoints.
- Added first-class export helpers and CLI commands for converting local or
  `hf://` model sources into canonical torch EMA artifacts and for converting
  external torch or JAX training checkpoints into native torch checkpoint
  bundles.
- Fixed artifact metadata written from config-driven training and checkpoint
  export so `model_config` always includes `num_classes`, which makes saved
  MAE and generator bundles fully reloadable through the public loader.
- Added round-trip parity coverage for exported JAX model artifacts and
  converted JAX training checkpoints, including full optimizer-state resume
  comparisons after conversion.
