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
