# Release Scope

This repository is a clean research release of the paper's method, training manifest, three SAGE result files, tests, and publication source.

## Included

- The complete dual-loop controller used by the main method.
- Deterministic realization of all 24 simulator-only interaction states.
- Rollout-group integration that shares one condition across group members.
- The 500-scenario training manifest and validation script.
- The Qwen3-8B reproduction launcher and paper hyperparameters.
- Sanitized per-scenario SAGE outputs for three independent runs.
- Unit tests for controller updates, persistence, priors, prompts, and sampling.
- Public paper source, referenced figures, and technical supplement.

## External Dependencies

- A compatible VERL multi-turn training runtime.
- Qwen3-8B weights obtained under the model's license.
- A user-simulator and emotion-verifier API compatible with the launcher's environment variables.
- SAGE, ESConv, ESC-Eval, and EIBench evaluation packages obtained from their official releases.

## Not Included

- Model checkpoints or optimizer states.
- API credentials, internal endpoints, OSS paths, or local filesystem paths.
- Third-party model weights and benchmark repositories.
- Obsolete prototypes, failed runs, generated caches, and duplicate experiment directories.

These exclusions keep the release auditable and avoid redistributing third-party assets. They do not change the method implementation or the reported released SAGE results.
