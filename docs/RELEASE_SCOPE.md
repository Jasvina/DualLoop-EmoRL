# Public Release Scope

This repository is the curated public release for the paper. It contains the
artifacts needed to inspect the method, validate its core logic, reconstruct
the released training configuration, and trace the reported evidence.

## Included

- The current complete SAGEVariant-PassCtrl runtime under
  `release/work_sagevar_passctrl_train500_n4`, including its 500 generated
  SAGE-style training profiles and generation utility.
- A transfer-ready ZIP of that same runtime under `artifacts/`.
- Method-specific controller and rollout-integration code.
- Exact 500-scenario ordered training manifest.
- Released Qwen3-8B launch configuration.
- Unit tests and validation utilities.
- Aggregate paper results and three sanitized complete-method SAGE runs.
- Controller evolution, 24-state, and 192-unit diagnostic exports.
- Interaction-state validation records and analysis outputs.
- Paper, bibliography, figure, technical-supplement, and appendix source.
- License, third-party notices, citation metadata, and artifact hashes.

## Obtained Separately

The following dependencies are not authored by this project and must be
obtained from their official providers under the applicable terms:

- Qwen3-8B model weights.
- A compatible VERL/RLVER training runtime.
- SAGE, ESConv/ESC-Eval, and EIBench evaluation packages and model weights.
- User-simulator and verifier API access.

## Intentionally Excluded

- Model checkpoints and optimizer states, because of size and upstream model
  licensing constraints.
- Credentials, service URLs, internal storage paths, experiment-tracking IDs,
  and machine-specific metadata.
- Raw server logs, transient SQLite WAL files, caches, and failed-run outputs.
- Superseded drafts, duplicate exports, inferred placeholders, and exploratory
  analyses that are not evidence for the public paper.
- Third-party repositories and benchmark bundles that should be obtained from
  their maintainers.
- Patent disclosures, invention-submission forms, and other confidential legal
  materials. These are not research reproducibility artifacts and publishing
  them can affect patent strategy.

Exclusion does not change the method definition. The public implementation,
configuration, manifest, tests, and supporting result exports are the
canonical released artifacts.

The generic RL runtime exclusion applies to the older method-only release at
the repository root. The current SAGEVariant-PassCtrl experiment is also
published as a self-contained runnable tree under `release/` at the user's
request. Both release tracks exclude credentials, internal endpoints, model
weights, checkpoints, generated caches, and obsolete three-difficulty data.
