# Released Results

This directory contains artifacts supporting the public paper:

- `paper_main_results.csv`: protocol-matched aggregate values in the main table.
- `ablation_summary.csv`: single-run SAGE component ablations reported by the paper.
- `sage/`: per-scenario outputs from three independent complete-method training
  runs and their recomputed summary.
- `controller/`: controller phase, state-level, and 192-unit snapshots.
- `state_validation/`: interaction-state recovery and human-review records.
- `analysis/`: aggregate intent and paired-scenario analyses.

SAGE `Success` and `Failure` are percentages. ESConv values in the aggregate
table are multiplied by 100. ESC-Eval uses the official 0--4 macro-average;
EIBench reports raw aggregate reward. Human evaluation uses a 0--4 scale.

The three complete-method SAGE run scores are 81.02, 78.58, and 78.11. Their
mean is 79.2367 and their sample standard deviation is 1.5622.

The raw SAGE files were sanitized for public release by removing service URLs,
timestamps, local paths, and experiment-tracking identifiers. Dialogue content
and evaluation outcomes were retained.
