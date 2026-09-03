# Evaluation Guide

The paper evaluates one policy checkpoint through four complementary automatic
protocols and a blinded human evaluation. Third-party benchmark repositories,
judge weights, and model weights are not redistributed here; obtain them from
their official releases and follow their licenses.

## Protocols

| Evaluation | Released protocol | Reported fields |
|---|---|---|
| SAGE | 100 held-out multi-turn scenarios | Mean final emotion, success, failure |
| ESConv | 2,895 reference responses | Strategy accuracy, BLEU-2, ROUGE-L, BERTScore, Distinct-2 |
| ESC-Eval | 331 fixed-role interactions, official InternLM2 + ESC-RANK pipeline | Seven dimensions and 0--4 macro-average |
| EIBench | 213 held-out scenarios | Overall raw reward and Support, Defense, Repair, Charm |
| Human | 100 held-out scenarios, three blinded annotators | Overall support quality on a 0--4 scale |

Training simulation and emotion verification use frozen DeepSeek-V3 services.
ESC-Eval uses its official InternLM2 + ESC-RANK judges, EIBench uses Qwen3-Max,
and ESConv is reference based. These evaluation surfaces are therefore
separate from the training verifier.

## SAGE thresholds

For a final emotion score `r`, the released summaries use:

- success: `r >= 100`;
- failure: `r < 10`;
- Overall: arithmetic mean of final emotion over all 100 scenarios.

The three complete-method per-scenario files are in `results/sage/`. Recompute
and cross-check all released SAGE aggregates without modifying those files:

```bash
python scripts/verify_released_results.py
```

## Scaling conventions

- ESConv values in `results/paper_main_results.csv` are multiplied by 100.
- ESC-Eval retains the official 0--4 scale.
- EIBench retains raw aggregate rewards; negative values are valid.
- Human evaluation retains the 0--4 scale.

## Comparison requirements

For protocol-matched comparisons, keep the benchmark split, simulator, judge,
generation configuration, and evaluation seed fixed across checkpoints. Do not
use held-out test outcomes for policy training, controller updates, checkpoint
selection, or curriculum construction.

The repository publishes aggregate comparison values and raw complete-method
SAGE outputs. It does not redistribute third-party evaluation implementations
or outputs belonging to checkpoints that are not part of the released method.
