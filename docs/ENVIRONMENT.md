# Environment

The released method code has a lightweight validation environment and a
separate full-training environment.

## Method validation

Python 3.10 or newer, NumPy, and the standard-library SQLite module are enough
to validate the controller, prompts, persistence, and training manifest:

```bash
python -m pip install -r requirements.txt
bash scripts/verify_release.sh
```

## Full training

The reported Qwen3-8B experiments used the following runtime snapshot:

| Component | Version or configuration |
|---|---|
| Hardware | 4 x NVIDIA A800 80GB |
| PyTorch | 2.5.1 with CUDA 12.4 support |
| Transformers | 4.48.3 |
| Ray | 2.42.1 |
| vLLM | Development build compatible with the multi-turn VERL runtime |
| Base policy | Qwen3-8B, non-thinking mode |
| Simulator / verifier | Frozen DeepSeek-V3 services |

Full training additionally requires the compatible multi-turn VERL/RLVER
runtime described in `INTEGRATION.md`. The repository does not pin a public
vLLM commit because the multi-turn runtime and its vLLM build must be installed
as a compatible pair. The exact optimization and controller overrides are
recorded in `launch/train_dual_loop_qwen3_8b_n4.sh`.

No API key, private endpoint, model weight, or checkpoint is included. Supply
those values through environment variables as shown in the root README.
