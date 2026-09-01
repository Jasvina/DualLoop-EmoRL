# Training Manifest

`train_scenarios.jsonl` is the exact ordered 500-record training manifest used
by the released configuration. It is derived from the public RLVER training
profiles and preserves the scenario identifiers, personas, complete events,
behavioral tendencies, and native support-intent labels consumed by training.

The manifest contains eight support intents. Interaction states are not baked
into these records; one of the 24 simulator-only states is composed with a
scenario online by `method/rollout_adapter.py`.

Validate the file with:

```bash
python scripts/validate_training_scenarios.py \
  --input data/train_scenarios.jsonl \
  --expected-rows 500 --expected-intents 8
```

SHA-256:

```text
82e09e73a89bf6e167fe8e56823aaa5e62b0bd55c1d08ef9618011aeb7fa53a3
```

The upstream release remains authoritative for dataset terms and provenance.
