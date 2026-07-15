# LLM4Rec Experiments

The current reproducible experiment batches are managed through
`run_experiments.py`. Model outputs and per-version YAML files stay in the same
version sequence; metadata is kept in one shared registry instead of per-batch
files.

## Layout

- `runs.json`: unified v31-v43 registry for trainable and derived models
- `data/manifest.json`: dataset recipes, counts, hashes, and source statistics
- `configs/`: one LLaMA-Factory config per run; multi-stage runs have numbered configs
- `outputs/<version>/`: final model, training logs, trainer results, and reproduction recipe
- `logs/summary.json`: structured training results
- `logs/summary.md`: human-readable combined summary
- `logs/supervisor_status.jsonl`: preparation, training, cleanup, and preflight events
- `source_data.py`: internal builder for the official Explorer source parquet data

Generated dataset JSONL files and everything under `outputs/` are ignored by
git. Intermediate `checkpoint-*` directories are disabled. A successful run
deletes its generated JSONL and matching Arrow caches, while retaining the raw
data, manifest, config, builder, recipe, logs, and final model.

## Commands

List every registered experiment without reading source data or training:

```bash
demo/LLaMA-Factory/.venv/bin/python experiments/run_experiments.py --list
```

Rebuild shared indexes after editing metadata; this does not prepare data or train:

```bash
demo/LLaMA-Factory/.venv/bin/python experiments/run_experiments.py --rebuild-index
```

Regenerate data and reproduce one model on a selected GPU:

```bash
demo/LLaMA-Factory/.venv/bin/python experiments/run_experiments.py \
  --single v34_v29_paper_mix_lr5e7_ep10 --gpu 0
```

Prepare datasets and configs without starting training:

```bash
demo/LLaMA-Factory/.venv/bin/python experiments/run_experiments.py \
  --batch v34-v43 --prepare-only
```

Run a full registered batch with two independent GPU workers:

```bash
demo/LLaMA-Factory/.venv/bin/python experiments/run_experiments.py --batch v34-v43
```

Running the script without a selection only prints help and never starts training.

## v34-v43 Design

The official source data lives at `data/Explorer_LLM_Rec_Competition`. These
runs preserve meaningful `/think` supervision and pure final output for
`/no_think`; target SIDs are forbidden from prompts and reasoning traces.

| version | base | hypothesis | lr | epochs |
|---|---|---|---:|---:|
| v34 | v29 | paper-like R0/R3/general mixture | 5e-7 | 1.0 |
| v35 | v29 | more compact R3 CoT with direct-route balance | 6e-7 | 1.0 |
| v36 | v29 | general-heavy forgetting guard | 4e-7 | 1.2 |
| v37 | v29 | R0-heavy caption/SID perception | 6e-7 | 1.0 |
| v38 | v29 | aggressive real-holdout R3 training | 7e-7 | 1.0 |
| v39 | v29 | R0/general then R3/general curriculum | 6e-7 -> 5e-7 | 0.8 + 0.8 |
| v40 | official pretrain | clean restart on the source mixture | 3e-6 | 1.5 |
| v41 | v29 | original competition plus general replay control | 4e-7 | 1.0 |
| v42 | v29 | user-format/evolution guard plus R3 | 5e-7 | 1.0 |
| v43 | v29 | same data as v34 with lower LR and longer exposure | 2.5e-7 | 1.5 |

The controlled comparisons and recommended evaluation order are recorded in
`logs/summary.md`. Each output also contains `experiment_recipe.json` with the
exact base, dataset manifest, configs, code hashes, and unified reproduction command.
