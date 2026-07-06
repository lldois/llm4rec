# LLM4Rec Experiments

This folder is managed by `run_experiments.py`.

Layout:

- `data/`: generated Alpaca jsonl variants and `manifest.json`
- `configs/`: per-run LLaMA-Factory yaml configs
- `outputs/<version>/`: model artifacts, `train.log`, tensorboard logs, trainer state
- `logs/`: supervisor status and summary files

The runner is designed for two independent RTX 4090 processes. Set
`EXPERIMENT_DEADLINE="YYYY-MM-DD HH:MM:SS"` when a hard stop is needed.

Intermediate `checkpoint-*` directories are intentionally disabled. Each run
keeps only the final model output to save disk space.

Generated dataset jsonl files are intentionally ignored by git and deleted after
each full experiment run. They can be recreated from the raw competition data by
rerunning the experiment script.
