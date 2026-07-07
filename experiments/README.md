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

Current active run set:

- `v11_item_cot_focus_lr2e5`: target `懂物料`
- `v12_user_cot_focus_lr15e6`: target `懂用户`
- `v13_rec_cot_focus_lr2e5`: target `懂推荐`
- `v14_world_cot_preserve_lr8e6`: target `懂世界` / general knowledge preservation

The v11-v14 designs are based on the observed baseline vs v7 evaluation logs:
v7 improved total score from 0.8184 to 0.8978, mainly by improving itemic
grounding and part of user/recommendation performance, while slightly reducing
one user subtask, video recommendation, and common-sense score.

Unlike v7, these runs keep the original `<think>...</think>` supervision because
reasoning traces are central to the OneReason/LLM-Rec setup.
