#!/usr/bin/env python3
"""Unify v29-derived recommendation experts with reproducible task arithmetic."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
OUTPUT_DIR = EXP / "outputs"
TOP_LOG = ROOT / "log.txt"

DEFAULT_NAME = "v33_task_arith_v29_live55_r325"
BASE_NAME = "v29_v19_user_world_guard_lr8e7_ep018"
EXPERTS = [
    ("v31_v29_live_specialist_r3_lr12e6_ep035", 0.55),
    ("v32_v29_balanced_r3_draft_lr8e7_ep020", 0.25),
]
COPY_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
]


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = OUTPUT_DIR / BASE_NAME
    expert_dirs = [(OUTPUT_DIR / name, scale) for name, scale in EXPERTS]
    out_dir = OUTPUT_DIR / args.name

    required = [base_dir, *(path for path, _ in expert_dirs)]
    missing = [str(path) for path in required if not (path / "model.safetensors").exists()]
    if missing:
        raise SystemExit(f"missing model.safetensors under: {missing}")
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    base = load_file(base_dir / "model.safetensors", device="cpu")
    dtypes = {key: tensor.dtype for key, tensor in base.items()}
    merged = {key: tensor.float().clone() for key, tensor in base.items()}
    expected_keys = set(base)

    for expert_dir, scale in expert_dirs:
        expert = load_file(expert_dir / "model.safetensors", device="cpu")
        if set(expert) != expected_keys:
            raise SystemExit(f"tensor keys differ for {expert_dir}")
        for key, tensor in expert.items():
            merged[key].add_(tensor.float().sub(base[key].float()), alpha=scale)
        del expert
        gc.collect()

    output = {key: tensor.to(dtype=dtypes[key]) for key, tensor in merged.items()}
    save_file(output, out_dir / "model.safetensors")
    for name in COPY_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    recipe = {
        "created_at": stamp(),
        "name": args.name,
        "method": "task_arithmetic",
        "formula": "theta_v29 + 0.55*(theta_v31-theta_v29) + 0.25*(theta_v32-theta_v29)",
        "base": str(base_dir.resolve()),
        "experts": [
            {"path": str(path.resolve()), "task_vector_scale": scale}
            for path, scale in expert_dirs
        ],
        "rationale": (
            "Approximate OneReason specialize-then-unify without leaderboard labels: retain 20% explicit "
            "v29 weight, emphasize the live-domain expert, and add a smaller balanced R3 cognition delta."
        ),
        "cot_policy": "All checkpoints descend from the CoT-native v29/v19 line; v7 is not used.",
        "reproduce": f"demo/LLaMA-Factory/.venv/bin/python experiments/task_arithmetic.py --name {args.name} --overwrite",
        "runtime_seconds": round(time.time() - start, 2),
    }
    for name in ["experiment_recipe.json", "merge_results.json"]:
        (out_dir / name).write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    with TOP_LOG.open("a", encoding="utf-8") as fp:
        fp.write(
            f"\n[{stamp()}] task-arithmetic model created\n"
            f"- {args.name}: {recipe['formula']}, output={out_dir.resolve()}\n"
        )
    print(json.dumps({"output": str(out_dir.resolve()), "runtime_seconds": recipe["runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
