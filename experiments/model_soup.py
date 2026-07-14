#!/usr/bin/env python3
"""Create a reproducible weighted model soup for LLM4Rec checkpoints."""

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

DEFAULT_NAME = "v30_soup_v19_v21_v26_w45_35_20"
DEFAULT_SOURCES = [
    ("v19_v15_dualmode_fast_lr5e6_ep055", 0.45),
    ("v21_v15_user_repair_rec_fast_lr4e6_ep055", 0.35),
    ("v26_v21_product_rec_rebalance_lr22e6_ep032", 0.20),
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


def append_log(text: str) -> None:
    with TOP_LOG.open("a", encoding="utf-8") as fp:
        fp.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_model_files(base_dir: Path, out_dir: Path) -> None:
    for name in COPY_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def load_scores() -> dict:
    return {
        "v19": {"total": 0.8855, "item": 0.2146, "user": [0.0855, 0.0337], "rec": [0.0768, 0.1020, 0.1344, 0.1143], "world": 0.1242, "eval_time_min": 48.11},
        "v21": {"total": 0.8800, "item": 0.2146, "user": [0.0881, 0.0347], "rec": [0.0576, 0.1122, 0.1358, 0.1098], "world": 0.1271, "eval_time_min": 47.61},
        "v26": {"total": 0.8804, "item": 0.2146, "user": [0.0895, 0.0344], "rec": [0.0768, 0.0986, 0.1330, 0.1116], "world": 0.1219, "eval_time_min": 45.73},
    }


def main() -> int:
    args = parse_args()
    sources = [(OUTPUT_DIR / name, weight) for name, weight in DEFAULT_SOURCES]
    out_dir = OUTPUT_DIR / args.name

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path, _ in sources if not (path / "model.safetensors").exists()]
    if missing:
        raise SystemExit(f"missing model.safetensors under: {missing}")
    total_weight = sum(weight for _, weight in sources)
    if abs(total_weight - 1.0) > 1e-8:
        raise SystemExit(f"weights must sum to 1.0, got {total_weight}")

    start = time.time()
    acc: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    expected_keys: set[str] | None = None

    for src_dir, weight in sources:
        tensors = load_file(src_dir / "model.safetensors", device="cpu")
        keys = set(tensors.keys())
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise SystemExit(f"tensor keys differ for {src_dir}")

        for key, tensor in tensors.items():
            if key not in dtypes:
                dtypes[key] = tensor.dtype
            weighted = tensor.float().mul_(weight)
            if key in acc:
                acc[key].add_(weighted)
            else:
                acc[key] = weighted
        del tensors
        gc.collect()

    merged = {key: tensor.to(dtype=dtypes[key]) for key, tensor in acc.items()}
    save_file(merged, out_dir / "model.safetensors")
    copy_model_files(sources[0][0], out_dir)

    recipe = {
        "created_at": stamp(),
        "name": args.name,
        "method": "weighted_model_soup",
        "sources": [
            {"path": str(path.resolve()), "weight": weight}
            for path, weight in sources
        ],
        "source_scores": load_scores(),
        "rationale": (
            "Average related CoT-native checkpoints from the same architecture to combine v19 total/rec4, "
            "v21 user/world signal, and v26 user1/rec1/speed without adding inference-time cost."
        ),
        "cot_policy": "No v7 final-only checkpoint is used. All sources are CoT-native continuations.",
        "reproduce": f"demo/LLaMA-Factory/.venv/bin/python experiments/model_soup.py --name {args.name} --overwrite",
        "runtime_seconds": round(time.time() - start, 2),
    }
    (out_dir / "experiment_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "merge_results.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")

    append_log(
        f"\n\n[{stamp()}] model soup created\n"
        f"- {args.name}: method=weighted_model_soup, weights=v19:0.45/v21:0.35/v26:0.20, output={out_dir.resolve()}\n"
        "- note=CoT-native soup, no v7 final-only source; intended to combine v19 total, v21 user/world, and v26 speed/user1/rec1 without inference overhead.\n"
    )
    print(json.dumps({"output": str(out_dir.resolve()), "runtime_seconds": recipe["runtime_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
