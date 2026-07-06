#!/usr/bin/env python3
"""Prepare and run LLM4Rec experiments on two GPUs.

This script intentionally keeps every experiment self-contained:
dataset variant, yaml config, output directory, train.log, and summary.

Set EXPERIMENT_DEADLINE="YYYY-MM-DD HH:MM:SS" to enforce a local hard stop.
Generated dataset jsonl files are removed after training finishes because they
can be recreated from the raw competition jsonl files.
"""

from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
DATA_DIR = EXP / "data"
CONFIG_DIR = EXP / "configs"
OUTPUT_DIR = EXP / "outputs"
LOG_DIR = EXP / "logs"
LF_DIR = ROOT / "demo" / "LLaMA-Factory"
VENV = LF_DIR / ".venv"
RAW_DATA_DIR = ROOT / "demo" / "baseline-data" / "dataset"
DATASET_INFO = LF_DIR / "data" / "dataset_info.json"
TOP_LOG = ROOT / "log.txt"
DEADLINE_TEXT = os.environ.get("EXPERIMENT_DEADLINE", "").strip()
DEADLINE = datetime.strptime(DEADLINE_TEXT, "%Y-%m-%d %H:%M:%S") if DEADLINE_TEXT else None


RAW_GROUPS = {
    "rec": ["懂推荐1.jsonl", "懂推荐2.jsonl", "懂推荐3.jsonl", "懂推荐4.jsonl"],
    "item": [
        "懂物料part1.jsonl",
        "懂物料part2.jsonl",
        "懂物料part3.jsonl",
        "懂物料part4.jsonl",
        "懂物料part5.jsonl",
        "懂物料part6.jsonl",
        "懂物料part7.jsonl",
    ],
    "user": ["懂用户.jsonl"],
}


RUNS = [
    {
        "name": "v01_all_lr1e5",
        "dataset": "all",
        "lr": "1.0e-5",
        "epochs": 1,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607061,
        "note": "All data, conservative LR to reduce forgetting.",
    },
    {
        "name": "v02_all_lr3e5",
        "dataset": "all",
        "lr": "3.0e-5",
        "epochs": 1,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607062,
        "note": "All data, aggressive LR for stronger task adaptation.",
    },
    {
        "name": "v03_all_lr1e5_ep2",
        "dataset": "all",
        "lr": "1.0e-5",
        "epochs": 2,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607063,
        "note": "All data, two epochs with low LR.",
    },
    {
        "name": "v04_balanced_core_lr2e5",
        "dataset": "balanced_core",
        "lr": "2.0e-5",
        "epochs": 1,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607064,
        "note": "Downsample rec and upweight user so the four-dimension evaluation is less rec-heavy.",
    },
    {
        "name": "v05_user_item_up_lr2e5",
        "dataset": "user_item_up",
        "lr": "2.0e-5",
        "epochs": 1,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607065,
        "note": "All data plus extra user and item examples.",
    },
    {
        "name": "v06_no_think_input_lr2e5",
        "dataset": "no_think_input",
        "lr": "2.0e-5",
        "epochs": 1,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607066,
        "note": "Strip /think or /no_think suffix from inputs while keeping CoT outputs.",
    },
    {
        "name": "v07_final_only_lr2e5",
        "dataset": "final_only",
        "lr": "2.0e-5",
        "epochs": 1,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607067,
        "note": "Remove leading <think>...</think> from outputs when possible; trains concise final answers.",
    },
    {
        "name": "v08_rec_focus_lr2e5",
        "dataset": "rec_focus",
        "lr": "2.0e-5",
        "epochs": 1,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607068,
        "note": "Recommendation-task files only; a specialist checkpoint.",
    },
    {
        "name": "v09_all_linear_lr2e5",
        "dataset": "all",
        "lr": "2.0e-5",
        "epochs": 1,
        "warmup": 0.05,
        "scheduler": "linear",
        "seed": 202607069,
        "note": "Baseline LR with linear decay and slightly longer warmup.",
    },
    {
        "name": "v10_balanced_core_lr1e5_ep2",
        "dataset": "balanced_core",
        "lr": "1.0e-5",
        "epochs": 2,
        "warmup": 0.03,
        "scheduler": "cosine",
        "seed": 202607070,
        "note": "Balanced data with low-LR second pass.",
    },
]


def now() -> datetime:
    return datetime.now()


def stamp() -> str:
    return now().strftime("%Y-%m-%d %H:%M:%S")


def deadline_reached() -> bool:
    return DEADLINE is not None and now() >= DEADLINE


def deadline_label() -> str:
    return f"{DEADLINE_TEXT} +0800" if DEADLINE is not None else "none"


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(text)


def load_raw_records() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for group, names in RAW_GROUPS.items():
        for name in names:
            path = RAW_DATA_DIR / name
            with path.open("r", encoding="utf-8") as fp:
                for line_no, line in enumerate(fp, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as exc:
                        print(f"[WARN] {path}:{line_no} JSON parse failed: {exc}", file=sys.stderr)
                        continue
                    if isinstance(obj, list):
                        if not obj:
                            continue
                        obj = obj[0]
                    if not isinstance(obj, dict):
                        continue
                    prompt = obj.get("prompt", "") or ""
                    response = obj.get("response", "") or ""
                    if not prompt and not response:
                        continue
                    grouped[group].append(
                        {
                            "instruction": obj.get("system", "") or "",
                            "input": prompt,
                            "output": response,
                            "history": [],
                            "_group": group,
                        }
                    )
    return dict(grouped)


def strip_internal(record: dict) -> dict:
    return {
        "instruction": record["instruction"],
        "input": record["input"],
        "output": record["output"],
        "history": record.get("history", []),
    }


def write_dataset(name: str, records: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    path = DATA_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for rec in shuffled:
            fp.write(json.dumps(strip_internal(rec), ensure_ascii=False) + "\n")
    groups = Counter(rec.get("_group", "unknown") for rec in shuffled)
    return {
        "name": name,
        "path": str(path.resolve()),
        "records": len(shuffled),
        "groups": dict(groups),
        "seed": seed,
    }


def prepare_datasets() -> dict[str, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grouped = load_raw_records()
    all_records = grouped["rec"] + grouped["item"] + grouped["user"]
    rng = random.Random(20260706)

    balanced_rec = rng.sample(grouped["rec"], min(12000, len(grouped["rec"])))
    balanced_core = balanced_rec + grouped["item"] + grouped["user"] * 4
    user_item_up = all_records + grouped["item"] + grouped["user"] * 3

    no_think_input = []
    for rec in all_records:
        new = dict(rec)
        new["input"] = re.sub(r"/(?:no_)?think\s*$", "", new["input"]).rstrip()
        no_think_input.append(new)

    think_re = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
    final_only = []
    stripped = 0
    for rec in all_records:
        new = dict(rec)
        output = think_re.sub("", new["output"], count=1).lstrip()
        if output and output != new["output"]:
            new["output"] = output
            stripped += 1
        final_only.append(new)

    variants = {
        "all": all_records,
        "balanced_core": balanced_core,
        "user_item_up": user_item_up,
        "no_think_input": no_think_input,
        "final_only": final_only,
        "rec_focus": grouped["rec"],
    }
    manifest = {
        name: write_dataset(name, records, seed=20260706 + i)
        for i, (name, records) in enumerate(variants.items(), 1)
    }
    manifest["_notes"] = {
        "raw_counts": {k: len(v) for k, v in grouped.items()},
        "final_only_outputs_stripped": stripped,
        "balanced_core_recipe": "sample 12000 rec + all item + user x4",
        "user_item_up_recipe": "all + item x1 extra + user x3 extra",
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def update_dataset_info(manifest: dict[str, dict]) -> None:
    info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    for name, meta in manifest.items():
        if name.startswith("_"):
            continue
        info[f"exp_{name}"] = {
            "file_name": meta["path"],
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
            },
        }
    DATASET_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def yaml_for(run: dict) -> str:
    out = (OUTPUT_DIR / run["name"]).resolve()
    dataset = f"exp_{run['dataset']}"
    return f"""### model
model_name_or_path: OpenOneRec/OneReason-0.8B-pretrain-competition
trust_remote_code: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
enable_liger_kernel: true

### dataset
dataset: {dataset}
dataset_dir: demo/LLaMA-Factory/data
template: qwen3_nothink
cutoff_len: 32768
packing: true
neat_packing: true
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: {out}
logging_steps: 5
save_strategy: "no"
save_total_limit: 1
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: tensorboard
logging_dir: {out}/tb

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: {run["lr"]}
num_train_epochs: {run["epochs"]}
lr_scheduler_type: {run["scheduler"]}
warmup_ratio: {run["warmup"]}
weight_decay: 0.0
bf16: true
pure_bf16: true
seed: {run["seed"]}
"""


def prepare_configs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        run["config"] = str((CONFIG_DIR / f"{run['name']}.yaml").resolve())
        (CONFIG_DIR / f"{run['name']}.yaml").write_text(yaml_for(run), encoding="utf-8")
    (EXP / "runs.json").write_text(json.dumps(RUNS, ensure_ascii=False, indent=2), encoding="utf-8")


def train_command(config: str) -> str:
    return (
        f"source {VENV}/bin/activate && "
        "export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=1 && "
        f"llamafactory-cli train {config}"
    )


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def terminate_process(proc: subprocess.Popen, grace: int = 45) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline and proc.poll() is None:
        time.sleep(1)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_result(run: dict) -> dict:
    out = OUTPUT_DIR / run["name"]
    result = {
        "name": run["name"],
        "dataset": run["dataset"],
        "lr": run["lr"],
        "epochs": run["epochs"],
        "scheduler": run["scheduler"],
        "note": run["note"],
        "output_dir": str(out.resolve()),
    }
    train_results = out / "train_results.json"
    state = out / "trainer_state.json"
    if train_results.exists():
        try:
            result.update(json.loads(train_results.read_text(encoding="utf-8")))
        except Exception as exc:
            result["train_results_error"] = repr(exc)
    if state.exists():
        try:
            st = json.loads(state.read_text(encoding="utf-8"))
            result["global_step"] = st.get("global_step")
            history = [x for x in st.get("log_history", []) if "loss" in x]
            if history:
                result["last_logged_loss"] = history[-1].get("loss")
                result["min_logged_loss"] = min(x["loss"] for x in history)
        except Exception as exc:
            result["trainer_state_error"] = repr(exc)
    if not train_results.exists():
        result["status"] = "no_train_results"
    return result


def write_summary(results: list[dict], manifest: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = LOG_DIR / "summary.json"
    summary_md = LOG_DIR / "summary.md"
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for item in results:
        rows.append(
            "| {name} | {dataset} | {lr} | {epochs} | {scheduler} | {train_loss} | {last_loss} | {runtime} | {status} |".format(
                name=item.get("name", ""),
                dataset=item.get("dataset", ""),
                lr=item.get("lr", ""),
                epochs=item.get("epochs", ""),
                scheduler=item.get("scheduler", ""),
                train_loss=f"{item.get('train_loss', ''):.4f}" if isinstance(item.get("train_loss"), (int, float)) else item.get("train_loss", ""),
                last_loss=f"{item.get('last_logged_loss', ''):.4f}" if isinstance(item.get("last_logged_loss"), (int, float)) else item.get("last_logged_loss", ""),
                runtime=f"{item.get('train_runtime', ''):.1f}s" if isinstance(item.get("train_runtime"), (int, float)) else item.get("train_runtime", ""),
                status=item.get("status", "ok" if item.get("train_loss") is not None else ""),
            )
        )

    dataset_lines = []
    for name, meta in manifest.items():
        if name.startswith("_"):
            continue
        dataset_lines.append(f"- `{name}`: {meta['records']} records, groups={meta['groups']}")

    md = [
        "# LLM4Rec Experiment Summary",
        "",
        f"Generated: {stamp()}",
        f"Hard deadline: {deadline_label()}",
        "",
        "## Dataset Variants",
        *dataset_lines,
        "",
        "## Runs",
        "| version | dataset | lr | epochs | scheduler | train_loss | last_logged_loss | runtime | status |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
        *rows,
        "",
        "## Notes",
        "- Ranking by train loss is only a rough sanity signal because no held-out leaderboard metric is available locally.",
        "- Prefer comparing generated behavior on the official validation/upload flow tomorrow if available.",
    ]
    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    append(
        TOP_LOG,
        f"\n\n[{stamp()}] experiments summary\n"
        f"- summary: {summary_md}\n"
        f"- configs: {CONFIG_DIR}\n"
        f"- outputs: {OUTPUT_DIR}\n"
        f"- deadline enforced: {deadline_label()}\n",
    )
    for item in results:
        append(
            TOP_LOG,
            "- {name}: dataset={dataset}, lr={lr}, epochs={epochs}, scheduler={scheduler}, "
            "train_loss={loss}, last_logged_loss={last}, runtime={runtime}, output={out}\n".format(
                name=item.get("name"),
                dataset=item.get("dataset"),
                lr=item.get("lr"),
                epochs=item.get("epochs"),
                scheduler=item.get("scheduler"),
                loss=item.get("train_loss"),
                last=item.get("last_logged_loss"),
                runtime=item.get("train_runtime"),
                out=item.get("output_dir"),
            ),
        )


def worker(gpu: int, queue: list[dict], lock: threading.Lock, results: list[dict], status_path: Path) -> None:
    while True:
        with lock:
            run = queue.pop(0) if queue else None
        if run is None:
            return
        if deadline_reached():
            with lock:
                results.append({**parse_result(run), "status": "skipped_deadline"})
            return

        out_dir = OUTPUT_DIR / run["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        train_log = out_dir / "train.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["WANDB_DISABLED"] = "1"

        append(
            status_path,
            json.dumps(
                {
                    "time": stamp(),
                    "event": "start",
                    "gpu": gpu,
                    "run": run["name"],
                    "config": run["config"],
                    "gpu_snapshot": gpu_snapshot(),
                },
                ensure_ascii=False,
            )
            + "\n",
        )
        start = time.time()
        with train_log.open("w", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(
                ["bash", "-lc", train_command(run["config"])],
                cwd=str(ROOT),
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                text=True,
            )
            while proc.poll() is None:
                if deadline_reached():
                    append(
                        status_path,
                        json.dumps(
                            {
                                "time": stamp(),
                                "event": "deadline_terminate",
                                "gpu": gpu,
                                "run": run["name"],
                                "pid": proc.pid,
                            },
                            ensure_ascii=False,
                        )
                        + "\n",
                    )
                    terminate_process(proc)
                    break
                time.sleep(30)
        rc = proc.poll()
        elapsed = time.time() - start
        parsed = parse_result(run)
        parsed["returncode"] = rc
        parsed["wall_seconds"] = elapsed
        if deadline_reached() and parsed.get("train_loss") is None:
            parsed["status"] = "terminated_deadline"
        elif rc != 0 and parsed.get("train_loss") is None:
            parsed["status"] = f"failed_rc_{rc}"
        else:
            parsed["status"] = "ok"
        with lock:
            results.append(parsed)
        append(
            status_path,
            json.dumps(
                {
                    "time": stamp(),
                    "event": "finish",
                    "gpu": gpu,
                    "run": run["name"],
                    "returncode": rc,
                    "wall_seconds": round(elapsed, 1),
                    "status": parsed["status"],
                    "gpu_snapshot": gpu_snapshot(),
                },
                ensure_ascii=False,
            )
            + "\n",
        )


def cleanup_generated_datasets(status_path: Path) -> None:
    removed = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        try:
            size = path.stat().st_size
            path.unlink()
            removed.append({"path": str(path.resolve()), "bytes": size})
        except FileNotFoundError:
            continue
    append(
        status_path,
        json.dumps(
            {
                "time": stamp(),
                "event": "cleanup_generated_datasets",
                "removed": removed,
            },
            ensure_ascii=False,
        )
        + "\n",
    )


def main() -> int:
    for path in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    append(
        TOP_LOG,
        f"\n\n[{stamp()}] start experiment preparation; deadline={deadline_label()}\n",
    )
    manifest = prepare_datasets()
    update_dataset_info(manifest)
    prepare_configs()

    status_path = LOG_DIR / "supervisor_status.jsonl"
    append(
        status_path,
        json.dumps(
            {
                "time": stamp(),
                "event": "prepared",
                "runs": [r["name"] for r in RUNS],
                "deadline": deadline_label(),
                "manifest": str((DATA_DIR / "manifest.json").resolve()),
            },
            ensure_ascii=False,
        )
        + "\n",
    )

    queue = list(RUNS)
    results: list[dict] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(0, queue, lock, results, status_path), daemon=False),
        threading.Thread(target=worker, args=(1, queue, lock, results, status_path), daemon=False),
    ]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        append(
            status_path,
            json.dumps(
                {
                    "time": stamp(),
                    "event": "heartbeat",
                    "queued": len(queue),
                    "finished": len(results),
                    "gpu_snapshot": gpu_snapshot(),
                },
                ensure_ascii=False,
            )
            + "\n",
        )
        time.sleep(300)
    for thread in threads:
        thread.join()

    run_order = {run["name"]: i for i, run in enumerate(RUNS)}
    results.sort(key=lambda x: run_order.get(x.get("name", ""), 999))
    write_summary(results, manifest)
    cleanup_generated_datasets(status_path)
    append(status_path, json.dumps({"time": stamp(), "event": "all_done"}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
