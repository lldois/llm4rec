#!/usr/bin/env python3
"""Prepare and run LLM4Rec experiments on two GPUs.

This script intentionally keeps every experiment self-contained:
dataset variant, yaml config, output directory, train.log, and summary.

Set EXPERIMENT_DEADLINE="YYYY-MM-DD HH:MM:SS" to enforce a local hard stop.
Set EXPERIMENT_HEARTBEAT_SECONDS to control supervisor polling logs.
Generated dataset jsonl files are removed after training finishes because they
can be recreated from the raw competition jsonl files.
"""

from __future__ import annotations

import argparse
import json
import hashlib
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
HEARTBEAT_SECONDS = int(os.environ.get("EXPERIMENT_HEARTBEAT_SECONDS", "300"))


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

ITEMIC_RE = re.compile(r"<\|(?:prod|video|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


RUNS = [
    {
        "name": "v31_v29_live_specialist_r3_lr12e6_ep035",
        "dataset": "v29_live_specialist_r3",
        "model_path": str((OUTPUT_DIR / "v29_v19_user_world_guard_lr8e7_ep018").resolve()),
        "lr": "1.2e-6",
        "epochs": 0.35,
        "warmup": 0.02,
        "scheduler": "cosine",
        "seed": 202607311,
        "note": "v29 continuation and live-domain specialist. Use diverse evidence-grounded three-level CoT plus strict no-think direct answers for every live-target record, with item/user/general-rec guards. Goal: raise recommendation-live without losing v29's speed or other dimensions.",
    },
    {
        "name": "v32_v29_balanced_r3_draft_lr8e7_ep020",
        "dataset": "v29_balanced_r3_draft",
        "model_path": str((OUTPUT_DIR / "v29_v19_user_world_guard_lr8e7_ep018").resolve()),
        "lr": "8.0e-7",
        "epochs": 0.20,
        "warmup": 0.02,
        "scheduler": "cosine",
        "seed": 202607321,
        "note": "v29 continuation and balanced four-domain R3 specialist. Equalize live/ad/product/video target supervision and train concise individualized interest-evolution-task CoT plus route-correct direct answers. Goal: reduce video dominance and improve recommendation balance while retaining meaningful CoT.",
    },
]

DERIVED_RUNS = [
    {
        "name": "v33_task_arith_v29_live55_r325",
        "batch": "v31-v33",
        "method": "task_arithmetic",
        "model_path": str((OUTPUT_DIR / "v29_v19_user_world_guard_lr8e7_ep018").resolve()),
        "sources": [
            "v31_v29_live_specialist_r3_lr12e6_ep035",
            "v32_v29_balanced_r3_draft_lr8e7_ep020",
        ],
        "formula": "theta_v29 + 0.55*(theta_v31-theta_v29) + 0.25*(theta_v32-theta_v29)",
        "note": "CoT-native task arithmetic from the shared v29 base; no v7 source.",
        "output_dir": str((OUTPUT_DIR / "v33_task_arith_v29_live55_r325").resolve()),
    }
]


def source_runs() -> list[dict]:
    import source_data

    return source_data.all_source_runs()


def registry_entry(run: dict, batch: str) -> dict:
    entry = dict(run)
    entry["batch"] = entry.get("batch", batch)
    if "model" in entry:
        entry["model_path"] = entry.pop("model")
    if "configs" not in entry:
        if entry.get("stages"):
            entry["configs"] = [
                str((CONFIG_DIR / f"{entry['name']}_stage{index}.yaml").resolve())
                for index, _ in enumerate(entry["stages"], 1)
            ]
        elif entry.get("method") is None:
            entry["configs"] = [str((CONFIG_DIR / f"{entry['name']}.yaml").resolve())]
    entry.pop("config", None)
    return entry


def all_runs() -> list[dict]:
    runs = [registry_entry(run, "v31-v32") for run in RUNS]
    runs.extend(registry_entry(run, run.get("batch", "v34-v43")) for run in source_runs())
    runs.extend(dict(run) for run in DERIVED_RUNS)
    return sorted(runs, key=lambda item: int(re.match(r"v(\d+)", item["name"]).group(1)))


def write_run_registry() -> None:
    (EXP / "runs.json").write_text(json.dumps(all_runs(), ensure_ascii=False, indent=2), encoding="utf-8")


def merge_manifest(entries: dict) -> dict:
    path = DATA_DIR / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    source_meta = entries.get("_source")
    for name, value in entries.items():
        if not name.startswith("_"):
            manifest[name] = value
        elif name != "_source":
            manifest[name] = value
    if source_meta:
        manifest.setdefault("_sources", {})["official_explorer"] = source_meta
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def merge_results(results: list[dict]) -> list[dict]:
    path = LOG_DIR / "summary.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = {item["name"]: item for item in current}
    registry = {item["name"]: item for item in all_runs()}
    for result in results:
        item = dict(result)
        run = registry.get(item["name"], {})
        for key in ("batch", "dataset", "model_path", "lr", "epochs", "stages", "note", "method", "output_dir"):
            if key not in item and run.get(key) is not None:
                item[key] = run[key]
        item.setdefault("output_dir", str((OUTPUT_DIR / item["name"]).resolve()))
        merged[item["name"]] = item
    ordered = sorted(merged.values(), key=lambda item: int(re.match(r"v(\d+)", item["name"]).group(1)))
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return ordered


def write_unified_summary() -> None:
    manifest_path = DATA_DIR / "manifest.json"
    results_path = LOG_DIR / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    registry = {item["name"]: item for item in all_runs()}

    lines = [
        "# LLM4Rec Experiment Summary",
        "",
        f"Updated: {stamp()}",
        "",
        "## Dataset Variants",
    ]
    for name, meta in manifest.items():
        if name.startswith("_"):
            continue
        lines.append(
            f"- `{name}`: {meta.get('records', '?')} records, groups={meta.get('groups', {})}, "
            f"sha256={meta.get('sha256', '')}"
        )
    lines.extend([
        "",
        "## Runs",
        "| version | batch | method | base | dataset | lr | epochs | steps | train loss | eval loss | runtime min | status |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for result in results:
        run = registry.get(result["name"], {})
        base = result.get("model_path", run.get("model_path", ""))
        if base == "OpenOneRec/OneReason-0.8B-pretrain-competition":
            base = "official-pretrain"
        elif base:
            base = Path(base).name
        stages = result.get("stages", run.get("stages"))
        epochs = " + ".join(str(stage["epochs"]) for stage in stages) if stages else result.get("epochs", run.get("epochs", ""))
        loss = result.get("train_loss", "")
        loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
        runtime = result.get("wall_seconds", result.get("train_runtime"))
        runtime_text = f"{runtime / 60:.2f}" if isinstance(runtime, (int, float)) else ""
        eval_loss = result.get("eval_loss", "")
        eval_loss_text = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss)
        lines.append(
            f"| {result['name']} | {result.get('batch', run.get('batch', ''))} | {result.get('method', run.get('method', 'full'))} | {base} | "
            f"{result.get('dataset', run.get('dataset', ''))} | {result.get('lr', run.get('lr', ''))} | "
            f"{epochs} | {result.get('global_step', '')} | {loss_text} | {eval_loss_text} | {runtime_text} | {result.get('status', '')} |"
        )

    lines.extend(["", "## Derived Models"])
    for run in DERIVED_RUNS:
        lines.append(f"- `{run['name']}`: {run['formula']}. {run['note']}")
    lines.extend(["", "## Run Notes"])
    for name, run in registry.items():
        if run.get("note"):
            lines.append(f"- `{name}`: {run['note']}")
    lines.extend([
        "",
        "## v34-v43 Evaluation Order",
        "",
        "1. v34 anchor paper mixture",
        "2. v38 aggressive R3",
        "3. v39 perception-first curriculum",
        "4. v43 low-LR long-dose anchor",
        "5. v36 general guard",
        "6. v35 CoT-heavy R3",
        "7. v37 perception-heavy",
        "8. v42 user guard",
        "9. v41 original-data control",
        "10. v40 scratch high-risk model",
        "",
        "## v44-v48 Clean-Data Evaluation Order",
        "",
        "1. v48 fused high-confidence candidate",
        "2. v44 strict live specialist",
        "3. v45 domain-aware R3 mixture",
        "4. v47 clean world guard",
        "5. v46 clean user-evolution specialist",
        "",
        "## v49-v53 Fixed-Data Tuning Order",
        "",
        "1. v51 rank-64 LoRA (official score 0.9188)",
        "2. v50 lower-LR 2-epoch LoRA (0.8971)",
        "3. v49 exact public anchor (0.8834)",
        "4. v52 1-epoch full SFT (0.8614)",
        "5. v53 3-epoch full SFT (0.7862)",
        "",
        "## v54-v63 Advanced Evaluation Order",
        "",
        "1. v63 real-label user/video/live guard",
        "2. v58 LoRA+",
        "3. v56 rank-stabilized LoRA",
        "4. v61 v29 bridge at 5e-5",
        "5. v54 rank-96 interpolation",
        "6. v55 standard rank-128",
        "7. v62 conservative v29 bridge",
        "8. v57 DoRA",
        "9. v59 rank-64 dropout-zero control",
        "10. v60 doubled LoRA scaling",
        "",
        "## v64-v68 Focused Evaluation Order",
        "",
        "1. v66 rsLoRA + LoRA+ combination",
        "2. v67 real-user replay + LoRA+",
        "3. v65 rank-128 rank-aware LoRA+",
        "4. v64 conservative ratio-8 LoRA+",
        "5. v68 PiSSA + LoRA+ exploration",
        "",
        "## Notes",
        "- Ranking by train loss is only a training-health signal; use leaderboard evaluation for model selection.",
        "- Generated experiment JSONL and model outputs are ignored by git; JSONL can be rebuilt through the unified runner.",
    ])
    (LOG_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def clone_record(record: dict, *, input_text: str | None = None, output_text: str | None = None, variant: str = "") -> dict:
    cloned = dict(record)
    if input_text is not None:
        cloned["input"] = input_text
    if output_text is not None:
        cloned["output"] = output_text
    if variant:
        cloned["_variant"] = variant
    return cloned


def split_think_output(text: str) -> tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>\s*(.*)\Z", text or "", flags=re.S)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", (text or "").strip()


def empty_think_json_output(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return f"<think>\n</think>\n{payload}"


def short_think_output(thought: str, final: str) -> str:
    return f"<think>\n{thought.strip()}\n</think>\n{final.strip()}"


def prompt_with_route(input_text: str, route: str) -> str:
    token = "/no_think" if route == "no_think" else "/think"
    if "/no_think" in input_text:
        return input_text.replace("/no_think", token, 1)
    if "/think" in input_text:
        return input_text.replace("/think", token, 1)
    return input_text.rstrip() + token


def prompt_route(input_text: str) -> str:
    if "/no_think" in input_text:
        return "no_think"
    if "/think" in input_text:
        return "think"
    return "none"


def compact_json(value, *, cap_logic_events: bool = False):
    if isinstance(value, list):
        return dedupe_json_list(value)
    if isinstance(value, dict) and cap_logic_events:
        value = json.loads(json.dumps(value, ensure_ascii=False))
        logic_chain = value.get("logic_chain")
        if isinstance(logic_chain, dict) and isinstance(logic_chain.get("events"), list):
            logic_chain["events"] = logic_chain["events"][:5]
    return value


def final_only_output(record: dict, *, compact_user_json: bool = False, cap_logic_events: bool = False) -> str:
    _, final = split_think_output(record["output"])
    if compact_user_json:
        try:
            parsed = json.loads(final)
        except Exception:
            return final.strip()
        parsed = compact_json(parsed, cap_logic_events=cap_logic_events)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return final.strip()


def make_rec_no_think_augments(records: list[dict]) -> list[dict]:
    augments: list[dict] = []
    for record in records:
        final = final_only_output(record)
        if not final or not ITEMIC_RE.search(final):
            continue
        augments.append(
            clone_record(
                record,
                input_text=prompt_with_route(record["input"], "no_think"),
                output_text=final,
                variant="rec_no_think_direct_final",
            )
        )
    return augments


def rec_target_label(final: str) -> str:
    if "视频" in final:
        return "视频"
    if "广告" in final:
        return "广告"
    if "商品" in final:
        return "商品"
    if "主播" in final or "直播" in final:
        return "直播"
    token = ITEMIC_RE.search(final)
    return infer_item_domain(token.group(0)) if token else "目标内容"


def make_rec_short_think_augments(records: list[dict]) -> list[dict]:
    augments: list[dict] = []
    for record in records:
        final = final_only_output(record)
        if not final or not ITEMIC_RE.search(final):
            continue
        label = rec_target_label(final)
        thought = (
            f"先按近期、高频、深度互动和跨域重复信号归纳用户兴趣，再筛掉与历史语义弱相关的候选；"
            f"最终只输出最可能命中的{label} itemic 结果。"
        )
        augments.append(
            clone_record(
                record,
                input_text=prompt_with_route(record["input"], "think"),
                output_text=short_think_output(thought, final),
                variant="rec_short_think_final",
            )
        )
    return augments


def user_json_payload(record: dict, *, cap_logic_events: bool = False) -> str | None:
    final = final_only_output(record)
    try:
        parsed = json.loads(final)
    except Exception:
        return None
    parsed = compact_json(parsed, cap_logic_events=cap_logic_events)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def make_user_route_augments(records: list[dict], *, cap_logic_events: bool = False, add_missing_no_think: bool = False) -> list[dict]:
    augments: list[dict] = []
    for record in records:
        payload = user_json_payload(record, cap_logic_events=cap_logic_events)
        if payload is None:
            continue
        route = prompt_route(record["input"])
        is_logic = '"logic_chain"' in payload or "logic_chain" in record["input"]
        if route == "think":
            thought = "按时间顺序筛选关键交互，抽取兴趣变化和行为转化链路，最终只返回合法 JSON。"
            output = short_think_output(thought, payload)
        else:
            output = payload
        augments.append(
            clone_record(
                record,
                input_text=strict_user_prompt(record["input"]),
                output_text=output,
                variant="user_strict_logic" if is_logic else "user_strict_array",
            )
        )
        if add_missing_no_think and route != "no_think":
            augments.append(
                clone_record(
                    record,
                    input_text=strict_user_prompt(prompt_with_route(record["input"], "no_think")),
                    output_text=payload,
                    variant="user_extra_no_think_logic" if is_logic else "user_extra_no_think_array",
                )
            )
    return augments


def make_item_route_augments(records: list[dict]) -> list[dict]:
    augments: list[dict] = []
    for record in records:
        final = final_only_output(record)
        token_match = ITEMIC_RE.search(final)
        if not token_match:
            continue
        token = token_match.group(0)
        route = prompt_route(record["input"])
        if route == "think":
            domain = infer_item_domain(token)
            thought = f"提取描述中的核心品类、属性、使用场景和人群，匹配最接近的{domain}语义标识。"
            output = short_think_output(thought, token)
            variant = "item_short_think"
        else:
            output = token
            variant = "item_no_think_direct_final"
        augments.append(clone_record(record, output_text=output, variant=variant))
    return augments


def strict_user_prompt(input_text: str) -> str:
    note = "\n\n强约束：最终答案必须是一个可被 json.loads 解析的 JSON，不能包含 Markdown、解释、重复键或 JSON 外额外字符。"
    if "强约束：最终答案必须是一个可被 json.loads 解析的 JSON" in input_text:
        return input_text
    if "/no_think" in input_text:
        return input_text.replace("/no_think", f"{note}/no_think", 1)
    if "/think" in input_text:
        return input_text.replace("/think", f"{note}/think", 1)
    return input_text + note


def dedupe_json_list(values: list) -> list:
    seen = set()
    deduped = []
    for item in values:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def make_user_json_augments(records: list[dict]) -> list[dict]:
    augments: list[dict] = []
    for record in records:
        _, final = split_think_output(record["output"])
        try:
            parsed = json.loads(final)
        except Exception:
            continue
        if isinstance(parsed, list):
            parsed = dedupe_json_list(parsed)
            variant = "user_array_json_strict"
        elif isinstance(parsed, dict):
            variant = "user_logic_json_strict" if "logic_chain" in parsed or "logic_chain" in record["input"] else "user_object_json_strict"
        else:
            continue
        augments.append(
            clone_record(
                record,
                input_text=strict_user_prompt(record["input"]),
                output_text=empty_think_json_output(parsed),
                variant=variant,
            )
        )
    return augments


def normalize_cot_text(text: str) -> str:
    text = (text or "").replace("**", "")
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ：:\n\t")


def truncate_text(text: str, limit: int) -> str:
    text = normalize_cot_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ["。", "；", ";", "，", ","]:
        pos = cut.rfind(sep)
        if pos >= int(limit * 0.55):
            return cut[: pos + 1]
    return cut.rstrip() + "..."


def extract_stage(text: str, label: str, next_labels: list[str]) -> str:
    label_pattern = rf"(?:【\s*{re.escape(label)}\s*】|{re.escape(label)}[:：])"
    start_match = re.search(label_pattern, text)
    if not start_match:
        return ""
    start = start_match.end()
    end = len(text)
    for next_label in next_labels:
        next_pattern = rf"(?:【\s*{re.escape(next_label)}\s*】|{re.escape(next_label)}[:：])"
        next_match = re.search(next_pattern, text[start:])
        if next_match:
            end = min(end, start + next_match.start())
    return text[start:end]


def make_rec_cot_clean_augments(records: list[dict]) -> list[dict]:
    augments: list[dict] = []
    for record in records:
        think, final = split_think_output(record["output"])
        if not think or not final or not ITEMIC_RE.search(final):
            continue
        full = normalize_cot_text(think)
        interest = extract_stage(think, "兴趣归纳", ["行为模式", "预测总结"])
        evidence = extract_stage(think, "行为模式", ["预测总结"])
        summary = extract_stage(think, "预测总结", [])
        if not interest:
            interest = full[:700]
        if not evidence:
            evidence = full[700:1300] or "重点参考高频、近期、深度互动以及跨域重复出现的语义簇。"
        if not summary:
            summary = full[-520:] or "根据近期强兴趣和目标场景，选择最可能产生互动的 itemic token。"
        cleaned_output = (
            "<think>\n"
            f"【兴趣归纳】{truncate_text(interest, 620)}\n"
            f"【行为证据】{truncate_text(evidence, 520)}\n"
            f"【预测总结】{truncate_text(summary, 420)}\n"
            "</think>\n"
            f"{final}"
        )
        augments.append(clone_record(record, output_text=cleaned_output, variant="rec_cot_pattern_clean"))
    return augments


def make_rec_r3_draft_augments(records: list[dict]) -> list[dict]:
    """Compress each original recommendation trace without replacing its evidence."""
    augments: list[dict] = []
    for record in records:
        think, final = split_think_output(record["output"])
        if not think or not final or not ITEMIC_RE.search(final):
            continue
        full = normalize_cot_text(think)
        interest = extract_stage(think, "兴趣归纳", ["行为模式", "预测总结"])
        evolution = extract_stage(think, "行为模式", ["预测总结"])
        decision = extract_stage(think, "预测总结", [])
        if not interest:
            interest = full[:520]
        if not evolution:
            evolution = full[520:940] or full[-420:]
        if not decision:
            decision = full[-360:]
        output = (
            "<think>\n"
            f"【兴趣证据】{truncate_text(interest, 420)}\n"
            f"【演化关系】{truncate_text(evolution, 340)}\n"
            f"【目标判断】{truncate_text(decision, 260)}\n"
            "</think>\n"
            f"{final}"
        )
        augments.append(clone_record(record, output_text=output, variant="rec_r3_draft_evidence"))
    return augments


def infer_item_domain(token: str) -> str:
    if token.startswith("<|prod_begin|>"):
        return "商品"
    if token.startswith("<|video_begin|>"):
        return "视频"
    if token.startswith("<|ad_begin|>"):
        return "广告"
    if token.startswith("<|living_begin|>"):
        return "直播"
    return "内容"


def make_item_compact_cot_augments(records: list[dict]) -> list[dict]:
    augments: list[dict] = []
    for record in records:
        _, final = split_think_output(record["output"])
        token_match = ITEMIC_RE.search(final)
        if not token_match:
            continue
        token = token_match.group(0)
        domain = infer_item_domain(token)
        cleaned_output = (
            "<think>\n"
            f"提取描述中的核心品类、关键属性、使用场景、风格和目标人群，匹配最接近的{domain}语义标识。\n"
            "</think>\n"
            f"{token}"
        )
        augments.append(clone_record(record, output_text=cleaned_output, variant="item_compact_cot"))
    return augments


def sample_records(records: list[dict], count: int, seed: int) -> list[dict]:
    if count >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, count)


def itemic_domain(record: dict) -> str:
    final = final_only_output(record)
    token_match = ITEMIC_RE.search(final)
    return infer_item_domain(token_match.group(0)) if token_match else ""


def sample_domain(records: list[dict], domains: set[str], count: int, seed: int) -> list[dict]:
    filtered = [record for record in records if itemic_domain(record) in domains]
    return sample_records(filtered, count, seed)


def write_dataset(name: str, records: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    path = DATA_DIR / f"{name}.jsonl"
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as fp:
        for rec in shuffled:
            line = json.dumps(strip_internal(rec), ensure_ascii=False) + "\n"
            digest.update(line.encode("utf-8"))
            fp.write(line)
    groups = Counter(rec.get("_group", "unknown") for rec in shuffled)
    variants = Counter(rec.get("_variant", "raw") for rec in shuffled)
    return {
        "name": name,
        "path": str(path.resolve()),
        "records": len(shuffled),
        "groups": dict(groups),
        "variants": dict(variants),
        "sha256": digest.hexdigest(),
        "seed": seed,
    }


def prepare_datasets() -> dict[str, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grouped = load_raw_records()
    all_records = grouped["rec"] + grouped["item"] + grouped["user"]
    user_json_aug = make_user_json_augments(grouped["user"])
    user_logic_aug = [rec for rec in user_json_aug if rec.get("_variant") == "user_logic_json_strict"]
    rec_clean_aug = make_rec_cot_clean_augments(grouped["rec"])
    item_compact_aug = make_item_compact_cot_augments(grouped["item"])
    rec_no_think_aug = make_rec_no_think_augments(grouped["rec"])
    rec_short_think_aug = make_rec_short_think_augments(grouped["rec"])
    rec_r3_draft_aug = make_rec_r3_draft_augments(grouped["rec"])
    user_strict_aug = make_user_route_augments(grouped["user"], cap_logic_events=True, add_missing_no_think=False)
    user_strict_dual_aug = make_user_route_augments(grouped["user"], cap_logic_events=True, add_missing_no_think=True)
    item_route_aug = make_item_route_augments(grouped["item"])

    variants = {
        "v29_live_specialist_r3": (
            sample_domain(grouped["rec"], {"直播"}, 2000, 202607311)
            + sample_domain(rec_r3_draft_aug, {"直播"}, 2000, 202607312)
            + sample_domain(rec_no_think_aug, {"直播"}, 2000, 202607313)
            + sample_records(rec_r3_draft_aug, 1800, 202607314)
            + sample_records(rec_no_think_aug, 1800, 202607315)
            + sample_records(item_route_aug, 3200, 202607316)
            + sample_records(user_strict_dual_aug, 2400, 202607317)
        ),
        "v29_balanced_r3_draft": (
            sample_domain(rec_r3_draft_aug, {"直播"}, 1300, 202607321)
            + sample_domain(rec_r3_draft_aug, {"广告"}, 1300, 202607322)
            + sample_domain(rec_r3_draft_aug, {"商品"}, 1300, 202607323)
            + sample_domain(rec_r3_draft_aug, {"视频"}, 1300, 202607324)
            + sample_domain(rec_no_think_aug, {"直播"}, 1300, 202607325)
            + sample_domain(rec_no_think_aug, {"广告"}, 1300, 202607326)
            + sample_domain(rec_no_think_aug, {"商品"}, 1300, 202607327)
            + sample_domain(rec_no_think_aug, {"视频"}, 1300, 202607328)
            + sample_records(item_route_aug, 3200, 202607329)
            + sample_records(user_strict_dual_aug, 2400, 202607330)
        ),
    }
    manifest = {
        name: write_dataset(name, records, seed=202607110 + i)
        for i, (name, records) in enumerate(variants.items(), 1)
    }
    manifest["_notes"] = {
        "raw_counts": {k: len(v) for k, v in grouped.items()},
        "baseline_scores": {
            "total": 0.8184,
            "item": 0.1840,
            "user": [0.0442, 0.0375],
            "rec": [0.0672, 0.1054, 0.1344, 0.1089],
            "world": 0.1368,
        },
        "v07_scores": {
            "total": 0.8978,
            "item": 0.2146,
            "user": [0.0656, 0.0315],
            "rec": [0.0672, 0.1360, 0.1428, 0.1044],
            "world": 0.1357,
        },
        "v11_v14_scores": {
            "v11": {"total": 0.8545, "item": 0.2146, "user": [0.0301, 0.0416], "rec": [0.0672, 0.1156, 0.1386, 0.1089], "world": 0.1379},
            "v12": {"total": 0.8755, "item": 0.2146, "user": [0.0780, 0.0412], "rec": [0.0672, 0.1088, 0.1176, 0.1098], "world": 0.1383},
            "v13": {"total": 0.7743, "item": 0.1533, "user": [0.0447, 0.0431], "rec": [0.0480, 0.1020, 0.1386, 0.1071], "world": 0.1375},
            "v14": {"total": 0.6743, "item": 0.1533, "user": [0.0036, 0.0290], "rec": [0.0288, 0.0816, 0.1330, 0.1116], "world": 0.1335},
        },
        "v19_v24_scores": {
            "v19": {"total": 0.8855, "item": 0.2146, "user": [0.0855, 0.0337], "rec": [0.0768, 0.1020, 0.1344, 0.1143], "world": 0.1242, "eval_time_min": 48.11},
            "v21": {"total": 0.8800, "item": 0.2146, "user": [0.0881, 0.0347], "rec": [0.0576, 0.1122, 0.1358, 0.1098], "world": 0.1271, "eval_time_min": 47.61},
            "v22": {"total": 0.8217, "item": 0.2146, "user": [0.0413, 0.0228], "rec": [0.0576, 0.0986, 0.1372, 0.1116], "world": 0.1379, "eval_time_min": 69.83},
            "v23": {"total": 0.6990, "item": 0.1533, "user": [0.0058, 0.0211], "rec": [0.0384, 0.0884, 0.1484, 0.1071], "world": 0.1364, "eval_time_min": 75.92},
            "v24": {"total": 0.8364, "item": 0.2146, "user": [0.0733, 0.0365], "rec": [0.0480, 0.0952, 0.1288, 0.1080], "world": 0.1320, "eval_time_min": 64.35},
        },
        "v25_v27_scores": {
            "v25": {"total": 0.8793, "item": 0.2146, "user": [0.0835, 0.0342], "rec": [0.0672, 0.1088, 0.1358, 0.1098], "world": 0.1253, "eval_time_min": 50.48},
            "v26": {"total": 0.8804, "item": 0.2146, "user": [0.0895, 0.0344], "rec": [0.0768, 0.0986, 0.1330, 0.1116], "world": 0.1219, "eval_time_min": 45.73},
            "v27": {"total": 0.8563, "item": 0.2146, "user": [0.0733, 0.0419], "rec": [0.0576, 0.0986, 0.1344, 0.1062], "world": 0.1297, "eval_time_min": 70.63},
        },
        "v28_v30_scores": {
            "v28": {"total": 0.8748, "item": 0.2146, "user": [0.0886, 0.0350], "rec": [0.0576, 0.1122, 0.1330, 0.1107], "world": 0.1230, "eval_time_min": 51.70},
            "v29": {"total": 0.9038, "item": 0.2146, "user": [0.0886, 0.0346], "rec": [0.0768, 0.1054, 0.1400, 0.1125], "world": 0.1312, "eval_time_min": 45.30},
            "v30": {"total": 0.8846, "item": 0.2146, "user": [0.0895, 0.0338], "rec": [0.0672, 0.1020, 0.1316, 0.1143], "world": 0.1316, "eval_time_min": 46.51},
        },
        "augmentation_counts": {
            "user_json_aug": len(user_json_aug),
            "user_logic_aug": len(user_logic_aug),
            "rec_clean_aug": len(rec_clean_aug),
            "item_compact_aug": len(item_compact_aug),
            "rec_no_think_aug": len(rec_no_think_aug),
            "rec_short_think_aug": len(rec_short_think_aug),
            "rec_r3_draft_aug": len(rec_r3_draft_aug),
            "user_strict_aug": len(user_strict_aug),
            "user_strict_dual_aug": len(user_strict_dual_aug),
            "item_route_aug": len(item_route_aug),
        },
        "recipes": {
            "v29_live_specialist_r3": "v29 live-domain specialist: all live-target raw CoT, individualized three-level R3 draft CoT, and no-think direct answers, plus balanced general-rec, item, and user guards.",
            "v29_balanced_r3_draft": "v29 four-domain cognition specialist: equal target counts for live/ad/product/video in both individualized R3 draft-thinking and direct no-thinking routes, plus item and user guards.",
        },
        "cot_policy": "Preserve /think reasoning supervision and do not use v7_final_only as a CoT training base. For /no_think prompts, train pure final answers without generated <think> tags. This is route-specific behavior, not global CoT removal.",
        "eval_observations": {
            "v07": "best local score so far: total=0.8978, eval_time≈47.3min; fast final outputs likely help.",
            "v15": "best CoT-preserving score so far: total=0.8778, eval_time≈70.1min; logs show repeated tokens, JSON shell errors, prompt leakage, and verbose /no_think outputs.",
            "v19": "best CoT-native continuation so far: total=0.8855, eval_time≈48.1min; user1 and rec4 improved but world dropped.",
            "v20": "v7 final-only continuation with CoT restore failed as a CoT route: total=0.8527, item fell to 0.1840; do not use v7 as future CoT base.",
            "v22": "scratch official-base 3 epoch clean CoT underperformed: total=0.8217; item/world preserved but user and rec2 are weak.",
            "v23": "scratch official-base 5 epoch low-LR guard failed badly: total=0.6990; item/user collapse suggests long scratch SFT is not viable with current data mix.",
            "v24": "v15 light repair is best among v22-v24 but still only total=0.8364; user2 improves but rec/world do not recover.",
            "v25": "v19 product-heavy repair did not beat v19: total=0.8793. Logs show heavy repeated product tokens and repeated short-think phrases; avoid this over-sampling pattern.",
            "v26": "best latest batch and fastest eval: total=0.8804, eval_time≈45.7min, best user1/rec1. It is useful as a base, but rec2/world dropped.",
            "v27": "v12 product/ad repair kept user2/world relatively better but was slow and template-heavy: total=0.8563, eval_time≈70.6min. Do not continue this exact direction.",
            "v28": "v26 no-template repair regressed to total=0.8748 and slowed to 51.7min; broad continuation from v26 did not recover world/rec balance.",
            "v29": "new best CoT-native model: total=0.9038, eval_time≈45.3min. Strong item/user1/ad/product/world, with live recommendation (0.1054) the clearest remaining gap. Logs still show occasional no-think tag leakage and a repeated generic live-reasoning sentence.",
            "v30": "three-source soup reached total=0.8846, below v29. It retained speed/world but diluted recommendation scores; do not repeat broad checkpoint averaging.",
            "v16": "CoT pattern rewrite failed: total=0.7912; logs show malformed user JSON and fragmented recommendation reasoning.",
            "v18": "low-LR mixed replay from v12 failed: total=0.8340; item/world dropped and rec outputs mixed text/itemic/think tags.",
        },
    }
    merge_manifest(manifest)
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
    model_path = run.get("model_path", "OpenOneRec/OneReason-0.8B-pretrain-competition")
    return f"""### model
model_name_or_path: {model_path}
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
    write_run_registry()


def write_experiment_recipes(manifest: dict[str, dict]) -> None:
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for run in RUNS:
        out = OUTPUT_DIR / run["name"]
        out.mkdir(parents=True, exist_ok=True)
        recipe = {
            "created_at": stamp(),
            "run": run,
            "dataset_manifest": manifest.get(run["dataset"], {}),
            "dataset_recipe": manifest.get("_notes", {}).get("recipes", {}).get(run["dataset"], ""),
            "cot_policy": manifest.get("_notes", {}).get("cot_policy", ""),
            "raw_counts": manifest.get("_notes", {}).get("raw_counts", {}),
            "eval_observations": manifest.get("_notes", {}).get("eval_observations", {}),
            "script": str(Path(__file__).resolve()),
            "script_sha256": script_sha,
            "deadline": deadline_label(),
            "reproduce": {
                "prepare_command": f"{VENV}/bin/python {Path(__file__).resolve()} --single {run['name']} --gpu <gpu> --prepare-only",
                "train_command": f"{VENV}/bin/python {Path(__file__).resolve()} --single {run['name']} --gpu <gpu>",
            },
        }
        (out / "experiment_recipe.json").write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
        "model_path": run.get("model_path", "OpenOneRec/OneReason-0.8B-pretrain-competition"),
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
    merge_manifest(manifest)
    merge_results(results)
    write_unified_summary()
    summary_md = LOG_DIR / "summary.md"

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
            "- {name}: base={base}, dataset={dataset}, lr={lr}, epochs={epochs}, scheduler={scheduler}, "
            "train_loss={loss}, last_logged_loss={last}, runtime={runtime}, output={out}\n"
            "  note={note}\n".format(
                name=item.get("name"),
                base=item.get("model_path"),
                dataset=item.get("dataset"),
                lr=item.get("lr"),
                epochs=item.get("epochs"),
                scheduler=item.get("scheduler"),
                loss=item.get("train_loss"),
                last=item.get("last_logged_loss"),
                runtime=item.get("train_runtime"),
                out=item.get("output_dir"),
                note=item.get("note"),
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


def run_legacy_experiments(*, single: str | None = None, gpu: int = 0, prepare_only: bool = False) -> int:
    for path in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    selected_runs = RUNS if single is None else [next(run for run in RUNS if run["name"] == single)]

    append(
        TOP_LOG,
        f"\n\n[{stamp()}] start experiment preparation; deadline={deadline_label()}\n",
    )
    manifest = prepare_datasets()
    update_dataset_info(manifest)
    prepare_configs()
    write_experiment_recipes(manifest)

    status_path = LOG_DIR / "supervisor_status.jsonl"
    append(
        status_path,
        json.dumps(
            {
                "time": stamp(),
                "event": "prepared",
                "batch": "v31-v32",
                "runs": [r["name"] for r in selected_runs],
                "deadline": deadline_label(),
                "heartbeat_seconds": HEARTBEAT_SECONDS,
                "manifest": str((DATA_DIR / "manifest.json").resolve()),
            },
            ensure_ascii=False,
        )
        + "\n",
    )

    if prepare_only or os.environ.get("EXPERIMENT_PREPARE_ONLY", "").strip() == "1":
        append(
            status_path,
            json.dumps(
                {
                    "time": stamp(),
                    "event": "prepare_only_exit",
                    "manifest": str((DATA_DIR / "manifest.json").resolve()),
                    "configs": str(CONFIG_DIR.resolve()),
                },
                ensure_ascii=False,
            )
            + "\n",
        )
        append(TOP_LOG, f"[{stamp()}] prepare-only completed; no training launched.\n")
        return 0

    queue = list(selected_runs)
    results: list[dict] = []
    lock = threading.Lock()
    gpus = [gpu] if single else [0, 1]
    threads = [
        threading.Thread(target=worker, args=(worker_gpu, queue, lock, results, status_path), daemon=False)
        for worker_gpu in gpus
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
        slept = 0
        while slept < HEARTBEAT_SECONDS and any(thread.is_alive() for thread in threads):
            interval = min(30, HEARTBEAT_SECONDS - slept)
            time.sleep(interval)
            slept += interval
    for thread in threads:
        thread.join()

    run_order = {run["name"]: i for i, run in enumerate(selected_runs)}
    results.sort(key=lambda x: run_order.get(x.get("name", ""), 999))
    write_summary(results, manifest)
    cleanup_generated_datasets(status_path)
    append(status_path, json.dumps({"time": stamp(), "event": "all_done"}, ensure_ascii=False) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    trainable = RUNS + source_runs()
    parser = argparse.ArgumentParser(description="Unified LLM4Rec experiment runner")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--batch",
        choices=["v31-v32", "v34-v43", "v44-v48", "v49-v53", "v54-v63", "v64-v68"],
    )
    selection.add_argument("--single", choices=[run["name"] for run in trainable])
    selection.add_argument("--list", action="store_true", help="list registered experiments without training")
    selection.add_argument("--rebuild-index", action="store_true", help="rebuild shared metadata files without preparing data or training")
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0, help="GPU used by --single")
    parser.add_argument("--prepare-only", action="store_true", help="generate data/configs but do not train")
    parser.add_argument(
        "--reuse-prepared", action="store_true",
        help="train v44-v48 from already hash-verified generated JSONL without rebuilding it",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list:
        for run in all_runs():
            print(f"{run['name']}\t{run.get('batch', '')}\t{run.get('method', 'train')}")
        return 0
    if args.rebuild_index:
        write_run_registry()
        write_unified_summary()
        return 0
    if not args.batch and not args.single:
        parser.print_help()
        return 0

    source_names = {run["name"] for run in source_runs()}
    clean_names = {run["name"] for run in source_runs() if run.get("batch") == "v44-v48"}
    tuning_names = {run["name"] for run in source_runs() if run.get("batch") == "v49-v53"}
    advanced_names = {run["name"] for run in source_runs() if run.get("batch") == "v54-v63"}
    focused_names = {run["name"] for run in source_runs() if run.get("batch") == "v64-v68"}
    if args.batch == "v64-v68" or args.single in focused_names:
        import source_data

        return source_data.run_focused_experiments(
            single=args.single,
            gpu=args.gpu,
            prepare_only=args.prepare_only,
        )
    if args.batch == "v54-v63" or args.single in advanced_names:
        import source_data

        return source_data.run_advanced_experiments(
            single=args.single,
            gpu=args.gpu,
            prepare_only=args.prepare_only,
        )
    if args.batch == "v49-v53" or args.single in tuning_names:
        import source_data

        return source_data.run_tuning_experiments(
            single=args.single,
            gpu=args.gpu,
            prepare_only=args.prepare_only,
        )
    if args.batch == "v44-v48" or args.single in clean_names:
        import source_data

        return source_data.run_clean_experiments(
            single=args.single,
            gpu=args.gpu,
            prepare_only=args.prepare_only,
            reuse_prepared=args.reuse_prepared,
        )
    if args.batch == "v34-v43" or args.single in source_names:
        import source_data

        return source_data.run_source_experiments(
            single=args.single,
            gpu=args.gpu,
            prepare_only=args.prepare_only,
        )
    return run_legacy_experiments(single=args.single, gpu=args.gpu, prepare_only=args.prepare_only)


if __name__ == "__main__":
    raise SystemExit(main())
