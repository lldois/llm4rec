"""High-precision data curation for the v44-v48 LLM4Rec experiments.

The previous source-data batch favored coverage.  This module favors label
precision: strict temporal holdouts, high-value targets, evidence-grounded
reasoning, canonical joins, route validation, and aggressive deduplication.
Generated JSONL files are still owned and cleaned up by ``source_data.py``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pads

import run_experiments as legacy


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "Explorer_LLM_Rec_Competition"
GENERAL_DIR = SOURCE / "OneReason_General"
PROFILE_DIR = SOURCE / "OneReason_UserProfile"
PID2SID_DIR = SOURCE / "OneReason_Pid2Sid"
PID2CAPTION_DIR = SOURCE / "OneReason_Pid2Caption"
PID2TAG_DIR = SOURCE / "OneReason_Pid2Tag"

SEED = 202607160
DAY_MS = 86_400_000
ITEMIC_RE = re.compile(r"<\|(?:prod|video|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
THINK_RE = re.compile(r"\s*<think>(.*?)</think>\s*(.*?)\s*\Z", re.S)
ZH_RE = re.compile(r"[\u4e00-\u9fff]")

DOMAIN_PREFIX = {
    "video/video": "video",
    "video/ad": "ad",
    "goods": "prod",
    "live": "living",
}
DOMAIN_CN = {
    "video/video": "视频",
    "video/ad": "广告",
    "goods": "商品",
    "live": "直播",
}
DOMAIN_SHORT = {
    "video/video": "video",
    "video/ad": "ad",
    "goods": "goods",
    "live": "live",
}


def stable_u64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def safe_list(value) -> list:
    return value if isinstance(value, list) else []


def number(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def integer(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def value_at(row: dict, field: str, index: int, default=None):
    values = safe_list(row.get(field))
    return values[index] if index < len(values) else default


def route_prompt(text: str, route: str) -> str:
    return legacy.prompt_with_route(text, route)


def record(instruction: str, prompt: str, output: str, group: str, variant: str, **meta) -> dict:
    value = {
        "instruction": instruction,
        "input": prompt,
        "output": output,
        "history": [],
        "_group": group,
        "_variant": variant,
    }
    value.update(meta)
    return value


def sid_token(domain: str, values) -> str | None:
    values = safe_list(values)
    if domain not in DOMAIN_PREFIX or len(values) != 3:
        return None
    parts = []
    for value in values:
        numeric = number(value, -1)
        if numeric < 0 or not numeric.is_integer():
            return None
        parts.append(int(numeric))
    return (
        f"<|{DOMAIN_PREFIX[domain]}_begin|>"
        f"<s_a_{parts[0]}><s_b_{parts[1]}><s_c_{parts[2]}>"
    )


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    if isinstance(content, list):
        return "".join(content_text(item) for item in content)
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def template_signature(prompt: str, final: str) -> str:
    value = normalize_key(prompt + "\n" + final)
    value = ITEMIC_RE.sub("<item>", value)
    value = re.sub(r"\d+(?:\.\d+)?", "<n>", value)
    value = re.sub(r"[a-f0-9]{12,}", "<id>", value)
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()


def is_choice_answer(prompt: str, final: str) -> bool:
    final = final.strip()
    has_options = len(re.findall(r"(?:^|\n)\s*[A-D][.、:：]", prompt)) >= 2
    return has_options and bool(re.fullmatch(r"(?:正确答案(?:是|为)?\s*)?[（(]?[A-D][）)]?[。.]?", final))


def build_clean_general_pools(limit: int = 40_000) -> tuple[dict[str, list[dict]], dict]:
    pools = {
        "general_cn_direct": [],
        "general_cn_think": [],
        "general_other_direct": [],
        "general_other_think": [],
    }
    stats = Counter()
    seen_pairs = set()
    signature_counts = Counter()
    source_counts = Counter()
    banned = (
        "<tool_call>", "<tool_response>", '"tool_calls"', "<|python_tag|>",
        "<response>", "<analysis>", "environment initialized", "sandbox initialized",
    )
    scanner = pads.dataset(GENERAL_DIR, format="parquet").scanner(
        columns=["source", "messages"], batch_size=1024, use_threads=True
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            stats["scanned"] += 1
            raw = row.get("messages")
            if not raw:
                stats["missing_messages"] += 1
                continue
            try:
                messages = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                stats["bad_json"] += 1
                continue
            if not isinstance(messages, list):
                stats["bad_schema"] += 1
                continue
            cleaned = []
            malformed = False
            for message in messages:
                if not isinstance(message, dict):
                    malformed = True
                    break
                role = message.get("role")
                if role == "human":
                    role = "user"
                if role not in {"system", "user", "assistant"}:
                    malformed = True
                    break
                cleaned.append((role, content_text(message.get("content")).strip()))
            if malformed:
                stats["unsupported_role"] += 1
                continue
            systems = [text for role, text in cleaned if role == "system"]
            turns = [(role, text) for role, text in cleaned if role != "system"]
            if len(turns) != 2 or turns[0][0] != "user" or turns[1][0] != "assistant":
                stats["not_strict_single_turn"] += 1
                continue
            prompt, answer = turns[0][1], turns[1][1]
            system = systems[0] if systems else ""
            match = THINK_RE.fullmatch(answer)
            if not match:
                stats["missing_cot_shell"] += 1
                continue
            thought, final = match.group(1).strip(), match.group(2).strip()
            joined = system + prompt + answer
            lowered = joined.lower()
            if any(token in lowered for token in banned):
                stats["banned_format"] += 1
                continue
            short_choice = is_choice_answer(prompt, final)
            if not (8 <= len(prompt) <= 3000 and 50 <= len(thought) <= 2500 and len(final) <= 1500):
                stats["length_reject"] += 1
                continue
            if len(final) < 12 and not short_choice:
                stats["short_final"] += 1
                continue
            if len(joined) > 7000 or answer.count("<think>") != 1 or answer.count("</think>") != 1:
                stats["shell_reject"] += 1
                continue
            pair = (normalize_key(prompt), normalize_key(final))
            if pair in seen_pairs:
                stats["exact_duplicate"] += 1
                continue
            signature = template_signature(prompt, final)
            if signature_counts[signature] >= 2:
                stats["template_duplicate"] += 1
                continue
            source = normalize_text(row.get("source") or "unknown")[:120]
            if source_counts[source] >= 4000:
                stats["source_cap"] += 1
                continue
            seen_pairs.add(pair)
            signature_counts[signature] += 1
            source_counts[source] += 1
            is_cn = len(ZH_RE.findall(prompt + final)) >= 8
            prefix = "general_cn" if is_cn else "general_other"
            pools[f"{prefix}_direct"].append(record(
                system, route_prompt(prompt, "no_think"), final,
                "general_clean", "strict_single_turn_direct", _source=source,
            ))
            pools[f"{prefix}_think"].append(record(
                system, route_prompt(prompt, "think"),
                f"<think>\n{thought}\n</think>\n{final}",
                "general_clean", "strict_single_turn_cot", _source=source,
            ))
            stats["accepted_cn" if is_cn else "accepted_other"] += 1
            if len(seen_pairs) >= limit:
                break
        if len(seen_pairs) >= limit:
            break
    stats["unique_pairs"] = len(seen_pairs)
    return pools, {"filter_counts": dict(stats), "source_counts": dict(source_counts.most_common(30))}


def suspicious_thought_end(text: str) -> bool:
    text = text.rstrip()
    return not text or text[-1] in "(（、，,/:：-" or text.endswith(("例如", "比如", "包括"))


def compact_piece(text: str, limit: int) -> str:
    value = legacy.normalize_cot_text(text)
    if len(value) <= limit:
        return value
    cut = value[:limit]
    positions = [cut.rfind(mark) for mark in "。；!?！？"]
    position = max(positions)
    return cut[: position + 1] if position >= int(limit * 0.55) else cut.rstrip() + "。"


def extract_old_rec_thought(thought: str) -> str | None:
    interest = legacy.extract_stage(thought, "兴趣归纳", ["行为模式", "预测总结"])
    evolution = legacy.extract_stage(thought, "行为模式", ["预测总结"])
    decision = legacy.extract_stage(thought, "预测总结", [])
    if not interest or not evolution or not decision:
        return None
    return (
        f"【画像抽象】{compact_piece(interest, 260)}\n"
        f"【兴趣展开】{compact_piece(evolution, 240)}\n"
        f"【转移判断】{compact_piece(decision, 180)}"
    )


def domain_from_token(token: str) -> str:
    if token.startswith("<|video_begin|>"):
        return "video"
    if token.startswith("<|ad_begin|>"):
        return "ad"
    if token.startswith("<|prod_begin|>"):
        return "goods"
    if token.startswith("<|living_begin|>"):
        return "live"
    return "unknown"


def build_clean_old_pools() -> tuple[dict[str, list[dict]], dict]:
    raw = legacy.load_raw_records()
    pools: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()

    # Original recommendation data contains many targets per identical history.
    # Keep at most one target per history and domain, deterministically.
    rec_candidates: dict[tuple[str, str], tuple[int, dict, str, str]] = {}
    for value in raw["rec"]:
        thought, final = legacy.split_think_output(value["output"])
        targets = ITEMIC_RE.findall(final)
        if len(targets) != 1:
            stats["rec_bad_target"] += 1
            continue
        target = targets[0]
        domain = domain_from_token(target)
        prompt_tokens = set(ITEMIC_RE.findall(value["input"]))
        thought_tokens = ITEMIC_RE.findall(thought)
        if target in value["input"] or target in thought:
            stats["rec_target_leakage"] += 1
            continue
        if not (220 <= len(thought) <= 2200) or suspicious_thought_end(thought):
            stats["rec_bad_thought"] += 1
            continue
        if len(set(thought_tokens)) < 2 or any(token not in prompt_tokens for token in thought_tokens):
            stats["rec_ungrounded_thought"] += 1
            continue
        cleaned_thought = extract_old_rec_thought(thought)
        if not cleaned_thought or target in cleaned_thought:
            stats["rec_missing_stages"] += 1
            continue
        key = (normalize_key(value["input"]), domain)
        rank = stable_u64(int(hashlib.blake2b(target.encode(), digest_size=8).hexdigest(), 16))
        candidate = (rank, value, target, cleaned_thought)
        if key not in rec_candidates or rank < rec_candidates[key][0]:
            rec_candidates[key] = candidate
    for _, value, target, thought in rec_candidates.values():
        domain = domain_from_token(target)
        system = value["instruction"]
        pools[f"old_rec_{domain}_direct"].append(record(
            system, route_prompt(value["input"], "no_think"), target,
            "old_rec_clean", f"old_rec_{domain}_direct", _domain=domain,
        ))
        pools[f"old_rec_{domain}_think"].append(record(
            system, route_prompt(value["input"], "think"), f"<think>\n{thought}\n</think>\n{target}",
            "old_rec_clean", f"old_rec_{domain}_cot", _domain=domain,
        ))
    stats["rec_selected_histories"] = len(rec_candidates)

    seen_user = set()
    for value in raw["user"]:
        thought, final = legacy.split_think_output(value["output"])
        try:
            payload = json.loads(final)
        except Exception:
            stats["user_bad_json"] += 1
            continue
        if isinstance(payload, list):
            deduped = legacy.dedupe_json_list(payload)
            kind = "array"
            payload = deduped
        elif isinstance(payload, dict):
            kind = "logic" if "logic_chain" in payload or "logic_chain" in value["input"] else "object"
        else:
            stats["user_bad_type"] += 1
            continue
        final_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        key = (normalize_key(value["input"]), final_json)
        if key in seen_user:
            stats["user_duplicate"] += 1
            continue
        seen_user.add(key)
        strict_prompt = legacy.strict_user_prompt(route_prompt(value["input"], "no_think"))
        pools[f"old_user_{kind}_direct"].append(record(
            value["instruction"], strict_prompt, final_json,
            "old_user_clean", f"old_user_{kind}_direct",
        ))
        if thought and 800 <= len(thought) <= 2200 and not suspicious_thought_end(thought):
            cleaned = compact_piece(thought, 900)
            pools[f"old_user_{kind}_think"].append(record(
                value["instruction"],
                legacy.strict_user_prompt(route_prompt(value["input"], "think")),
                f"<think>\n{cleaned}\n</think>\n{final_json}",
                "old_user_clean", f"old_user_{kind}_cot",
            ))
    stats["user_selected"] = len(seen_user)

    seen_item = set()
    for value in raw["item"]:
        thought, final = legacy.split_think_output(value["output"])
        targets = ITEMIC_RE.findall(final)
        if len(targets) != 1:
            stats["item_bad_target"] += 1
            continue
        target = targets[0]
        key = (normalize_key(value["input"]), target)
        if key in seen_item:
            stats["item_duplicate"] += 1
            continue
        seen_item.add(key)
        domain = domain_from_token(target)
        pools[f"old_item_{domain}_direct"].append(record(
            value["instruction"], route_prompt(value["input"], "no_think"), target,
            "old_item_clean", f"old_item_{domain}_direct", _domain=domain,
        ))
        if thought and 30 <= len(thought) <= 700 and target not in thought and not suspicious_thought_end(thought):
            pools[f"old_item_{domain}_think"].append(record(
                value["instruction"], route_prompt(value["input"], "think"),
                f"<think>\n{compact_piece(thought, 360)}\n</think>\n{target}",
                "old_item_clean", f"old_item_{domain}_cot", _domain=domain,
            ))
    stats["item_selected"] = len(seen_item)
    return dict(pools), {"filter_counts": dict(stats), "pool_counts": {k: len(v) for k, v in pools.items()}}


def live_timestamp(value) -> int | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{8}", text):
        return None
    try:
        parsed = datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def add_event(events: list[dict], domain: str, pid, action: str, ts, strength: float) -> None:
    pid_value = integer(pid)
    ts_value = integer(ts)
    if pid_value is None or ts_value is None or ts_value <= 0 or strength <= 0:
        return
    events.append({
        "key": (domain, pid_value),
        "domain": domain,
        "action": action,
        "ts": ts_value,
        "strength": round(float(strength), 3),
        "repeat": 1,
    })


PROFILE_COLUMNS = [
    "video_sampled_pid_list", "video_ts_list", "video_neg_feedback_list", "video_like_list",
    "video_comment_list", "video_forward_list", "video_collect_list", "video_watch_time_list",
    "video_play_done_list", "video_duration_list", "video_history_sampled_pid_list",
    "video_history_ts_list", "video_history_like_list", "video_history_comment_list",
    "video_history_forward_list", "video_history_collect_list", "video_history_play_done_list",
    "video_history_neg_feedback_list", "video_history_watch_time_list", "video_history_duration_list",
    "ec_time_ms", "ec_item_id_list", "ec_cvr_label_list", "ec_time_ms_list",
    "ec_colossus_rs_item_id_list", "ec_colossus_rs_lagv1_list", "ec_colossus_rs_lagv2_list",
    "ec_colossus_rs_is_click_list", "ec_colossus_rs_is_cart_list", "ec_colossus_rs_is_buy_list",
    "ec_good_click_item_id_list_extend", "ec_trunc_clk_lag", "ec_good_order_item_id_list_extend",
    "ec_trunc_buy_lag", "live_hist_author_id_list", "live_hist_timestamp_list",
    "live_hist_follow_author_cnt_list", "live_hist_comment_cnt_list", "live_hist_like_cnt_list",
    "live_hist_valid_play_duration_list", "outer_loop_history_action_pid_list_pos",
    "outer_loop_history_action_pid_list_pos_ts", "outer_loop_history_action_pid_list_click",
    "outer_loop_history_action_pid_list_click_ts", "outer_loop_history_action_pid_list_click_type",
    "outer_loop_deep_target_pid", "outer_loop_deep_target_pid_ts",
]


def compact_clean_profile(row: dict, profile_id: int) -> dict | None:
    events: list[dict] = []
    targets: dict[str, dict] = {}

    # Video target: a later sampled item with an explicit interaction or strong completion.
    video_candidates = []
    for i, pid in enumerate(safe_list(row.get("video_sampled_pid_list"))):
        ts = integer(value_at(row, "video_ts_list", i))
        if pid is None or ts is None:
            continue
        neg = number(value_at(row, "video_neg_feedback_list", i))
        watch = number(value_at(row, "video_watch_time_list", i))
        duration = number(value_at(row, "video_duration_list", i))
        completion = watch / duration if duration > 0 else 0
        score = (
            3 * number(value_at(row, "video_like_list", i))
            + 4 * number(value_at(row, "video_comment_list", i))
            + 4 * number(value_at(row, "video_forward_list", i))
            + 5 * number(value_at(row, "video_collect_list", i))
            + 2 * number(value_at(row, "video_play_done_list", i))
            + (1 if completion >= 0.75 else 0)
            - 6 * neg
        )
        if score >= 2:
            video_candidates.append((ts, score, int(pid)))
    if video_candidates:
        ts, score, pid = max(video_candidates, key=lambda value: (value[0], value[1]))
        targets["video/video"] = {"key": ("video/video", pid), "ts": ts, "strength": score}

    video_history = safe_list(row.get("video_history_sampled_pid_list"))
    for i, pid in enumerate(video_history):
        if pid is None or number(value_at(row, "video_history_neg_feedback_list", i)) > 0:
            continue
        labels = []
        strength = 0.0
        for field, label, weight in [
            ("video_history_like_list", "点赞", 3), ("video_history_comment_list", "评论", 4),
            ("video_history_forward_list", "转发", 4), ("video_history_collect_list", "收藏", 5),
            ("video_history_play_done_list", "完播", 2),
        ]:
            if number(value_at(row, field, i)) > 0:
                labels.append(label)
                strength += weight
        watch = number(value_at(row, "video_history_watch_time_list", i))
        duration = number(value_at(row, "video_history_duration_list", i))
        if duration > 0 and watch / duration >= 0.75:
            labels.append("长播")
            strength += 1
        if labels:
            add_event(events, "video/video", pid, "/".join(dict.fromkeys(labels)), value_at(row, "video_history_ts_list", i), strength)

    # Product target: the most recent unambiguous click, preferring a clicked item that was ordered.
    now_ms = integer(row.get("ec_time_ms"))
    click_ids = safe_list(row.get("ec_good_click_item_id_list_extend"))
    click_lags = safe_list(row.get("ec_trunc_clk_lag"))
    order_ids = {int(pid) for pid in safe_list(row.get("ec_good_order_item_id_list_extend")) if pid is not None}
    click_candidates = []
    if now_ms:
        for i, pid in enumerate(click_ids):
            lag = number(click_lags[i] if i < len(click_lags) else None, -1)
            if pid is None or lag < 0:
                continue
            pid_value = int(pid)
            click_candidates.append((lag, 8.0 if pid_value in order_ids else 2.0, pid_value))
            add_event(events, "goods", pid_value, "点击", now_ms - int(lag * DAY_MS), 2)
        order_lags = safe_list(row.get("ec_trunc_buy_lag"))
        for i, pid in enumerate(safe_list(row.get("ec_good_order_item_id_list_extend"))):
            lag = number(order_lags[i] if i < len(order_lags) else None, -1)
            if pid is not None and lag >= 0:
                add_event(events, "goods", pid, "购买", now_ms - int(lag * DAY_MS), 8)
    if click_candidates:
        min_lag = min(value[0] for value in click_candidates)
        latest = {value[2]: value for value in click_candidates if value[0] == min_lag}
        ordered_latest = [value for value in latest.values() if value[1] >= 8]
        chosen = ordered_latest[0] if len(ordered_latest) == 1 else next(iter(latest.values())) if len(latest) == 1 else None
        if chosen and now_ms:
            lag, score, pid = chosen
            targets["goods"] = {"key": ("goods", pid), "ts": now_ms - int(lag * DAY_MS), "strength": score}

    for i, pid in enumerate(safe_list(row.get("ec_item_id_list"))):
        label = number(value_at(row, "ec_cvr_label_list", i))
        add_event(
            events, "goods", pid, "购买" if label > 0 else "点击",
            value_at(row, "ec_time_ms_list", i), 6 if label > 0 else 1,
        )
    if now_ms:
        for i, pid in enumerate(safe_list(row.get("ec_colossus_rs_item_id_list"))):
            lag = number(value_at(row, "ec_colossus_rs_lagv1_list", i), -1)
            if lag < 0:
                lag = number(value_at(row, "ec_colossus_rs_lagv2_list", i), -1)
            if lag < 0:
                continue
            labels = []
            strength = 0.0
            for field, label, weight in [
                ("ec_colossus_rs_is_buy_list", "购买", 8),
                ("ec_colossus_rs_is_cart_list", "加购", 5),
                ("ec_colossus_rs_is_click_list", "点击", 2),
            ]:
                if number(value_at(row, field, i)) > 0:
                    labels.append(label)
                    strength += weight
            if labels:
                add_event(events, "goods", pid, "/".join(labels), now_ms - int(lag * DAY_MS), strength)

    # Live target: latest unambiguous strong interaction. Same-day ties are dropped.
    live_candidates = []
    for i, pid in enumerate(safe_list(row.get("live_hist_author_id_list"))):
        ts = live_timestamp(value_at(row, "live_hist_timestamp_list", i))
        if pid is None or ts is None:
            continue
        follow = number(value_at(row, "live_hist_follow_author_cnt_list", i))
        comment = number(value_at(row, "live_hist_comment_cnt_list", i))
        like = number(value_at(row, "live_hist_like_cnt_list", i))
        duration_ms = number(value_at(row, "live_hist_valid_play_duration_list", i))
        labels = []
        if follow > 0:
            labels.append("关注")
        if comment > 0:
            labels.append("评论")
        if like > 0:
            labels.append("点赞")
        if duration_ms >= 60_000:
            labels.append("深度观看")
        score = 6 * min(follow, 1) + 4 * min(comment, 2) + 3 * min(like, 2) + min(duration_ms / 60_000, 3)
        if labels:
            add_event(events, "live", pid, "/".join(labels), ts, score)
        if follow > 0 or comment >= 2 or like >= 3:
            live_candidates.append((ts, score, int(pid)))
    if live_candidates:
        latest_ts = max(value[0] for value in live_candidates)
        latest = sorted((value for value in live_candidates if value[0] == latest_ts), key=lambda value: value[1], reverse=True)
        if len(latest) == 1 or latest[0][1] - latest[1][1] >= 2:
            ts, score, pid = latest[0]
            targets["live"] = {"key": ("live", pid), "ts": ts, "strength": score}

    # Ad target: deep target or an explicit conversion/payment, never a plain click.
    ad_high_value = []
    for i, pid in enumerate(safe_list(row.get("outer_loop_deep_target_pid"))):
        ts = integer(value_at(row, "outer_loop_deep_target_pid_ts", i))
        if pid is not None and ts:
            ad_high_value.append((ts, 9.0, int(pid)))
    click_types = safe_list(row.get("outer_loop_history_action_pid_list_click_type"))
    for i, pid in enumerate(safe_list(row.get("outer_loop_history_action_pid_list_click"))):
        ts = integer(value_at(row, "outer_loop_history_action_pid_list_click_ts", i))
        kind = str(click_types[i] if i < len(click_types) else "")
        if "PAY" in kind or "CONVERSION" in kind:
            action, strength = "转化", 8.0
        elif "KEY_INAPP" in kind or "PRIVATE_MESSAGE" in kind:
            action, strength = "深度转化", 6.0
        else:
            action, strength = "点击", 2.0
        add_event(events, "video/ad", pid, action, ts, strength)
        if strength >= 6 and pid is not None and ts:
            ad_high_value.append((ts, strength, int(pid)))
    for i, pid in enumerate(safe_list(row.get("outer_loop_history_action_pid_list_pos"))):
        add_event(events, "video/ad", pid, "深度转化", value_at(row, "outer_loop_history_action_pid_list_pos_ts", i), 6)
    if ad_high_value:
        ts, score, pid = max(ad_high_value, key=lambda value: (value[0], value[1]))
        targets["video/ad"] = {"key": ("video/ad", pid), "ts": ts, "strength": score}

    if not targets or not events:
        return None
    # Mapping every raw exposure is expensive and mostly adds weak noise. Keep
    # the union of the 12 most recent and 8 strongest events per domain.
    pruned = []
    for domain in DOMAIN_PREFIX:
        domain_events = [event for event in events if event["domain"] == domain]
        recent = sorted(domain_events, key=lambda value: value["ts"], reverse=True)[:12]
        strongest = sorted(domain_events, key=lambda value: (value["strength"], value["ts"]), reverse=True)[:8]
        seen = set()
        for event in recent + strongest:
            key = (event["key"], event["ts"], event["action"])
            if key not in seen:
                seen.add(key)
                pruned.append(event)
    return {"profile_id": profile_id, "events": pruned, "targets": targets}


def sample_clean_profiles() -> tuple[list[dict], set[tuple[str, int]], dict]:
    profiles = []
    needed: set[tuple[str, int]] = set()
    stats = Counter()
    scanner = pads.dataset(PROFILE_DIR, format="parquet").scanner(
        columns=PROFILE_COLUMNS, batch_size=512, use_threads=True
    )
    row_idx = 0
    for batch in scanner.to_batches():
        indices = [index for index in range(batch.num_rows) if stable_u64(row_idx + index + SEED) % 17 == 0]
        base_idx = row_idx
        row_idx += batch.num_rows
        if not indices:
            continue
        selected = batch.take(pa.array(indices, type=pa.int32()))
        for offset, row in zip(indices, selected.to_pylist()):
            stats["sampled_rows"] += 1
            profile = compact_clean_profile(row, base_idx + offset)
            if profile is None:
                stats["empty_profiles"] += 1
                continue
            profiles.append(profile)
            for event in profile["events"]:
                needed.add(event["key"])
            for target in profile["targets"].values():
                needed.add(target["key"])
                stats[f"raw_target_{target['key'][0]}"] += 1
    stats["kept_profiles"] = len(profiles)
    stats["needed_keys"] = len(needed)
    return profiles, needed, dict(stats)


def clean_caption(domain: str, caption) -> str | None:
    if not isinstance(caption, str):
        return None
    value = caption.strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(parsed, list):
            return None
        parts = [normalize_text(item) for item in parsed if isinstance(item, str) and normalize_text(item)]
        if len(parts) < 3:
            return None
        value = "；".join(parts)
    value = normalize_text(value)
    if value.lower() in {"none", "null", "unknown", "n/a", "无", "未知"}:
        return None
    minimum = {"goods": 8, "live": 12, "video/ad": 20, "video/video": 20}.get(domain, 20)
    if len(value) < minimum or not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", value):
        return None
    if len(value) > 600:
        cut = value[:600]
        position = max(cut.rfind(mark) for mark in "。！？；")
        if position < 300:
            return None
        value = cut[: position + 1]
    return value


def scan_clean_captions(needed: set[tuple[str, int]]) -> tuple[dict, dict, dict]:
    values = {}
    conflicts = set()
    r0_keys = set()
    stats = Counter()
    scanner = pads.dataset(PID2CAPTION_DIR, format="parquet").scanner(
        columns=["pid", "domain", "caption"], batch_size=32768, use_threads=True
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            domain, pid = row.get("domain"), integer(row.get("pid"))
            if domain not in DOMAIN_PREFIX or pid is None:
                continue
            key = (domain, pid)
            use_for_profile = key in needed
            use_for_r0 = stable_u64(pid + SEED + len(domain)) % 509 == 0
            if not use_for_profile and not use_for_r0:
                continue
            caption = clean_caption(domain, row.get("caption"))
            if caption is None:
                stats["caption_rejected"] += 1
                continue
            if key in values and values[key] != caption:
                conflicts.add(key)
                stats["caption_conflict"] += 1
                continue
            values[key] = caption
            if use_for_r0:
                r0_keys.add(key)
    for key in conflicts:
        values.pop(key, None)
        r0_keys.discard(key)
    r0_captions = {key: values[key] for key in r0_keys if key in values}
    profile_captions = {key: value for key, value in values.items() if key in needed}
    needed.update(r0_captions)
    stats["profile_captions"] = len(profile_captions)
    stats["r0_captions"] = len(r0_captions)
    return profile_captions, r0_captions, dict(stats)


def scan_clean_tags(needed: set[tuple[str, int]]) -> tuple[dict, dict]:
    values = {}
    conflicts = set()
    stats = Counter()
    scanner = pads.dataset(PID2TAG_DIR, format="parquet").scanner(
        columns=["pid", "domain", "tag_lv3"], batch_size=65536, use_threads=True
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            domain, pid = row.get("domain"), integer(row.get("pid"))
            if domain not in DOMAIN_PREFIX or pid is None or (domain, pid) not in needed:
                continue
            tag = normalize_text(row.get("tag_lv3"))
            levels = [part.strip() for part in tag.split("-") if part.strip()]
            if len(levels) < 2 or any(len(part) > 60 for part in levels):
                stats["tag_rejected"] += 1
                continue
            canonical = "-".join(levels[:3])
            key = (domain, pid)
            if key in values and values[key] != canonical:
                conflicts.add(key)
                stats["tag_conflict"] += 1
                continue
            values[key] = canonical
    for key in conflicts:
        values.pop(key, None)
    stats["tags"] = len(values)
    return values, dict(stats)


def scan_clean_sids(needed: set[tuple[str, int]]) -> tuple[dict, dict]:
    values = {}
    conflicts = set()
    stats = Counter()
    scanner = pads.dataset(PID2SID_DIR, format="parquet").scanner(
        columns=["pid", "domain", "sid_three"], batch_size=65536, use_threads=True
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            domain, pid = row.get("domain"), integer(row.get("pid"))
            if domain not in DOMAIN_PREFIX or pid is None or (domain, pid) not in needed:
                continue
            token = sid_token(domain, row.get("sid_three"))
            if token is None:
                stats["sid_rejected"] += 1
                continue
            key = (domain, pid)
            if key in values and values[key] != token:
                conflicts.add(key)
                stats["sid_conflict"] += 1
                continue
            values[key] = token
    for key in conflicts:
        values.pop(key, None)
    stats["sids"] = len(values)
    return values, dict(stats)


def tag_levels(tag: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(tag or "").split("-") if part.strip())


def build_clean_r0_pools(captions: dict, tags: dict, sids: dict) -> tuple[dict[str, list[dict]], dict]:
    pools = {"r0_ground_direct": [], "r0_caption_direct": [], "r0_caption_think": []}
    stats = Counter()
    caption_targets: dict[str, set[str]] = defaultdict(set)
    for key, caption in captions.items():
        token = sids.get(key)
        if token:
            caption_targets[normalize_key(caption)].add(token)
    tag_counts = Counter()
    for key, caption in captions.items():
        token = sids.get(key)
        if not token:
            stats["missing_sid"] += 1
            continue
        if len(caption_targets[normalize_key(caption)]) != 1:
            stats["caption_collision"] += 1
            continue
        domain = key[0]
        cn = DOMAIN_CN[domain]
        pools["r0_ground_direct"].append(record(
            f"你是一位{cn}内容分析专家，负责将自然语言描述映射为精确的 itemic token。",
            route_prompt(f"请根据以下{cn}描述输出唯一匹配的{cn}token：{caption}", "no_think"),
            token, "r0_clean", "caption_to_sid_collision_free", _domain=DOMAIN_SHORT[domain],
        ))
        pools["r0_caption_direct"].append(record(
            f"你是一名专业的{cn}内容理解助手。",
            route_prompt(f"请准确描述{cn}token {token} 所表示的内容。", "no_think"),
            caption, "r0_clean", "sid_to_caption_direct", _domain=DOMAIN_SHORT[domain],
        ))
        levels = tag_levels(tags.get(key))
        signature = tuple(levels[:3])
        if len(levels) >= 3 and tag_counts[signature] < 30:
            tag_counts[signature] += 1
            thought = (
                f"先由粗粒度语义确定“{levels[0]}”，再用中粒度信息收敛到“{levels[1]}”，"
                f"最后结合细粒度线索“{levels[2]}”核对内容；三层证据共同支持最终描述。"
            )
            pools["r0_caption_think"].append(record(
                f"你是一名专业的{cn}内容理解助手。",
                route_prompt(f"请准确描述{cn}token {token} 所表示的内容。", "think"),
                f"<think>\n{thought}\n</think>\n{caption}",
                "r0_clean", "sid_to_caption_grounded_cot", _domain=DOMAIN_SHORT[domain],
            ))
    return pools, {"filter_counts": dict(stats), "pool_counts": {key: len(value) for key, value in pools.items()}}


def merge_visible_events(events: list[dict], target: dict, target_sid: str, sids: dict) -> list[dict]:
    merged = {}
    for event in events:
        if event["ts"] >= target["ts"] or event["key"] == target["key"]:
            continue
        token = sids.get(event["key"])
        if not token or token == target_sid:
            continue
        if event["key"] not in merged:
            merged[event["key"]] = dict(event, token=token, actions={event["action"]})
            continue
        current = merged[event["key"]]
        current["repeat"] += 1
        current["strength"] = min(current["strength"] + event["strength"], 20)
        current["ts"] = max(current["ts"], event["ts"])
        current["actions"].add(event["action"])
    result = []
    for event in merged.values():
        event["action"] = "/".join(sorted(event.pop("actions")))
        result.append(event)
    return sorted(result, key=lambda value: value["ts"])


def caption_terms(caption: str | None) -> set[str]:
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(caption or "")).lower()
    terms = {text[index:index + 2] for index in range(max(len(text) - 1, 0))}
    generic = {
        "用户", "视频", "商品", "直播", "广告", "内容", "展示", "介绍", "使用",
        "主要", "一款", "一个", "一种", "进行", "相关", "通过", "以及", "适合",
    }
    return {term for term in terms if term not in generic and not term.isdigit()}


def semantic_relation(
    event_key: tuple[str, int],
    target_key: tuple[str, int],
    tags: dict,
    captions: dict,
) -> int:
    event_tag = tags.get(event_key)
    target_tag = tags.get(target_key)
    event = tag_levels(event_tag)
    target = tag_levels(target_tag)
    if event and target and event[0] == target[0]:
        if len(event) >= 2 and len(target) >= 2 and event[1] == target[1]:
            return 2
        return 1
    overlap = caption_terms(captions.get(event_key)) & caption_terms(captions.get(target_key))
    if len(overlap) >= 3:
        return 2
    if overlap:
        return 1
    return 0


def semantic_label(key: tuple[str, int], tags: dict, captions: dict) -> str:
    levels = tag_levels(tags.get(key))
    if len(levels) >= 2:
        return levels[1]
    if levels:
        return levels[0]
    caption = normalize_text(captions.get(key, ""))
    caption = re.sub(r"^(这是一段|这是一个|本视频|该商品|该直播)", "", caption)
    return caption[:16].strip("，。；：: ") or "近期兴趣"


def select_evidence(
    visible: list[dict],
    target_key: tuple[str, int],
    tags: dict,
    captions: dict,
) -> tuple[list[dict], list[dict]] | None:
    if len(visible) < 6 or len({event["domain"] for event in visible}) < 2:
        return None
    max_ts = max(event["ts"] for event in visible)
    min_ts = min(event["ts"] for event in visible)
    span = max(max_ts - min_ts, 1)
    for event in visible:
        relation = semantic_relation(event["key"], target_key, tags, captions)
        recency = (event["ts"] - min_ts) / span
        event["relation"] = relation
        event["rank_score"] = relation * 10 + min(event["strength"], 10) + recency * 3 + min(event["repeat"], 3)
    aligned = sorted((event for event in visible if event["relation"] > 0), key=lambda value: value["rank_score"], reverse=True)
    same_l2 = [event for event in aligned if event["relation"] == 2]
    same_l1_keys = {event["key"] for event in aligned}
    if not same_l2 and len(same_l1_keys) < 2:
        return None
    chosen = []
    for event in aligned + sorted(visible, key=lambda value: value["rank_score"], reverse=True):
        if event["key"] not in {item["key"] for item in chosen}:
            chosen.append(event)
        if len(chosen) >= 16:
            break
    if len(chosen) < 6:
        return None
    chosen = sorted(chosen, key=lambda value: value["ts"])
    return chosen, aligned


def build_profile_prompt(events: list[dict], target_domain: str) -> str:
    lines = ["用户多域历史行为（已按时间从早到晚排列）："]
    total = len(events)
    for index, event in enumerate(events, 1):
        if index > total * 0.7:
            period = "近期"
        elif index > total * 0.35:
            period = "中期"
        else:
            period = "较早"
        repeat = f"，累计{event['repeat']}次" if event["repeat"] > 1 else ""
        lines.append(
            f"{index}. [{period}][{DOMAIN_CN[event['domain']]}-{event['action']}{repeat}] {event['token']}"
        )
    lines.append(f"请推断用户接下来最可能交互的{DOMAIN_CN[target_domain]}，最终只输出一个 itemic token。")
    return "\n".join(lines)


def build_grounded_cot(
    events: list[dict],
    aligned: list[dict],
    target_domain: str,
    tags: dict,
    captions: dict,
    profile_id: int,
) -> str | None:
    if len(aligned) < 2:
        return None
    groups: dict[str, list[dict]] = defaultdict(list)
    broad = Counter()
    for event in events:
        levels = tag_levels(tags.get(event["key"]))
        label = semantic_label(event["key"], tags, captions)
        broad[levels[0] if levels else label] += event["strength"] + event["repeat"]
        groups[label].append(event)
    if not groups:
        return None
    target_direction = semantic_label(aligned[0]["key"], tags, captions)
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: max(value["rank_score"] for value in item[1]),
        reverse=True,
    )[:3]
    directions = []
    for label, group in ranked_groups:
        evidence = max(group, key=lambda value: value["rank_score"])
        directions.append(f"“{label}”由{evidence['token']}的{evidence['action']}支持")
    strongest = aligned[0]
    second = next((event for event in aligned[1:] if event["key"] != strongest["key"]), None)
    if second is None:
        return None
    persona = "、".join(label for label, _ in broad.most_common(2))
    variants = [
        (
            f"【画像抽象】较稳定的兴趣集中在{persona}；{strongest['token']}的{strongest['action']}与"
            f"{second['token']}的{second['action']}构成两条独立证据。\n"
            f"【兴趣展开】保留少量有历史依据的方向：{'；'.join(directions)}。\n"
            f"【转移判断】“{target_direction}”同时具备近期性、动作强度和重复语义支持，其他方向证据较弱；"
            f"因此面向下一次{DOMAIN_CN[target_domain]}交互，沿该已出现的兴趣方向预测。"
        ),
        (
            f"【画像抽象】历史呈现{persona}偏好，其中{strongest['token']}和{second['token']}都产生了"
            f"{strongest['action']}/{second['action']}等主动信号。\n"
            f"【兴趣展开】按时间连续性展开三个以内候选：{'；'.join(directions)}。\n"
            f"【转移判断】比较近期程度、交互深度与跨条目复现后，“{target_direction}”证据最完整；"
            f"下一次{DOMAIN_CN[target_domain]}选择应延续这一历史可见方向。"
        ),
        (
            f"【画像抽象】去除孤立行为后，{persona}是主要兴趣背景；关键依据是{strongest['token']}的"
            f"{strongest['action']}和{second['token']}的{second['action']}。\n"
            f"【兴趣展开】当前可验证的候选包括：{'；'.join(directions)}。\n"
            f"【转移判断】“{target_direction}”在时间上更近、信号上更强，并有至少两条历史证据闭合；"
            f"故对{DOMAIN_CN[target_domain]}域做该方向的下一交互预测。"
        ),
    ]
    thought = variants[stable_u64(profile_id + len(target_domain)) % len(variants)]
    if len(thought) > 520:
        return None
    refs = ITEMIC_RE.findall(thought)
    prompt_tokens = {event["token"] for event in events}
    if len(set(refs)) < 2 or any(token not in prompt_tokens for token in refs):
        return None
    return thought


def build_clean_r3_pools(
    profiles: list[dict],
    captions: dict,
    tags: dict,
    sids: dict,
) -> tuple[dict[str, list[dict]], dict]:
    pools: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()
    balanced_counts = Counter()
    balanced_candidates = []
    system = "你是推荐系统助手，需要基于多域时序行为进行证据充分、无目标泄漏的下一交互预测。"
    for profile in profiles:
        valid = []
        for domain, target in profile["targets"].items():
            target_sid = sids.get(target["key"])
            if not target_sid or (target["key"] not in tags and target["key"] not in captions):
                stats["missing_target_metadata"] += 1
                continue
            visible = merge_visible_events(profile["events"], target, target_sid, sids)
            selected = select_evidence(visible, target["key"], tags, captions)
            if selected is None:
                stats["weak_or_short_history"] += 1
                continue
            evidence, aligned = selected
            prompt = build_profile_prompt(evidence, domain)
            thought = build_grounded_cot(evidence, aligned, domain, tags, captions, profile["profile_id"])
            if not thought or target_sid in prompt or target_sid in thought:
                stats["cot_or_leakage_reject"] += 1
                continue
            if any(token not in prompt for token in ITEMIC_RE.findall(thought)):
                stats["ungrounded_reference"] += 1
                continue
            short = DOMAIN_SHORT[domain]
            direct = record(
                system, route_prompt(prompt, "no_think"), target_sid,
                "r3_clean", f"strict_{short}_direct", _domain=short,
                _profile_id=profile["profile_id"], _target_sid=target_sid,
            )
            cot = record(
                system, route_prompt(prompt, "think"), f"<think>\n{thought}\n</think>\n{target_sid}",
                "r3_clean", f"strict_{short}_cot", _domain=short,
                _profile_id=profile["profile_id"], _target_sid=target_sid,
            )
            valid.append((domain, direct, cot, target["strength"]))
            if domain == "live":
                pools["r3_live_specialist_direct"].append(direct)
                pools["r3_live_specialist_think"].append(cot)
            stats[f"valid_{domain}"] += 1
        if not valid:
            continue
        # One target per profile in balanced mixtures. Prefer the currently
        # underrepresented domain, then the stronger target.
        chosen = min(
            valid,
            key=lambda value: (
                balanced_counts[value[0]],
                -value[3],
                stable_u64(profile["profile_id"] + len(value[0])),
            ),
        )
        domain, direct, cot, _ = chosen
        balanced_counts[domain] += 1
        balanced_candidates.append((domain, direct, cot))
    for domain, direct, cot in balanced_candidates:
        short = DOMAIN_SHORT[domain]
        pools[f"r3_balanced_{short}_direct"].append(direct)
        pools[f"r3_balanced_{short}_think"].append(cot)
    stats["balanced_profiles"] = len(balanced_candidates)
    return dict(pools), {
        "filter_counts": dict(stats),
        "balanced_domains": dict(balanced_counts),
        "pool_counts": {key: len(value) for key, value in pools.items()},
    }


def build_clean_pools() -> tuple[dict[str, list[dict]], dict]:
    general, general_stats = build_clean_general_pools()
    old, old_stats = build_clean_old_pools()
    profiles, needed, profile_stats = sample_clean_profiles()
    profile_captions, r0_captions, caption_stats = scan_clean_captions(needed)
    tags, tag_stats = scan_clean_tags(needed)
    sids, sid_stats = scan_clean_sids(needed)
    r0, r0_stats = build_clean_r0_pools(r0_captions, tags, sids)
    r3, r3_stats = build_clean_r3_pools(profiles, profile_captions, tags, sids)
    pools = {**general, **old, **r0, **r3}
    stats = {
        "general": general_stats,
        "old_competition": old_stats,
        "profiles": profile_stats,
        "captions": caption_stats,
        "tags": tag_stats,
        "sids": sid_stats,
        "r0": r0_stats,
        "r3": r3_stats,
        "pool_counts": {key: len(value) for key, value in pools.items()},
        "policy": (
            "Strict single-turn general QA; canonical metadata joins; one balanced R3 target per profile; "
            "history timestamps strictly precede targets; high-value targets; tag-supported evidence; "
            "target SID forbidden from prompt and CoT; every CoT itemic reference must occur in history."
        ),
    }
    return pools, stats
