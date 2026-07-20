"""Official-source data builder used by the unified experiment runner.

This module derives reproducible SFT data from the original competition JSONL
and official source parquet data. Invoke it through ``run_experiments.py`` so
all runs, manifests, statuses, and summaries stay in the shared experiment
registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pads

import run_experiments as legacy


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DATA_DIR = EXP / "data"
CONFIG_DIR = EXP / "configs"
OUTPUT_DIR = EXP / "outputs"
LOG_DIR = EXP / "logs"
LF_DIR = ROOT / "demo" / "LLaMA-Factory"
VENV = LF_DIR / ".venv"
DATASET_INFO = LF_DIR / "data" / "dataset_info.json"
TOP_LOG = ROOT / "log.txt"
UNIFIED_RUNNER = EXP / "run_experiments.py"
STATUS_PATH = LOG_DIR / "supervisor_status.jsonl"

SOURCE = ROOT / "data" / "Explorer_LLM_Rec_Competition"
GENERAL_DIR = SOURCE / "OneReason_General"
PROFILE_DIR = SOURCE / "OneReason_UserProfile"
PID2SID_DIR = SOURCE / "OneReason_Pid2Sid"
PID2CAPTION_DIR = SOURCE / "OneReason_Pid2Caption"
PID2TAG_DIR = SOURCE / "OneReason_Pid2Tag"

BASE_V29 = OUTPUT_DIR / "v29_v19_user_world_guard_lr8e7_ep018"
OFFICIAL_BASE = "OpenOneRec/OneReason-0.8B-pretrain-competition"
SEED = 202607150
PUBLIC_091_DIR = ROOT / "data" / "external" / "kuaishou-llmrec-sft-baseline-0.91"
PUBLIC_091_SOURCE = PUBLIC_091_DIR / "train.jsonl"
PUBLIC_091_URL = (
    "https://huggingface.co/datasets/Frinkleko/"
    "kuaishou-llmrec-sft-baseline-0.91/resolve/main/train.jsonl?download=true"
)
PUBLIC_091_SHA256 = "4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8"
PUBLIC_091_DATASET = "v49_public_091_exact"
PUBLIC_GUARD_DATASET = "v63_public_user_video_live_guard"
PUBLIC_USER_REPLAY_DATASET = "v67_public_user_replay"

ITEMIC_RE = re.compile(r"<\|(?:prod|video|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
THINK_RE = re.compile(r"\s*<think>(.*?)</think>\s*(.*?)\s*\Z", re.S)
TOKEN_RE = re.compile(r"<s_([abc])_(\d+)>")

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
TARGET_ORDER = {
    "goods": ["live", "video/ad", "video/video", "goods"],
    "live": ["goods", "video/ad", "video/video", "live"],
    "video/video": ["live", "goods", "video/ad", "video/video"],
    "video/ad": ["live", "goods", "video/video", "video/ad"],
}


DATASET_SPECS = {
    "v34_paper_mix": {
        "general": 30400, "r0": 19000, "r3_think": 6000, "r3_direct": 12000,
        "old_rec": 5000, "old_user": 3800, "old_item": 3800,
    },
    "v35_cot_r3_balanced": {
        "general": 24000, "r0": 12000, "r3_think": 18000, "r3_direct": 14000,
        "old_rec": 5000, "old_user": 3500, "old_item": 3500,
    },
    "v36_general_guard": {
        "general": 36000, "r0": 12000, "r3_think": 7000, "r3_direct": 10000,
        "old_rec": 6000, "old_user": 4500, "old_item": 4500,
    },
    "v37_r0_perception": {
        "general": 24000, "r0": 28000, "r3_think": 8000, "r3_direct": 8000,
        "old_rec": 4000, "old_user": 4000, "old_item": 4000,
    },
    "v38_r3_aggressive": {
        "general": 20000, "r0": 8000, "r3_think": 20000, "r3_direct": 22000,
        "old_rec": 4000, "old_user": 3000, "old_item": 3000,
    },
    "v39_curriculum_r0": {
        "general": 32000, "r0": 28000, "old_rec": 4000, "old_user": 3000, "old_item": 3000,
    },
    "v39_curriculum_r3": {
        "general": 22000, "r0": 6000, "r3_think": 16000, "r3_direct": 18000,
        "old_rec": 3000, "old_user": 2500, "old_item": 2500,
    },
    "v41_baseline_general": {
        "general": 32000, "r0": 8000, "old_rec": 22000, "old_user": 8000, "old_item": 10000,
    },
    "v42_user_evolution": {
        "general": 26000, "r0": 6000, "r3_think": 12000, "r3_direct": 12000,
        "old_rec": 5000, "old_user": 9000, "old_item": 5000,
    },
}


SOURCE_RUNS = [
    {
        "name": "v34_v29_paper_mix_lr5e7_ep10", "dataset": "v34_paper_mix",
        "model": str(BASE_V29), "lr": "5.0e-7", "epochs": 1.0, "seed": SEED + 34,
        "note": "Anchor run using a scaled OneReason-like R0/R3/general mixture plus old-task replay.",
    },
    {
        "name": "v35_v29_cot_r3_balanced_lr6e7_ep10", "dataset": "v35_cot_r3_balanced",
        "model": str(BASE_V29), "lr": "6.0e-7", "epochs": 1.0, "seed": SEED + 35,
        "note": "Higher R3 CoT ratio with compact Persona-Expansion-Transition traces and direct-route balance.",
    },
    {
        "name": "v36_v29_general_guard_lr4e7_ep12", "dataset": "v36_general_guard",
        "model": str(BASE_V29), "lr": "4.0e-7", "epochs": 1.2, "seed": SEED + 36,
        "note": "General-heavy replay explicitly targeting the world-score forgetting seen in v31-v33.",
    },
    {
        "name": "v37_v29_r0_perception_lr6e7_ep10", "dataset": "v37_r0_perception",
        "model": str(BASE_V29), "lr": "6.0e-7", "epochs": 1.0, "seed": SEED + 37,
        "note": "R0-heavy bidirectional caption/SID grounding to improve item perception and downstream recommendation semantics.",
    },
    {
        "name": "v38_v29_r3_aggressive_lr7e7_ep10", "dataset": "v38_r3_aggressive",
        "model": str(BASE_V29), "lr": "7.0e-7", "epochs": 1.0, "seed": SEED + 38,
        "note": "Highest R3 share, trained on real temporal holdouts from official user profiles with target leakage blocked.",
    },
    {
        "name": "v39_v29_curriculum_r0_to_r3_lr6e7_ep08x2", "dataset": "v39_curriculum_r3",
        "model": str(BASE_V29), "lr": "5.0e-7", "epochs": 0.8, "seed": SEED + 39,
        "note": "Two-stage curriculum: 0.8 epoch R0/general alignment followed by 0.8 epoch R3/general cognition.",
        "stages": [
            {"dataset": "v39_curriculum_r0", "lr": "6.0e-7", "epochs": 0.8},
            {"dataset": "v39_curriculum_r3", "lr": "5.0e-7", "epochs": 0.8},
        ],
    },
    {
        "name": "v40_scratch_source_paper_mix_lr3e6_ep15", "dataset": "v34_paper_mix",
        "model": OFFICIAL_BASE, "lr": "3.0e-6", "epochs": 1.5, "seed": SEED + 40,
        "note": "High-risk restart from the official pretrained checkpoint using the new source-data mixture.",
    },
    {
        "name": "v41_v29_baseline_general_replay_lr4e7_ep10", "dataset": "v41_baseline_general",
        "model": str(BASE_V29), "lr": "4.0e-7", "epochs": 1.0, "seed": SEED + 41,
        "note": "Control run dominated by original competition supervision and general replay, with no profile-derived R3.",
    },
    {
        "name": "v42_v29_user_evolution_guard_lr5e7_ep10", "dataset": "v42_user_evolution",
        "model": str(BASE_V29), "lr": "5.0e-7", "epochs": 1.0, "seed": SEED + 42,
        "note": "Protects structured user extraction/evolution formats while adding real-holdout R3 and general replay.",
    },
    {
        "name": "v43_v29_paper_mix_low_lr25e7_ep15", "dataset": "v34_paper_mix",
        "model": str(BASE_V29), "lr": "2.5e-7", "epochs": 1.5, "seed": SEED + 43,
        "note": "Controlled long, low-LR exposure on exactly the same data as v34.",
    },
]


# High-precision, low-dose continuations after the v34-v43 data audit.  These
# are intentionally separate so rebuilding the older batch remains unchanged.
CLEAN_DATASET_SPECS = {
    "v44_clean_live_temporal": {
        "r3_live_specialist_direct": 1300, "r3_live_specialist_think": 1800,
        "old_rec_live_direct": 350, "old_rec_live_think": 250,
        "old_rec_ad_direct": 250, "old_rec_ad_think": 100,
        "old_rec_goods_direct": 250, "old_rec_goods_think": 100,
        "old_rec_video_direct": 500, "old_rec_video_think": 200,
        "old_user_array_direct": 600, "old_user_logic_direct": 550, "old_user_logic_think": 180,
        "old_item_goods_direct": 500, "old_item_goods_think": 120,
        "old_item_live_direct": 350, "old_item_live_think": 100,
        "old_item_ad_direct": 500, "old_item_ad_think": 120,
        "old_item_video_direct": 500, "old_item_video_think": 100,
        "general_cn_direct": 500, "general_cn_think": 250,
        "general_other_direct": 350, "general_other_think": 150,
    },
    "v45_clean_r3_domainmix": {
        "r3_balanced_ad_direct": 700, "r3_balanced_ad_think": 250,
        "r3_balanced_live_direct": 550, "r3_balanced_live_think": 700,
        "r3_balanced_goods_direct": 300, "r3_balanced_goods_think": 500,
        "r3_balanced_video_direct": 550, "r3_balanced_video_think": 550,
        "old_rec_live_direct": 300, "old_rec_live_think": 150,
        "old_rec_ad_direct": 300, "old_rec_ad_think": 150,
        "old_rec_goods_direct": 300, "old_rec_goods_think": 150,
        "old_rec_video_direct": 300, "old_rec_video_think": 150,
        "old_user_array_direct": 400, "old_user_logic_direct": 400, "old_user_logic_think": 150,
        "old_item_goods_direct": 300, "old_item_goods_think": 75,
        "old_item_live_direct": 300, "old_item_live_think": 75,
        "old_item_ad_direct": 300, "old_item_ad_think": 75,
        "old_item_video_direct": 300, "old_item_video_think": 75,
        "general_cn_direct": 500, "general_cn_think": 250,
        "general_other_direct": 300, "general_other_think": 150,
    },
    "v46_clean_user_evolution": {
        "old_user_array_direct": 1200, "old_user_logic_direct": 1100, "old_user_logic_think": 450,
        "old_rec_live_direct": 300, "old_rec_live_think": 150,
        "old_rec_ad_direct": 300, "old_rec_ad_think": 150,
        "old_rec_goods_direct": 300, "old_rec_goods_think": 150,
        "old_rec_video_direct": 300, "old_rec_video_think": 150,
        "old_item_goods_direct": 300, "old_item_goods_think": 50,
        "old_item_live_direct": 300, "old_item_live_think": 50,
        "old_item_ad_direct": 300, "old_item_ad_think": 50,
        "old_item_video_direct": 300, "old_item_video_think": 50,
        "r3_balanced_ad_direct": 250, "r3_balanced_ad_think": 100,
        "r3_balanced_live_direct": 350, "r3_balanced_live_think": 450,
        "r3_balanced_goods_direct": 150, "r3_balanced_goods_think": 300,
        "r3_balanced_video_direct": 250, "r3_balanced_video_think": 250,
        "general_cn_direct": 500, "general_cn_think": 250,
        "general_other_direct": 300, "general_other_think": 150,
    },
    "v47_clean_world_guard": {
        "general_cn_direct": 1300, "general_cn_think": 900,
        "general_other_direct": 1600, "general_other_think": 1100,
        "old_rec_live_direct": 250, "old_rec_live_think": 100,
        "old_rec_ad_direct": 250, "old_rec_ad_think": 100,
        "old_rec_goods_direct": 250, "old_rec_goods_think": 100,
        "old_rec_video_direct": 250, "old_rec_video_think": 100,
        "old_user_array_direct": 350, "old_user_logic_direct": 350, "old_user_logic_think": 100,
        "old_item_goods_direct": 250, "old_item_goods_think": 50,
        "old_item_live_direct": 250, "old_item_live_think": 50,
        "old_item_ad_direct": 250, "old_item_ad_think": 50,
        "old_item_video_direct": 250, "old_item_video_think": 50,
        "r3_balanced_ad_direct": 150, "r3_balanced_ad_think": 50,
        "r3_balanced_live_direct": 200, "r3_balanced_live_think": 250,
        "r3_balanced_goods_direct": 100, "r3_balanced_goods_think": 150,
        "r3_balanced_video_direct": 150, "r3_balanced_video_think": 150,
    },
    "v48_clean_fused": {
        "general_cn_direct": 800, "general_cn_think": 400,
        "general_other_direct": 800, "general_other_think": 400,
        "r0_ground_direct": 1000, "r0_caption_direct": 400, "r0_caption_think": 250,
        "r3_balanced_ad_direct": 600, "r3_balanced_ad_think": 200,
        "r3_balanced_live_direct": 400, "r3_balanced_live_think": 500,
        "r3_balanced_goods_direct": 250, "r3_balanced_goods_think": 500,
        "r3_balanced_video_direct": 550, "r3_balanced_video_think": 600,
        "old_rec_live_direct": 250, "old_rec_live_think": 100,
        "old_rec_ad_direct": 250, "old_rec_ad_think": 100,
        "old_rec_goods_direct": 250, "old_rec_goods_think": 100,
        "old_rec_video_direct": 250, "old_rec_video_think": 150,
        "old_user_array_direct": 450, "old_user_logic_direct": 400, "old_user_logic_think": 150,
        "old_item_goods_direct": 300, "old_item_goods_think": 75,
        "old_item_live_direct": 300, "old_item_live_think": 75,
        "old_item_ad_direct": 300, "old_item_ad_think": 75,
        "old_item_video_direct": 300, "old_item_video_think": 75,
    },
}


CLEAN_RUNS = [
    {
        "name": "v44_v29_clean_live_temporal_lr6e7_ep045", "dataset": "v44_clean_live_temporal",
        "model": str(BASE_V29), "lr": "6.0e-7", "epochs": 0.45, "seed": SEED + 44,
        "batch": "v44-v48",
        "note": "Strict live specialist: future high-value live target, chronological pre-target history, semantic support, and 58/42 CoT/direct live supervision with clean guards.",
    },
    {
        "name": "v45_v29_clean_r3_domainmix_lr55e7_ep045", "dataset": "v45_clean_r3_domainmix",
        "model": str(BASE_V29), "lr": "5.5e-7", "epochs": 0.45, "seed": SEED + 45,
        "batch": "v44-v48",
        "note": "Strict balanced R3 with paper-motivated domain ratios: ad direct-heavy, product CoT-heavy, live moderately CoT-rich, and video balanced.",
    },
    {
        "name": "v46_v29_clean_user_evolution_lr5e7_ep045", "dataset": "v46_clean_user_evolution",
        "model": str(BASE_V29), "lr": "5.0e-7", "epochs": 0.45, "seed": SEED + 46,
        "batch": "v44-v48",
        "note": "User/evolution specialist using parseable deduplicated JSON, retained evidence-bearing user CoT, strict R3 temporal examples, and balanced item/rec guards.",
    },
    {
        "name": "v47_v29_clean_world_guard_lr35e7_ep035", "dataset": "v47_clean_world_guard",
        "model": str(BASE_V29), "lr": "3.5e-7", "epochs": 0.35, "seed": SEED + 47,
        "batch": "v44-v48",
        "note": "World-knowledge guard with strict single-turn paired general data, exact/template deduplication, bounded traces, and a low-dose task-format replay shell.",
    },
    {
        "name": "v48_v29_clean_fused_lr45e7_ep040", "dataset": "v48_clean_fused",
        "model": str(BASE_V29), "lr": "4.5e-7", "epochs": 0.40, "seed": SEED + 48,
        "batch": "v44-v48",
        "note": "Fused high-confidence candidate combining clean general, collision-free R0, domain-aware strict R3, and deduplicated original-task guards.",
    },
]

CLEAN_EVALUATIONS = {
    "v44_v29_clean_live_temporal_lr6e7_ep045": {
        "total": 0.8732, "item": 0.2146, "user": [0.0874, 0.0370],
        "recommendation": [0.0576, 0.1122, 0.1330, 0.1143], "world": 0.1171,
        "generation_minutes": 52.25,
    },
    "v45_v29_clean_r3_domainmix_lr55e7_ep045": {
        "total": 0.8774, "item": 0.2146, "user": [0.0871, 0.0369],
        "recommendation": [0.0672, 0.1088, 0.1358, 0.1125], "world": 0.1145,
        "generation_minutes": 52.53,
    },
    "v46_v29_clean_user_evolution_lr5e7_ep045": {
        "total": 0.8787, "item": 0.2146, "user": [0.0862, 0.0360],
        "recommendation": [0.0768, 0.1020, 0.1344, 0.1116], "world": 0.1171,
        "generation_minutes": 52.41,
    },
    "v47_v29_clean_world_guard_lr35e7_ep035": {
        "total": 0.8753, "item": 0.2146, "user": [0.0873, 0.0352],
        "recommendation": [0.0768, 0.0986, 0.1358, 0.1125], "world": 0.1145,
    },
    "v48_v29_clean_fused_lr45e7_ep040": {
        "total": 0.8731, "item": 0.2146, "user": [0.0876, 0.0371],
        "recommendation": [0.0672, 0.1020, 0.1330, 0.1134], "world": 0.1182,
        "generation_minutes": 51.91,
    },
}
for clean_run in CLEAN_RUNS:
    clean_run["evaluation"] = CLEAN_EVALUATIONS[clean_run["name"]]


# This batch deliberately keeps data fixed and changes optimization only.  It
# separates under-training from data-quality effects and includes two published
# anchors: the 0.9107 LoRA recipe and a converged 3-epoch full-SFT recipe.
TUNING_RUNS = [
    {
        "name": "v49_public091_lora_r32_lr2e4_ep1", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "2.0e-4", "epochs": 1.0,
        "seed": SEED + 49, "batch": "v49-v53", "weight_decay": 0.001,
        "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.05,
        "validation_size": 0.0,
        "note": "Exact public 0.9107 reproduction anchor: published 32,705-row clean data and LoRA r32/alpha32/dropout0.05 recipe.",
    },
    {
        "name": "v50_public091_lora_r32_lr1e4_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "1.0e-4", "epochs": 2.0,
        "seed": SEED + 50, "batch": "v49-v53", "weight_decay": 0.001,
        "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.05,
        "validation_size": 0.0,
        "note": "Same public data and LoRA capacity as v49, with half LR and two epochs to test whether the one-epoch public anchor is optimization-limited.",
    },
    {
        "name": "v51_public091_lora_r64_lr1e4_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "1.0e-4", "epochs": 2.0,
        "seed": SEED + 51, "batch": "v49-v53", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "validation_size": 0.0,
        "note": "Higher-capacity LoRA on fixed data; isolates adapter rank from the longer schedule while retaining the public alpha/rank scaling.",
    },
    {
        "name": "v52_public091_full_lr2e5_ep1", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "full", "lr": "2.0e-5", "epochs": 1.0,
        "seed": SEED + 52, "batch": "v49-v53", "weight_decay": 0.0,
        "validation_size": 0.0,
        "note": "Full-SFT counterpart using v7's successful 2e-5 scale, now on balanced CoT/direct clean data instead of final-only data.",
    },
    {
        "name": "v53_public091_full_lr1e5_ep3", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "full", "lr": "1.0e-5", "epochs": 3.0,
        "seed": SEED + 53, "batch": "v49-v53", "weight_decay": 0.0,
        "validation_size": 0.0,
        "note": "Convergence anchor matching the published full-SFT 1e-5/3-epoch schedule whose validation loss fell through the third epoch.",
    },
]

TUNING_EVALUATIONS = {
    "v49_public091_lora_r32_lr2e4_ep1": {
        "total": 0.8834, "item": 0.2146, "user": [0.0554, 0.0395],
        "recommendation": [0.0768, 0.1088, 0.1302, 0.1071], "world": 0.1509,
        "evaluation_minutes": 75.44,
    },
    "v50_public091_lora_r32_lr1e4_ep2": {
        "total": 0.8971, "item": 0.2453, "user": [0.0622, 0.0399],
        "recommendation": [0.0384, 0.1122, 0.1498, 0.1062], "world": 0.1431,
        "evaluation_minutes": 71.56,
    },
    "v51_public091_lora_r64_lr1e4_ep2": {
        "total": 0.9188, "item": 0.2146, "user": [0.0790, 0.0381],
        "recommendation": [0.0576, 0.1156, 0.1526, 0.1071], "world": 0.1543,
        "evaluation_minutes": 68.58,
    },
    "v52_public091_full_lr2e5_ep1": {
        "total": 0.8614, "item": 0.1840, "user": [0.0538, 0.0424],
        "recommendation": [0.0768, 0.1122, 0.1246, 0.1125], "world": 0.1550,
        "evaluation_minutes": 71.41,
    },
    "v53_public091_full_lr1e5_ep3": {
        "total": 0.7862, "item": 0.2146, "user": [0.0003, 0.0410],
        "recommendation": [0.0384, 0.1054, 0.1260, 0.1089], "world": 0.1517,
        "evaluation_minutes": 71.53,
    },
}
for tuning_run in TUNING_RUNS:
    tuning_run["evaluation"] = TUNING_EVALUATIONS[tuning_run["name"]]


ADVANCED_RUNS = [
    {
        "name": "v54_public091_lora_r96_lr1e4_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "1.0e-4", "epochs": 2.0,
        "seed": SEED + 54, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 96, "lora_alpha": 96, "lora_dropout": 0.05,
        "note": "Interpolate between rank-64 v51 and rank-128 while keeping unit LoRA scaling and the successful two-epoch schedule.",
    },
    {
        "name": "v55_public091_lora_r128_lr8e5_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "8.0e-5", "epochs": 2.0,
        "seed": SEED + 55, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 128, "lora_alpha": 128, "lora_dropout": 0.05,
        "note": "Higher-capacity standard LoRA with a modestly reduced LR to test whether v51 remains rank-limited.",
    },
    {
        "name": "v56_public091_rslora_r128_a16_lr6e5_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "6.0e-5", "epochs": 2.0,
        "seed": SEED + 56, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 128, "lora_alpha": 16, "lora_dropout": 0.05,
        "use_rslora": True, "merge_to_full": True,
        "note": "Rank-stabilized LoRA branch; alpha/sqrt(rank) avoids the high-rank scaling collapse and is merged for submission compatibility.",
    },
    {
        "name": "v57_public091_dora_r64_lr8e5_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "8.0e-5", "epochs": 2.0,
        "seed": SEED + 57, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "use_dora": True, "merge_to_full": True,
        "note": "DoRA magnitude/direction adaptation at v51 rank, merged to a standard full checkpoint after training.",
    },
    {
        "name": "v58_public091_loraplus_r64_lr2e5_x16_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "2.0e-5", "epochs": 2.0,
        "seed": SEED + 58, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "loraplus_lr_ratio": 16.0,
        "note": "LoRA+ uses separate A/B learning rates; the conservative base LR gives the B matrix a 3.2e-4 update rate.",
    },
    {
        "name": "v59_public091_lora_r64_drop0_lr1e4_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "1.0e-4", "epochs": 2.0,
        "seed": SEED + 59, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.0,
        "note": "Exact v51 control with adapter dropout removed, matching the recent public ORPO/GRPO adapter convention.",
    },
    {
        "name": "v60_public091_lora_r64_a128_lr5e5_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "5.0e-5", "epochs": 2.0,
        "seed": SEED + 60, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 128, "lora_dropout": 0.05,
        "note": "Double LoRA scaling with half v51 LR, isolating update geometry from nominal learning rate.",
    },
    {
        "name": "v61_v29_public091_lora_r64_lr5e5_ep1", "dataset": PUBLIC_091_DATASET,
        "model": str(BASE_V29), "method": "lora", "lr": "5.0e-5", "epochs": 1.0,
        "seed": SEED + 61, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "merge_to_full": True,
        "note": "Public clean-data LoRA over the strong v29 full checkpoint, targeting v29 user/video retention plus v51 ad/world gains.",
    },
    {
        "name": "v62_v29_public091_lora_r64_lr2e5_ep1", "dataset": PUBLIC_091_DATASET,
        "model": str(BASE_V29), "method": "lora", "lr": "2.0e-5", "epochs": 1.0,
        "seed": SEED + 62, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "merge_to_full": True,
        "note": "Lower-dose v29-based counterpart intended to preserve its structured-user and video behavior more conservatively.",
    },
    {
        "name": "v63_public_guard_lora_r64_lr8e5_ep2", "dataset": PUBLIC_GUARD_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "8.0e-5", "epochs": 2.0,
        "seed": SEED + 63, "batch": "v54-v63", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "note": "Real-label task guard: duplicate valid user JSON and add at most one distinct original CoT target per video/live prompt without synthetic traces.",
    },
]


FOCUSED_RUNS = [
    {
        "name": "v64_public091_loraplus_r64_lr2e5_x8_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "2.0e-5", "epochs": 2.0,
        "seed": SEED + 58, "batch": "v64-v68", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "loraplus_lr_ratio": 8.0,
        "note": "LoRA+ ratio-8 neighbor of v58; keeps A at 2e-5 while reducing B from 3.2e-4 to 1.6e-4.",
    },
    {
        "name": "v65_public091_loraplus_r128_lr1e5_x16_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "1.0e-5", "epochs": 2.0,
        "seed": SEED + 58, "batch": "v64-v68", "weight_decay": 0.001,
        "lora_rank": 128, "lora_alpha": 128, "lora_dropout": 0.05,
        "loraplus_lr_ratio": 16.0,
        "note": "Rank-128 LoRA+ with both learning rates halved relative to v58 for rank-aware capacity scaling.",
    },
    {
        "name": "v66_public091_rslora_loraplus_r128_a16_lr1e5_x16_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "1.0e-5", "epochs": 2.0,
        "seed": SEED + 58, "batch": "v64-v68", "weight_decay": 0.001,
        "lora_rank": 128, "lora_alpha": 16, "lora_dropout": 0.05,
        "use_rslora": True, "loraplus_lr_ratio": 16.0, "merge_to_full": True,
        "note": "Combines v58 LoRA+ optimizer geometry with v56 rank-stabilized high-rank scaling; targets ad and world gains.",
    },
    {
        "name": "v67_public_user_replay_loraplus_r64_lr2e5_x16_ep2", "dataset": PUBLIC_USER_REPLAY_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "2.0e-5", "epochs": 2.0,
        "seed": SEED + 58, "batch": "v64-v68", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "loraplus_lr_ratio": 16.0,
        "note": "Applies v58 LoRA+ to exact public data plus one replay of each real parseable user target, isolating the useful v63 signal.",
    },
    {
        "name": "v68_public091_pissa_loraplus_r64_lr2e5_x8_ep2", "dataset": PUBLIC_091_DATASET,
        "model": OFFICIAL_BASE, "method": "lora", "lr": "2.0e-5", "epochs": 2.0,
        "seed": SEED + 58, "batch": "v64-v68", "weight_decay": 0.001,
        "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.05,
        "loraplus_lr_ratio": 8.0, "pissa_init": True, "pissa_iter": 16,
        "merge_to_full": True,
        "note": "PiSSA FSVD-16 initialization with the conservative LoRA+ ratio-8 optimizer; merged for submission compatibility.",
    },
]


def all_source_runs() -> list[dict]:
    return SOURCE_RUNS + CLEAN_RUNS + TUNING_RUNS + ADVANCED_RUNS + FOCUSED_RUNS


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def safe_list(value) -> list:
    return value if isinstance(value, list) else []


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


def sid_token(domain: str, sid_values) -> str | None:
    values = safe_list(sid_values)
    if domain not in DOMAIN_PREFIX or len(values) < 3:
        return None
    try:
        a, b, c = (int(float(values[i])) for i in range(3))
    except (TypeError, ValueError, OverflowError):
        return None
    if min(a, b, c) < 0:
        return None
    return f"<|{DOMAIN_PREFIX[domain]}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    if isinstance(content, list):
        return "".join(content_text(item) for item in content)
    return ""


def build_general_pool(limit: int = 60000) -> list[dict]:
    pool: list[dict] = []
    scanner = pads.dataset(GENERAL_DIR, format="parquet").scanner(
        columns=["messages"], batch_size=1024, use_threads=True
    )
    global_idx = 0
    banned = ("<tool_call>", "<tool_response>", '"tool_calls"', "<|python_tag|>")
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            raw = row.get("messages")
            global_idx += 1
            if not raw:
                continue
            try:
                messages = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            if not isinstance(messages, list) or any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict)):
                continue
            system = ""
            users = []
            assistants = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                text = content_text(msg.get("content"))
                role = msg.get("role")
                if role == "system" and not system:
                    system = text
                elif role in {"user", "human"}:
                    users.append(text)
                elif role == "assistant":
                    assistants.append(text)
            if not users or not assistants:
                continue
            prompt = users[0].strip()
            answer = assistants[-1].strip()
            match = THINK_RE.fullmatch(answer)
            if not match:
                continue
            thought, final = match.group(1).strip(), match.group(2).strip()
            joined = system + prompt + answer
            if any(token in joined for token in banned):
                continue
            if not (5 <= len(final) <= 1800 and 20 <= len(thought) <= 4500 and len(prompt) <= 5000 and len(joined) <= 10000):
                continue
            pool.append(record(system, route_prompt(prompt, "no_think"), final, "general", "official_general_direct"))
            pool.append(record(system, route_prompt(prompt, "think"), f"<think>\n{thought}\n</think>\n{final}", "general", "official_general_cot"))
            if len(pool) >= limit:
                random.Random(SEED).shuffle(pool)
                return pool[:limit]
    return pool


def aligned_events(pids, actions, timestamps=None, *, domain: str, cap: int = 24) -> list[dict]:
    ids = safe_list(pids)
    acts = safe_list(actions)
    times = safe_list(timestamps)
    events = []
    for i, pid in enumerate(ids):
        if pid is None:
            continue
        action = acts[i] if i < len(acts) else "浏览"
        ts = times[i] if i < len(times) else i
        events.append({"key": (domain, int(pid)), "action": str(action), "ts": ts if ts is not None else i})
    return events[-cap:]


def compact_profile(row: dict) -> dict | None:
    events: dict[str, list[dict]] = {domain: [] for domain in DOMAIN_PREFIX}
    targets: dict[str, tuple[str, int]] = {}

    video_ids = safe_list(row.get("video_sampled_pid_list"))
    video_ts = safe_list(row.get("video_ts_list"))
    video_neg = safe_list(row.get("video_neg_feedback_list"))
    video_strength = []
    for i, pid in enumerate(video_ids):
        if pid is None or (i < len(video_neg) and float(video_neg[i] or 0) > 0):
            continue
        score = 0.0
        for field in ["video_like_list", "video_comment_list", "video_forward_list", "video_collect_list", "video_play_done_list"]:
            values = safe_list(row.get(field))
            if i < len(values):
                score += float(values[i] or 0)
        ts = video_ts[i] if i < len(video_ts) else i
        video_strength.append((ts, score, int(pid)))
    if video_strength:
        targets["video/video"] = ("video/video", max(video_strength, key=lambda x: (x[0], x[1]))[2])

    history_ids = safe_list(row.get("video_history_sampled_pid_list"))
    history_ts = safe_list(row.get("video_history_ts_list"))
    history_actions = []
    for i in range(len(history_ids)):
        labels = []
        for field, label in [
            ("video_history_like_list", "点赞"), ("video_history_comment_list", "评论"),
            ("video_history_forward_list", "转发"), ("video_history_collect_list", "收藏"),
            ("video_history_play_done_list", "完播"), ("video_history_neg_feedback_list", "负反馈"),
        ]:
            values = safe_list(row.get(field))
            if i < len(values) and float(values[i] or 0) > 0:
                labels.append(label)
        if not labels:
            watch = safe_list(row.get("video_history_watch_time_list"))
            duration = safe_list(row.get("video_history_duration_list"))
            if i < len(watch) and i < len(duration) and float(duration[i] or 0) > 0 and float(watch[i] or 0) / float(duration[i]) >= 0.75:
                labels.append("长播")
        history_actions.append("/".join(labels) if labels else "浏览")
    events["video/video"] = aligned_events(history_ids, history_actions, history_ts, domain="video/video")

    ec_orders = safe_list(row.get("ec_good_order_item_id_list_extend"))
    ec_order_lags = safe_list(row.get("ec_trunc_buy_lag"))
    ec_clicks = safe_list(row.get("ec_good_click_item_id_list_extend"))
    ec_click_lags = safe_list(row.get("ec_trunc_clk_lag"))
    order_candidates = [
        (float(ec_order_lags[i] or 0) if i < len(ec_order_lags) else float(i), int(pid))
        for i, pid in enumerate(ec_orders) if pid is not None
    ]
    click_candidates = [
        (float(ec_click_lags[i] or 0) if i < len(ec_click_lags) else float(i), int(pid))
        for i, pid in enumerate(ec_clicks) if pid is not None
    ]
    if order_candidates:
        targets["goods"] = ("goods", min(order_candidates, key=lambda x: x[0])[1])
    elif click_candidates:
        targets["goods"] = ("goods", min(click_candidates, key=lambda x: x[0])[1])
    else:
        ec_ids = safe_list(row.get("ec_item_id_list"))
        ec_labels = safe_list(row.get("ec_cvr_label_list"))
        ec_times = safe_list(row.get("ec_time_ms_list"))
        ec_candidates = []
        for i, pid in enumerate(ec_ids):
            if pid is None:
                continue
            label = int(ec_labels[i] or 0) if i < len(ec_labels) else 0
            ts = ec_times[i] if i < len(ec_times) else i
            ec_candidates.append((label, ts, int(pid)))
        if ec_candidates:
            positive = [item for item in ec_candidates if item[0] > 0]
            chosen = max(positive or ec_candidates, key=lambda x: (x[1], x[0]))
            targets["goods"] = ("goods", chosen[2])
    ec_hist = safe_list(row.get("ec_colossus_rs_item_id_list"))
    ec_actions = []
    for i in range(len(ec_hist)):
        labels = []
        for field, label in [
            ("ec_colossus_rs_is_buy_list", "购买"),
            ("ec_colossus_rs_is_cart_list", "加购"),
            ("ec_colossus_rs_is_click_list", "点击"),
        ]:
            values = safe_list(row.get(field))
            if i < len(values) and int(values[i] or 0) > 0:
                labels.append(label)
        ec_actions.append("/".join(labels) if labels else "浏览")
    events["goods"] = (
        aligned_events(ec_orders, ["购买"] * len(ec_orders), [-float(value or 0) for value in ec_order_lags], domain="goods", cap=10)
        + aligned_events(ec_clicks, ["点击"] * len(ec_clicks), [-float(value or 0) for value in ec_click_lags], domain="goods", cap=10)
        + aligned_events(ec_hist, ec_actions, domain="goods", cap=12)
    )[-24:]

    live_ids = safe_list(row.get("live_hist_author_id_list"))
    live_times = safe_list(row.get("live_hist_timestamp_list"))
    live_actions = []
    live_candidates = []
    for i, pid in enumerate(live_ids):
        if pid is None:
            live_actions.append("观看")
            continue
        labels = []
        for field, label in [
            ("live_hist_follow_author_cnt_list", "关注"),
            ("live_hist_comment_cnt_list", "评论"),
            ("live_hist_like_cnt_list", "点赞"),
        ]:
            values = safe_list(row.get(field))
            if i < len(values) and int(values[i] or 0) > 0:
                labels.append(label)
        action = "/".join(labels) if labels else "观看"
        live_actions.append(action)
        ts = live_times[i] if i < len(live_times) else str(i)
        live_candidates.append((ts, len(labels), int(pid)))
    if live_candidates:
        targets["live"] = ("live", max(live_candidates, key=lambda x: (x[0], x[1]))[2])
    events["live"] = aligned_events(live_ids, live_actions, live_times, domain="live")

    deep_ids = safe_list(row.get("outer_loop_deep_target_pid"))
    deep_times = safe_list(row.get("outer_loop_deep_target_pid_ts"))
    click_ids = safe_list(row.get("outer_loop_history_action_pid_list_click"))
    click_times = safe_list(row.get("outer_loop_history_action_pid_list_click_ts"))
    if deep_ids:
        pairs = [(deep_times[i] if i < len(deep_times) else i, int(pid)) for i, pid in enumerate(deep_ids) if pid is not None]
        if pairs:
            targets["video/ad"] = ("video/ad", max(pairs)[1])
    elif click_ids:
        pairs = [(click_times[i] if i < len(click_times) else i, int(pid)) for i, pid in enumerate(click_ids) if pid is not None]
        if pairs:
            targets["video/ad"] = ("video/ad", max(pairs)[1])
    pos_events = aligned_events(
        row.get("outer_loop_history_action_pid_list_pos"),
        ["深度转化"] * len(safe_list(row.get("outer_loop_history_action_pid_list_pos"))),
        row.get("outer_loop_history_action_pid_list_pos_ts"), domain="video/ad", cap=12,
    )
    click_types = safe_list(row.get("outer_loop_history_action_pid_list_click_type"))
    click_actions = ["转化" if "CONVERSION" in str(value) else "点击" for value in click_types]
    events["video/ad"] = (pos_events + aligned_events(click_ids, click_actions, click_times, domain="video/ad", cap=18))[-24:]

    if not targets:
        return None
    target_keys = set(targets.values())
    for domain in events:
        events[domain] = [event for event in events[domain] if event["key"] not in target_keys]
    return {"events": events, "targets": targets}


PROFILE_COLUMNS = [
    "video_sampled_pid_list", "video_ts_list", "video_neg_feedback_list", "video_like_list",
    "video_comment_list", "video_forward_list", "video_collect_list", "video_play_done_list",
    "video_history_sampled_pid_list", "video_history_ts_list", "video_history_like_list",
    "video_history_comment_list", "video_history_forward_list", "video_history_collect_list",
    "video_history_play_done_list", "video_history_neg_feedback_list", "video_history_watch_time_list",
    "video_history_duration_list", "ec_item_id_list", "ec_cvr_label_list", "ec_time_ms_list",
    "ec_colossus_rs_item_id_list", "ec_colossus_rs_is_click_list", "ec_colossus_rs_is_cart_list",
    "ec_colossus_rs_is_buy_list", "ec_good_click_item_id_list_extend", "ec_trunc_clk_lag",
    "ec_good_order_item_id_list_extend", "ec_trunc_buy_lag", "live_hist_author_id_list", "live_hist_timestamp_list",
    "live_hist_follow_author_cnt_list", "live_hist_comment_cnt_list", "live_hist_like_cnt_list",
    "outer_loop_history_action_pid_list_pos", "outer_loop_history_action_pid_list_pos_ts",
    "outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts",
    "outer_loop_history_action_pid_list_click_type", "outer_loop_deep_target_pid", "outer_loop_deep_target_pid_ts",
]


def sample_profiles() -> tuple[list[dict], set[tuple[str, int]]]:
    profiles = []
    needed: set[tuple[str, int]] = set()
    scanner = pads.dataset(PROFILE_DIR, format="parquet").scanner(
        columns=PROFILE_COLUMNS, batch_size=512, use_threads=True
    )
    row_idx = 0
    for batch in scanner.to_batches():
        indices = [index for index in range(batch.num_rows) if stable_u64(row_idx + index + SEED) % 17 == 0]
        row_idx += batch.num_rows
        if not indices:
            continue
        selected_batch = batch.take(pa.array(indices, type=pa.int32()))
        for row in selected_batch.to_pylist():
            compact = compact_profile(row)
            if compact is None:
                continue
            profiles.append(compact)
            needed.update(compact["targets"].values())
            for values in compact["events"].values():
                needed.update(event["key"] for event in values)
    return profiles, needed


def scan_captions(needed: set[tuple[str, int]]) -> tuple[dict, dict]:
    profile_captions = {}
    r0_captions = {}
    per_domain = Counter()
    scanner = pads.dataset(PID2CAPTION_DIR, format="parquet").scanner(
        columns=["pid", "domain", "caption"], batch_size=32768, use_threads=True
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            domain, pid, caption = row.get("domain"), row.get("pid"), row.get("caption")
            if domain not in DOMAIN_PREFIX or pid is None or not isinstance(caption, str):
                continue
            caption = re.sub(r"\s+", " ", caption).strip()
            if not (6 <= len(caption) <= 500):
                continue
            key = (domain, int(pid))
            if key in needed and key not in profile_captions:
                profile_captions[key] = caption
            if per_domain[domain] < 35000 and stable_u64(int(pid) + SEED + len(domain)) % 257 == 0:
                r0_captions[key] = caption
                per_domain[domain] += 1
    needed.update(r0_captions)
    return profile_captions, r0_captions


def scan_tags(needed: set[tuple[str, int]]) -> dict:
    tags = {}
    scanner = pads.dataset(PID2TAG_DIR, format="parquet").scanner(
        columns=["pid", "domain", "tag_lv3"], batch_size=65536, use_threads=True
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            domain, pid, tag = row.get("domain"), row.get("pid"), row.get("tag_lv3")
            if pid is None or not isinstance(tag, str):
                continue
            key = (domain, int(pid))
            if key in needed:
                tags[key] = re.sub(r"\s+", " ", tag).strip()[:180]
    return tags


def scan_sids(needed: set[tuple[str, int]]) -> dict:
    sids = {}
    scanner = pads.dataset(PID2SID_DIR, format="parquet").scanner(
        columns=["pid", "domain", "sid_three"], batch_size=65536, use_threads=True
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            domain, pid = row.get("domain"), row.get("pid")
            if pid is None:
                continue
            key = (domain, int(pid))
            if key not in needed:
                continue
            token = sid_token(domain, row.get("sid_three"))
            if token:
                sids[key] = token
    return sids


def build_r0_pool(r0_captions: dict, tags: dict, sids: dict) -> list[dict]:
    pool = []
    systems = {
        "video/video": "你是一位视频数据分析专家，负责将视频文本映射为精确的视频token。",
        "video/ad": "你是一位广告内容分析专家，负责将广告文本映射为精确的广告token。",
        "goods": "你是一位商品内容分析专家，负责将商品文本映射为精确的商品token。",
        "live": "你是一位直播内容分析专家，负责将主播描述映射为精确的主播token。",
    }
    for key, caption in r0_captions.items():
        token = sids.get(key)
        if not token:
            continue
        domain = key[0]
        cn = DOMAIN_CN[domain]
        pool.append(record(
            systems[domain], route_prompt(f"请根据以下{cn}描述生成唯一匹配的{cn}token：{caption}", "no_think"),
            token, "r0", "caption_to_sid_direct",
        ))
        if stable_u64(key[1] + SEED) % 2 == 0:
            pool.append(record(
                f"你是一名专业的{cn}内容理解助手，请根据itemic token生成准确描述。",
                route_prompt(f"请描述{cn}token {token} 所表示的内容。", "no_think"),
                caption, "r0", "sid_to_caption_direct",
            ))
        else:
            tag = tags.get(key, "")
            levels = [part.strip() for part in re.split(r"[-/|]", tag) if part.strip()]
            if len(levels) >= 3:
                thought = f"第一级语义定位到{levels[0]}，第二级缩小到{levels[1]}，第三级细化为{levels[2]}；三级组合后形成具体内容描述。"
            else:
                thought = f"先由第一级token判断{cn}大类，再用第二级token缩小主题与场景，最后由第三级token补充细粒度属性，三级语义共同对应给定内容。"
            pool.append(record(
                f"你是一名专业的{cn}内容理解助手，请根据itemic token生成准确描述。",
                route_prompt(f"请描述{cn}token {token} 所表示的内容。", "think"),
                f"<think>\n{thought}\n</think>\n{caption}", "r0", "sid_to_caption_cot",
            ))
    return pool


def display_semantic(key: tuple[str, int], captions: dict, tags: dict) -> str:
    value = tags.get(key) or captions.get(key) or ""
    value = re.sub(r"[\r\n]+", " ", value).strip()
    return value[:72]


def event_groups(events: list[dict], sids: dict, excluded_sid: str) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for event in reversed(events):
        token = sids.get(event["key"])
        if not token or token == excluded_sid or token in seen:
            continue
        seen.add(token)
        grouped[event["action"]].append(token)
    return [(action, tokens[:12]) for action, tokens in grouped.items() if tokens]


def build_profile_prompt(events_by_domain: dict, target_domain: str, target_sid: str, sids: dict) -> tuple[str, list[dict]]:
    lines = ["用户多域历史行为："]
    visible_events = []
    for domain in TARGET_ORDER[target_domain]:
        groups = event_groups(events_by_domain.get(domain, []), sids, target_sid)
        if not groups:
            continue
        parts = []
        for action, tokens in groups:
            parts.append(f"{action}过的{DOMAIN_CN[domain]}有 " + ", ".join(tokens))
        lines.append(f"用户在{DOMAIN_CN[domain]}域: " + "；".join(parts) + "。")
        for event in events_by_domain.get(domain, []):
            token = sids.get(event["key"])
            if token and token != target_sid:
                visible_events.append(event)
    lines.append(f"请推断用户接下来最可能交互的{DOMAIN_CN[target_domain]}。")
    return "\n".join(lines), visible_events


def build_r3_cot(visible_events: list[dict], target_domain: str, sids: dict, captions: dict, tags: dict) -> str:
    candidates = []
    seen = set()
    for event in reversed(visible_events):
        token = sids.get(event["key"])
        if not token or token in seen or "负反馈" in event["action"]:
            continue
        seen.add(token)
        semantic = display_semantic(event["key"], captions, tags)
        candidates.append((token, event["action"], semantic))
        if len(candidates) >= 3:
            break
    if not candidates:
        return "先压缩多域历史中的近期和深度交互，再比较少量候选兴趣方向，最后选择与目标域最匹配的延续方向。"
    evidence = "；".join(
        f"{token} 的{action}信号" + (f"对应{semantic}" if semantic else "")
        for token, action, semantic in candidates[:2]
    )
    hypotheses = "；".join(
        f"方向{i + 1}由{token}的{action}行为支持" + (f"，主题集中在{semantic}" if semantic else "")
        for i, (token, action, semantic) in enumerate(candidates)
    )
    strongest = candidates[0]
    return (
        f"【画像压缩】多域历史中较强且较新的证据包括{evidence}，孤立浏览和负反馈不作为稳定偏好。\n"
        f"【兴趣展开】保留三个以内的候选：{hypotheses}。\n"
        f"【转移判断】近期{strongest[1]}信号优先级最高，并与{DOMAIN_CN[target_domain]}目标域兼容；"
        "因此沿最强兴趣方向继续预测，答案只在思考结束后给出。"
    )


def build_r3_pools(profiles: list[dict], captions: dict, tags: dict, sids: dict) -> tuple[list[dict], list[dict], dict]:
    think_pool = []
    direct_pool = []
    domain_counts = Counter()
    leakage = 0
    system = "你是一个推荐系统助手，擅长根据用户属性与多域历史行为预测用户的内容偏好。"
    for profile in profiles:
        for target_domain, target_key in profile["targets"].items():
            target_sid = sids.get(target_key)
            if not target_sid:
                continue
            prompt, visible = build_profile_prompt(profile["events"], target_domain, target_sid, sids)
            if len(visible) < 4 or target_sid in prompt:
                leakage += int(target_sid in prompt)
                continue
            thought = build_r3_cot(visible, target_domain, sids, captions, tags)
            if target_sid in thought:
                leakage += 1
                continue
            direct_pool.append(record(
                system, route_prompt(prompt, "no_think"), target_sid,
                "r3", f"profile_{target_domain}_direct", _target_sid=target_sid,
            ))
            think_pool.append(record(
                system, route_prompt(prompt, "think"), f"<think>\n{thought}\n</think>\n{target_sid}",
                "r3", f"profile_{target_domain}_cot", _target_sid=target_sid,
            ))
            domain_counts[target_domain] += 1
    return think_pool, direct_pool, {"domain_targets": dict(domain_counts), "target_leakage_dropped": leakage}


def build_old_pools() -> dict[str, list[dict]]:
    grouped = legacy.load_raw_records()
    rec_direct = legacy.make_rec_no_think_augments(grouped["rec"])
    user_strict = legacy.make_user_route_augments(grouped["user"], cap_logic_events=True, add_missing_no_think=True)
    user_json = legacy.make_user_json_augments(grouped["user"])
    item_routes = legacy.make_item_route_augments(grouped["item"])
    pools = {
        "old_rec": list(grouped["rec"]) + rec_direct,
        "old_user": list(grouped["user"]) + user_strict + user_json,
        "old_item": list(grouped["item"]) + item_routes,
    }
    for group, values in pools.items():
        for value in values:
            value["_group"] = group
            value.setdefault("_variant", "original_competition")
    return pools


def strip_internal(value: dict) -> dict:
    return {key: value[key] for key in ["instruction", "input", "output", "history"]}


def sample_component(values: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(values):
        raise RuntimeError(f"component has {len(values)} records but {count} requested")
    return random.Random(seed).sample(values, count)


def write_dataset(name: str, spec: dict[str, int], pools: dict[str, list[dict]], seed: int) -> dict:
    selected = []
    for index, (pool_name, count) in enumerate(spec.items()):
        selected.extend(sample_component(pools[pool_name], count, seed + index * 101))
    random.Random(seed).shuffle(selected)
    path = DATA_DIR / f"{name}.jsonl"
    digest = hashlib.sha256()
    groups = Counter()
    variants = Counter()
    routes = Counter()
    leakage = 0
    with path.open("w", encoding="utf-8") as fp:
        for value in selected:
            clean = strip_internal(value)
            target_sid = value.get("_target_sid")
            if target_sid:
                thought = THINK_RE.fullmatch(clean["output"])
                reason = thought.group(1) if thought else ""
                if target_sid in clean["input"] or target_sid in reason:
                    leakage += 1
            line = json.dumps(clean, ensure_ascii=False) + "\n"
            fp.write(line)
            digest.update(line.encode("utf-8"))
            groups[value.get("_group", "unknown")] += 1
            variants[value.get("_variant", "unknown")] += 1
            routes["no_think" if "/no_think" in clean["input"] else "think" if "/think" in clean["input"] else "none"] += 1
    if leakage:
        raise RuntimeError(f"{name}: detected {leakage} target leakage records")
    return {
        "name": name,
        "path": str(path.resolve()),
        "records": len(selected),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "requested_components": spec,
        "groups": dict(groups),
        "variants": dict(variants),
        "routes": dict(routes),
        "target_leakage": leakage,
        "seed": seed,
    }


def validate_clean_record(value: dict) -> tuple[str, str]:
    prompt = value["input"]
    output = value["output"].strip()
    route_count = prompt.count("/think") + prompt.count("/no_think")
    if route_count != 1:
        raise RuntimeError(f"{value.get('_variant')}: expected one route marker, found {route_count}")
    route = "no_think" if "/no_think" in prompt else "think"
    thought = ""
    final = output
    if route == "think":
        match = THINK_RE.fullmatch(output)
        if not match or not match.group(1).strip():
            raise RuntimeError(f"{value.get('_variant')}: malformed or empty CoT output")
        thought, final = match.group(1).strip(), match.group(2).strip()
        if output.count("<think>") != 1 or output.count("</think>") != 1:
            raise RuntimeError(f"{value.get('_variant')}: nested or repeated think tags")
    elif "<think>" in output or "</think>" in output:
        raise RuntimeError(f"{value.get('_variant')}: no-think output leaked think tags")

    group = value.get("_group", "")
    variant = value.get("_variant", "")
    if group in {"r3_clean", "old_rec_clean", "old_item_clean"} or variant == "caption_to_sid_collision_free":
        if not ITEMIC_RE.fullmatch(final):
            raise RuntimeError(f"{variant}: final output is not exactly one itemic token")
    if group == "old_user_clean":
        try:
            json.loads(final)
        except Exception as exc:
            raise RuntimeError(f"{variant}: invalid final JSON: {exc}") from exc
    target_sid = value.get("_target_sid")
    if target_sid and (target_sid in prompt or target_sid in thought):
        raise RuntimeError(f"{variant}: target SID leakage")
    if group == "r3_clean":
        references = ITEMIC_RE.findall(thought)
        if route == "think" and (len(set(references)) < 2 or any(token not in prompt for token in references)):
            raise RuntimeError(f"{variant}: ungrounded or insufficient CoT evidence")
    return route, final


def write_clean_dataset(name: str, spec: dict[str, int], pools: dict[str, list[dict]], seed: int) -> dict:
    selected = []
    seen = set()
    for index, (pool_name, count) in enumerate(spec.items()):
        values = list(pools.get(pool_name, []))
        random.Random(seed + index * 101).shuffle(values)
        accepted = 0
        for value in values:
            key = (
                re.sub(r"\s+", " ", value["instruction"]).strip(),
                re.sub(r"\s+", " ", value["input"]).strip(),
                re.sub(r"\s+", " ", value["output"]).strip(),
            )
            if key in seen:
                continue
            validate_clean_record(value)
            seen.add(key)
            selected.append(value)
            accepted += 1
            if accepted >= count:
                break
        if accepted != count:
            raise RuntimeError(
                f"{name}: pool {pool_name} supplied {accepted}/{count} unique valid records "
                f"from {len(values)} candidates"
            )
    random.Random(seed).shuffle(selected)
    path = DATA_DIR / f"{name}.jsonl"
    digest = hashlib.sha256()
    groups = Counter()
    variants = Counter()
    routes = Counter()
    domains = Counter()
    output_chars = []
    with path.open("w", encoding="utf-8") as fp:
        for value in selected:
            route, _ = validate_clean_record(value)
            clean = strip_internal(value)
            line = json.dumps(clean, ensure_ascii=False) + "\n"
            fp.write(line)
            digest.update(line.encode("utf-8"))
            groups[value.get("_group", "unknown")] += 1
            variants[value.get("_variant", "unknown")] += 1
            routes[route] += 1
            if value.get("_domain"):
                domains[value["_domain"]] += 1
            output_chars.append(len(clean["output"]))
    ordered_lengths = sorted(output_chars)
    return {
        "name": name,
        "path": str(path.resolve()),
        "records": len(selected),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "requested_components": spec,
        "groups": dict(groups),
        "variants": dict(variants),
        "routes": dict(routes),
        "domains": dict(domains),
        "canonical_duplicates": len(selected) - len(seen),
        "output_chars_p50": ordered_lengths[len(ordered_lengths) // 2],
        "output_chars_p90": ordered_lengths[int(len(ordered_lengths) * 0.9)],
        "seed": seed,
    }


def write_clean_audit(stats: dict, manifest: dict) -> None:
    json_path = LOG_DIR / "data_audit_v44_v48.json"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# v44-v48 Data Audit",
        "",
        f"Updated: {stamp()}",
        "",
        "## Policy",
        "",
        stats["policy"],
        "",
        "## Pool Counts",
        "",
    ]
    for name, count in sorted(stats["pool_counts"].items()):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Dataset Recipes", ""])
    for name, value in manifest.items():
        if name.startswith("_"):
            continue
        lines.append(
            f"- `{name}`: records={value['records']}, routes={value['routes']}, "
            f"groups={value['groups']}, domains={value['domains']}, sha256={value['sha256']}"
        )
    lines.extend([
        "",
        "## R3 Filters",
        "",
        "```json",
        json.dumps(stats["r3"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## General Filters",
        "",
        "```json",
        json.dumps(stats["general"], ensure_ascii=False, indent=2),
        "```",
    ])
    (LOG_DIR / "data_audit_v44_v48.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_clean_data() -> tuple[dict, dict]:
    import clean_data

    batch = "v44-v48"
    append(TOP_LOG, f"\n[{stamp()}] high-precision {batch} data preparation started.\n")
    pools, stats = clean_data.build_clean_pools()
    append(STATUS_PATH, json.dumps({
        "time": stamp(), "batch": batch, "event": "clean_pools_ready",
        "counts": stats["pool_counts"],
    }, ensure_ascii=False) + "\n")
    minimums = {
        pool: max(spec.get(pool, 0) for spec in CLEAN_DATASET_SPECS.values())
        for pool in {name for spec in CLEAN_DATASET_SPECS.values() for name in spec}
    }
    shortages = {
        name: {"available": len(pools.get(name, [])), "required": required}
        for name, required in minimums.items()
        if len(pools.get(name, [])) < required
    }
    if shortages:
        raise RuntimeError(f"clean pool shortages: {json.dumps(shortages, ensure_ascii=False)}")
    manifest = {
        name: write_clean_dataset(name, spec, pools, SEED + 100 + index)
        for index, (name, spec) in enumerate(CLEAN_DATASET_SPECS.items(), 1)
    }
    manifest["_cleaning_v44_v48"] = {
        "external_root": str(SOURCE),
        "cleaner": str((EXP / "clean_data.py").resolve()),
        "cleaner_sha256": sha256_file(EXP / "clean_data.py"),
        "stats_log": str((LOG_DIR / "data_audit_v44_v48.json").resolve()),
        "policy": stats["policy"],
    }
    write_clean_audit(stats, manifest)
    legacy.merge_manifest(manifest)
    return manifest, pools


def prepare_data() -> tuple[dict, dict]:
    append(TOP_LOG, f"\n[{stamp()}] unified source-data experiment preparation started.\n")
    old_pools = build_old_pools()
    append(STATUS_PATH, json.dumps({"time": stamp(), "batch": "v34-v43", "event": "old_pools_ready", "counts": {k: len(v) for k, v in old_pools.items()}}, ensure_ascii=False) + "\n")
    general = build_general_pool()
    append(STATUS_PATH, json.dumps({"time": stamp(), "batch": "v34-v43", "event": "general_ready", "count": len(general)}, ensure_ascii=False) + "\n")
    profiles, needed = sample_profiles()
    append(STATUS_PATH, json.dumps({"time": stamp(), "batch": "v34-v43", "event": "profiles_ready", "profiles": len(profiles), "keys": len(needed)}, ensure_ascii=False) + "\n")
    profile_captions, r0_captions = scan_captions(needed)
    tags = scan_tags(needed)
    sids = scan_sids(needed)
    r0 = build_r0_pool(r0_captions, tags, sids)
    r3_think, r3_direct, r3_stats = build_r3_pools(profiles, profile_captions, tags, sids)
    pools = {
        **old_pools,
        "general": general,
        "r0": r0,
        "r3_think": r3_think,
        "r3_direct": r3_direct,
    }
    minimums = {pool: max(spec.get(pool, 0) for spec in DATASET_SPECS.values()) for pool in pools}
    for pool_name, minimum in minimums.items():
        if len(pools[pool_name]) < minimum:
            raise RuntimeError(f"pool {pool_name} has {len(pools[pool_name])}, needs {minimum}")
    manifest = {
        name: write_dataset(name, spec, pools, SEED + index)
        for index, (name, spec) in enumerate(DATASET_SPECS.items(), 1)
    }
    manifest["_source"] = {
        "external_root": str(SOURCE),
        "profiles_sampled": len(profiles),
        "required_pid_domain_keys": len(needed),
        "profile_captions": len(profile_captions),
        "r0_caption_candidates": len(r0_captions),
        "mapped_sids": len(sids),
        "tags": len(tags),
        "pool_counts": {key: len(value) for key, value in pools.items()},
        "r3": r3_stats,
        "policy": "CoT retained; target SID forbidden in prompt and reasoning; direct /no_think outputs contain only the final target.",
    }
    legacy.merge_manifest(manifest)
    return manifest, pools


def prepare_public_091_data() -> dict:
    """Convert and validate the published 0.9107 dataset without changing rows."""
    if not PUBLIC_091_SOURCE.exists():
        raise RuntimeError(
            f"missing public dataset: {PUBLIC_091_SOURCE}\n"
            f"download with: curl -L --fail -o {PUBLIC_091_SOURCE} '{PUBLIC_091_URL}'"
        )
    source_hash = sha256_file(PUBLIC_091_SOURCE)
    if source_hash != PUBLIC_091_SHA256:
        raise RuntimeError(f"public dataset hash mismatch: {source_hash} != {PUBLIC_091_SHA256}")

    output = DATA_DIR / f"{PUBLIC_091_DATASET}.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    groups = Counter()
    routes = Counter()
    thought_modes = Counter()
    prompt_prefixes = {
        "请阅读下面的用户行为记录，并推断该用户在各推荐场景中的目标内容。",
        "下面给出用户历史行为线索：",
        "已知如下用户跨场景行为：",
        "以下是一个用户的多域历史行为信息：",
    }
    world_final = re.compile(r"正确答案是\s*\([A-D]\)")

    with PUBLIC_091_SOURCE.open("r", encoding="utf-8") as source_fp, temporary.open(
        "w", encoding="utf-8"
    ) as output_fp:
        for line_no, line in enumerate(source_fp, 1):
            value = json.loads(line)
            if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
                raise RuntimeError(f"unexpected public row structure at line {line_no}")
            row = value[0]
            system = row.get("system", "")
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            if not all(isinstance(part, str) for part in (system, prompt, response)):
                raise RuntimeError(f"non-string public row field at line {line_no}")

            match = THINK_RE.fullmatch(response)
            if not match:
                raise RuntimeError(f"unbalanced or missing think wrapper at line {line_no}")
            thought = match.group(1).strip()
            final = match.group(2).strip()
            route = "no_think" if "/no_think" in prompt else "think" if "/think" in prompt else "none"
            routes[route] += 1
            thought_modes["filled" if thought else "empty"] += 1
            if route == "no_think" and thought:
                raise RuntimeError(f"filled thought on /no_think row {line_no}")
            if route == "think" and not thought:
                raise RuntimeError(f"empty thought on /think row {line_no}")

            first_line = prompt.splitlines()[0] if prompt else ""
            if first_line in prompt_prefixes:
                group = "rec"
                if len(ITEMIC_RE.findall(final)) != 1:
                    raise RuntimeError(f"recommendation row {line_no} does not have one final itemic token")
            elif world_final.fullmatch(re.sub(r"\s+", "", final)):
                group = "world"
            else:
                try:
                    parsed = json.loads(final)
                except Exception:
                    parsed = None
                group = "user" if isinstance(parsed, (dict, list)) else "item"
            groups[group] += 1

            converted = {
                "instruction": system,
                "input": prompt,
                "output": response,
                "history": [],
            }
            output_fp.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")

    expected_groups = {"rec": 18651, "item": 9684, "user": 2792, "world": 1578}
    expected_routes = {"no_think": 20934, "think": 11771}
    if dict(groups) != expected_groups:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"public group audit failed: {dict(groups)} != {expected_groups}")
    if dict(routes) != expected_routes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"public route audit failed: {dict(routes)} != {expected_routes}")
    temporary.replace(output)

    meta = {
        "path": str(output.resolve()),
        "records": sum(groups.values()),
        "groups": dict(groups),
        "routes": dict(routes),
        "thought_modes": dict(thought_modes),
        "sha256": sha256_file(output),
        "source_path": str(PUBLIC_091_SOURCE.resolve()),
        "source_url": PUBLIC_091_URL,
        "source_sha256": source_hash,
        "source_license": "Apache-2.0",
        "transform": "Lossless field mapping from system/prompt/response to instruction/input/output; original seed-42 order retained.",
    }
    audit = {
        "created_at": stamp(),
        "dataset": PUBLIC_091_DATASET,
        "published_score": 0.9107,
        **meta,
    }
    (LOG_DIR / "data_audit_v49_v53.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        PUBLIC_091_DATASET: meta,
        "_public_091": {
            "source_url": PUBLIC_091_URL,
            "source_sha256": source_hash,
            "published_score": 0.9107,
            "published_recipe": "LoRA r32 alpha32 dropout0.05, lr=2e-4, wd=0.001, cosine, warmup=0.03, 1 epoch.",
        },
    }
    legacy.merge_manifest(manifest)
    append(
        TOP_LOG,
        f"\n[{stamp()}] v49-v53 fixed-data tuning batch prepared\n"
        f"- source={PUBLIC_091_SOURCE} sha256={source_hash}\n"
        f"- dataset={PUBLIC_091_DATASET} records={sum(groups.values())} groups={dict(groups)} routes={dict(routes)}\n"
        "- strategy=fixed clean data; compare exact public LoRA, longer LoRA, higher-rank LoRA, 1-epoch full SFT, and converged 3-epoch full SFT.\n"
        "- checkpoint_policy=no intermediate checkpoints; retain final weights only.\n",
    )
    return manifest


def prepare_advanced_data() -> dict:
    manifest = prepare_public_091_data()
    public_rows = []
    public_triples = set()
    user_rows = []
    with PUBLIC_091_SOURCE.open("r", encoding="utf-8") as fp:
        for line in fp:
            source = json.loads(line)[0]
            triple = (source.get("system", ""), source.get("prompt", ""), source.get("response", ""))
            public_triples.add(triple)
            row = {
                "instruction": triple[0], "input": triple[1], "output": triple[2], "history": [],
            }
            public_rows.append(row)
            _, final = legacy.split_think_output(row["output"])
            try:
                parsed = json.loads(final)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                user_rows.append(dict(row))

    rng = random.Random(SEED + 63)
    candidates = defaultdict(list)
    guard_rejections = Counter()
    for record in legacy.load_raw_records()["rec"]:
        triple = (record["instruction"], record["input"], record["output"])
        if triple in public_triples or legacy.prompt_route(record["input"]) != "think":
            continue
        match = THINK_RE.fullmatch(record["output"])
        if not match or not match.group(1).strip():
            continue
        final_tokens = ITEMIC_RE.findall(match.group(2))
        if len(final_tokens) != 1:
            continue
        token = ITEMIC_RE.search(match.group(2)).group(0)
        domain = "video" if token.startswith("<|video_begin|>") else "living" if token.startswith("<|living_begin|>") else None
        if domain is None:
            continue
        if token in record["instruction"] or token in record["input"]:
            guard_rejections["target_in_prompt"] += 1
            continue
        if token in match.group(1):
            guard_rejections["target_in_thought"] += 1
            continue
        candidates[(domain, record["instruction"], record["input"])].append(legacy.strip_internal(record))

    domain_additions = Counter()
    selected = []
    for key in sorted(candidates):
        values = sorted(
            candidates[key],
            key=lambda row: hashlib.sha256(row["output"].encode("utf-8")).hexdigest(),
        )
        choice = values[rng.randrange(len(values))]
        selected.append(choice)
        domain_additions[key[0]] += 1

    records = public_rows + user_rows + selected
    rng.shuffle(records)
    output = DATA_DIR / f"{PUBLIC_GUARD_DATASET}.jsonl"
    with output.open("w", encoding="utf-8") as fp:
        for row in records:
            fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    meta = {
        "path": str(output.resolve()),
        "records": len(records),
        "groups": {
            "public_exact": len(public_rows),
            "user_json_replay": len(user_rows),
            "video_distinct_cot": domain_additions["video"],
            "live_distinct_cot": domain_additions["living"],
        },
        "integrity": {
            "target_in_prompt": 0,
            "target_in_thought": 0,
            "rejected_before_prompt_dedup": dict(guard_rejections),
        },
        "sha256": sha256_file(output),
        "source_path": str(PUBLIC_091_SOURCE.resolve()),
        "source_sha256": PUBLIC_091_SHA256,
        "transform": (
            "Published clean rows plus one intentional replay of every parseable user target and at most one "
            "strict original CoT target per video/live prompt; no synthetic response text."
        ),
    }
    manifest[PUBLIC_GUARD_DATASET] = meta
    manifest["_advanced_v54_v63"] = {
        "guard_dataset": meta,
        "policy": "Fixed public control for v54-v62; only v63 changes data, using real labeled records.",
    }
    legacy.merge_manifest({PUBLIC_GUARD_DATASET: meta, "_advanced_v54_v63": manifest["_advanced_v54_v63"]})
    (LOG_DIR / "data_audit_v54_v63.json").write_text(
        json.dumps({"created_at": stamp(), **manifest["_advanced_v54_v63"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    append(
        TOP_LOG,
        f"\n[{stamp()}] v54-v63 advanced batch prepared\n"
        f"- fixed_dataset={PUBLIC_091_DATASET} records={manifest[PUBLIC_091_DATASET]['records']}\n"
        f"- guard_dataset={PUBLIC_GUARD_DATASET} records={len(records)} groups={meta['groups']} sha256={meta['sha256']}\n"
        "- strategy=v51-centered PEFT search, two v29-base bridges, and one real-label user/video/live data guard.\n"
        "- checkpoint_policy=no intermediate checkpoints; merge nonstandard/v29-based adapters to full models.\n",
    )
    return manifest


def prepare_focused_data() -> dict:
    """Prepare the fixed public control and a real-label user replay variant."""
    manifest = prepare_public_091_data()
    public_rows = []
    user_rows = []
    with PUBLIC_091_SOURCE.open("r", encoding="utf-8") as fp:
        for line in fp:
            source = json.loads(line)[0]
            row = {
                "instruction": source.get("system", ""),
                "input": source.get("prompt", ""),
                "output": source.get("response", ""),
                "history": [],
            }
            public_rows.append(row)
            _, final = legacy.split_think_output(row["output"])
            try:
                parsed = json.loads(final)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                user_rows.append(dict(row))

    records = public_rows + user_rows
    random.Random(SEED + 58).shuffle(records)
    output = DATA_DIR / f"{PUBLIC_USER_REPLAY_DATASET}.jsonl"
    with output.open("w", encoding="utf-8") as fp:
        for row in records:
            fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    meta = {
        "path": str(output.resolve()),
        "records": len(records),
        "groups": {
            "public_exact": len(public_rows),
            "user_json_replay": len(user_rows),
        },
        "sha256": sha256_file(output),
        "source_path": str(PUBLIC_091_SOURCE.resolve()),
        "source_sha256": PUBLIC_091_SHA256,
        "transform": (
            "Published clean rows plus one intentional replay of every parseable real user target; "
            "no synthetic response text and no video/live additions."
        ),
    }
    manifest[PUBLIC_USER_REPLAY_DATASET] = meta
    manifest["_focused_v64_v68"] = {
        "user_replay_dataset": meta,
        "policy": "Fixed public control for four runs; only v67 adds a single replay of real user labels.",
    }
    legacy.merge_manifest({
        PUBLIC_USER_REPLAY_DATASET: meta,
        "_focused_v64_v68": manifest["_focused_v64_v68"],
    })
    (LOG_DIR / "data_audit_v64_v68.json").write_text(
        json.dumps({"created_at": stamp(), **manifest["_focused_v64_v68"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    append(
        TOP_LOG,
        f"\n[{stamp()}] v64-v68 focused batch prepared\n"
        f"- fixed_dataset={PUBLIC_091_DATASET} records={manifest[PUBLIC_091_DATASET]['records']}\n"
        f"- user_replay_dataset={PUBLIC_USER_REPLAY_DATASET} records={len(records)} groups={meta['groups']} sha256={meta['sha256']}\n"
        "- strategy=v58-centered LoRA+ search, one rsLoRA combination, one real-user replay, and one PiSSA initialization.\n"
        "- checkpoint_policy=no intermediate checkpoints; merge rsLoRA/PiSSA adapters to full models.\n",
    )
    return manifest


def register_datasets(manifest: dict) -> None:
    info = json.loads(DATASET_INFO.read_text(encoding="utf-8"))
    for name, value in manifest.items():
        if name.startswith("_"):
            continue
        info[f"exp_{name}"] = {
            "file_name": value["path"],
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
        }
    DATASET_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def yaml_text(
    model: str,
    dataset: str,
    output: Path,
    lr: str,
    epochs: float,
    seed: int,
    *,
    method: str = "full",
    weight_decay: float = 0.0,
    validation_size: float = 0.0,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_rslora: bool = False,
    use_dora: bool = False,
    loraplus_lr_ratio: float | None = None,
    pissa_init: bool = False,
    pissa_iter: int = 16,
    gradient_accumulation_steps: int = 4,
) -> str:
    if method not in {"full", "lora"}:
        raise ValueError(f"unsupported fine-tuning method: {method}")
    method_lines = f"""finetuning_type: {method}
enable_liger_kernel: true"""
    if method == "lora":
        method_lines += f"""
lora_rank: {lora_rank}
lora_alpha: {lora_alpha}
lora_dropout: {lora_dropout}
lora_target: all"""
        if use_rslora:
            method_lines += "\nuse_rslora: true"
        if use_dora:
            method_lines += "\nuse_dora: true"
        if loraplus_lr_ratio is not None:
            method_lines += f"\nloraplus_lr_ratio: {loraplus_lr_ratio}"
        if pissa_init:
            method_lines += f"\npissa_init: true\npissa_iter: {pissa_iter}"
    eval_lines = ""
    if validation_size:
        eval_lines = f"""

### eval
val_size: {validation_size}
per_device_eval_batch_size: 1
eval_strategy: epoch
"""
    return f"""### model
model_name_or_path: {model}
trust_remote_code: true
flash_attn: fa2

### method
stage: sft
do_train: true
{method_lines}

### dataset
dataset: exp_{dataset}
dataset_dir: demo/LLaMA-Factory/data
template: qwen3_nothink
cutoff_len: 32768
packing: true
neat_packing: true
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: {output.resolve()}
logging_steps: 10
save_strategy: \"no\"
save_total_limit: 1
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: tensorboard
logging_dir: {(output / 'tb').resolve()}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: {lr}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: {weight_decay}
max_grad_norm: 1.0
bf16: true
pure_bf16: true
seed: {seed}
{eval_lines}"""


def run_yaml_text(run: dict, dataset: str, output: Path, model: str, lr: str, epochs: float, seed: int) -> str:
    return yaml_text(
        model,
        dataset,
        output,
        lr,
        epochs,
        seed,
        method=run.get("method", "full"),
        weight_decay=run.get("weight_decay", 0.0),
        validation_size=run.get("validation_size", 0.0),
        lora_rank=run.get("lora_rank", 32),
        lora_alpha=run.get("lora_alpha", 32),
        lora_dropout=run.get("lora_dropout", 0.05),
        use_rslora=run.get("use_rslora", False),
        use_dora=run.get("use_dora", False),
        loraplus_lr_ratio=run.get("loraplus_lr_ratio"),
        pissa_init=run.get("pissa_init", False),
        pissa_iter=run.get("pissa_iter", 16),
        gradient_accumulation_steps=run.get("gradient_accumulation_steps", 4),
    )


def prepare_configs_and_recipes(manifest: dict, runs: list[dict] | None = None) -> None:
    runs = SOURCE_RUNS if runs is None else runs
    runner_hash = sha256_file(UNIFIED_RUNNER)
    builder_hash = sha256_file(Path(__file__))
    cleaner_path = EXP / "clean_data.py"
    cleaner_hash = sha256_file(cleaner_path) if cleaner_path.exists() else None
    for run in runs:
        output = OUTPUT_DIR / run["name"]
        configs = []
        if "stages" in run:
            previous_model = run["model"]
            for index, stage in enumerate(run["stages"], 1):
                stage_output = output if index == len(run["stages"]) else OUTPUT_DIR / f".{run['name']}_stage{index}_temporary"
                config = CONFIG_DIR / f"{run['name']}_stage{index}.yaml"
                config.write_text(
                    run_yaml_text(
                        run, stage["dataset"], stage_output, previous_model,
                        stage["lr"], stage["epochs"], run["seed"] + index,
                    ),
                    encoding="utf-8",
                )
                configs.append(str(config.resolve()))
                previous_model = str(stage_output.resolve())
        else:
            config = CONFIG_DIR / f"{run['name']}.yaml"
            config.write_text(
                run_yaml_text(
                    run, run["dataset"], output, run["model"],
                    run["lr"], run["epochs"], run["seed"],
                ),
                encoding="utf-8",
            )
            configs.append(str(config.resolve()))
        run["configs"] = configs
        recipe = {
            "created_at": stamp(),
            "name": run["name"],
            "base": run["model"],
            "dataset": run["dataset"],
            "dataset_manifest": manifest[run["dataset"]],
            "learning_rate": run["lr"],
            "epochs": run["epochs"],
            "method": run.get("method", "full"),
            "weight_decay": run.get("weight_decay", 0.0),
            "validation_size": run.get("validation_size", 0.0),
            "lora": {
                "rank": run.get("lora_rank"),
                "alpha": run.get("lora_alpha"),
                "dropout": run.get("lora_dropout"),
                "target": "all" if run.get("method") == "lora" else None,
                "use_rslora": run.get("use_rslora", False),
                "use_dora": run.get("use_dora", False),
                "loraplus_lr_ratio": run.get("loraplus_lr_ratio"),
                "pissa_init": run.get("pissa_init", False),
                "pissa_iter": run.get("pissa_iter"),
            },
            "merge_to_full": run.get("merge_to_full", False),
            "stages": run.get("stages"),
            "note": run["note"],
            "evaluation": run.get("evaluation"),
            "cot_policy": "Preserve meaningful CoT for /think and pure final outputs for /no_think.",
            "checkpoint_policy": "save_strategy=no; only the final model is retained.",
            "script": str(UNIFIED_RUNNER.resolve()),
            "script_sha256": runner_hash,
            "data_builder": str(Path(__file__).resolve()),
            "data_builder_sha256": builder_hash,
            "data_cleaner": str(cleaner_path.resolve()) if run.get("batch") == "v44-v48" else None,
            "data_cleaner_sha256": cleaner_hash if run.get("batch") == "v44-v48" else None,
            "configs": configs,
            "reproduce": f"{VENV}/bin/python {UNIFIED_RUNNER.resolve()} --single {run['name']} --gpu <gpu>",
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "experiment_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy.write_run_registry()


def parse_train_result(output: Path) -> dict:
    path = output / "train_results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    eval_path = output / "eval_results.json"
    if eval_path.exists():
        eval_result = json.loads(eval_path.read_text(encoding="utf-8"))
        result.update({key: value for key, value in eval_result.items() if key.startswith("eval_")})
    state_path = output / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        result["global_step"] = state.get("global_step")
        history = [value for value in state.get("log_history", []) if "loss" in value]
        if history:
            losses = [value["loss"] for value in history]
            tail_size = max(1, len(losses) // 10)
            result["first_logged_loss"] = losses[0]
            result["last_logged_loss"] = history[-1]["loss"]
            result["min_logged_loss"] = min(losses)
            result["tail_loss_mean"] = sum(losses[-tail_size:]) / tail_size
            result["logged_loss_change"] = losses[-1] - losses[0]
        eval_history = [value for value in state.get("log_history", []) if "eval_loss" in value]
        if eval_history:
            result["eval_loss_history"] = [
                {"step": value.get("step"), "epoch": value.get("epoch"), "eval_loss": value["eval_loss"]}
                for value in eval_history
            ]
    return result


def merge_adapter_for_submission(run: dict, output: Path, log_fp) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    log_fp.write(f"[{stamp()}] merging adapter into base model for full-checkpoint submission\n")
    log_fp.flush()
    base = AutoModelForCausalLM.from_pretrained(
        run["model"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, output, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    adapter_repro = output / "adapter_repro"
    adapter_repro.mkdir(exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        source = output / name
        if source.exists():
            shutil.move(str(source), adapter_repro / name)
    del merged, peft_model, base
    log_fp.write(
        f"[{stamp()}] merged model saved to {output / 'model.safetensors'}; "
        f"reproducibility adapter moved to {adapter_repro}\n"
    )
    log_fp.flush()


def train_run(run: dict, gpu: int, batch: str = "v34-v43") -> dict:
    output = OUTPUT_DIR / run["name"]
    log_path = output / "train.log"
    started = time.time()
    result = {"name": run["name"], "gpu": gpu, "started_at": stamp(), "status": "running"}
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log_fp:
        for index, config in enumerate(run["configs"], 1):
            log_fp.write(f"[{stamp()}] stage {index}/{len(run['configs'])}: {config}\n")
            log_fp.flush()
            completed = subprocess.run(
                [str(VENV / "bin" / "llamafactory-cli"), "train", config],
                cwd=ROOT, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                result["status"] = f"failed_stage_{index}_rc_{completed.returncode}"
                break
            if index < len(run["configs"]):
                temporary = Path(json.loads((output / "experiment_recipe.json").read_text(encoding="utf-8"))["configs"][index - 1])
                del temporary
        else:
            result["status"] = "ok"
        if result["status"] == "ok" and run.get("merge_to_full"):
            try:
                merge_adapter_for_submission(run, output, log_fp)
                result["merged_to_full"] = True
            except Exception as exc:
                log_fp.write(f"[{stamp()}] adapter merge failed: {type(exc).__name__}: {exc}\n")
                log_fp.flush()
                result["status"] = f"failed_merge_{type(exc).__name__}"
    result["wall_seconds"] = round(time.time() - started, 2)
    result["finished_at"] = stamp()
    result.update(parse_train_result(output))
    result["batch"] = batch
    recipe_path = output / "experiment_recipe.json"
    if recipe_path.exists():
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipe["training_result"] = result
        recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    append(STATUS_PATH, json.dumps(result, ensure_ascii=False) + "\n")
    return result


def worker(
    gpu: int,
    queue: list[dict],
    lock: threading.Lock,
    results: list[dict],
    batch: str = "v34-v43",
) -> None:
    while True:
        with lock:
            if not queue:
                return
            run = queue.pop(0)
        append(STATUS_PATH, json.dumps({"time": stamp(), "batch": batch, "event": "launch", "gpu": gpu, "name": run["name"]}, ensure_ascii=False) + "\n")
        results.append(train_run(run, gpu, batch))
        if "stages" in run:
            temp = OUTPUT_DIR / f".{run['name']}_stage1_temporary"
            if temp.exists():
                shutil.rmtree(temp)


def run_all(runs: list[dict] | None = None, batch: str = "v34-v43") -> list[dict]:
    runs = SOURCE_RUNS if runs is None else runs
    queue = list(runs)
    lock = threading.Lock()
    results: list[dict] = []
    threads = [threading.Thread(target=worker, args=(gpu, queue, lock, results, batch), daemon=False) for gpu in [0, 1]]
    for thread in threads:
        thread.start()
    last_heartbeat = 0.0
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=5)
        if time.time() - last_heartbeat >= 1800:
            last_heartbeat = time.time()
            append(STATUS_PATH, json.dumps({"time": stamp(), "batch": batch, "event": "heartbeat", "completed": len(results), "remaining": len(queue)}, ensure_ascii=False) + "\n")
    order = {run["name"]: i for i, run in enumerate(runs)}
    return sorted(results, key=lambda item: order[item["name"]])


def validate_outputs(results: list[dict], expected_runs: list[dict] | None = None) -> list[str]:
    errors = []
    result_by_name = {item["name"]: item for item in results}
    for run in expected_runs or SOURCE_RUNS:
        output = OUTPUT_DIR / run["name"]
        if result_by_name.get(run["name"], {}).get("status") != "ok":
            errors.append(f"{run['name']}: training status is not ok")
        if run.get("merge_to_full"):
            weights = output / "model.safetensors"
            if not weights.exists() or weights.stat().st_size < 100_000_000:
                errors.append(f"{run['name']}: missing or undersized merged model.safetensors")
            if (output / "adapter_config.json").exists():
                errors.append(f"{run['name']}: root adapter_config.json would mask the merged full model")
            adapter_repro = output / "adapter_repro"
            if not (adapter_repro / "adapter_config.json").exists() or not (
                adapter_repro / "adapter_model.safetensors"
            ).exists():
                errors.append(f"{run['name']}: missing reproducibility adapter under adapter_repro")
        elif run.get("method") == "lora":
            weights = output / "adapter_model.safetensors"
            config = output / "adapter_config.json"
            if not weights.exists() or weights.stat().st_size < 1_000_000:
                errors.append(f"{run['name']}: missing or undersized adapter_model.safetensors")
            if not config.exists():
                errors.append(f"{run['name']}: missing adapter_config.json")
        else:
            weights = output / "model.safetensors"
            if not weights.exists() or weights.stat().st_size < 100_000_000:
                errors.append(f"{run['name']}: missing or undersized model.safetensors")
        checkpoints = list(output.glob("checkpoint-*"))
        if checkpoints:
            errors.append(f"{run['name']}: unexpected intermediate checkpoints")
    return errors


def cleanup_generated(manifest: dict) -> dict:
    removed_data = []
    for name, value in manifest.items():
        if name.startswith("_"):
            continue
        path = Path(value["path"])
        if path.exists():
            removed_data.append({"path": str(path), "bytes": path.stat().st_size})
            path.unlink()
    removed_caches = []
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "json"
    names = {
        Path(value["path"]).name
        for name, value in manifest.items()
        if not name.startswith("_") and isinstance(value, dict) and value.get("path")
    }
    if cache_root.exists():
        for info_path in cache_root.glob("default-*/0.0.0/*/dataset_info.json"):
            try:
                text = info_path.read_text(encoding="utf-8")
            except Exception:
                continue
            if not any(name in text for name in names):
                continue
            directory = info_path.parent
            size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            removed_caches.append({"path": str(directory), "bytes": size})
            shutil.rmtree(directory)
    return {"datasets": removed_data, "hf_caches": removed_caches}


def write_summary(
    results: list[dict],
    manifest: dict,
    cleanup: dict,
    errors: list[str],
    runs: list[dict] | None = None,
    batch: str = "v34-v43",
) -> None:
    runs = SOURCE_RUNS if runs is None else runs
    legacy.merge_results(results)
    legacy.write_unified_summary()
    append(TOP_LOG, f"\n[{stamp()}] {batch} batch completed\n")
    append(TOP_LOG, f"- summary={LOG_DIR / 'summary.md'}\n- manifest={DATA_DIR / 'manifest.json'}\n")
    run_by_name = {run["name"]: run for run in runs}
    for result in results:
        run = run_by_name[result["name"]]
        append(TOP_LOG, f"- {run['name']}: base={run['model']}, dataset={run['dataset']}, lr={run['lr']}, epochs={run['epochs']}, train_loss={result.get('train_loss', '-')}, runtime={result.get('wall_seconds', '-')}s, status={result.get('status')}, output={OUTPUT_DIR / run['name']}\n  note={run['note']}\n")
    append(TOP_LOG, f"- cleanup: generated_jsonl={len(cleanup['datasets'])}, arrow_caches={len(cleanup['hf_caches'])}; checkpoints=none expected.\n")
    append(TOP_LOG, f"- validation_errors={errors}\n")


def run_source_experiments(*, single: str | None = None, gpu: int = 0, prepare_only: bool = False) -> int:
    for path in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    if not BASE_V29.exists():
        raise SystemExit(f"missing v29 base: {BASE_V29}")
    for path in [GENERAL_DIR, PROFILE_DIR, PID2SID_DIR, PID2CAPTION_DIR, PID2TAG_DIR]:
        if not path.exists():
            raise SystemExit(f"missing source data: {path}")
    manifest, _ = prepare_data()
    register_datasets(manifest)
    prepare_configs_and_recipes(manifest)
    if prepare_only:
        append(STATUS_PATH, json.dumps({"time": stamp(), "batch": "v34-v43", "event": "prepare_only_exit"}, ensure_ascii=False) + "\n")
        return 0
    if single:
        run = next(item for item in SOURCE_RUNS if item["name"] == single)
        results = [train_run(run, gpu)]
        errors = validate_outputs(results, [run])
        cleanup = cleanup_generated(manifest)
        legacy.merge_results(results)
        legacy.write_unified_summary()
        (OUTPUT_DIR / run["name"] / "single_run_result.json").write_text(
            json.dumps({"result": results[0], "errors": errors, "cleanup": cleanup}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1 if errors else 0
    results = run_all()
    errors = validate_outputs(results)
    cleanup = cleanup_generated(manifest)
    write_summary(results, manifest, cleanup, errors)
    return 1 if errors else 0


def load_prepared_clean_manifest() -> dict:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise RuntimeError("--reuse-prepared requested but manifest.json is missing")
    shared = json.loads(path.read_text(encoding="utf-8"))
    manifest = {}
    for name in CLEAN_DATASET_SPECS:
        value = shared.get(name)
        if not value:
            raise RuntimeError(f"--reuse-prepared requested but {name} is absent from manifest")
        data_path = Path(value["path"])
        if not data_path.exists():
            raise RuntimeError(f"--reuse-prepared requested but dataset is missing: {data_path}")
        digest = sha256_file(data_path)
        if digest != value["sha256"]:
            raise RuntimeError(f"prepared dataset hash mismatch for {name}: {digest} != {value['sha256']}")
        manifest[name] = value
    if shared.get("_cleaning_v44_v48"):
        manifest["_cleaning_v44_v48"] = shared["_cleaning_v44_v48"]
    return manifest


def run_clean_experiments(
    *,
    single: str | None = None,
    gpu: int = 0,
    prepare_only: bool = False,
    reuse_prepared: bool = False,
) -> int:
    batch = "v44-v48"
    for path in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    if not BASE_V29.exists():
        raise SystemExit(f"missing v29 base: {BASE_V29}")
    for path in [GENERAL_DIR, PROFILE_DIR, PID2SID_DIR, PID2CAPTION_DIR, PID2TAG_DIR]:
        if not path.exists():
            raise SystemExit(f"missing source data: {path}")
    if reuse_prepared:
        manifest = load_prepared_clean_manifest()
        append(STATUS_PATH, json.dumps({
            "time": stamp(), "batch": batch, "event": "reuse_prepared",
            "datasets": list(CLEAN_DATASET_SPECS),
        }, ensure_ascii=False) + "\n")
    else:
        manifest, _ = prepare_clean_data()
    register_datasets(manifest)
    prepare_configs_and_recipes(manifest, CLEAN_RUNS)
    if prepare_only:
        append(STATUS_PATH, json.dumps({
            "time": stamp(), "batch": batch, "event": "prepare_only_exit",
        }, ensure_ascii=False) + "\n")
        return 0
    if single:
        run = next(item for item in CLEAN_RUNS if item["name"] == single)
        results = [train_run(run, gpu, batch)]
        errors = validate_outputs(results, [run])
        cleanup = cleanup_generated(manifest)
        legacy.merge_results(results)
        legacy.write_unified_summary()
        (OUTPUT_DIR / run["name"] / "single_run_result.json").write_text(
            json.dumps({"result": results[0], "errors": errors, "cleanup": cleanup}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1 if errors else 0
    results = run_all(CLEAN_RUNS, batch)
    errors = validate_outputs(results, CLEAN_RUNS)
    cleanup = cleanup_generated(manifest)
    write_summary(results, manifest, cleanup, errors, CLEAN_RUNS, batch)
    return 1 if errors else 0


def run_tuning_experiments(
    *,
    single: str | None = None,
    gpu: int = 0,
    prepare_only: bool = False,
) -> int:
    batch = "v49-v53"
    for path in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    manifest = prepare_public_091_data()
    register_datasets(manifest)
    prepare_configs_and_recipes(manifest, TUNING_RUNS)
    if prepare_only:
        append(STATUS_PATH, json.dumps({
            "time": stamp(), "batch": batch, "event": "prepare_only_exit",
        }, ensure_ascii=False) + "\n")
        return 0
    if single:
        run = next(item for item in TUNING_RUNS if item["name"] == single)
        results = [train_run(run, gpu, batch)]
        errors = validate_outputs(results, [run])
        cleanup = cleanup_generated(manifest)
        legacy.merge_results(results)
        legacy.write_unified_summary()
        (OUTPUT_DIR / run["name"] / "single_run_result.json").write_text(
            json.dumps({"result": results[0], "errors": errors, "cleanup": cleanup}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1 if errors else 0

    by_name = {run["name"]: run for run in TUNING_RUNS}
    execution_order = [
        by_name["v53_public091_full_lr1e5_ep3"],
        by_name["v51_public091_lora_r64_lr1e4_ep2"],
        by_name["v52_public091_full_lr2e5_ep1"],
        by_name["v50_public091_lora_r32_lr1e4_ep2"],
        by_name["v49_public091_lora_r32_lr2e4_ep1"],
    ]
    results = run_all(execution_order, batch)
    errors = validate_outputs(results, TUNING_RUNS)
    cleanup = cleanup_generated(manifest)
    write_summary(results, manifest, cleanup, errors, TUNING_RUNS, batch)
    return 1 if errors else 0


def run_advanced_experiments(
    *,
    single: str | None = None,
    gpu: int = 0,
    prepare_only: bool = False,
) -> int:
    batch = "v54-v63"
    for path in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    if not BASE_V29.exists():
        raise SystemExit(f"missing v29 base: {BASE_V29}")
    manifest = prepare_advanced_data()
    register_datasets(manifest)
    prepare_configs_and_recipes(manifest, ADVANCED_RUNS)
    if prepare_only:
        append(STATUS_PATH, json.dumps({
            "time": stamp(), "batch": batch, "event": "prepare_only_exit",
        }, ensure_ascii=False) + "\n")
        return 0
    if single:
        run = next(item for item in ADVANCED_RUNS if item["name"] == single)
        results = [train_run(run, gpu, batch)]
        errors = validate_outputs(results, [run])
        cleanup = cleanup_generated(manifest)
        legacy.merge_results(results)
        legacy.write_unified_summary()
        (OUTPUT_DIR / run["name"] / "single_run_result.json").write_text(
            json.dumps({"result": results[0], "errors": errors, "cleanup": cleanup}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1 if errors else 0

    by_name = {run["name"]: run for run in ADVANCED_RUNS}
    execution_order = [
        by_name["v55_public091_lora_r128_lr8e5_ep2"],
        by_name["v56_public091_rslora_r128_a16_lr6e5_ep2"],
        by_name["v57_public091_dora_r64_lr8e5_ep2"],
        by_name["v63_public_guard_lora_r64_lr8e5_ep2"],
        by_name["v54_public091_lora_r96_lr1e4_ep2"],
        by_name["v58_public091_loraplus_r64_lr2e5_x16_ep2"],
        by_name["v59_public091_lora_r64_drop0_lr1e4_ep2"],
        by_name["v60_public091_lora_r64_a128_lr5e5_ep2"],
        by_name["v61_v29_public091_lora_r64_lr5e5_ep1"],
        by_name["v62_v29_public091_lora_r64_lr2e5_ep1"],
    ]
    results = run_all(execution_order, batch)
    errors = validate_outputs(results, ADVANCED_RUNS)
    cleanup = cleanup_generated(manifest)
    write_summary(results, manifest, cleanup, errors, ADVANCED_RUNS, batch)
    return 1 if errors else 0


def run_focused_experiments(
    *,
    single: str | None = None,
    gpu: int = 0,
    prepare_only: bool = False,
) -> int:
    batch = "v64-v68"
    for path in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    manifest = prepare_focused_data()
    register_datasets(manifest)
    prepare_configs_and_recipes(manifest, FOCUSED_RUNS)
    if prepare_only:
        append(STATUS_PATH, json.dumps({
            "time": stamp(), "batch": batch, "event": "prepare_only_exit",
        }, ensure_ascii=False) + "\n")
        return 0
    if single:
        run = next(item for item in FOCUSED_RUNS if item["name"] == single)
        results = [train_run(run, gpu, batch)]
        errors = validate_outputs(results, [run])
        cleanup = cleanup_generated(manifest)
        legacy.merge_results(results)
        legacy.write_unified_summary()
        (OUTPUT_DIR / run["name"] / "single_run_result.json").write_text(
            json.dumps({"result": results[0], "errors": errors, "cleanup": cleanup}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1 if errors else 0

    by_name = {run["name"]: run for run in FOCUSED_RUNS}
    execution_order = [
        by_name["v66_public091_rslora_loraplus_r128_a16_lr1e5_x16_ep2"],
        by_name["v67_public_user_replay_loraplus_r64_lr2e5_x16_ep2"],
        by_name["v65_public091_loraplus_r128_lr1e5_x16_ep2"],
        by_name["v68_public091_pissa_loraplus_r64_lr2e5_x8_ep2"],
        by_name["v64_public091_loraplus_r64_lr2e5_x8_ep2"],
    ]
    results = run_all(execution_order, batch)
    errors = validate_outputs(results, FOCUSED_RUNS)
    cleanup = cleanup_generated(manifest)
    write_summary(results, manifest, cleanup, errors, FOCUSED_RUNS, batch)
    return 1 if errors else 0
