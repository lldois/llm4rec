# LLM4Rec Experiment Summary

Updated: 2026-07-20 03:55:30

## Dataset Variants
- `v29_live_specialist_r3`: 13013 records, groups={'rec': 7413, 'user': 2400, 'item': 3200}, sha256=5710cee8621dde0b19e5418cfe81cc5aea2933d432478968c498ff0f00e24934
- `v29_balanced_r3_draft`: 15942 records, groups={'rec': 10342, 'user': 2400, 'item': 3200}, sha256=a7e50c88eef370f6de88e1afa70995c7712afa750acc711b490d34d700d44f31
- `v34_paper_mix`: 80000 records, groups={'general': 30400, 'old_rec': 5000, 'r3': 18000, 'r0': 19000, 'old_item': 3800, 'old_user': 3800}, sha256=114013600aa0113ef0ef441bc1ecbd16a8c01f1a77cf76b9d0de16040656da8f
- `v35_cot_r3_balanced`: 80000 records, groups={'r3': 32000, 'old_user': 3500, 'general': 24000, 'old_rec': 5000, 'r0': 12000, 'old_item': 3500}, sha256=4207a66e3c1e6fe035067e219a3a45243c46a9167f1b912eca13d62ec1cdf192
- `v36_general_guard`: 80000 records, groups={'general': 36000, 'r0': 12000, 'r3': 17000, 'old_item': 4500, 'old_user': 4500, 'old_rec': 6000}, sha256=8aa50ef29b36de4764d1255110a3d1eba6c62e3eaa22636cd92f351cb59d72c2
- `v37_r0_perception`: 80000 records, groups={'r3': 16000, 'general': 24000, 'r0': 28000, 'old_item': 4000, 'old_rec': 4000, 'old_user': 4000}, sha256=71579dd6021ab3045524e80275a793d3adcb1b552884209837e77c149d880266
- `v38_r3_aggressive`: 80000 records, groups={'r3': 42000, 'general': 20000, 'old_item': 3000, 'old_user': 3000, 'old_rec': 4000, 'r0': 8000}, sha256=1b53237ca7e35d4c0019b89966b7eec83e6945b643ddc0fe86ed057d90f7b81c
- `v39_curriculum_r0`: 70000 records, groups={'r0': 28000, 'old_user': 3000, 'general': 32000, 'old_item': 3000, 'old_rec': 4000}, sha256=3ea92f76bbf5ed386ef57430df75b428bd43cbb0a1d306de01669d639eca889a
- `v39_curriculum_r3`: 70000 records, groups={'general': 22000, 'r3': 34000, 'old_rec': 3000, 'old_user': 2500, 'r0': 6000, 'old_item': 2500}, sha256=fb38ffd99b1ed61bbca834a85af32e3bb557aeba6a8b05f134ab5b99c16a22da
- `v41_baseline_general`: 80000 records, groups={'general': 32000, 'old_item': 10000, 'old_user': 8000, 'r0': 8000, 'old_rec': 22000}, sha256=82e09565761de669ade898e8a1f743ab8e72da7e34b6effa1c780cdf12ecf007
- `v42_user_evolution`: 75000 records, groups={'general': 26000, 'r3': 24000, 'old_user': 9000, 'old_item': 5000, 'old_rec': 5000, 'r0': 6000}, sha256=d4dcc5f06f9e32b0d781e29d2b6965ffc8683fc9b63e1d2533322535cfd73fde
- `v44_clean_live_temporal`: 9970 records, groups={'old_rec_clean': 2000, 'old_user_clean': 1330, 'old_item_clean': 2290, 'r3_clean': 3100, 'general_clean': 1250}, sha256=b5dca87857669c62efbfc40d14a3e6d71b0ecd3b6b3d848e54f88b469253a1c3
- `v45_clean_r3_domainmix`: 9550 records, groups={'old_rec_clean': 1800, 'general_clean': 1200, 'r3_clean': 4100, 'old_item_clean': 1500, 'old_user_clean': 950}, sha256=4c3dca009ec1070a50416a29ad6162f7c2251a7df59c7ab52408c2f9cecdd484
- `v46_clean_user_evolution`: 9250 records, groups={'old_item_clean': 1400, 'old_rec_clean': 1800, 'r3_clean': 2100, 'old_user_clean': 2750, 'general_clean': 1200}, sha256=44a179f71b7ecbe7f4d3df99088bbed4b249e7283fe0937448c63f628452907c
- `v47_clean_world_guard`: 9500 records, groups={'old_item_clean': 1200, 'general_clean': 4900, 'old_rec_clean': 1400, 'r3_clean': 1200, 'old_user_clean': 800}, sha256=876ac568a0c26891e5ad3475a7636ef37fe784f7b956d0c3285efc2d3a1e2c90
- `v48_clean_fused`: 11600 records, groups={'old_user_clean': 1000, 'r0_clean': 1650, 'general_clean': 2400, 'r3_clean': 3600, 'old_rec_clean': 1450, 'old_item_clean': 1500}, sha256=d080d8786abbd5f20116c97bfee6d8d073bcecb38143880985452fe0fe503af2
- `v49_public_091_exact`: 32705 records, groups={'rec': 18651, 'item': 9684, 'user': 2792, 'world': 1578}, sha256=cdedea13c560d3453697f4a3c9b96a2303bd708c84833034fa953d16482b9925
- `v63_public_user_video_live_guard`: 38855 records, groups={'public_exact': 32705, 'user_json_replay': 2792, 'video_distinct_cot': 3017, 'live_distinct_cot': 341}, sha256=d6e668bb3eb7c3d248248bf96a307540cf2937811b9438d4e64c7b0a09c3c6d2
- `v67_public_user_replay`: 35497 records, groups={'public_exact': 32705, 'user_json_replay': 2792}, sha256=974801db6aadac76657e92f59d3281709cdd0b801da20ee3df483de22271a8ae

## Runs
| version | batch | method | base | dataset | lr | epochs | steps | train loss | eval loss | runtime min | status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| v31_v29_live_specialist_r3_lr12e6_ep035 | v31-v32 | full | v29_v19_user_world_guard_lr8e7_ep018 | v29_live_specialist_r3 | 1.2e-6 | 0.35 | 61 | 1.5557 |  | 9.51 | ok |
| v32_v29_balanced_r3_draft_lr8e7_ep020 | v31-v32 | full | v29_v19_user_world_guard_lr8e7_ep018 | v29_balanced_r3_draft | 8.0e-7 | 0.2 | 40 | 1.7310 |  | 6.01 | ok |
| v34_v29_paper_mix_lr5e7_ep10 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v34_paper_mix | 5.0e-7 | 1.0 |  | 1.8975 |  | 63.25 | ok |
| v35_v29_cot_r3_balanced_lr6e7_ep10 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v35_cot_r3_balanced | 6.0e-7 | 1.0 |  | 2.0760 |  | 62.97 | ok |
| v36_v29_general_guard_lr4e7_ep12 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v36_general_guard | 4.0e-7 | 1.2 |  | 1.8869 |  | 85.13 | ok |
| v37_v29_r0_perception_lr6e7_ep10 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v37_r0_perception | 6.0e-7 | 1.0 |  | 1.9282 |  | 59.50 | ok |
| v38_v29_r3_aggressive_lr7e7_ep10 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v38_r3_aggressive | 7.0e-7 | 1.0 |  | 2.1436 |  | 59.84 | ok |
| v39_v29_curriculum_r0_to_r3_lr6e7_ep08x2 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v39_curriculum_r3 | 5.0e-7 | 0.8 + 0.8 |  | 2.1088 |  | 82.93 | ok |
| v40_scratch_source_paper_mix_lr3e6_ep15 | v34-v43 | full | official-pretrain | v34_paper_mix | 3.0e-6 | 1.5 |  | 1.8137 |  | 94.49 | ok |
| v41_v29_baseline_general_replay_lr4e7_ep10 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v41_baseline_general | 4.0e-7 | 1.0 |  | 1.7103 |  | 103.31 | ok |
| v42_v29_user_evolution_guard_lr5e7_ep10 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v42_user_evolution | 5.0e-7 | 1.0 |  | 1.9070 |  | 90.68 | ok |
| v43_v29_paper_mix_low_lr25e7_ep15 | v34-v43 | full | v29_v19_user_world_guard_lr8e7_ep018 | v34_paper_mix | 2.5e-7 | 1.5 |  | 1.9072 |  | 92.70 | ok |
| v44_v29_clean_live_temporal_lr6e7_ep045 | v44-v48 | full | v29_v19_user_world_guard_lr8e7_ep018 | v44_clean_live_temporal | 6.0e-7 | 0.45 | 40 | 2.0323 |  | 5.98 | ok |
| v45_v29_clean_r3_domainmix_lr55e7_ep045 | v44-v48 | full | v29_v19_user_world_guard_lr8e7_ep018 | v45_clean_r3_domainmix | 5.5e-7 | 0.45 | 34 | 2.2282 |  | 4.98 | ok |
| v46_v29_clean_user_evolution_lr5e7_ep045 | v44-v48 | full | v29_v19_user_world_guard_lr8e7_ep018 | v46_clean_user_evolution | 5.0e-7 | 0.45 | 60 | 1.7778 |  | 9.05 | ok |
| v47_v29_clean_world_guard_lr35e7_ep035 | v44-v48 | full | v29_v19_user_world_guard_lr8e7_ep018 | v47_clean_world_guard | 3.5e-7 | 0.35 | 26 | 1.9200 |  | 3.90 | ok |
| v48_v29_clean_fused_lr45e7_ep040 | v44-v48 | full | v29_v19_user_world_guard_lr8e7_ep018 | v48_clean_fused | 4.5e-7 | 0.4 | 32 | 2.1477 |  | 4.75 | ok |
| v49_public091_lora_r32_lr2e4_ep1 | v49-v53 | lora | official-pretrain | v49_public_091_exact | 2.0e-4 | 1.0 | 326 | 1.5965 |  | 53.92 | ok |
| v50_public091_lora_r32_lr1e4_ep2 | v49-v53 | lora | official-pretrain | v49_public_091_exact | 1.0e-4 | 2.0 | 652 | 1.5554 |  | 108.31 | ok |
| v51_public091_lora_r64_lr1e4_ep2 | v49-v53 | lora | official-pretrain | v49_public_091_exact | 1.0e-4 | 2.0 | 652 | 1.4956 |  | 108.22 | ok |
| v52_public091_full_lr2e5_ep1 | v49-v53 | full | official-pretrain | v49_public_091_exact | 2.0e-5 | 1.0 | 326 | 1.6356 |  | 43.52 | ok |
| v53_public091_full_lr1e5_ep3 | v49-v53 | full | official-pretrain | v49_public_091_exact | 1.0e-5 | 3.0 | 978 | 1.6426 |  | 132.30 | ok |
| v54_public091_lora_r96_lr1e4_ep2 | v54-v63 | lora | official-pretrain | v49_public_091_exact | 1.0e-4 | 2.0 | 652 | 1.4611 |  | 109.35 | ok |
| v55_public091_lora_r128_lr8e5_ep2 | v54-v63 | lora | official-pretrain | v49_public_091_exact | 8.0e-5 | 2.0 | 652 | 1.4589 |  | 110.04 | ok |
| v56_public091_rslora_r128_a16_lr6e5_ep2 | v54-v63 | lora | official-pretrain | v49_public_091_exact | 6.0e-5 | 2.0 | 652 | 1.4548 |  | 109.29 | ok |
| v57_public091_dora_r64_lr8e5_ep2 | v54-v63 | lora | official-pretrain | v49_public_091_exact | 8.0e-5 | 2.0 | 652 | 1.5141 |  | 223.51 | ok |
| v58_public091_loraplus_r64_lr2e5_x16_ep2 | v54-v63 | lora | official-pretrain | v49_public_091_exact | 2.0e-5 | 2.0 | 652 | 1.4333 |  | 107.93 | ok |
| v59_public091_lora_r64_drop0_lr1e4_ep2 | v54-v63 | lora | official-pretrain | v49_public_091_exact | 1.0e-4 | 2.0 | 652 | 1.4926 |  | 96.49 | ok |
| v60_public091_lora_r64_a128_lr5e5_ep2 | v54-v63 | lora | official-pretrain | v49_public_091_exact | 5.0e-5 | 2.0 | 652 | 1.5032 |  | 107.99 | ok |
| v61_v29_public091_lora_r64_lr5e5_ep1 | v54-v63 | lora | v29_v19_user_world_guard_lr8e7_ep018 | v49_public_091_exact | 5.0e-5 | 1.0 | 326 | 1.4662 |  | 54.71 | ok |
| v62_v29_public091_lora_r64_lr2e5_ep1 | v54-v63 | lora | v29_v19_user_world_guard_lr8e7_ep018 | v49_public_091_exact | 2.0e-5 | 1.0 | 326 | 1.4832 |  | 54.39 | ok |
| v63_public_guard_lora_r64_lr8e5_ep2 | v54-v63 | lora | official-pretrain | v63_public_user_video_live_guard | 8.0e-5 | 2.0 | 952 | 1.4271 |  | 162.75 | ok |
| v64_public091_loraplus_r64_lr2e5_x8_ep2 | v64-v68 | lora | official-pretrain | v49_public_091_exact | 2.0e-5 | 2.0 | 652 | 1.4903 |  | 108.80 | ok |
| v65_public091_loraplus_r128_lr1e5_x16_ep2 | v64-v68 | lora | official-pretrain | v49_public_091_exact | 1.0e-5 | 2.0 | 652 | 1.4310 |  | 109.68 | ok |
| v66_public091_rslora_loraplus_r128_a16_lr1e5_x16_ep2 | v64-v68 | lora | official-pretrain | v49_public_091_exact | 1.0e-5 | 2.0 | 652 | 1.4016 |  | 110.14 | ok |
| v67_public_user_replay_loraplus_r64_lr2e5_x16_ep2 | v64-v68 | lora | official-pretrain | v67_public_user_replay | 2.0e-5 | 2.0 | 856 | 1.3518 |  | 146.05 | ok |
| v68_public091_pissa_loraplus_r64_lr2e5_x8_ep2 | v64-v68 | lora | official-pretrain | v49_public_091_exact | 2.0e-5 | 2.0 | 652 | 1.4986 |  | 108.12 | ok |

## Derived Models
- `v33_task_arith_v29_live55_r325`: theta_v29 + 0.55*(theta_v31-theta_v29) + 0.25*(theta_v32-theta_v29). CoT-native task arithmetic from the shared v29 base; no v7 source.

## Run Notes
- `v31_v29_live_specialist_r3_lr12e6_ep035`: v29 continuation and live-domain specialist. Use diverse evidence-grounded three-level CoT plus strict no-think direct answers for every live-target record, with item/user/general-rec guards. Goal: raise recommendation-live without losing v29's speed or other dimensions.
- `v32_v29_balanced_r3_draft_lr8e7_ep020`: v29 continuation and balanced four-domain R3 specialist. Equalize live/ad/product/video target supervision and train concise individualized interest-evolution-task CoT plus route-correct direct answers. Goal: reduce video dominance and improve recommendation balance while retaining meaningful CoT.
- `v33_task_arith_v29_live55_r325`: CoT-native task arithmetic from the shared v29 base; no v7 source.
- `v34_v29_paper_mix_lr5e7_ep10`: Anchor run using a scaled OneReason-like R0/R3/general mixture plus old-task replay.
- `v35_v29_cot_r3_balanced_lr6e7_ep10`: Higher R3 CoT ratio with compact Persona-Expansion-Transition traces and direct-route balance.
- `v36_v29_general_guard_lr4e7_ep12`: General-heavy replay explicitly targeting the world-score forgetting seen in v31-v33.
- `v37_v29_r0_perception_lr6e7_ep10`: R0-heavy bidirectional caption/SID grounding to improve item perception and downstream recommendation semantics.
- `v38_v29_r3_aggressive_lr7e7_ep10`: Highest R3 share, trained on real temporal holdouts from official user profiles with target leakage blocked.
- `v39_v29_curriculum_r0_to_r3_lr6e7_ep08x2`: Two-stage curriculum: 0.8 epoch R0/general alignment followed by 0.8 epoch R3/general cognition.
- `v40_scratch_source_paper_mix_lr3e6_ep15`: High-risk restart from the official pretrained checkpoint using the new source-data mixture.
- `v41_v29_baseline_general_replay_lr4e7_ep10`: Control run dominated by original competition supervision and general replay, with no profile-derived R3.
- `v42_v29_user_evolution_guard_lr5e7_ep10`: Protects structured user extraction/evolution formats while adding real-holdout R3 and general replay.
- `v43_v29_paper_mix_low_lr25e7_ep15`: Controlled long, low-LR exposure on exactly the same data as v34.
- `v44_v29_clean_live_temporal_lr6e7_ep045`: Strict live specialist: future high-value live target, chronological pre-target history, semantic support, and 58/42 CoT/direct live supervision with clean guards.
- `v45_v29_clean_r3_domainmix_lr55e7_ep045`: Strict balanced R3 with paper-motivated domain ratios: ad direct-heavy, product CoT-heavy, live moderately CoT-rich, and video balanced.
- `v46_v29_clean_user_evolution_lr5e7_ep045`: User/evolution specialist using parseable deduplicated JSON, retained evidence-bearing user CoT, strict R3 temporal examples, and balanced item/rec guards.
- `v47_v29_clean_world_guard_lr35e7_ep035`: World-knowledge guard with strict single-turn paired general data, exact/template deduplication, bounded traces, and a low-dose task-format replay shell.
- `v48_v29_clean_fused_lr45e7_ep040`: Fused high-confidence candidate combining clean general, collision-free R0, domain-aware strict R3, and deduplicated original-task guards.
- `v49_public091_lora_r32_lr2e4_ep1`: Exact public 0.9107 reproduction anchor: published 32,705-row clean data and LoRA r32/alpha32/dropout0.05 recipe.
- `v50_public091_lora_r32_lr1e4_ep2`: Same public data and LoRA capacity as v49, with half LR and two epochs to test whether the one-epoch public anchor is optimization-limited.
- `v51_public091_lora_r64_lr1e4_ep2`: Higher-capacity LoRA on fixed data; isolates adapter rank from the longer schedule while retaining the public alpha/rank scaling.
- `v52_public091_full_lr2e5_ep1`: Full-SFT counterpart using v7's successful 2e-5 scale, now on balanced CoT/direct clean data instead of final-only data.
- `v53_public091_full_lr1e5_ep3`: Convergence anchor matching the published full-SFT 1e-5/3-epoch schedule whose validation loss fell through the third epoch.
- `v54_public091_lora_r96_lr1e4_ep2`: Interpolate between rank-64 v51 and rank-128 while keeping unit LoRA scaling and the successful two-epoch schedule.
- `v55_public091_lora_r128_lr8e5_ep2`: Higher-capacity standard LoRA with a modestly reduced LR to test whether v51 remains rank-limited.
- `v56_public091_rslora_r128_a16_lr6e5_ep2`: Rank-stabilized LoRA branch; alpha/sqrt(rank) avoids the high-rank scaling collapse and is merged for submission compatibility.
- `v57_public091_dora_r64_lr8e5_ep2`: DoRA magnitude/direction adaptation at v51 rank, merged to a standard full checkpoint after training.
- `v58_public091_loraplus_r64_lr2e5_x16_ep2`: LoRA+ uses separate A/B learning rates; the conservative base LR gives the B matrix a 3.2e-4 update rate.
- `v59_public091_lora_r64_drop0_lr1e4_ep2`: Exact v51 control with adapter dropout removed, matching the recent public ORPO/GRPO adapter convention.
- `v60_public091_lora_r64_a128_lr5e5_ep2`: Double LoRA scaling with half v51 LR, isolating update geometry from nominal learning rate.
- `v61_v29_public091_lora_r64_lr5e5_ep1`: Public clean-data LoRA over the strong v29 full checkpoint, targeting v29 user/video retention plus v51 ad/world gains.
- `v62_v29_public091_lora_r64_lr2e5_ep1`: Lower-dose v29-based counterpart intended to preserve its structured-user and video behavior more conservatively.
- `v63_public_guard_lora_r64_lr8e5_ep2`: Real-label task guard: duplicate valid user JSON and add at most one distinct original CoT target per video/live prompt without synthetic traces.
- `v64_public091_loraplus_r64_lr2e5_x8_ep2`: LoRA+ ratio-8 neighbor of v58; keeps A at 2e-5 while reducing B from 3.2e-4 to 1.6e-4.
- `v65_public091_loraplus_r128_lr1e5_x16_ep2`: Rank-128 LoRA+ with both learning rates halved relative to v58 for rank-aware capacity scaling.
- `v66_public091_rslora_loraplus_r128_a16_lr1e5_x16_ep2`: Combines v58 LoRA+ optimizer geometry with v56 rank-stabilized high-rank scaling; targets ad and world gains.
- `v67_public_user_replay_loraplus_r64_lr2e5_x16_ep2`: Applies v58 LoRA+ to exact public data plus one replay of each real parseable user target, isolating the useful v63 signal.
- `v68_public091_pissa_loraplus_r64_lr2e5_x8_ep2`: PiSSA FSVD-16 initialization with the conservative LoRA+ ratio-8 optimizer; merged for submission compatibility.

## v34-v43 Evaluation Order

1. v34 anchor paper mixture
2. v38 aggressive R3
3. v39 perception-first curriculum
4. v43 low-LR long-dose anchor
5. v36 general guard
6. v35 CoT-heavy R3
7. v37 perception-heavy
8. v42 user guard
9. v41 original-data control
10. v40 scratch high-risk model

## v44-v48 Clean-Data Evaluation Order

1. v48 fused high-confidence candidate
2. v44 strict live specialist
3. v45 domain-aware R3 mixture
4. v47 clean world guard
5. v46 clean user-evolution specialist

## v49-v53 Fixed-Data Tuning Order

1. v51 rank-64 LoRA (official score 0.9188)
2. v50 lower-LR 2-epoch LoRA (0.8971)
3. v49 exact public anchor (0.8834)
4. v52 1-epoch full SFT (0.8614)
5. v53 3-epoch full SFT (0.7862)

## v54-v63 Advanced Evaluation Order

1. v63 real-label user/video/live guard
2. v58 LoRA+
3. v56 rank-stabilized LoRA
4. v61 v29 bridge at 5e-5
5. v54 rank-96 interpolation
6. v55 standard rank-128
7. v62 conservative v29 bridge
8. v57 DoRA
9. v59 rank-64 dropout-zero control
10. v60 doubled LoRA scaling

## v64-v68 Focused Evaluation Order

1. v66 rsLoRA + LoRA+ combination
2. v67 real-user replay + LoRA+
3. v65 rank-128 rank-aware LoRA+
4. v64 conservative ratio-8 LoRA+
5. v68 PiSSA + LoRA+ exploration

## Notes
- Ranking by train loss is only a training-health signal; use leaderboard evaluation for model selection.
- Generated experiment JSONL and model outputs are ignored by git; JSONL can be rebuilt through the unified runner.
