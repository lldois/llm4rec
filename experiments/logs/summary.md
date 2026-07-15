# LLM4Rec Experiment Summary

Updated: 2026-07-15 23:15:37

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

## Runs
| version | batch | base | dataset | lr | epochs | train loss | runtime min | status |
|---|---|---|---|---:|---:|---:|---:|---|
| v31_v29_live_specialist_r3_lr12e6_ep035 | v31-v32 | v29_v19_user_world_guard_lr8e7_ep018 | v29_live_specialist_r3 | 1.2e-6 | 0.35 | 1.5557 | 9.51 | ok |
| v32_v29_balanced_r3_draft_lr8e7_ep020 | v31-v32 | v29_v19_user_world_guard_lr8e7_ep018 | v29_balanced_r3_draft | 8.0e-7 | 0.2 | 1.7310 | 6.01 | ok |
| v34_v29_paper_mix_lr5e7_ep10 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v34_paper_mix | 5.0e-7 | 1.0 | 1.8975 | 63.25 | ok |
| v35_v29_cot_r3_balanced_lr6e7_ep10 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v35_cot_r3_balanced | 6.0e-7 | 1.0 | 2.0760 | 62.97 | ok |
| v36_v29_general_guard_lr4e7_ep12 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v36_general_guard | 4.0e-7 | 1.2 | 1.8869 | 85.13 | ok |
| v37_v29_r0_perception_lr6e7_ep10 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v37_r0_perception | 6.0e-7 | 1.0 | 1.9282 | 59.50 | ok |
| v38_v29_r3_aggressive_lr7e7_ep10 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v38_r3_aggressive | 7.0e-7 | 1.0 | 2.1436 | 59.84 | ok |
| v39_v29_curriculum_r0_to_r3_lr6e7_ep08x2 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v39_curriculum_r3 | 5.0e-7 | 0.8 + 0.8 | 2.1088 | 82.93 | ok |
| v40_scratch_source_paper_mix_lr3e6_ep15 | v34-v43 | official-pretrain | v34_paper_mix | 3.0e-6 | 1.5 | 1.8137 | 94.49 | ok |
| v41_v29_baseline_general_replay_lr4e7_ep10 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v41_baseline_general | 4.0e-7 | 1.0 | 1.7103 | 103.31 | ok |
| v42_v29_user_evolution_guard_lr5e7_ep10 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v42_user_evolution | 5.0e-7 | 1.0 | 1.9070 | 90.68 | ok |
| v43_v29_paper_mix_low_lr25e7_ep15 | v34-v43 | v29_v19_user_world_guard_lr8e7_ep018 | v34_paper_mix | 2.5e-7 | 1.5 | 1.9072 | 92.70 | ok |

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

## Notes
- Ranking by train loss is only a training-health signal; use leaderboard evaluation for model selection.
- Generated experiment JSONL and model outputs are ignored by git; JSONL can be rebuilt through the unified runner.
