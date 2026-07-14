# LLM4Rec Experiment Summary

Generated: 2026-07-13 03:38:13
Hard deadline: none

## Dataset Variants
- `v29_live_specialist_r3`: 13013 records, groups={'rec': 7413, 'user': 2400, 'item': 3200}, variants={'raw': 1271, 'user_strict_array': 1080, 'user_extra_no_think_logic': 424, 'rec_no_think_direct_final': 3071, 'item_no_think_direct_final': 1659, 'rec_r3_draft_evidence': 3071, 'item_short_think': 1541, 'user_strict_logic': 896}, sha256=5710cee8621dde0b19e5418cfe81cc5aea2933d432478968c498ff0f00e24934
- `v29_balanced_r3_draft`: 15942 records, groups={'rec': 10342, 'user': 2400, 'item': 3200}, variants={'rec_r3_draft_evidence': 5171, 'user_strict_logic': 892, 'rec_no_think_direct_final': 5171, 'item_no_think_direct_final': 1563, 'user_strict_array': 1087, 'user_extra_no_think_logic': 421, 'item_short_think': 1637}, sha256=a7e50c88eef370f6de88e1afa70995c7712afa750acc711b490d34d700d44f31

## Runs
| version | dataset | lr | epochs | scheduler | train_loss | last_logged_loss | runtime | status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| v31_v29_live_specialist_r3_lr12e6_ep035 | v29_live_specialist_r3 | 1.2e-6 | 0.35 | cosine | 1.5557 | 1.5307 | 514.5s | ok |
| v32_v29_balanced_r3_draft_lr8e7_ep020 | v29_balanced_r3_draft | 8.0e-7 | 0.2 | cosine | 1.7310 | 1.7678 | 323.2s | ok |

## Run Notes
- `v31_v29_live_specialist_r3_lr12e6_ep035`: v29 continuation and live-domain specialist. Use diverse evidence-grounded three-level CoT plus strict no-think direct answers for every live-target record, with item/user/general-rec guards. Goal: raise recommendation-live without losing v29's speed or other dimensions.
- `v32_v29_balanced_r3_draft_lr8e7_ep020`: v29 continuation and balanced four-domain R3 specialist. Equalize live/ad/product/video target supervision and train concise individualized interest-evolution-task CoT plus route-correct direct answers. Goal: reduce video dominance and improve recommendation balance while retaining meaningful CoT.

## Unified Model
- `v33_task_arith_v29_live55_r325`: `theta_v29 + 0.55*(theta_v31-theta_v29) + 0.25*(theta_v32-theta_v29)`.
- Method: CoT-native task arithmetic from the shared v29 base; v7 is not used.
- Recipe: `/home/ll/llm4rec/experiments/outputs/v33_task_arith_v29_live55_r325/experiment_recipe.json`.
- Suggested evaluation order: v33, v31, v32.

## Notes
- Ranking by train loss is only a rough sanity signal because no held-out leaderboard metric is available locally.
- Prefer comparing generated behavior on the official validation/upload flow tomorrow if available.
