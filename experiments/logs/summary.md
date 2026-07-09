# LLM4Rec Experiment Summary

Generated: 2026-07-09 01:09:19
Hard deadline: 2026-07-09 09:00:00 +0800

## Dataset Variants
- `v15_dualmode_fast`: 42493 records, groups={'rec': 28804, 'item': 10797, 'user': 2892}, variants={'rec_short_think_final': 9600, 'rec_no_think_direct_final': 19204, 'item_no_think_direct_final': 5425, 'item_short_think': 5372, 'user_strict_array': 1588, 'user_strict_logic': 1304}, sha256=8fca17210405bf265088c3002986c52a0d8290bb3b9ec08fdc24fb39d7b20c43
- `v7_cot_restore_dualmode`: 41095 records, groups={'rec': 32004, 'item': 5597, 'user': 3494}, variants={'rec_no_think_direct_final': 19204, 'rec_short_think_final': 12800, 'item_no_think_direct_final': 2805, 'user_strict_logic': 1304, 'item_short_think': 2792, 'user_strict_array': 1588, 'user_extra_no_think_logic': 602}, sha256=653fee03d173f5ef8921b6eb428ea090f16a0beded20cb4ced0aadccc586d76d
- `v15_user_repair_rec_fast`: 37587 records, groups={'rec': 25604, 'item': 5597, 'user': 6386}, variants={'rec_no_think_direct_final': 19204, 'item_no_think_direct_final': 2805, 'user_extra_no_think_logic': 602, 'user_strict_logic': 2608, 'item_short_think': 2792, 'rec_short_think_final': 6400, 'user_strict_array': 3176}, sha256=c6fde985378739a2f685d3993a080c58c0dfcefee55dffefbd6577a53a02ad80

## Runs
| version | dataset | lr | epochs | scheduler | train_loss | last_logged_loss | runtime | status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| v19_v15_dualmode_fast_lr5e6_ep055 | v15_dualmode_fast | 5.0e-6 | 0.55 | cosine | 0.9286 | 0.7683 | 1707.0s | ok |
| v20_v7_cot_restore_dualmode_lr3e6_ep045 | v7_cot_restore_dualmode | 3.0e-6 | 0.45 | cosine | 1.6411 | 1.0497 | 1547.0s | ok |
| v21_v15_user_repair_rec_fast_lr4e6_ep055 | v15_user_repair_rec_fast | 4.0e-6 | 0.55 | cosine | 1.0272 | 0.8512 | 2147.0s | ok |

## Run Notes
- `v19_v15_dualmode_fast_lr5e6_ep055`: v15 continuation. Dual-mode fast correction: add rec /no_think direct-final supervision, short stable rec /think CoT, strict user JSON replay, and item replay. Goal: keep CoT route while reducing verbose/invalid generation time.
- `v20_v7_cot_restore_dualmode_lr3e6_ep045`: v7 continuation. Restore acceptable short CoT on /think while preserving v7's fast direct itemic behavior on /no_think. Highest-upside submit candidate if it keeps v7 speed and gains reasoning-route robustness.
- `v21_v15_user_repair_rec_fast_lr4e6_ep055`: v15 continuation. User JSON/logic-chain repair plus rec fast-route supervision. Goal: raise 懂用户 without losing v15 item score or rec speed.

## Notes
- Ranking by train loss is only a rough sanity signal because no held-out leaderboard metric is available locally.
- Prefer comparing generated behavior on the official validation/upload flow tomorrow if available.
