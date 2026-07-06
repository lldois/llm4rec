# LLM4Rec Experiment Summary

Generated: 2026-07-06 07:55:53
Hard deadline: 2026-07-06 08:00:00 +0800

## Dataset Variants
- `all`: 32480 records, groups={'item': 10384, 'rec': 19204, 'user': 2892}
- `balanced_core`: 33952 records, groups={'item': 10384, 'user': 11568, 'rec': 12000}
- `user_item_up`: 51540 records, groups={'rec': 19204, 'item': 20768, 'user': 11568}
- `no_think_input`: 32480 records, groups={'item': 10384, 'rec': 19204, 'user': 2892}
- `final_only`: 32480 records, groups={'item': 10384, 'rec': 19204, 'user': 2892}
- `rec_focus`: 19204 records, groups={'rec': 19204}

## Runs
| version | dataset | lr | epochs | scheduler | train_loss | last_logged_loss | runtime | status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| v01_all_lr1e5 | all | 1.0e-5 | 1 | cosine | 1.7491 | 1.7104 | 3250.6s | ok |
| v02_all_lr3e5 | all | 3.0e-5 | 1 | cosine | 1.4728 | 1.3591 | 3211.0s | ok |
| v03_all_lr1e5_ep2 | all | 1.0e-5 | 2 | cosine | 1.6446 | 1.5597 | 6416.9s | ok |
| v04_balanced_core_lr2e5 | balanced_core | 2.0e-5 | 1 | cosine | 1.4642 | 1.4023 | 5382.2s | ok |
| v05_user_item_up_lr2e5 | user_item_up | 2.0e-5 | 1 | cosine | 1.4734 | 1.3570 | 6290.4s | ok |
| v06_no_think_input_lr2e5 | no_think_input | 2.0e-5 | 1 | cosine | 1.5678 | 1.4073 | 3206.5s | ok |
| v07_final_only_lr2e5 | final_only | 2.0e-5 | 1 | cosine | 1.2550 | 1.3084 | 2346.3s | ok |
| v08_rec_focus_lr2e5 | rec_focus | 2.0e-5 | 1 | cosine | 1.5747 | 1.4710 | 2170.5s | ok |
| v09_all_linear_lr2e5 | all | 2.0e-5 | 1 | linear | 1.5957 | 1.4411 | 3205.2s | ok |
| v10_balanced_core_lr1e5_ep2 | balanced_core | 1.0e-5 | 2 | cosine | 1.5350 | 1.3439 | 10749.8s | ok |

## Notes
- Ranking by train loss is only a rough sanity signal because no held-out leaderboard metric is available locally.
- Prefer comparing generated behavior on the official validation/upload flow tomorrow if available.
