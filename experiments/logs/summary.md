# LLM4Rec Experiment Summary

Generated: 2026-07-07 06:47:17
Hard deadline: 2026-07-07 09:00:00 +0800

## Dataset Variants
- `item_cot_focus`: 74016 records, groups={'item': 51920, 'user': 2892, 'rec': 19204}
- `user_cot_focus`: 46940 records, groups={'rec': 19204, 'user': 17352, 'item': 10384}
- `rec_cot_focus`: 51684 records, groups={'item': 10384, 'rec': 38408, 'user': 2892}
- `world_cot_preserve`: 32480 records, groups={'rec': 19204, 'item': 10384, 'user': 2892}

## Runs
| version | dataset | lr | epochs | scheduler | train_loss | last_logged_loss | runtime | status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| v11_item_cot_focus_lr2e5 | item_cot_focus | 2.0e-5 | 1 | cosine | 1.6237 | 1.5290 | 3645.9s | ok |
| v12_user_cot_focus_lr15e6 | user_cot_focus | 1.5e-5 | 1 | cosine | 1.4710 | 1.3636 | 8039.0s | ok |
| v13_rec_cot_focus_lr2e5 | rec_cot_focus | 2.0e-5 | 1 | cosine | 1.4704 | 1.4336 | 5410.0s | ok |
| v14_world_cot_preserve_lr8e6 | world_cot_preserve | 8.0e-6 | 1 | linear | 1.8378 | 1.6936 | 3207.2s | ok |

## Notes
- Ranking by train loss is only a rough sanity signal because no held-out leaderboard metric is available locally.
- Prefer comparing generated behavior on the official validation/upload flow tomorrow if available.
