# LLM4Rec Experiment Summary

Generated: 2026-07-11 00:06:59
Hard deadline: 2026-07-11 09:00:00 +0800

## Dataset Variants
- `v19_product_world_guard`: 27772 records, groups={'rec': 13378, 'item': 11194, 'user': 3200}, variants={'rec_cot_pattern_clean': 2400, 'item_short_think': 2792, 'item_compact_cot': 5597, 'rec_no_think_direct_final': 6289, 'user_strict_logic': 1208, 'rec_short_think_final': 4689, 'item_no_think_direct_final': 2805, 'user_extra_no_think_logic': 550, 'user_strict_array': 1442}, sha256=6e45bc1b371e8ca509768327592079c144491f55adf772007ca24b1859acd29a
- `v21_product_rec_rebalance`: 37594 records, groups={'rec': 24000, 'item': 11194, 'user': 2400}, variants={'rec_short_think_final': 9600, 'item_no_think_direct_final': 2805, 'rec_no_think_direct_final': 14400, 'user_strict_logic': 904, 'user_strict_array': 1094, 'item_short_think': 2792, 'item_compact_cot': 5597, 'user_extra_no_think_logic': 402}, sha256=5b58f82b4dfaed0a3bee1dc5ae24ee4406ae40c70038e6bca271ca61b8fe50fa
- `v12_product_ad_rec_lite`: 26924 records, groups={'rec': 13330, 'item': 11194, 'user': 2400}, variants={'rec_short_think_final': 5465, 'rec_no_think_direct_final': 6265, 'item_no_think_direct_final': 2805, 'user_extra_no_think_logic': 410, 'item_compact_cot': 5597, 'user_strict_array': 1096, 'rec_cot_pattern_clean': 1600, 'user_strict_logic': 894, 'item_short_think': 2792}, sha256=8d8285f5ea337154d720a4f1cae44ae9e00228264effaa9ebc527248cb291eff

## Runs
| version | dataset | lr | epochs | scheduler | train_loss | last_logged_loss | runtime | status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| v25_v19_product_world_guard_lr18e6_ep028 | v19_product_world_guard | 1.8e-6 | 0.28 | cosine | 1.3358 | 1.4775 | 616.3s | ok |
| v26_v21_product_rec_rebalance_lr22e6_ep032 | v21_product_rec_rebalance | 2.2e-6 | 0.32 | cosine | 0.7456 | 0.8161 | 819.7s | ok |
| v27_v12_product_ad_rec_lite_lr16e6_ep03 | v12_product_ad_rec_lite | 1.6e-6 | 0.3 | cosine | 2.1623 | 2.0355 | 564.0s | ok |

## Run Notes
- `v25_v19_product_world_guard_lr18e6_ep028`: v19 continuation. Ultra-light product-rec repair with item replay and small user guard. Goal: keep v19 fast/item/user1 behavior while recovering product recommendation and avoiding further world-score erosion.
- `v26_v21_product_rec_rebalance_lr22e6_ep032`: v21 continuation. Rec rebalance from the best user1 checkpoint: heavier product/video rec route supervision, item replay, and only light user replay. Goal: preserve user1 while raising rec1/rec2.
- `v27_v12_product_ad_rec_lite_lr16e6_ep03`: v12 continuation. Tiny product/ad recommendation repair from the strongest user2/world CoT checkpoint. Goal: test whether v12 can gain rec2/rec3 without losing its user2 and world advantages.

## Notes
- Ranking by train loss is only a rough sanity signal because no held-out leaderboard metric is available locally.
- Prefer comparing generated behavior on the official validation/upload flow tomorrow if available.
