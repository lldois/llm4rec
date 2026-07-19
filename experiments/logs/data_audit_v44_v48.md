# v44-v48 Data Audit

Updated: 2026-07-16 01:28:12

## Policy

Strict single-turn general QA; canonical metadata joins; one balanced R3 target per profile; history timestamps strictly precede targets; high-value targets; tag-supported evidence; target SID forbidden from prompt and CoT; every CoT itemic reference must occur in history.

## Pool Counts

- `general_cn_direct`: 1741
- `general_cn_think`: 1741
- `general_other_direct`: 2259
- `general_other_think`: 2259
- `old_item_ad_direct`: 1581
- `old_item_ad_think`: 519
- `old_item_goods_direct`: 1611
- `old_item_goods_think`: 531
- `old_item_live_direct`: 784
- `old_item_live_think`: 284
- `old_item_video_direct`: 1621
- `old_item_video_think`: 418
- `old_rec_ad_direct`: 793
- `old_rec_ad_think`: 793
- `old_rec_goods_direct`: 630
- `old_rec_goods_think`: 630
- `old_rec_live_direct`: 671
- `old_rec_live_think`: 671
- `old_rec_video_direct`: 3107
- `old_rec_video_think`: 3107
- `old_user_array_direct`: 1588
- `old_user_logic_direct`: 1304
- `old_user_logic_think`: 598
- `r0_caption_direct`: 40414
- `r0_caption_think`: 6390
- `r0_ground_direct`: 40414
- `r3_balanced_ad_direct`: 6153
- `r3_balanced_ad_think`: 6153
- `r3_balanced_goods_direct`: 5761
- `r3_balanced_goods_think`: 5761
- `r3_balanced_live_direct`: 6146
- `r3_balanced_live_think`: 6146
- `r3_balanced_video_direct`: 6482
- `r3_balanced_video_think`: 6482
- `r3_live_specialist_direct`: 8856
- `r3_live_specialist_think`: 8856

## Dataset Recipes

- `v44_clean_live_temporal`: records=9970, routes={'no_think': 6500, 'think': 3470}, groups={'old_rec_clean': 2000, 'old_user_clean': 1330, 'old_item_clean': 2290, 'r3_clean': 3100, 'general_clean': 1250}, domains={'video': 1300, 'live': 4150, 'ad': 970, 'goods': 970}, sha256=b5dca87857669c62efbfc40d14a3e6d71b0ecd3b6b3d848e54f88b469253a1c3
- `v45_clean_r3_domainmix`: records=9550, routes={'no_think': 6100, 'think': 3450}, groups={'old_rec_clean': 1800, 'general_clean': 1200, 'r3_clean': 4100, 'old_item_clean': 1500, 'old_user_clean': 950}, domains={'live': 2075, 'video': 1925, 'goods': 1625, 'ad': 1775}, sha256=4c3dca009ec1070a50416a29ad6162f7c2251a7df59c7ab52408c2f9cecdd484
- `v46_clean_user_evolution`: records=9250, routes={'no_think': 6500, 'think': 2750}, groups={'old_item_clean': 1400, 'old_rec_clean': 1800, 'r3_clean': 2100, 'old_user_clean': 2750, 'general_clean': 1200}, domains={'video': 1300, 'live': 1600, 'ad': 1150, 'goods': 1250}, sha256=44a179f71b7ecbe7f4d3df99088bbed4b249e7283fe0937448c63f628452907c
- `v47_clean_world_guard`: records=9500, routes={'no_think': 6200, 'think': 3300}, groups={'old_item_clean': 1200, 'general_clean': 4900, 'old_rec_clean': 1400, 'r3_clean': 1200, 'old_user_clean': 800}, domains={'goods': 900, 'ad': 850, 'live': 1100, 'video': 950}, sha256=876ac568a0c26891e5ad3475a7636ef37fe784f7b956d0c3285efc2d3a1e2c90
- `v48_clean_fused`: records=11600, routes={'no_think': 7850, 'think': 3750}, groups={'old_user_clean': 1000, 'r0_clean': 1650, 'general_clean': 2400, 'r3_clean': 3600, 'old_rec_clean': 1450, 'old_item_clean': 1500}, domains={'ad': 1654, 'goods': 2148, 'video': 2764, 'live': 1634}, sha256=d080d8786abbd5f20116c97bfee6d8d073bcecb38143880985452fe0fe503af2

## R3 Filters

```json
{
  "filter_counts": {
    "valid_video/ad": 12421,
    "valid_video/video": 18618,
    "valid_live": 8856,
    "valid_goods": 5777,
    "weak_or_short_history": 20458,
    "missing_target_metadata": 3382,
    "cot_or_leakage_reject": 807,
    "balanced_profiles": 24542
  },
  "balanced_domains": {
    "video/ad": 6153,
    "video/video": 6482,
    "live": 6146,
    "goods": 5761
  },
  "pool_counts": {
    "r3_live_specialist_direct": 8856,
    "r3_live_specialist_think": 8856,
    "r3_balanced_ad_direct": 6153,
    "r3_balanced_ad_think": 6153,
    "r3_balanced_video_direct": 6482,
    "r3_balanced_video_think": 6482,
    "r3_balanced_live_direct": 6146,
    "r3_balanced_live_think": 6146,
    "r3_balanced_goods_direct": 5761,
    "r3_balanced_goods_think": 5761
  }
}
```

## General Filters

```json
{
  "filter_counts": {
    "scanned": 152005,
    "accepted_cn": 1741,
    "length_reject": 109918,
    "not_strict_single_turn": 22224,
    "accepted_other": 2259,
    "missing_cot_shell": 2105,
    "short_final": 351,
    "banned_format": 21,
    "exact_duplicate": 432,
    "source_cap": 12954,
    "unique_pairs": 4000
  },
  "source_counts": {
    "stepfun_general": 4000
  }
}
```
