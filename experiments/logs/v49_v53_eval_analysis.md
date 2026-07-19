# v49-v53 evaluation analysis and v54-v63 plan

Generated: 2026-07-19

## Official evaluation summary

| Version | Total | Item | User select | User generation | Video | Product | Ad | Live | World | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v49 | 0.8834 | 0.2146 | 0.0554 | 0.0395 | 0.0768 | 0.1088 | 0.1302 | 0.1071 | 0.1509 | 75.44 min |
| v50 | 0.8971 | 0.2453 | 0.0622 | 0.0399 | 0.0384 | 0.1122 | 0.1498 | 0.1062 | 0.1431 | 71.56 min |
| v51 | **0.9188** | 0.2146 | 0.0790 | 0.0381 | 0.0576 | 0.1156 | 0.1526 | 0.1071 | 0.1543 | 68.58 min |
| v52 | 0.8614 | 0.1840 | 0.0538 | 0.0424 | 0.0768 | 0.1122 | 0.1246 | 0.1125 | 0.1550 | 71.41 min |
| v53 | 0.7862 | 0.2146 | 0.0003 | 0.0410 | 0.0384 | 0.1054 | 0.1260 | 0.1089 | 0.1517 | 71.53 min |

## Conclusions from scores and generations

1. v51 is the new local best. Moving from LoRA rank 32 to rank 64 is much more useful than full SFT on this data.
2. Full SFT is not merely weaker: three epochs in v53 nearly erase user-selection ability. Training loss is therefore not a leaderboard proxy.
3. v51 improves product, ad and world knowledge over v29, but remains behind v29 on user selection, video and live recommendation. These are the targets for the data-guard branch.
4. v51 still contains rare prompt echo and runaway generations, including one extremely long video answer. Faster evaluation is a useful signal, but output validity alone does not predict Pass@64 item hits.
5. v50 has the best item score but the worst video score in this group. Candidate-token validity and ranking quality must be treated separately.

## v54-v63 experiment matrix

| Version | Controlled change | Main question |
| --- | --- | --- |
| v54 | public data, LoRA r96/a96, lr 1e-4, 2 epochs | Does capacity continue to improve beyond v51 r64? |
| v55 | public data, LoRA r128/a128, lr 8e-5, 2 epochs | Is a larger adapter useful with a slightly safer LR? |
| v56 | public data, rsLoRA r128/a16, lr 6e-5, 2 epochs | Can rank-stabilized scaling use high rank without excessive update magnitude? |
| v57 | public data, DoRA r64/a64, lr 8e-5, 2 epochs | Does magnitude/direction decomposition improve stability and ranking? |
| v58 | public data, LoRA+ r64/a64, base LR 2e-5, B/A ratio 16, 2 epochs | Can asymmetric LoRA learning rates converge better? |
| v59 | exact v51 except dropout 0 | Is dropout suppressing useful memorization on deterministic recommendation labels? |
| v60 | public data, LoRA r64/a128, lr 5e-5, 2 epochs | Does a larger adapter scale help at lower LR? |
| v61 | v29 full model base, public data, r64/a64, lr 5e-5, 1 epoch | Can public clean data retain v29 user/video strengths while learning v51 strengths? |
| v62 | same v29 bridge at lr 2e-5 | Is conservative continuation safer for inherited capabilities? |
| v63 | public data plus real user replay and one clean CoT video/live label per prompt | Can targeted real-data weighting repair v51's weak tasks without synthetic labels? |

The v63 guard set has 38,855 records: 32,705 exact public rows, 2,792 user-label replays, 3,017 distinct video CoT rows, and 341 distinct live CoT rows. Independent checks found zero target tokens in prompts or thoughts. No intermediate checkpoints are saved for any run.

## Completed training results

| Version | Steps | Train loss | Tail loss mean | Artifact |
| --- | ---: | ---: | ---: | --- |
| v54 | 652 | 1.4611 | 1.3334 | LoRA adapter |
| v55 | 652 | 1.4589 | 1.3448 | LoRA adapter |
| v56 | 652 | 1.4548 | 1.3237 | merged full model + reproducibility adapter |
| v57 | 652 | 1.5141 | 1.3979 | merged full model + reproducibility adapter |
| v58 | 652 | **1.4333** | **1.3121** | LoRA adapter |
| v59 | 652 | 1.4926 | 1.3609 | LoRA adapter |
| v60 | 652 | 1.5032 | 1.3970 | LoRA adapter |
| v61 | 326 | 1.4662 | 1.5045 | merged v29-based full model + reproducibility adapter |
| v62 | 326 | 1.4832 | 1.4817 | merged v29-based full model + reproducibility adapter |
| v63 | 952 | **1.4271** | **1.3155** | LoRA adapter |

Training loss is not used as a proxy for official score: v53 already proved that lower or longer optimization can destroy a task. Suggested evaluation order is **v63, v58, v56, v61, v54, v55, v62, v57, v59, v60**. The first three respectively test targeted task repair, a better optimizer geometry, and rank-stabilized high capacity.

## Research basis

- LoRA+: https://arxiv.org/abs/2402.12354
- DoRA: https://arxiv.org/abs/2402.09353
- rsLoRA implementation semantics: https://huggingface.co/docs/peft/package_reference/lora
- Public 0.9107 SFT dataset: https://huggingface.co/datasets/Frinkleko/kuaishou-llmrec-sft-baseline-0.91
- Competition FAQ: https://www.streamlake.com/document/WANQING/mq57afym1d7p20atnau
