# v44-v48 Evaluation and Convergence Analysis

Updated: 2026-07-17

## Scores

Official column order is item, user-select, user-topic, recommendation-video,
recommendation-product, recommendation-ad, recommendation-live, and world.

| version | total | item | user select | user topic | rec video | rec product | rec ad | rec live | world | generation min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v29 | 0.9038 | 0.2146 | 0.0886 | 0.0346 | 0.0768 | 0.1054 | 0.1400 | 0.1125 | 0.1312 | 45.30 |
| v44 | 0.8732 | 0.2146 | 0.0874 | 0.0370 | 0.0576 | 0.1122 | 0.1330 | 0.1143 | 0.1171 | 52.25 |
| v45 | 0.8774 | 0.2146 | 0.0871 | 0.0369 | 0.0672 | 0.1088 | 0.1358 | 0.1125 | 0.1145 | 52.53 |
| v46 | 0.8787 | 0.2146 | 0.0862 | 0.0360 | 0.0768 | 0.1020 | 0.1344 | 0.1116 | 0.1171 | 52.41 |
| v47 | 0.8753 | 0.2146 | 0.0873 | 0.0352 | 0.0768 | 0.0986 | 0.1358 | 0.1125 | 0.1145 | n/a |
| v48 | 0.8731 | 0.2146 | 0.0876 | 0.0371 | 0.0672 | 0.1020 | 0.1330 | 0.1134 | 0.1182 | 51.91 |

## Reasoning Log Findings

- Item grounding remains healthy: the inspected 320 candidates are unique,
  correctly shaped itemic tokens, consistent with the unchanged maximum item score.
- The new recommendation CoT frequently repeats the synthetic sentence beginning
  with "先按近期、高频、深度互动...". It often emits an itemic token before closing
  the thought, reopens `<think>`, or continues the same template after the final token.
  This turns one candidate into several malformed candidates and reduces effective
  Pass@64 diversity.
- v44 and v45 show the strongest template-copy behavior. v46 is relatively cleaner
  in the ad task and is also the best of the batch, but it still regresses on product,
  ad, live, user-select, and world relative to v29.
- v47 and v48 contain extreme runaway generations in the printed recommendation
  samples, including outputs tens of thousands of characters long. This explains
  much of the roughly seven-minute evaluation slowdown relative to v29.
- User-task outputs sometimes continue beyond the requested JSON object/list or
  repeat long itemic histories. The specialist data did not improve either user
  metric over v29.
- Common-sense sample formatting is usually a valid single letter, but world scores
  fall by 0.0130-0.0167 versus v29. The clean continuation therefore still causes
  measurable knowledge forgetting.

## Training Audit

| version | optimizer steps | train loss | observed trajectory | assessment |
|---|---:|---:|---|---|
| v44 | 40 | 2.032 | about 2.214 to 1.947 | still descending when LR reached zero |
| v45 | 34 | 2.228 | 2.240, 2.277, 2.081 | too few observations and not converged |
| v46 | 60 | 1.778 | about 1.72-1.91 late | more exposure, but no stable late improvement |
| v47 | 26 | 1.920 | 1.945 to 1.843 | clearly too short |
| v48 | 32 | 2.148 | 2.188 to 2.012 | still descending when LR reached zero |

The user's under-training concern is correct. Absolute loss is not directly
comparable across different packed data mixtures, but four runs end while their
logged loss is still falling. At the same time, all five lose 0.0251-0.0307 total
score from the same v29 base, so longer training on the same synthetic distribution
is likely to strengthen its template artifacts and forgetting rather than recover
the leaderboard score.

## v49-v53 Decision

All five runs restart from `OpenOneRec/OneReason-0.8B-pretrain-competition` and use
the same published 32,705-row dataset. It has balanced `<think>` wrappers, 11,771
filled `/think` examples, 20,934 empty-thought `/no_think` examples, valid JSON user
targets, and one final itemic token for each recommendation row. The published LoRA
recipe reports 0.9107, so v49 is an external reproducibility anchor rather than a
new unverified data hypothesis.

| version | method | LR | epochs | convergence evidence | purpose |
|---|---|---:|---:|---:|---|
| v49 | LoRA r32/a32 | 2e-4 | 1 | full loss curve | exact published 0.9107 anchor |
| v50 | LoRA r32/a32 | 1e-4 | 2 | per-epoch/tail loss | test longer convergence at lower LR |
| v51 | LoRA r64/a64 | 1e-4 | 2 | per-epoch/tail loss | test adapter capacity independently |
| v52 | full SFT | 2e-5 | 1 | full loss curve | transfer v7's successful LR to clean CoT data |
| v53 | full SFT | 1e-5 | 3 | per-epoch/tail loss | published full-SFT convergence schedule |

No intermediate checkpoint is saved. A first attempt at 1% framework evaluation
was rejected after the training epoch because Transformers materialized 32K-token
full-vocabulary FP32 logits and requested an extra 13-21 GB on a 24 GB card. The
failed runs were stopped and restarted from the official base with evaluation
disabled; convergence is assessed from all logged training windows and the complete
one-, two-, or three-epoch cosine schedule. Final recipes retain the source URL/hash,
configuration, actual optimizer steps, loss summaries, runtime, and output file
requirements.

## Completed Training Results

| version | steps | average loss | tail-10 loss | minimum window | runtime min | final artifact |
|---|---:|---:|---:|---:|---:|---|
| v49 | 326 | 1.5965 | 1.4549 | 1.4245 | 53.61 | LoRA adapter, 80.8 MB |
| v50 | 652 | 1.5554 | 1.4340 | 1.3921 | 108.01 | LoRA adapter, 80.8 MB |
| v51 | 652 | 1.4956 | 1.3773 | 1.3266 | 107.54 | LoRA adapter, 161.5 MB |
| v52 | 326 | 1.6356 | 1.5478 | 1.5008 | 43.25 | full model, 1.60 GB |
| v53 | 978 | 1.6426 | 1.5556 | 1.4619 | 131.62 | full model, 1.60 GB |

Recommended evaluation order is v49, v51, v52, v50, then v53. v49 comes first
because the exact data/parameter recipe has an external 0.9107 leaderboard result.
v51 is the strongest new optimization result. v52 is the better full-SFT candidate:
one epoch at 2e-5 reaches essentially the same tail loss as v53's three epochs, so
the long 1e-5 schedule adds compute without a clear optimization gain on this data.

## Sources

- Official competition FAQ: https://www.streamlake.com/document/WANQING/mq57afym1d7p20atnau
- Public 0.9107 data and LoRA recipe: https://huggingface.co/datasets/Frinkleko/kuaishou-llmrec-sft-baseline-0.91
- Public 3-epoch full-SFT convergence record: https://huggingface.co/ZeroZeroSeven/onereason-0.8b-kuaishou-full-sft
