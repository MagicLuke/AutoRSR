# D9 ablation - does emitting CHAT `xxx` change downstream AutoRSR DLD screening?

Same xxx-SFT model (`qwen_xxx.jsonl`, greedy). Only variable = the `xxx` token.
Cohort: 132 speakers | DLD(CELF<86)=58 TD=74. Per-item alignment to known target (eval/item_map.csv). 193/2065 utterances carry >=1 `xxx`.
Threshold cohort (non-N/A AutoRSR norm decision): 126 scored, 6 N/A (age outside 5-9y norms), identical across A/B/C.

## Per-subject AutoRSR total/32 and continuous-total AUC

Positive class = DLD(CELF<86); AUC score = -(total/32) (lower recall -> more DLD-like); all 132 subjects. DeLong 95% CI.

| variant | mean total/32 (all) | DLD mean | TD mean | TD-DLD gap | AUC | DeLong 95% CI |
|---|---|---|---|---|---|---|
| A (xxx-kept) | 13.28 | 9.00 | 16.64 | 7.64 | 0.786 | [0.706, 0.866] |
| B (xxx-stripped) | 13.35 | 9.09 | 16.69 | 7.60 | 0.785 | [0.705, 0.865] |
| C (hybrid maxent>2.0->xxx) | 13.22 | 8.95 | 16.57 | 7.62 | 0.786 | [0.706, 0.866] |

## Sensitivity / specificity / Youden J at AutoRSR norm percentiles

| variant | cutoff | scored | TP | FP | TN | FN | Sens | Spec | YoudenJ |
|---|---|---|---|---|---|---|---|---|---|
| A | 5th | 126 | 39 | 12 | 60 | 15 | 0.722 | 0.833 | 0.556 |
| A | 10th | 126 | 46 | 25 | 47 | 8 | 0.852 | 0.653 | 0.505 |
| A | 15th | 126 | 48 | 31 | 41 | 6 | 0.889 | 0.569 | 0.458 |
| B | 5th | 126 | 39 | 11 | 61 | 15 | 0.722 | 0.847 | 0.569 |
| B | 10th | 126 | 46 | 25 | 47 | 8 | 0.852 | 0.653 | 0.505 |
| B | 15th | 126 | 49 | 31 | 41 | 5 | 0.907 | 0.569 | 0.477 |
| C | 5th | 126 | 39 | 12 | 60 | 15 | 0.722 | 0.833 | 0.556 |
| C | 10th | 126 | 46 | 25 | 47 | 8 | 0.852 | 0.653 | 0.505 |
| C | 15th | 126 | 48 | 31 | 41 | 6 | 0.889 | 0.569 | 0.458 |

## xxx effect: A (kept) minus B (stripped)

- Per-subject total/32: mean A 13.28 vs B 13.35 -> A-B = -0.068 pts (sum -9/32). 11/132 subjects change (10 lower under kept, 1 higher).
- Per-utterance points (pre best-per-item agg): 13/2065 utts change (mean A-B -0.0044; 11 lose pts under kept, 2 gain).
- 5th pctile: 1 decision flips (B->A: 0 ->Pass, 1 ->Fail). dSens=+0.000 dSpec=-0.014 dYoudenJ=-0.014.
- 10th pctile: 0 decision flips (B->A: 0 ->Pass, 0 ->Fail). dSens=+0.000 dSpec=+0.000 dYoudenJ=+0.000.
- 15th pctile: 1 decision flips (B->A: 1 ->Pass, 0 ->Fail). dSens=-0.019 dSpec=+0.000 dYoudenJ=-0.019.
- Continuous AUC: A 0.786 vs B 0.785 -> dAUC +0.0013 (DeLong z=+0.47, p=0.636).
- Bootstrap dAUC(A-B) 95% CI [-0.0044, +0.0074] (excludes 0: no).
- (C secondary) mean total A 13.28 vs C 13.22 (C adds entropy-xxx to 114 utts); AUC A 0.786 vs C 0.786.

