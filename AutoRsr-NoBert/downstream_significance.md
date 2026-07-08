# Downstream DLD-screening: statistical significance of ASR-backbone differences

Cohort: 132 subjects | DLD(ref)=58 TD=74.
Threshold-screening cohort (non-N/A norm decision): 126 (DLD=54 TD=72); 6 subjects N/A (age outside 5-9y norms) in ALL backbones identically.
Bootstrap: 2000 subject resamples, percentile 95% CIs, paired (same resample indices across backbones). AUC uses continuous total/32 over all 132 subjects.

## Point-estimate Youden J (per-item alignment), screening cohort

| cutoff | v3 | base | whisper | human |
|---|---|---|---|---|
| 5th | 0.546 | 0.528 | 0.495 | 0.486 |
| 10th | 0.495 | 0.389 | 0.417 | 0.495 |
| 15th | 0.477 | 0.361 | 0.361 | 0.514 |

## Task 1 - McNemar's test (paired correct/incorrect screening decisions)

b10 = #subjects v3 classifies CORRECTLY but the other backbone gets WRONG; b01 = the reverse. Exact two-sided binomial p (primary), chi2 w/ continuity correction (ref).

| cutoff | comparison | n_pairs | b10(v3 right) | b01(other right) | discordant | p_exact | p_chi2cc |
|---|---|---|---|---|---|---|---|
| 5th | v3 vs base | 126 | 12 | 8 | 20 | 0.5034 | 0.5023 |
| 5th | v3 vs whisper | 126 | 12 | 7 | 19 | 0.3593 | 0.3588 |
| 10th | v3 vs base | 126 | 13 | 4 | 17 | 0.0490 | 0.0523 |
| 10th | v3 vs whisper | 126 | 14 | 7 | 21 | 0.1892 | 0.1904 |
| 15th | v3 vs base | 126 | 11 | 2 | 13 | 0.0225 | 0.0265 |
| 15th | v3 vs whisper | 126 | 11 | 2 | 13 | 0.0225 | 0.0265 |

## Task 2 - Subject bootstrap of Youden J (95% CI) and paired J difference

### Per-backbone Youden J with 95% bootstrap CI

| cutoff | backbone | J (point) | J (boot mean) | 95% CI |
|---|---|---|---|---|
| 5th | v3 | 0.546 | 0.544 | [0.394, 0.683] |
| 5th | base | 0.528 | 0.525 | [0.392, 0.661] |
| 5th | whisper | 0.495 | 0.493 | [0.337, 0.631] |
| 5th | human | 0.486 | 0.483 | [0.330, 0.623] |
| 10th | v3 | 0.495 | 0.494 | [0.348, 0.632] |
| 10th | base | 0.389 | 0.387 | [0.257, 0.516] |
| 10th | whisper | 0.417 | 0.415 | [0.285, 0.543] |
| 10th | human | 0.495 | 0.494 | [0.336, 0.638] |
| 15th | v3 | 0.477 | 0.475 | [0.339, 0.608] |
| 15th | base | 0.361 | 0.359 | [0.233, 0.487] |
| 15th | whisper | 0.361 | 0.359 | [0.234, 0.489] |
| 15th | human | 0.514 | 0.511 | [0.358, 0.649] |

### Paired J difference (bootstrap): does 95% CI exclude 0?

| cutoff | diff | point | boot mean | 95% CI | excludes 0? | P(diff>0) |
|---|---|---|---|---|---|---|
| 5th | v3-base | +0.019 | +0.019 | [-0.108, +0.143] | no | 0.619 |
| 5th | v3-whisper | +0.051 | +0.051 | [-0.080, +0.182] | no | 0.774 |
| 10th | v3-base | +0.106 | +0.107 | [-0.013, +0.220] | no | 0.957 |
| 10th | v3-whisper | +0.079 | +0.080 | [-0.055, +0.207] | no | 0.886 |
| 15th | v3-base | +0.116 | +0.116 | [+0.016, +0.217] | YES | 0.988 |
| 15th | v3-whisper | +0.116 | +0.116 | [+0.018, +0.217] | YES | 0.989 |

## Task 3 - ROC/AUC of CONTINUOUS AutoRSR total/32 (DLD-vs-TD discrimination)

Positive class = DLD(ref). Score = -(total/32): lower recall total -> more DLD-like. All 132 subjects (continuous total has no N/A). DeLong CI + DeLong paired test.

| backbone | AUC | DeLong 95% CI | bootstrap 95% CI |
|---|---|---|---|
| v3 | 0.780 | [0.700, 0.861] | [0.698, 0.855] |
| base | 0.738 | [0.652, 0.824] | [0.652, 0.817] |
| whisper | 0.734 | [0.647, 0.820] | [0.648, 0.816] |
| human | 0.788 | [0.710, 0.866] | [0.710, 0.860] |

### Paired AUC comparisons (DeLong) + bootstrap diff CI

| comparison | AUC_a | AUC_b | dAUC | DeLong z | DeLong p | boot dAUC 95% CI | excl 0? |
|---|---|---|---|---|---|---|---|
| v3 vs base | 0.780 | 0.738 | +0.043 | +1.631 | 0.1030 | [-0.012, +0.097] | no |
| v3 vs whisper | 0.780 | 0.734 | +0.047 | +1.795 | 0.0727 | [-0.006, +0.101] | no |
| v3 vs human | 0.780 | 0.788 | -0.008 | -0.587 | 0.5569 | [-0.035, +0.020] | no |
| base vs whisper | 0.738 | 0.734 | +0.004 | +0.551 | 0.5818 | [-0.012, +0.022] | no |
| base vs human | 0.738 | 0.788 | -0.050 | -1.767 | 0.0773 | [-0.108, +0.007] | no |
| whisper vs human | 0.734 | 0.788 | -0.055 | -1.939 | 0.0525 | [-0.111, +0.001] | no |

## Task 4 - Verdict

**Which instrument separates the backbones?** All three are pinned near the human-RSR Youden ceiling (J~0.49-0.51 through AutoRSR norm thresholds). The ANSWER depends entirely on the readout:

- **Fixed-threshold Youden J at the headline 5th-pctile cutoff:** v3 0.546 > base 0.528 > whisper 0.495, but NOT significant. McNemar p=0.50 (vs base) / 0.36 (vs whisper); paired-J bootstrap CIs straddle 0 ([-0.108,+0.143] and [-0.080,+0.182]). At 5th pctile v3 is statistically indistinguishable from base/whisper -- the gaps are noise around the ceiling.
- **Fixed-threshold J at the looser 10th/15th-pctile cutoffs:** v3's edge over base/whisper GROWS and becomes significant. McNemar v3-vs-base p=0.049 (10th) and v3-vs-base/whisper p=0.0225 (15th); paired-J bootstrap CIs EXCLUDE 0 at 15th ([+0.016,+0.217] and [+0.018,+0.217]). So a real separation exists, but only at the more lenient cutoffs where base/whisper J collapses to ~0.36 while v3 holds ~0.48. This is a threshold-placement effect, not a discriminability effect.
- **Continuous total/32 AUC (discriminability, decoupled from threshold):** v3 0.780, base 0.738, whisper 0.734, human 0.788. v3 > base/whisper by ~0.04-0.05 but NOT significant (DeLong p=0.103 / 0.073; bootstrap dAUC CIs include 0). No backbone's AUC differs significantly from any other, and crucially NONE differs from human (v3-vs-human dAUC -0.008, p=0.56). The ASR pipelines already match the human RSR test's intrinsic DLD/TD separability.

**Plain answer.** At the screening level v3 is NOT robustly distinguishable from base/whisper: at the reported 5th-pctile operating point the differences are within noise around the ~0.5 human ceiling, and on the threshold-free AUC the three backbones (and human) are statistically tied (~0.73-0.79). The ONLY place v3 separates significantly is fixed-threshold J at the looser 10/15th cutoffs -- a brittle, threshold-dependent artifact (base/whisper happen to place subjects just across those particular norm lines), not evidence of better underlying discriminability. The instrument that actually separates backbones is therefore none of the downstream screening readouts in a robust way; any real backbone ranking has to come from the upstream morpheme/RS scoring fidelity, not from this DLD-screening endpoint.

**Recommendation for the paper.** Report the continuous-total AUC with DeLong CIs as the primary downstream discriminability metric (v3 0.780 [0.700,0.861], base 0.738, whisper 0.734, human 0.788) and state explicitly that the three ASR backbones are statistically indistinguishable from each other AND from the human RSR ceiling. Present the 5th-pctile Youden J only as a calibrated operating point with its bootstrap CI and the McNemar non-significance, NOT as evidence v3 > base/whisper. If the 10/15th-pctile significant McNemar/J gaps are shown, frame them honestly as a threshold-placement sensitivity, not a discriminability win, and lead the backbone comparison with the upstream RS/morpheme metrics instead.

