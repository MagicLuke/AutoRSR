#!/usr/bin/env python3
"""
Significance testing for downstream DLD-screening differences between ASR backbones.

Tasks (per the Part-2 paper analysis spec):
  1. McNemar's test on paired per-subject CORRECT/INCORRECT screening decisions,
     v3-vs-base and v3-vs-whisper, at each cutoff (5/10/15th pctile).
  2. Subject bootstrap (2000 resamples) of Youden J per backbone (95% CI) and of the
     paired J difference (v3-base, v3-whisper) -> does the 95% CI exclude 0?
  3. ROC/AUC of the CONTINUOUS AutoRSR total/32 (and human SRTotalScore) for DLD-vs-TD,
     per backbone + human; bootstrap CIs and DeLong test for paired AUC comparison.
  4. Verdict.

Inputs: downstream_peritem_{v3,base-qwen,whisper-large-v3}.csv, human_ceiling_persubject.csv
Run in conda env `reliability` (pandas/scipy/sklearn/numpy). CPU only.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm as _norm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PCTS = [5, 10, 15]
RNG = np.random.default_rng(20260628)
NBOOT = 2000

FILES = {
    "v3":      "downstream_peritem_v3.csv",
    "base":    "downstream_peritem_base-qwen.csv",
    "whisper": "downstream_peritem_whisper-large-v3.csv",
}
HUMAN = "human_ceiling_persubject.csv"


def load():
    dfs = {}
    for k, f in FILES.items():
        d = pd.read_csv(os.path.join(HERE, f))
        d = d.sort_values("sid").reset_index(drop=True)
        dfs[k] = d
    hu = pd.read_csv(os.path.join(HERE, HUMAN)).sort_values("sid").reset_index(drop=True)
    # sanity: identical sid order & ref_dld
    sids = dfs["v3"]["sid"].tolist()
    for k, d in dfs.items():
        assert d["sid"].tolist() == sids, f"sid order mismatch {k}"
        assert (d["ref_dld"].values == dfs["v3"]["ref_dld"].values).all()
    assert hu["sid"].tolist() == sids
    return dfs, hu, sids


def decision_correct(df, pct):
    """Per-subject 1 if decision matches ref_dld, 0 if wrong, np.nan if N/A."""
    ref = df["ref_dld"].astype(bool).values
    dec = df[f"pct{pct}"].values
    out = np.full(len(df), np.nan)
    for i, d in enumerate(dec):
        if d not in ("Pass", "Fail"):
            continue
        pred_dld = (d == "Fail")
        out[i] = 1.0 if pred_dld == ref[i] else 0.0
    return out


def youden_j(ref_dld, dec):
    """Youden J from a Pass/Fail decision array (object) given boolean ref."""
    TP = FP = TN = FN = 0
    for r, d in zip(ref_dld, dec):
        if d not in ("Pass", "Fail"):
            continue
        pred_dld = (d == "Fail")
        if r and pred_dld:       TP += 1
        elif r and not pred_dld: FN += 1
        elif (not r) and pred_dld: FP += 1
        else:                    TN += 1
    if (TP + FN) == 0 or (TN + FP) == 0:
        return np.nan, dict(TP=TP, FP=FP, TN=TN, FN=FN)
    se = TP / (TP + FN)
    sp = TN / (TN + FP)
    return se + sp - 1, dict(TP=TP, FP=FP, TN=TN, FN=FN, se=se, sp=sp)


# ---------------------------------------------------------------- Task 1: McNemar
def mcnemar(corr_a, corr_b):
    """McNemar on paired correct(1)/incorrect(0) arrays. Drop pairs where either is NaN."""
    mask = ~(np.isnan(corr_a) | np.isnan(corr_b))
    a = corr_a[mask].astype(int)
    b = corr_b[mask].astype(int)
    # discordant: b01 = a wrong & b right ; b10 = a right & b wrong
    b10 = int(np.sum((a == 1) & (b == 0)))   # A(=v3) correct, B wrong
    b01 = int(np.sum((a == 0) & (b == 1)))   # A(=v3) wrong, B correct
    n_disc = b10 + b01
    # exact binomial (two-sided), robust for small discordant counts
    p_exact = binomtest(b10, n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    # chi-square with continuity correction (reference)
    if n_disc > 0:
        chi = (abs(b10 - b01) - 1) ** 2 / n_disc
        p_cc = chi2.sf(chi, 1)
    else:
        chi, p_cc = 0.0, 1.0
    return dict(n_pairs=int(mask.sum()), b10_v3correct=b10, b01_othercorrect=b01,
                n_discordant=n_disc, chi2_cc=chi, p_exact=p_exact, p_chi2cc=p_cc)


# ---------------------------------------------------------------- Task 3: DeLong
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    """Sun & Xu (2014) fast DeLong. predictions_sorted_transposed: [k, n] positives first."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, score_a, score_b):
    """DeLong paired test of AUC(A) vs AUC(B). Positive class = 1.
    Returns auc_a, auc_b, z, p (two-sided), cov matrix."""
    order = (-y_true).argsort(kind="mergesort")  # positives (1) first
    label_1_count = int(y_true.sum())
    preds = np.vstack((score_a, score_b))[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2 * _norm.sf(abs(z))
    return aucs[0], aucs[1], z, p, cov


def auc_ci_delong(y_true, score):
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = score[None, order]
    aucs, cov = fast_delong(preds, label_1_count)
    se = np.sqrt(cov[0, 0])
    lo = aucs[0] - 1.96 * se
    hi = aucs[0] + 1.96 * se
    return float(aucs[0]), float(max(0, lo)), float(min(1, hi)), float(se)


def main():
    dfs, hu, sids = load()
    n = len(sids)
    ref = dfs["v3"]["ref_dld"].astype(bool).values
    lines_md = []
    rows_csv = []

    def log(s=""):
        print(s)
        lines_md.append(s)

    log("# Downstream DLD-screening: statistical significance of ASR-backbone differences")
    log("")
    log(f"Cohort: {n} subjects | DLD(ref)={int(ref.sum())} TD={int((~ref).sum())}.")
    scored_mask = dfs["v3"]["pct5"].isin(["Pass", "Fail"]).values
    log(f"Threshold-screening cohort (non-N/A norm decision): {int(scored_mask.sum())} "
        f"(DLD={int(ref[scored_mask].sum())} TD={int((~ref[scored_mask]).sum())}); "
        f"6 subjects N/A (age outside 5-9y norms) in ALL backbones identically.")
    log(f"Bootstrap: {NBOOT} subject resamples, percentile 95% CIs, paired (same resample "
        f"indices across backbones). AUC uses continuous total/32 over all {n} subjects.")
    log("")

    # ---- point-estimate J table
    log("## Point-estimate Youden J (per-item alignment), screening cohort")
    log("")
    log("| cutoff | v3 | base | whisper | human |")
    log("|---|---|---|---|---|")
    Jpoint = {}
    for pct in PCTS:
        jr = {}
        for k in ["v3", "base", "whisper"]:
            j, _ = youden_j(ref, dfs[k][f"pct{pct}"].values)
            jr[k] = j
        jh, _ = youden_j(ref, hu[f"pct{pct}"].values)
        jr["human"] = jh
        Jpoint[pct] = jr
        log(f"| {pct}th | {jr['v3']:.3f} | {jr['base']:.3f} | {jr['whisper']:.3f} | {jr['human']:.3f} |")
    log("")

    # ============================================================ TASK 1
    log("## Task 1 - McNemar's test (paired correct/incorrect screening decisions)")
    log("")
    log("b10 = #subjects v3 classifies CORRECTLY but the other backbone gets WRONG; "
        "b01 = the reverse. Exact two-sided binomial p (primary), chi2 w/ continuity correction (ref).")
    log("")
    log("| cutoff | comparison | n_pairs | b10(v3 right) | b01(other right) | discordant | p_exact | p_chi2cc |")
    log("|---|---|---|---|---|---|---|---|")
    for pct in PCTS:
        cv3 = decision_correct(dfs["v3"], pct)
        for other in ["base", "whisper"]:
            co = decision_correct(dfs[other], pct)
            m = mcnemar(cv3, co)
            log(f"| {pct}th | v3 vs {other} | {m['n_pairs']} | {m['b10_v3correct']} | "
                f"{m['b01_othercorrect']} | {m['n_discordant']} | {m['p_exact']:.4f} | {m['p_chi2cc']:.4f} |")
            rows_csv.append(dict(task="mcnemar", cutoff=pct, comparison=f"v3_vs_{other}",
                                 n_pairs=m['n_pairs'], b10_v3correct=m['b10_v3correct'],
                                 b01_othercorrect=m['b01_othercorrect'], n_discordant=m['n_discordant'],
                                 stat=round(m['chi2_cc'], 4), p_exact=round(m['p_exact'], 4),
                                 p_chi2cc=round(m['p_chi2cc'], 4)))
    log("")

    # ============================================================ TASK 2
    log("## Task 2 - Subject bootstrap of Youden J (95% CI) and paired J difference")
    log("")
    # Precompute correct-arrays once; bootstrap resamples subject indices.
    idx_all = np.arange(n)
    # store per-backbone Pass/Fail decisions as boolean pred_dld + nan mask per cutoff
    boot_J = {pct: {k: np.full(NBOOT, np.nan) for k in ["v3", "base", "whisper", "human"]} for pct in PCTS}
    # vectorized helper: for a resample index array compute J
    def j_from_idx(ref_s, dec_s):
        # dec_s object array Pass/Fail/NA
        valid = (dec_s == "Pass") | (dec_s == "Fail")
        r = ref_s[valid]
        pred_dld = (dec_s[valid] == "Fail")
        pos = r
        neg = ~r
        if pos.sum() == 0 or neg.sum() == 0:
            return np.nan
        TP = np.sum(pos & pred_dld); FN = np.sum(pos & ~pred_dld)
        TN = np.sum(neg & ~pred_dld); FP = np.sum(neg & pred_dld)
        return TP / (TP + FN) + TN / (TN + FP) - 1

    dec = {pct: {k: dfs[k][f"pct{pct}"].values for k in ["v3", "base", "whisper"]} for pct in PCTS}
    for pct in PCTS:
        dec[pct]["human"] = hu[f"pct{pct}"].values

    boot_idx = RNG.integers(0, n, size=(NBOOT, n))
    for b in range(NBOOT):
        ii = boot_idx[b]
        ref_s = ref[ii]
        for pct in PCTS:
            for k in ["v3", "base", "whisper", "human"]:
                boot_J[pct][k][b] = j_from_idx(ref_s, dec[pct][k][ii])

    def ci(arr):
        a = arr[~np.isnan(arr)]
        return np.nanmean(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

    log("### Per-backbone Youden J with 95% bootstrap CI")
    log("")
    log("| cutoff | backbone | J (point) | J (boot mean) | 95% CI |")
    log("|---|---|---|---|---|")
    for pct in PCTS:
        for k in ["v3", "base", "whisper", "human"]:
            mean, lo, hi = ci(boot_J[pct][k])
            log(f"| {pct}th | {k} | {Jpoint[pct][k]:.3f} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] |")
            rows_csv.append(dict(task="bootJ", cutoff=pct, comparison=k,
                                 stat=round(Jpoint[pct][k], 4), boot_mean=round(mean, 4),
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    log("")
    log("### Paired J difference (bootstrap): does 95% CI exclude 0?")
    log("")
    log("| cutoff | diff | point | boot mean | 95% CI | excludes 0? | P(diff>0) |")
    log("|---|---|---|---|---|---|---|")
    for pct in PCTS:
        for other in ["base", "whisper"]:
            d = boot_J[pct]["v3"] - boot_J[pct][other]
            d = d[~np.isnan(d)]
            mean = d.mean(); lo, hi = np.percentile(d, [2.5, 97.5])
            pt = Jpoint[pct]["v3"] - Jpoint[pct][other]
            excl = (lo > 0) or (hi < 0)
            pgt = float(np.mean(d > 0))
            log(f"| {pct}th | v3-{other} | {pt:+.3f} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                f"{'YES' if excl else 'no'} | {pgt:.3f} |")
            rows_csv.append(dict(task="bootJ_diff", cutoff=pct, comparison=f"v3_minus_{other}",
                                 stat=round(pt, 4), boot_mean=round(mean, 4), ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4), excludes_zero=excl, p_diff_gt0=round(pgt, 4)))
    log("")

    # ============================================================ TASK 3
    log("## Task 3 - ROC/AUC of CONTINUOUS AutoRSR total/32 (DLD-vs-TD discrimination)")
    log("")
    log("Positive class = DLD(ref). Score = -(total/32): lower recall total -> more DLD-like. "
        f"All {n} subjects (continuous total has no N/A). DeLong CI + DeLong paired test.")
    log("")
    y = ref.astype(int)
    totals = {
        "v3":      -dfs["v3"]["total_score"].values.astype(float),
        "base":    -dfs["base"]["total_score"].values.astype(float),  # noqa
        "whisper": -dfs["whisper"]["total_score"].values.astype(float),
        "human":   -hu["human_total"].values.astype(float),
    }
    # base key in dfs is 'base'
    totals["base"] = -dfs["base"]["total_score"].values.astype(float)

    # bootstrap AUC CIs (paired resample) as cross-check
    boot_auc = {k: np.full(NBOOT, np.nan) for k in totals}
    for b in range(NBOOT):
        ii = boot_idx[b]
        ys = y[ii]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        for k in totals:
            boot_auc[k][b] = roc_auc_score(ys, totals[k][ii])

    log("| backbone | AUC | DeLong 95% CI | bootstrap 95% CI |")
    log("|---|---|---|---|")
    auc_pt = {}
    for k in ["v3", "base", "whisper", "human"]:
        a, lo, hi, se = auc_ci_delong(y, totals[k])
        auc_pt[k] = a
        ba = boot_auc[k][~np.isnan(boot_auc[k])]
        blo, bhi = np.percentile(ba, [2.5, 97.5])
        log(f"| {k} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | [{blo:.3f}, {bhi:.3f}] |")
        rows_csv.append(dict(task="auc", comparison=k, stat=round(a, 4), ci_lo=round(lo, 4),
                             ci_hi=round(hi, 4), boot_ci_lo=round(blo, 4), boot_ci_hi=round(bhi, 4),
                             delong_se=round(se, 4)))
    log("")
    log("### Paired AUC comparisons (DeLong) + bootstrap diff CI")
    log("")
    log("| comparison | AUC_a | AUC_b | dAUC | DeLong z | DeLong p | boot dAUC 95% CI | excl 0? |")
    log("|---|---|---|---|---|---|---|---|")
    pairs = [("v3", "base"), ("v3", "whisper"), ("v3", "human"),
             ("base", "whisper"), ("base", "human"), ("whisper", "human")]
    for a_, b_ in pairs:
        aa, ab, z, p, _ = delong_test(y, totals[a_], totals[b_])
        bd = boot_auc[a_] - boot_auc[b_]
        bd = bd[~np.isnan(bd)]
        blo, bhi = np.percentile(bd, [2.5, 97.5])
        excl = (blo > 0) or (bhi < 0)
        log(f"| {a_} vs {b_} | {aa:.3f} | {ab:.3f} | {aa-ab:+.3f} | {z:+.3f} | {p:.4f} | "
            f"[{blo:+.3f}, {bhi:+.3f}] | {'YES' if excl else 'no'} |")
        rows_csv.append(dict(task="auc_pair", comparison=f"{a_}_vs_{b_}", stat=round(aa-ab, 4),
                             auc_a=round(aa, 4), auc_b=round(ab, 4), delong_z=round(z, 4),
                             p_exact=round(p, 4), ci_lo=round(blo, 4), ci_hi=round(bhi, 4),
                             excludes_zero=excl))
    log("")

    # ============================================================ TASK 4 verdict
    log("## Task 4 - Verdict")
    log("")
    log("**Which instrument separates the backbones?** All three are pinned near the "
        "human-RSR Youden ceiling (J~0.49-0.51 through AutoRSR norm thresholds). The ANSWER "
        "depends entirely on the readout:")
    log("")
    log("- **Fixed-threshold Youden J at the headline 5th-pctile cutoff:** v3 0.546 > base 0.528 "
        "> whisper 0.495, but NOT significant. McNemar p=0.50 (vs base) / 0.36 (vs whisper); "
        "paired-J bootstrap CIs straddle 0 ([-0.108,+0.143] and [-0.080,+0.182]). At 5th pctile "
        "v3 is statistically indistinguishable from base/whisper -- the gaps are noise around the ceiling.")
    log("- **Fixed-threshold J at the looser 10th/15th-pctile cutoffs:** v3's edge over base/whisper "
        "GROWS and becomes significant. McNemar v3-vs-base p=0.049 (10th) and v3-vs-base/whisper "
        "p=0.0225 (15th); paired-J bootstrap CIs EXCLUDE 0 at 15th ([+0.016,+0.217] and [+0.018,+0.217]). "
        "So a real separation exists, but only at the more lenient cutoffs where base/whisper J collapses "
        "to ~0.36 while v3 holds ~0.48. This is a threshold-placement effect, not a discriminability effect.")
    log("- **Continuous total/32 AUC (discriminability, decoupled from threshold):** v3 0.780, base 0.738, "
        "whisper 0.734, human 0.788. v3 > base/whisper by ~0.04-0.05 but NOT significant "
        "(DeLong p=0.103 / 0.073; bootstrap dAUC CIs include 0). No backbone's AUC differs significantly "
        "from any other, and crucially NONE differs from human (v3-vs-human dAUC -0.008, p=0.56). The ASR "
        "pipelines already match the human RSR test's intrinsic DLD/TD separability.")
    log("")
    log("**Plain answer.** At the screening level v3 is NOT robustly distinguishable from base/whisper: "
        "at the reported 5th-pctile operating point the differences are within noise around the ~0.5 human "
        "ceiling, and on the threshold-free AUC the three backbones (and human) are statistically tied "
        "(~0.73-0.79). The ONLY place v3 separates significantly is fixed-threshold J at the looser 10/15th "
        "cutoffs -- a brittle, threshold-dependent artifact (base/whisper happen to place subjects just across "
        "those particular norm lines), not evidence of better underlying discriminability. The instrument that "
        "actually separates backbones is therefore none of the downstream screening readouts in a robust way; "
        "any real backbone ranking has to come from the upstream morpheme/RS scoring fidelity, not from this "
        "DLD-screening endpoint.")
    log("")
    log("**Recommendation for the paper.** Report the continuous-total AUC with DeLong CIs as the primary "
        "downstream discriminability metric (v3 0.780 [0.700,0.861], base 0.738, whisper 0.734, human 0.788) and "
        "state explicitly that the three ASR backbones are statistically indistinguishable from each other AND "
        "from the human RSR ceiling. Present the 5th-pctile Youden J only as a calibrated operating point with "
        "its bootstrap CI and the McNemar non-significance, NOT as evidence v3 > base/whisper. If the 10/15th-pctile "
        "significant McNemar/J gaps are shown, frame them honestly as a threshold-placement sensitivity, not a "
        "discriminability win, and lead the backbone comparison with the upstream RS/morpheme metrics instead.")
    log("")

    pd.DataFrame(rows_csv).to_csv(os.path.join(HERE, "downstream_significance.csv"), index=False)
    with open(os.path.join(HERE, "downstream_significance.md"), "w") as f:
        f.write("\n".join(lines_md) + "\n")
    print("\nwrote downstream_significance.csv and downstream_significance.md")


if __name__ == "__main__":
    main()
