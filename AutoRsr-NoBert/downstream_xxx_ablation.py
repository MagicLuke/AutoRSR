#!/usr/bin/env python3
"""
D9 ablation: does emitting the CHAT `xxx` (unintelligible) marker in the ASR
transcript change downstream AutoRSR DLD screening?

CLEAN A/B (same xxx-SFT model, only the xxx token differs):
  A  xxx-kept     : model hypothesis as-is (its own xxx tokens kept)
  B  xxx-stripped : same hypothesis with `xxx` tokens removed
  C  hybrid (2nd) : hypothesis with every word whose max_entropy>2.0 ALSO mapped to xxx
                    (model's xxx kept; = model-xxx UNION entropy>2.0-xxx)

Each per-utterance transcript is scored through AutoRSR's OWN per-item machinery
(fuzzy_align_response_to_gt -> standarize -> score_rsr_errors -> score_rsr), exactly
like downstream_peritem.py, except the target item index comes from eval/item_map.csv
(joined on segment_id) instead of the parquet, and age/group from eval/id_map.csv.

Per subject: best (max) points per item k in 1..16, summed -> total/32 -> AutoRSR
age/percentile Pass/Fail. Metrics vs the CELF<86 (group) reference: sens/spec/YoudenJ
@5/10/15th pctile, plus continuous total/32 AUC (DeLong CI), for A vs B (vs C).

Run in conda env csr4rsr-dev (ftfy+transformers+pandas+numpy+scipy+sklearn). CPU only.
"""
import os, io, re, sys, json, types, glob, contextlib, statistics as st
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.modules.setdefault("whisperx", types.ModuleType("whisperx"))
import auto_rsr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PCTS = [5, 10, 15]
ENT_THRESH = 2.0
NBOOT = 2000
RNG = np.random.default_rng(20260628)

JSONL    = "/ws/ifp-54_1/hasegawa/haolong2/AI4EE/ASR4RSR/runs/probe/qwen_xxx.jsonl"
ITEM_MAP = "/ws/ifp-54_1/hasegawa/haolong2/AI4EE/eval/item_map.csv"
ID_MAP   = "/ws/ifp-54_1/hasegawa/haolong2/AI4EE/eval/id_map.csv"

GT_LINES = [
    "The big football player washed the car with the hose.",
    "All of the pictures were colored by his little sister.",
    "The rose bushes were planted yesterday by the girl scouts.",
    "The happy little girl kicked the ball over the fence.",
    "His little brother cleaned the dirty dishes and cups.",
    "A special cage was made to hold the dangerous animals.",
    "Everybody in my school colored Easter eggs for the picnic.",
    "A new hole was dug for the kid's swimming pool.",
    "Only the first graders made a birdhouse for their parents.",
    "My little sister's dog caught the ball on the first bounce.",
    "The soccer ball was kicked into the school's parking lot.",
    "The lion's teeth were cleaned with a giant toothbrush.",
    "Some of the kids dug holes in the sand two feet deep.",
    "The little white mouse was caught by our neighbor's cat.",
    "The second grade students planted coconuts in the garden.",
    "The dirty clothes were washed with soap one more time.",
]

_core = lambda t: re.sub(r"[^a-z]", "", t.lower())


def build_variants():
    rows = [json.loads(l) for l in open(JSONL)]
    itm = pd.read_csv(ITEM_MAP)
    seg2item = dict(zip(itm["segment_id"], itm["item"]))
    idm = pd.read_csv(ID_MAP)
    spk2age = dict(zip(idm["speaker_id"], idm["age_months"]))
    spk2dld = dict(zip(idm["speaker_id"], idm["dld"].astype(int)))

    recs = []
    for r in rows:
        seg = os.path.splitext(os.path.basename(r["audio"]))[0]
        item = int(seg2item[seg])
        toks = r["hypothesis"].split()
        words = r["words"]
        # A: as-is
        predA = r["hypothesis"]
        # B: drop xxx tokens (handles 'xxx' and 'xxx.')
        predB = " ".join(t for t in toks if _core(t) != "xxx")
        # C: hypothesis + map every maxent>ENT_THRESH word to xxx (model xxx kept)
        predC = " ".join("xxx" if (_core(w[0]) == "xxx" or w[3] > ENT_THRESH) else w[0]
                         for w in words)
        recs.append(dict(seg=seg, sid=r["speaker_id"], item=item,
                         age=spk2age[r["speaker_id"]], dld=spk2dld[r["speaker_id"]],
                         predA=predA, predB=predB, predC=predC,
                         had_xxx=any(_core(t) == "xxx" for t in toks),
                         c_added=any(_core(w[0]) != "xxx" and w[3] > ENT_THRESH for w in words)))
    return recs


_cache = {}
def score_item(pred, gt):
    key = (pred, gt)
    if key in _cache:
        return _cache[key]
    with contextlib.redirect_stdout(io.StringIO()):
        aligned = auto_rsr.fuzzy_align_response_to_gt(gt, pred or "")
        gt_w = auto_rsr.standarize(gt).split()
        al_w = auto_rsr.standarize(aligned).split()
        edits = auto_rsr.score_rsr_errors(al_w, gt_w)
        err = sum(len(v) for v in edits.values())
        pts = auto_rsr.score_rsr([err])
    _cache[key] = pts
    return pts


def score_variant(recs, predkey):
    """Per-subject best-per-item sum; returns DataFrame sorted by sid."""
    subj = OrderedDict()
    for r in recs:
        s = subj.setdefault(r["sid"], {"age": r["age"], "dld": r["dld"],
                                       "items": defaultdict(int), "n_utt": 0})
        pts = score_item(str(r[predkey]) if r[predkey] else "", GT_LINES[r["item"] - 1])
        if pts > s["items"][r["item"]]:
            s["items"][r["item"]] = pts
        s["n_utt"] += 1
    out = []
    for sid, s in subj.items():
        total = sum(s["items"].get(k, 0) for k in range(1, 17))
        try:
            age_m = int(float(s["age"]))
        except (ValueError, TypeError):
            age_m = -1
        rec = {"sid": sid, "age_m": age_m, "ref_dld": bool(s["dld"]),
               "n_items_attempted": sum(1 for k in range(1, 17) if k in s["items"]),
               "n_utt": s["n_utt"], "total_score": total}
        for p in PCTS:
            rec[f"pct{p}"] = auto_rsr.evaluate_rsr_result(total, age_m, p)
        out.append(rec)
    return pd.DataFrame(out).sort_values("sid").reset_index(drop=True)


def confusion(df, pct):
    TP = FP = TN = FN = na = 0
    for _, r in df.iterrows():
        d = r[f"pct{pct}"]
        if d not in ("Pass", "Fail"):
            na += 1; continue
        pred_dld = (d == "Fail")
        if r["ref_dld"] and pred_dld:       TP += 1
        elif r["ref_dld"] and not pred_dld: FN += 1
        elif not r["ref_dld"] and pred_dld: FP += 1
        else:                               TN += 1
    se = TP / (TP + FN) if (TP + FN) else float("nan")
    sp = TN / (TN + FP) if (TN + FP) else float("nan")
    j = se + sp - 1 if (TP + FN and TN + FP) else float("nan")
    return dict(TP=TP, FP=FP, TN=TN, FN=FN, na=na, scored=TP + FP + TN + FN, se=se, sp=sp, j=j)


# ---- DeLong (Sun & Xu 2014), ported from downstream_significance.py ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def _fast_delong(preds_sorted_T, m):
    n = preds_sorted_T.shape[1] - m
    pos = preds_sorted_T[:, :m]; neg = preds_sorted_T[:, m:]; k = preds_sorted_T.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted_T[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def auc_ci(y, score):
    order = (-y).argsort(kind="mergesort"); m = int(y.sum())
    aucs, cov = _fast_delong(score[None, order], m)
    se = float(np.sqrt(cov[0, 0]))
    return float(aucs[0]), max(0.0, aucs[0] - 1.96 * se), min(1.0, aucs[0] + 1.96 * se), se


def delong_paired(y, sa, sb):
    from scipy.stats import norm
    order = (-y).argsort(kind="mergesort"); m = int(y.sum())
    aucs, cov = _fast_delong(np.vstack((sa, sb))[:, order], m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var); p = 2 * norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def main():
    recs = build_variants()
    n_had = sum(r["had_xxx"] for r in recs)
    n_cadd = sum(r["c_added"] for r in recs)
    print(f"utts={len(recs)} | xxx-emitting utts (A)={n_had} | utts C adds entropy-xxx to={n_cadd}")

    variants = OrderedDict(A="predA", B="predB", C="predC")
    dfs = {v: score_variant(recs, k) for v, k in variants.items()}
    labels = {"A": "xxx-kept", "B": "xxx-stripped", "C": "hybrid maxent>2.0->xxx"}

    # sanity: identical cohorts
    base = dfs["A"]
    for v in variants:
        assert dfs[v]["sid"].tolist() == base["sid"].tolist()
        assert (dfs[v]["ref_dld"].values == base["ref_dld"].values).all()

    ref = base["ref_dld"].astype(bool).values
    y = ref.astype(int)
    n = len(base)

    md = []
    def log(s=""): print(s); md.append(s)
    rows_csv = []

    log("# D9 ablation - does emitting CHAT `xxx` change downstream AutoRSR DLD screening?")
    log("")
    log(f"Same xxx-SFT model (`qwen_xxx.jsonl`, greedy). Only variable = the `xxx` token.")
    log(f"Cohort: {n} speakers | DLD(CELF<86)={int(ref.sum())} TD={int((~ref).sum())}. "
        f"Per-item alignment to known target (eval/item_map.csv). "
        f"{n_had}/{len(recs)} utterances carry >=1 `xxx`.")
    scored5 = confusion(dfs['A'], 5)['scored']
    log(f"Threshold cohort (non-N/A AutoRSR norm decision): {scored5} scored, "
        f"{n - scored5} N/A (age outside 5-9y norms), identical across A/B/C.")
    log("")

    # ---------- mean total / discrimination ----------
    log("## Per-subject AutoRSR total/32 and continuous-total AUC")
    log("")
    log("Positive class = DLD(CELF<86); AUC score = -(total/32) (lower recall -> more DLD-like); "
        f"all {n} subjects. DeLong 95% CI.")
    log("")
    log("| variant | mean total/32 (all) | DLD mean | TD mean | TD-DLD gap | AUC | DeLong 95% CI |")
    log("|---|---|---|---|---|---|---|")
    aucs = {}
    for v in variants:
        d = dfs[v]
        tot = d["total_score"].values.astype(float)
        dld = tot[ref]; td = tot[~ref]
        a, lo, hi, se = auc_ci(y, -tot)
        aucs[v] = (a, lo, hi, tot)
        log(f"| {v} ({labels[v]}) | {tot.mean():.2f} | {dld.mean():.2f} | {td.mean():.2f} | "
            f"{td.mean()-dld.mean():.2f} | {a:.3f} | [{lo:.3f}, {hi:.3f}] |")
        rows_csv.append(dict(metric="summary", variant=v, label=labels[v],
                             mean_total=round(float(tot.mean()), 3),
                             dld_mean=round(float(dld.mean()), 3), td_mean=round(float(td.mean()), 3),
                             gap=round(float(td.mean()-dld.mean()), 3),
                             auc=round(a, 4), auc_lo=round(lo, 4), auc_hi=round(hi, 4)))
    log("")

    # ---------- sens/spec/J table ----------
    log("## Sensitivity / specificity / Youden J at AutoRSR norm percentiles")
    log("")
    log("| variant | cutoff | scored | TP | FP | TN | FN | Sens | Spec | YoudenJ |")
    log("|---|---|---|---|---|---|---|---|---|---|")
    conf = {v: {} for v in variants}
    for v in variants:
        for p in PCTS:
            c = confusion(dfs[v], p); conf[v][p] = c
            log(f"| {v} | {p}th | {c['scored']} | {c['TP']} | {c['FP']} | {c['TN']} | {c['FN']} | "
                f"{c['se']:.3f} | {c['sp']:.3f} | {c['j']:.3f} |")
            rows_csv.append(dict(metric="screen", variant=v, label=labels[v], pct=p,
                                 scored=c['scored'], TP=c['TP'], FP=c['FP'], TN=c['TN'], FN=c['FN'],
                                 sens=round(c['se'], 4), spec=round(c['sp'], 4), youden=round(c['j'], 4)))
    log("")

    # ---------- A vs B (vs C) deltas ----------
    log("## xxx effect: A (kept) minus B (stripped)")
    log("")
    totA = dfs["A"]["total_score"].values.astype(float)
    totB = dfs["B"]["total_score"].values.astype(float)
    dtot = totA - totB
    n_changed = int((dtot != 0).sum()); n_lower = int((dtot < 0).sum()); n_higher = int((dtot > 0).sum())
    log(f"- Per-subject total/32: mean A {totA.mean():.2f} vs B {totB.mean():.2f} "
        f"-> A-B = {dtot.mean():+.3f} pts (sum {dtot.sum():+.0f}/32). "
        f"{n_changed}/{n} subjects change ({n_lower} lower under kept, {n_higher} higher).")
    # per-utterance points delta
    putA = [score_item(r["predA"], GT_LINES[r["item"]-1]) for r in recs]
    putB = [score_item(r["predB"], GT_LINES[r["item"]-1]) for r in recs]
    dput = np.array(putA) - np.array(putB)
    log(f"- Per-utterance points (pre best-per-item agg): {int((dput!=0).sum())}/{len(recs)} utts change "
        f"(mean A-B {dput.mean():+.4f}; {int((dput<0).sum())} lose pts under kept, {int((dput>0).sum())} gain).")
    # decision flips at pct5
    for p in PCTS:
        decA = dfs["A"][f"pct{p}"].values; decB = dfs["B"][f"pct{p}"].values
        flips = [(dfs['A']['sid'][i], decB[i], decA[i], ref[i])
                 for i in range(n) if decA[i] != decB[i] and decA[i] in ("Pass","Fail") and decB[i] in ("Pass","Fail")]
        toFail = sum(1 for f in flips if f[2] == "Fail")
        toPass = sum(1 for f in flips if f[2] == "Pass")
        dj = conf["A"][p]["j"] - conf["B"][p]["j"]
        dse = conf["A"][p]["se"] - conf["B"][p]["se"]
        dsp = conf["A"][p]["sp"] - conf["B"][p]["sp"]
        log(f"- {p}th pctile: {len(flips)} decision flips (B->A: {toPass} ->Pass, {toFail} ->Fail). "
            f"dSens={dse:+.3f} dSpec={dsp:+.3f} dYoudenJ={dj:+.3f}.")
        rows_csv.append(dict(metric="A_minus_B", pct=p, n_flips=len(flips),
                             to_fail=toFail, to_pass=toPass,
                             d_sens=round(dse, 4), d_spec=round(dsp, 4), d_youden=round(dj, 4)))
    # DeLong paired A vs B
    aa, ab, z, pv = delong_paired(y, -totA, -totB)
    log(f"- Continuous AUC: A {aa:.3f} vs B {ab:.3f} -> dAUC {aa-ab:+.4f} (DeLong z={z:+.2f}, p={pv:.3f}).")
    # paired bootstrap dAUC
    bi = RNG.integers(0, n, size=(NBOOT, n)); bd = []
    for b in range(NBOOT):
        ii = bi[b]; ys = y[ii]
        if ys.sum() in (0, len(ys)): continue
        bd.append(roc_auc_score(ys, -totA[ii]) - roc_auc_score(ys, -totB[ii]))
    bd = np.array(bd); blo, bhi = np.percentile(bd, [2.5, 97.5])
    log(f"- Bootstrap dAUC(A-B) 95% CI [{blo:+.4f}, {bhi:+.4f}] (excludes 0: {'YES' if (blo>0 or bhi<0) else 'no'}).")
    rows_csv.append(dict(metric="A_minus_B_auc", auc_A=round(aa,4), auc_B=round(ab,4),
                         dAUC=round(aa-ab,4), delong_z=round(z,4), delong_p=round(pv,4),
                         boot_lo=round(float(blo),4), boot_hi=round(float(bhi),4)))
    # A vs C summary delta
    totC = dfs["C"]["total_score"].values.astype(float)
    log(f"- (C secondary) mean total A {totA.mean():.2f} vs C {totC.mean():.2f} "
        f"(C adds entropy-xxx to {n_cadd} utts); AUC A {aucs['A'][0]:.3f} vs C {aucs['C'][0]:.3f}.")
    log("")

    # write outputs
    persubj = base[["sid", "age_m", "ref_dld"]].copy()
    for v in variants:
        persubj[f"total_{v}"] = dfs[v]["total_score"].values
        persubj[f"pct5_{v}"] = dfs[v]["pct5"].values
    persubj.to_csv(os.path.join(HERE, "downstream_xxx_ablation_persubject.csv"), index=False)
    pd.DataFrame(rows_csv).to_csv(os.path.join(HERE, "downstream_xxx_ablation.csv"), index=False)
    with open(os.path.join(HERE, "downstream_xxx_ablation.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\nwrote downstream_xxx_ablation.{csv,md} and _persubject.csv")


if __name__ == "__main__":
    main()
