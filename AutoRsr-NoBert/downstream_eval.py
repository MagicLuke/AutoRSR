#!/usr/bin/env python3
"""
G4 downstream DLD-screening eval — switch our best ASR (Qwen3-ASR-1.7B v3) into AutoRSR.

We already have v3's per-utterance transcripts (the leaderboard predictions.csv), so we bypass
AutoRSR's WhisperX `transcribe()` entirely and inject those transcripts into the *scoring* half of
AutoRSR (align -> edit-distance -> points -> age/percentile pass/fail). The ASR is then the only
swapped component; everything downstream is AutoRSR exactly as designed.

Reference = the leaderboard cohort's own CELF labels (the `group`/`dld` columns in predictions.csv):
Fail => screen-positive (DLD), Pass => negative (typical). We report sensitivity / specificity /
the paper's diagnostic-screening metrics (Se/Sp/PPV/NPV/PLR/NLR/Youden J) at the 5th/10th/15th percentile
cutoffs, plus coverage (AutoRSR's norms only cover ages 5-9).

NOTE: this is the leaderboard's CELF cohort, NOT Part-1's CELF-4 198-subject reference, so these
numbers sit *next to* (not head-to-head with) Part-1's Whisper 0.768/0.784.

Run from this directory (AutoRsr-NoBert/) so AutoRSR's relative `english.json` resolves:
    python downstream_eval.py
Optionally override the predictions path:  PRED_CSV=/path/to/predictions.csv python downstream_eval.py
"""
import sys, os, csv, io, types, contextlib
from collections import OrderedDict

# Bypass WhisperX: stub the module so `import auto_rsr` doesn't require torch/whisperx/GPU.
# We never call auto_rsr.transcribe(); we only use its pure-Python scoring functions.
sys.modules.setdefault("whisperx", types.ModuleType("whisperx"))

import auto_rsr  # noqa: E402  (after the stub)

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_CSV = os.environ.get(
    "PRED_CSV",
    os.path.join(HERE, "..", "..", "RSR-leaderboard", "models", "qwen3-asr-1.7b", "v3", "predictions.csv"),
)
PERCENTILES = [5, 10, 15]
MODEL_LABEL = os.environ.get("MODEL_LABEL", "v3")


def load_subjects(path):
    """Group the per-utterance predictions into one record per subject (preserving file order)."""
    subjects = OrderedDict()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["speaker_id"]
            s = subjects.setdefault(sid, {
                "preds": [], "age": row["age"], "group": row["group"],
                "dld": row.get("dld", ""), "age_bin": row.get("age_bin", ""),
            })
            s["preds"].append(row["prediction"])
    return subjects


def score_subject(preds):
    """Concatenate the subject's 16 predicted utterances and run AutoRSR's align+score."""
    response = " ".join(p.strip() for p in preds if p and p.strip())
    with contextlib.redirect_stdout(io.StringIO()):  # silence auto_rsr's debug prints
        aligned = auto_rsr.align_transcription_to_ground_truth({"Transcription": response})
        scored = auto_rsr.generate_edit_sequences_and_score(aligned)
    return scored["Total Score"]


NAN, INF = float("nan"), float("inf")


def fmt_ratio(x, w=6):
    if x != x:   return f"{'nan':>{w}}"
    if x == INF: return f"{'inf':>{w}}"
    return f"{x:{w}.2f}"


def metrics(rows, pct):
    """The paper's diagnostic-screening metrics at one percentile cutoff.

    Se=TP/(TP+FN), Sp=TN/(TN+FP), PPV=TP/(TP+FP), NPV=TN/(TN+FN),
    PLR=Se/(1-Sp), NLR=(1-Se)/Sp, Youden J=Se+Sp-1  (Liu et al., Part-1 AutoRSR paper).
    """
    TP = FP = TN = FN = na = 0
    for r in rows:
        d = r[f"pct{pct}"]
        if d not in ("Pass", "Fail"):       # "N/A" or "" (age outside AutoRSR's 5-9 norms)
            na += 1
            continue
        pred_dld = (d == "Fail")            # Fail on the RSR => screen-positive for DLD
        if r["ref_dld"] and pred_dld:       TP += 1
        elif r["ref_dld"] and not pred_dld: FN += 1
        elif not r["ref_dld"] and pred_dld: FP += 1
        else:                               TN += 1
    se  = TP / (TP + FN) if (TP + FN) else NAN
    sp  = TN / (TN + FP) if (TN + FP) else NAN
    ppv = TP / (TP + FP) if (TP + FP) else NAN
    npv = TN / (TN + FN) if (TN + FN) else NAN
    ok = (se == se and sp == sp)
    plr = (INF if sp >= 1 else se / (1 - sp)) if ok else NAN
    nlr = (INF if sp <= 0 else (1 - se) / sp) if ok else NAN
    j = (se + sp - 1) if ok else NAN
    return dict(pct=pct, TP=TP, FP=FP, TN=TN, FN=FN, na=na, scored=TP + FP + TN + FN,
                se=se, sp=sp, ppv=ppv, npv=npv, plr=plr, nlr=nlr, j=j)


def main():
    subjects = load_subjects(PRED_CSV)
    rows = []
    for sid, s in subjects.items():
        total = score_subject(s["preds"])
        try:
            age_m = int(float(s["age"]))
        except (ValueError, TypeError):
            age_m = -1
        ref_dld = "below_86" in s["group"].lower()   # CELF_Below_86 => DLD; else typical
        rec = {"sid": sid, "age_m": age_m, "age_bin": s["age_bin"], "ref_dld": ref_dld,
               "n_sent": len(s["preds"]), "total_score": total}
        for p in PERCENTILES:
            rec[f"pct{p}"] = auto_rsr.evaluate_rsr_result(total, age_m, p)
        rows.append(rec)

    n_dld = sum(r["ref_dld"] for r in rows)
    print(f"model: {MODEL_LABEL}")
    print(f"predictions: {PRED_CSV}")
    print(f"subjects: {len(rows)}  |  reference: DLD={n_dld}  typical={len(rows) - n_dld}\n")

    print(f"Downstream screening ({MODEL_LABEL} ASR -> AutoRSR), vs cohort CELF labels")
    print("  metrics: Se=TP/(TP+FN)  Sp=TN/(TN+FP)  PPV=TP/(TP+FP)  NPV=TN/(TN+FN)  "
          "PLR=Se/(1-Sp)  NLR=(1-Se)/Sp  J=Se+Sp-1\n")
    print(f"  {'cutoff':>7} {'scored':>6} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'NA':>3} "
          f"{'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6} {'PLR':>6} {'NLR':>6} {'J':>6}")
    ms = [metrics(rows, p) for p in PERCENTILES]
    for m in ms:
        print(f"  {str(m['pct'])+'th':>7} {m['scored']:>6} {m['TP']:>3} {m['FP']:>3} {m['TN']:>3} "
              f"{m['FN']:>3} {m['na']:>3} {m['se']:>6.3f} {m['sp']:>6.3f} {m['ppv']:>6.3f} "
              f"{m['npv']:>6.3f} {fmt_ratio(m['plr'])} {fmt_ratio(m['nlr'])} {m['j']:>6.3f}")
    best = max(ms, key=lambda m: m['j'] if m['j'] == m['j'] else -9)
    print(f"\n  Best (max-J) operating point -> {best['pct']}th %ile: "
          f"Se={best['se']:.3f}  Sp={best['sp']:.3f}  PPV={best['ppv']:.3f}  "
          f"NPV={best['npv']:.3f}  PLR={fmt_ratio(best['plr']).strip()}  J={best['j']:.3f}")
    print("  (Part-1 paper, 130-child CELF-4 cohort — Whisper 5th: Se .768/Sp .784/J .552 ; "
          "Reverb-fusion 5th: Se .873/Sp .730/J .603)")

    # aggregate paper-metrics table (per cutoff) — for building the paper's Table
    magg = os.path.join(HERE, f"downstream_metrics_{MODEL_LABEL}.csv")
    cols = ["model", "pct", "scored", "TP", "FP", "TN", "FN", "na",
            "se", "sp", "ppv", "npv", "plr", "nlr", "j"]
    with open(magg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in ms:
            w.writerow({"model": MODEL_LABEL, **{k: m[k] for k in cols[1:]}})
    print(f"  wrote metrics table -> {magg}")

    # spot-check: lowest- and highest-scoring subjects
    by_score = sorted(rows, key=lambda r: r["total_score"])
    print("\n  spot-check (sid, age_m, ref, total/32, 5th-%ile decision):")
    for r in by_score[:3] + by_score[-3:]:
        print(f"    {r['sid']}  age={r['age_m']:>3}  ref={'DLD' if r['ref_dld'] else 'TD ':<3}  "
              f"total={r['total_score']:>2}/32  ->  {r['pct5']}")

    out = os.path.join(HERE, f"downstream_eval_{MODEL_LABEL}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote per-subject results -> {out}")


if __name__ == "__main__":
    main()
