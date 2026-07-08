#!/usr/bin/env python3
"""
HUMAN-RSR UPPER BOUND (Task 3).

Apply AutoRSR's *own* pass/fail thresholds (auto_rsr.rsr_pass_fail) to the HUMAN-scored
RSR totals (SRTotalScore, 0-32) for the same 132 test speakers, using each child's age in
months. This is the screening ceiling the ASR pipelines are chasing: if a perfect ASR fed
AutoRSR's scorer, this is the best Youden J the AutoRSR norm thresholds themselves permit on
this cohort (limited by the human RSR test + the published age/percentile cutoffs).

Reference labels = the same CELF (group=CELF_Below_86) DLD/TD labels used in the ASR eval.
Input: human_scores_join.csv (produced via openpyxl; ID_number = int(RSR_XXXX)).
Run in csr4rsr-dev so auto_rsr imports.
"""
import sys, os, types, csv
sys.modules.setdefault("whisperx", types.ModuleType("whisperx"))
import auto_rsr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JOIN = os.path.join(HERE, "human_scores_join.csv")
PERCENTILES = [5, 10, 15]


def metrics(rows, key):
    TP = FP = TN = FN = na = 0
    for r in rows:
        d = r[key]
        if d not in ("Pass", "Fail"):
            na += 1
            continue
        pred_dld = (d == "Fail")
        if r["ref_dld"] and pred_dld:       TP += 1
        elif r["ref_dld"] and not pred_dld: FN += 1
        elif not r["ref_dld"] and pred_dld: FP += 1
        else:                               TN += 1
    se = TP / (TP + FN) if (TP + FN) else float("nan")
    sp = TN / (TN + FP) if (TN + FP) else float("nan")
    j = (se + sp - 1) if (TP + FN and TN + FP) else float("nan")
    return dict(TP=TP, FP=FP, TN=TN, FN=FN, na=na, scored=TP + FP + TN + FN, se=se, sp=sp, j=j)


def main():
    rows = []
    with open(JOIN, newline="") as f:
        for r in csv.DictReader(f):
            age_m = int(float(r["age_m"]))
            total = float(r["SRTotalScore"])
            ref_dld = r["ref_dld"].strip().lower() in ("true", "1")
            rec = {"sid": r["speaker_id"], "age_m": age_m, "ref_dld": ref_dld,
                   "human_total": total}
            for p in PERCENTILES:
                rec[f"pct{p}"] = auto_rsr.rsr_pass_fail(total, age_m, p)
            rows.append(rec)

    n_dld = sum(r["ref_dld"] for r in rows)
    print(f"HUMAN-RSR ceiling  (subjects={len(rows)}, DLD={n_dld}, TD={len(rows)-n_dld})")
    print(f"  {'cutoff':>7} {'scored':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'N/A':>4} "
          f"{'Sens':>7} {'Spec':>7} {'YoudenJ':>8}")
    summ = []
    for p in PERCENTILES:
        m = metrics(rows, f"pct{p}")
        print(f"  {str(p)+'th':>7} {m['scored']:>6} {m['TP']:>4} {m['FP']:>4} {m['TN']:>4} "
              f"{m['FN']:>4} {m['na']:>4} {m['se']:>7.3f} {m['sp']:>7.3f} {m['j']:>8.3f}")
        summ.append({"scoring": "human", "pct": p, **{k: m[k] for k in
                     ("scored", "TP", "FP", "TN", "FN", "na", "se", "sp", "j")}})

    import statistics as st
    dld = [r["human_total"] for r in rows if r["ref_dld"]]
    td = [r["human_total"] for r in rows if not r["ref_dld"]]
    print(f"\n  human total/32  DLD mean={st.mean(dld):.2f}  TD mean={st.mean(td):.2f}  "
          f"gap={st.mean(td)-st.mean(dld):.2f}")

    out = os.path.join(HERE, "human_ceiling_persubject.csv")
    import csv as _csv
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(HERE, "human_ceiling_summary.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    print(f"\nwrote {out} and human_ceiling_summary.csv")


if __name__ == "__main__":
    main()
