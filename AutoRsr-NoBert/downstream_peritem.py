#!/usr/bin/env python3
"""
PER-ITEM downstream DLD-screening eval (Tasks 1 & 2).

AutoRSR's stock downstream pipeline (downstream_eval.py) concatenates a subject's ~16
utterances into one blob and lets difflib re-pick which of the 16 hardcoded RSR targets
each chunk best matches. That can cross-assign chunks. Here we instead use the KNOWN
per-utterance item index (`sentence_id` from the HF Redmond-Sentence-Recall test parquet,
== the `[+ NN]` CHAT marker in manifest_test.jsonl) to score EACH utterance against its
CORRECT target item, then sum per subject.

We reuse AutoRSR's own scoring machinery unchanged:
    fuzzy_align_response_to_gt(GT[k], prediction)  -> trim prediction to its target k
    standarize(...) ; score_rsr_errors(...) ; score_rsr([err])   (2 / 1 / 0 pts per item)
The ONLY change vs stock is *which* target each utterance is aligned to (known, not guessed).

Per subject: for each item k in 1..16 take the best (max-points) attempt among the
utterances tagged with that sentence_id (a few children repeat an item); items the child
never produced score 0. Sum -> total/32 -> AutoRSR age/percentile pass/fail.

Run in conda env csr4rsr-dev (ftfy + transformers + pandas + pyarrow). From this directory.
"""
import sys, os, io, types, contextlib, glob
from collections import OrderedDict, defaultdict
import pandas as pd

sys.modules.setdefault("whisperx", types.ModuleType("whisperx"))
import auto_rsr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PERCENTILES = [5, 10, 15]

# The 16 hardcoded RSR targets, in the same order AutoRSR uses; sentence_id k -> GT[k-1].
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

PARQUET_GLOB = ("/ws/ifp-54_1/hasegawa/haolong2/cache/huggingface/hub/"
                "datasets--MagicLuke--Redmond-Sentence-Recall/snapshots/*/sentence/test-*.parquet")

MODELS = {
    "v3":               "/ws/ifp-54_1/hasegawa/haolong2/AI4EE/RSR-leaderboard/models/qwen3-asr-1.7b/v3/predictions.csv",
    "base-qwen":        os.path.join(HERE, "base_qwen_preds.csv"),
    "whisper-large-v3": os.path.join(HERE, "whisper_largev3_preds.csv"),
}


def load_sentence_ids():
    """Per-row sentence_id (item index) from the parquet, in dataset order."""
    files = sorted(glob.glob(PARQUET_GLOB))
    pq = pd.concat([pd.read_parquet(f, columns=["speaker_id", "text", "sentence_id"])
                    for f in files], ignore_index=True)
    return pq


def score_item(pred_text, gt_text):
    """AutoRSR per-item points for ONE utterance vs its known target (2 / 1 / 0)."""
    with contextlib.redirect_stdout(io.StringIO()):
        aligned = auto_rsr.fuzzy_align_response_to_gt(gt_text, pred_text or "")
        gt_w = auto_rsr.standarize(gt_text).split()
        al_w = auto_rsr.standarize(aligned).split()
        edits = auto_rsr.score_rsr_errors(al_w, gt_w)
        err = sum(len(v) for v in edits.values())
        return auto_rsr.score_rsr([err])


def score_subjects_peritem(pred_csv, pq):
    df = pd.read_csv(pred_csv)
    assert len(df) == len(pq), f"row count mismatch {len(df)} vs {len(pq)}"
    assert (df["speaker_id"].values == pq["speaker_id"].values).all(), "speaker order mismatch"
    df = df.reset_index(drop=True)
    df["sentence_id"] = pq["sentence_id"].values

    # per-subject: best points per item, summed over 16
    subj = OrderedDict()
    for _, row in df.iterrows():
        sid = row["speaker_id"]
        s = subj.setdefault(sid, {"age": row["age"], "group": row["group"],
                                  "items": defaultdict(int), "n_utt": 0})
        k = int(row["sentence_id"])
        pts = score_item(str(row["prediction"]) if pd.notna(row["prediction"]) else "",
                         GT_LINES[k - 1])
        if pts > s["items"][k]:          # keep best attempt for that item
            s["items"][k] = pts
        s["n_utt"] += 1

    rows = []
    for sid, s in subj.items():
        total = sum(s["items"].get(k, 0) for k in range(1, 17))
        try:
            age_m = int(float(s["age"]))
        except (ValueError, TypeError):
            age_m = -1
        ref_dld = "below_86" in str(s["group"]).lower()
        rec = {"sid": sid, "age_m": age_m, "ref_dld": ref_dld,
               "n_items_attempted": sum(1 for k in range(1, 17) if k in s["items"]),
               "n_utt": s["n_utt"], "total_score": total}
        for p in PERCENTILES:
            rec[f"pct{p}"] = auto_rsr.evaluate_rsr_result(total, age_m, p)
        rows.append(rec)
    return rows


def metrics(rows, pct):
    TP = FP = TN = FN = na = 0
    for r in rows:
        d = r[f"pct{pct}"]
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
    pq = load_sentence_ids()
    summary = []
    per_subject_all = {}
    for label, path in MODELS.items():
        rows = score_subjects_peritem(path, pq)
        per_subject_all[label] = {r["sid"]: r for r in rows}
        n_dld = sum(r["ref_dld"] for r in rows)
        print(f"\n==== PER-ITEM  {label}  (subjects={len(rows)}, DLD={n_dld}, TD={len(rows)-n_dld}) ====")
        print(f"  {'cutoff':>7} {'scored':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'N/A':>4} "
              f"{'Sens':>7} {'Spec':>7} {'YoudenJ':>8}")
        for p in PERCENTILES:
            m = metrics(rows, p)
            print(f"  {str(p)+'th':>7} {m['scored']:>6} {m['TP']:>4} {m['FP']:>4} {m['TN']:>4} "
                  f"{m['FN']:>4} {m['na']:>4} {m['se']:>7.3f} {m['sp']:>7.3f} {m['j']:>8.3f}")
            summary.append({"model": label, "scoring": "per_item", "pct": p, **{k: m[k] for k in
                            ("scored", "TP", "FP", "TN", "FN", "na", "se", "sp", "j")}})
        # DLD vs TD total distributions
        dld = [r["total_score"] for r in rows if r["ref_dld"]]
        td = [r["total_score"] for r in rows if not r["ref_dld"]]
        import statistics as st
        print(f"  total/32  DLD mean={st.mean(dld):.2f} median={st.median(dld):.1f}  |  "
              f"TD mean={st.mean(td):.2f} median={st.median(td):.1f}  gap={st.mean(td)-st.mean(dld):.2f}")
        out = os.path.join(HERE, f"downstream_peritem_{label}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  wrote {out}")

    pd.DataFrame(summary).to_csv(os.path.join(HERE, "downstream_peritem_summary.csv"), index=False)
    print("\nwrote downstream_peritem_summary.csv")
    # stash combined per-subject totals for the flip analysis
    sids = list(per_subject_all["v3"].keys())
    comb = []
    for sid in sids:
        rec = {"sid": sid, "ref_dld": per_subject_all["v3"][sid]["ref_dld"],
               "age_m": per_subject_all["v3"][sid]["age_m"]}
        for label in MODELS:
            rec[f"total_{label}"] = per_subject_all[label][sid]["total_score"]
            rec[f"pct5_{label}"] = per_subject_all[label][sid]["pct5"]
        comb.append(rec)
    pd.DataFrame(comb).to_csv(os.path.join(HERE, "downstream_peritem_combined.csv"), index=False)
    print("wrote downstream_peritem_combined.csv")


if __name__ == "__main__":
    main()
