#!/usr/bin/env python3
"""
Build an eval-ready predictions CSV for a BASELINE ASR, apples-to-apples with our v3.

Reuses the v3 leaderboard CSV for the cohort + metadata (speaker_id, age, group, dld) and swaps in the
baseline model's `prediction`, matched per utterance by (speaker_id, turn_index). The v3 CSV is
turn-ordered per speaker, so a row's turn = its 1-based position within its speaker group; the JSONL
carries turn_index directly. (We match on turn, not reference text, because the JSONL `reference`
keeps raw CHAT markup -- um@x, &-um, (.), <...>[/] -- that the CSV `clean_transcription` strips.)
Result: exact same 132 subjects / 2,065 utterances / metadata as v3 -- only the ASR transcript differs.

Usage:  python build_baseline_csv.py <baseline_predictions.jsonl> <out.csv>
"""
import sys, csv, json, os

V3_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                      "RSR-leaderboard", "models", "qwen3-asr-1.7b", "v3", "predictions.csv")


def main():
    jsonl_path, out_path = sys.argv[1], sys.argv[2]

    pred = {}  # (source_id, turn_index) -> prediction
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sr = d.get("source_row", {})
            sid = sr.get("source_id") or d.get("source_id")
            turn = sr.get("turn_index")
            if sid is not None and turn is not None:
                pred[(sid, int(turn))] = d.get("prediction", "")

    cols = ["speaker_id", "age", "age_bin", "dld", "group", "prediction", "clean_transcription"]
    posn, matched, missing = {}, 0, 0
    with open(V3_CSV, newline="") as fin, open(out_path, "w", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=cols)
        w.writeheader()
        for row in r:
            sid = row["speaker_id"]
            posn[sid] = posn.get(sid, 0) + 1
            turn = posn[sid]                      # 1-based turn within the speaker (CSV is turn-ordered)
            if (sid, turn) in pred:
                matched += 1
                p = pred[(sid, turn)]
            else:
                missing += 1
                p = ""                            # true miss -> contributes errors (fair)
            w.writerow({
                "speaker_id": sid, "age": row["age"], "age_bin": row["age_bin"],
                "dld": row["dld"], "group": row["group"], "prediction": p,
                "clean_transcription": row["clean_transcription"],
            })
    print(f"{out_path}: matched={matched} missing={missing}")


if __name__ == "__main__":
    main()
