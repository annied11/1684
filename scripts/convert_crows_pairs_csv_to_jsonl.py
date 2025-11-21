import csv, json, sys, pathlib

inp = sys.argv[1]  # e.g., data/crows_pairs.csv
out = sys.argv[2]  # e.g., data/crows_pairs.jsonl

pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(inp, newline='', encoding='utf-8') as fin, open(out, 'w', encoding='utf-8') as fout:
    r = csv.DictReader(fin)
    for row in r:
        obj = {"sent_more": row["sent_more"], "sent_less": row["sent_less"]}
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"Wrote {out}")
