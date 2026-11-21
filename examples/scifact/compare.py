"""Compare SciFact runs at the claim level and optionally check NIST parity."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean
import subprocess

METHODS = ("bm25", "trigram", "rrf")


def compare(data, runs, trec_eval):
    metrics = defaultdict(dict)
    with (runs / "per-query.tsv").open(encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            metrics[row["qid"]][row["method"]] = row
    gain = {q: float(v["rrf"]["ndcg10"]) - float(v["bm25"]["ndcg10"]) for q, v in metrics.items()}
    print("RRF versus BM25, nDCG@10:",
          sum(d > 0 for d in gain.values()), "wins,",
          sum(d == 0 for d in gain.values()), "ties,",
          sum(d < 0 for d in gain.values()), "losses")
    oracle = mean(max(float(v[m]["ndcg10"]) for m in ("bm25", "trigram")) for v in metrics.values())
    print(f"Choosing the better source with qrels per claim would give {oracle:.4f} nDCG@10.")

    relevant = defaultdict(set)
    for row in (data / "qrels.trec").read_text().splitlines():
        q, _, doc, grade = row.split()
        if int(grade) > 0:
            relevant[q].add(doc)
    ranks = {m: defaultdict(dict) for m in METHODS}
    for method in METHODS:
        for row in (runs / f"{method}.run").read_text().splitlines():
            q, _, doc, pos, _, _ = row.split()
            ranks[method][q][doc] = int(pos)

    missing = {q: docs - ranks["bm25"][q].keys() - ranks["trigram"][q].keys()
               for q, docs in relevant.items()}
    print(f"Candidate union missed {sum(map(len, missing.values()))} known relevant documents;",
          f"{sum(missing[q] == docs for q, docs in relevant.items())} claims had no known relevant candidate.")
    # Break tied extremes by numeric claim ID so the example can be replayed.
    for q in (min(gain, key=lambda q: (gain[q], int(q))),
              max(gain, key=lambda q: (gain[q], -int(q)))):
        for doc in sorted(relevant[q]):
            positions = [ranks[m][q].get(doc, "absent") for m in METHODS]
            print(f"claim {q}, abstract {doc}: BM25 {positions[0]}, trigram {positions[1]}, RRF {positions[2]}")

    if trec_eval:
        keys = {"ndcg_cut_10": "ndcg10", "recall_10": "recall10", "recall_100": "recall100"}
        for method in METHODS:
            output = subprocess.check_output(
                [trec_eval, "-q", "-c", "-m", "ndcg_cut.10", "-m", "recall.10,100",
                 data / "qrels.trec", runs / f"{method}.run"], text=True)
            seen = set()
            for line in output.splitlines():
                measure, q, expected = line.split()
                key = keys[measure]
                value = (mean(float(v[method][key]) for v in metrics.values())
                         if q == "all" else float(metrics[q][method][key]))
                if abs(value - float(expected)) > 0.0000500001:
                    raise ValueError(f"{method}, {q}, {measure}: rankops={value}, NIST={expected}")
                seen.add((q, measure))
            if len(seen) != (len(metrics) + 1) * len(keys):
                raise ValueError(f"Incomplete trec_eval output for {method}")
            print(f"{method}: {len(seen)} NIST values agree to four decimal places.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--trec-eval", help="path to the compiled NIST trec_eval")
    args = parser.parse_args()
    compare(args.data, args.runs, args.trec_eval)
