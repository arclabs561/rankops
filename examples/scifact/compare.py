"""Compare BEIR runs at the query level and optionally check NIST parity."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from random import Random
from statistics import mean
import subprocess

METHODS = ("bm25", "trigram", "rrf")


def compare(data, runs, trec_eval):
    metrics = defaultdict(dict)
    with (runs / "per-query.tsv").open(encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            for key in ("ndcg10", "recall10", "recall100"):
                value = float(row[key])
                if not 0 <= value <= 1:
                    raise ValueError(f"{row['method']}, {row['qid']}, {key}: invalid score {value}")
            method = row["method"]
            scores = metrics[row["qid"]]
            if method in scores:
                raise ValueError(f"duplicate metric row: {row['qid']}, {method}")
            scores[method] = row

    relevant = defaultdict(set)
    for row in (data / "qrels.trec").read_text().splitlines():
        q, _, doc, grade = row.split()
        docs = relevant[q]
        if int(grade) > 0:
            docs.add(doc)
    if metrics.keys() != relevant.keys() or not metrics:
        raise ValueError("per-query metric query IDs must match the nonempty qrels query set")
    for qid, scores in metrics.items():
        if scores.keys() != set(METHODS):
            raise ValueError(f"{qid}: expected exactly one metric row per method")

    gain = {q: float(v["rrf"]["ndcg10"]) - float(v["bm25"]["ndcg10"]) for q, v in metrics.items()}
    for measure in ("ndcg10", "recall100"):
        diffs = [float(v["rrf"][measure]) - float(v["bm25"][measure]) for v in metrics.values()]
        rng = Random(561)
        n = len(diffs)
        # Resample paired queries. A query's two rankings always travel together.
        means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(10_000))
        print(f"RRF minus BM25, {measure}: {mean(diffs):+.4f}, "
              f"95% paired bootstrap [{means[249]:+.4f}, {means[9749]:+.4f}];",
              sum(d > 0 for d in diffs), "wins,", sum(d == 0 for d in diffs), "ties,",
              sum(d < 0 for d in diffs), "losses")
    oracle = mean(max(float(v[m]["ndcg10"]) for m in ("bm25", "trigram")) for v in metrics.values())
    print(f"Choosing the better source with qrels per query would give {oracle:.4f} nDCG@10.")

    ranks = {m: defaultdict(dict) for m in METHODS}
    for method in METHODS:
        for row in (runs / f"{method}.run").read_text().splitlines():
            q, _, doc, pos, _, _ = row.split()
            ranks[method][q][doc] = int(pos)

    missing = {q: docs - ranks["bm25"][q].keys() - ranks["trigram"][q].keys()
               for q, docs in relevant.items()}
    print(f"Candidate union missed {sum(map(len, missing.values()))} known relevant query–document pairs;",
          f"{sum(bool(docs) and missing[q] == docs for q, docs in relevant.items())} queries had no known relevant candidate.")
    # Break tied extremes by the lowest native query ID in these collections.
    for q in (min(gain, key=lambda q: (gain[q], len(q), q)),
              min(gain, key=lambda q: (-gain[q], len(q), q))):
        for doc in sorted(relevant[q]):
            positions = [ranks[m][q].get(doc, "absent") for m in METHODS]
            print(f"query {q}, document {doc}: BM25 {positions[0]}, trigram {positions[1]}, RRF {positions[2]}")

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
                if not abs(value - float(expected)) <= 0.0000500001:
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
