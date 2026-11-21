"""Prepare a frozen BEIR test split. Uses the Python standard library."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

SCIFACT = "https://huggingface.co/datasets/BeIR/scifact/resolve/984eed826375f18d27936c4a32bf0f8491e3f414"
SCIFACT_QRELS = "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
NFCORPUS = "https://huggingface.co/datasets/BeIR/nfcorpus/resolve/e28763e68d85db0fa71652ba3c4afabf8c5b3bb7"
NFCORPUS_QRELS = "https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65"
SCIFACT_INPUTS = {
    "corpus.jsonl.gz": (SCIFACT, "2f4a2a264485e5b5b95ae9cd5534817447757423991a4767d6da811539c0853e"),
    "queries.jsonl.gz": (SCIFACT, "c29574b9cb8fdf043df8494f689de15f4604ac4b585eb34bda9b6a389aa6c4d3"),
    "test.tsv": (SCIFACT_QRELS, "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c"),
}
NFCORPUS_INPUTS = {
    "corpus.jsonl.gz": (NFCORPUS, "ae4d6ab2ab88d880e7e180776c956a09c19c43a703457742cd850d2422035802"),
    "queries.jsonl.gz": (NFCORPUS, "baba3504a2736b0e32265962f04acc4fecfaf9649e0ea08e49ff441246f75364"),
    "test.tsv": (NFCORPUS_QRELS, "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a"),
}


def prepare(directory, dataset="scifact"):
    directory.mkdir(parents=True, exist_ok=True)
    inputs = {"scifact": SCIFACT_INPUTS, "nfcorpus": NFCORPUS_INPUTS}[dataset]
    for name, (base, digest) in inputs.items():
        path = directory / name
        if path.exists():
            content = path.read_bytes()
        else:
            with urlopen(f"{base}/{name}", timeout=60) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Unexpected SHA-256 for {name}")
        if not path.exists():
            path.write_bytes(content)

    with (directory / "test.tsv").open(encoding="utf-8") as handle:
        judgments = list(csv.DictReader(handle, delimiter="\t"))
    test_ids = {row["query-id"] for row in judgments}
    for kind in ("corpus", "queries"):
        with gzip.open(directory / f"{kind}.jsonl.gz", "rt", encoding="utf-8") as rows, (
            directory / f"{kind}.tsv"
        ).open("w", encoding="utf-8") as output:
            for line in rows:
                row = json.loads(line)
                if kind == "queries" and row["_id"] not in test_ids:
                    continue
                # Ignore judgment-bearing query metadata. The retrievers only
                # receive the query text, or the document title and abstract.
                text = " ".join((row.get("title", "") + " " + row["text"]).split())
                print(row["_id"], text, sep="\t", file=output)
    with (directory / "qrels.trec").open("w", encoding="utf-8") as output:
        for row in judgments:
            print(row["query-id"], 0, row["corpus-id"], row["score"], file=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset", choices=("scifact", "nfcorpus"), default="scifact")
    args = parser.parse_args()
    prepare(args.directory, args.dataset)
