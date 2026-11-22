"""Parity inputs must be valid even when the NIST executable is omitted."""
import csv
from pathlib import Path
import tempfile
import unittest

from compare import compare


class CompareTest(unittest.TestCase):
    def test_invalid_per_query_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for bad in ("nan", "inf", "-inf", "-0.01", "1.01"):
                with self.subTest(bad=bad):
                    with (path / "per-query.tsv").open("w", newline="") as f:
                        writer = csv.writer(f, delimiter="\t")
                        writer.writerow(("qid", "method", "ndcg10", "recall10", "recall100"))
                        writer.writerow(("q", "bm25", 1, bad, 1))
                    with self.assertRaisesRegex(ValueError, "bm25, q, recall10"):
                        compare(path, path, None)
