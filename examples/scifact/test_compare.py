"""Parity inputs must be valid even when the NIST executable is omitted."""
from contextlib import redirect_stdout
import csv
from io import StringIO
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

    def test_omitted_judged_query_fails_before_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "qrels.trec").write_text("q1 0 a 1\nq2 0 b 1\n")
            (path / "per-query.tsv").write_text(
                "qid\tmethod\tndcg10\trecall10\trecall100\n"
                + "".join(f"q1\t{method}\t1\t1\t1\n" for method in ("bm25", "trigram", "rrf"))
            )
            for method in ("bm25", "trigram", "rrf"):
                (path / f"{method}.run").write_text("q1 Q0 a 1 1 run\n")
            output = StringIO()
            with redirect_stdout(output), self.assertRaisesRegex(ValueError, "query IDs"):
                compare(path, path, None)
            self.assertEqual(output.getvalue(), "")
