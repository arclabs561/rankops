# Does the SciFact result transfer to NFCorpus?

On SciFact, BM25 found better evidence for a reader inspecting ten results. Equal-weight reciprocal-rank fusion (RRF) found a little more evidence at depth 100. NFCorpus gives the same retrievers a different task. Readers ask about familiar topics such as “shelf life” and the documents use the language of research abstracts. The original dataset derives graded relevance from citations, links and topic tags. [SciFact comparison](README.md) · [NFCorpus paper](https://www.cl.uni-heidelberg.de/~sokolov/pubs/boteva16full.pdf)

## Run

From the repository root, run the same scorer in a separate data directory. The preparation script checks the corpus, query and judgment files against frozen revisions. The runs retain external IDs such as `PLAIN-2` and `MED-10`. Equal native scores break by ascending numeric document suffix, including ties at the 100-document cutoff. Retrieval uses the same tokenization, BM25 parameters, exact 256-bin character-trigram cosine and RRF constant as the SciFact example. [Frozen corpus and queries](https://huggingface.co/datasets/BeIR/nfcorpus/tree/e28763e68d85db0fa71652ba3c4afabf8c5b3bb7) · [Frozen test judgments](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/tree/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65)

```sh
python3 examples/scifact/prepare.py target/nf-data --dataset nfcorpus
cargo run --release --no-default-features --features composed-example --example scifact -- target/nf-data target/nf-runs
python3 examples/scifact/compare.py target/nf-data target/nf-runs
```

## Results

The BEIR split has 3,633 abstracts, 323 test queries and 12,334 distinct judged query–document pairs. The grades are 1 for 11,758 pairs and 2 for 576 pairs. nDCG uses linear gains. Recall counts both grades as relevant and averages across queries. Unjudged results have unknown relevance.

| Run | nDCG@10 | Recall@10 | Judged@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.3062 | 0.1486 | 0.2158 | 0.2360 |
| Character trigrams | 0.1503 | 0.0679 | 0.1084 | 0.1593 |
| RRF | 0.2574 | 0.1257 | 0.1901 | 0.2464 |

For nDCG@10, fusion improved 54 queries, tied 125 and lost 144. Its mean change against BM25 was −0.0487, with a 95% paired bootstrap interval of [−0.0640, −0.0339]. At depth 100, recall improved on 81, tied on 180 and fell on 62. The mean recall change was +0.0105 [−0.0014, +0.0227]. The comparison script resamples whole queries 10,000 times with seed 561. Deltas use full precision before rounding.

| Query | Judged abstract | Grade | BM25 rank | Trigram rank | RRF rank |
| --- | --- | ---: | ---: | ---: | ---: |
| `PLAIN-2081`, “shelf life” | `MED-4303`, shelf-life evaluation of culinary oils | 1 | 1 | outside 100 | 15 |
| `PLAIN-3026`, “Vitamin C-Enriched Bacon” | `MED-4470`, vitamins C and E and N-nitroso compound formation | 2 | 11 | 4 | 2 |

These are the largest per-query nDCG loss and gain. For “shelf life,” fusion pushed the sole judged abstract off the first page. For the bacon query, both retrievers retrieved the evidence and fusion promoted it to the first page. Across the whole split, the union of the two candidate lists still missed 10,085 judged pairs. For 51 queries, neither source supplied any judged relevant abstract.

**Use BM25 for the first ten results from these two retrievers on both collections.** Both collections showed a small positive recall change at 100, with uncertainty on each. A reranker would need to demonstrate that the additional candidates improve its final ranking. The experiment uses a surface-form trigram representation. Learned scientific embeddings require a separate comparison.

With NIST trec_eval v10.0 built as shown in the [SciFact instructions](README.md#independent-metric-check), run:

```sh
python3 examples/scifact/compare.py target/nf-data target/nf-runs --trec-eval target/trec_eval/trec_eval
```

Rankops and NIST agreed on all 2,916 values to four decimal places across nDCG@10, Recall@10 and Recall@100. Two full replays produced byte-identical TREC runs and per-query tables.

### What if only the highest grade counts?

Only 119 of the 323 queries have a grade-2 judgment. On those queries, BM25 and RRF had nearly identical grade-2 recall at 100 (0.5627 and 0.5629). At ten, BM25 retrieved more of that evidence (0.3322 versus 0.2642). This changes both the relevance threshold and the population being averaged. It provides a second check on the apparent recall gain.

```sh
awk '$4 == 2' target/nf-data/qrels.trec > target/nf-data/grade2.trec
for run in bm25 rrf; do
  echo "$run"
  target/trec_eval/trec_eval -c -m recall.10,100 target/nf-data/grade2.trec "target/nf-runs/$run.run"
done
```
