# Which SciFact claims does fusion help?

A system returning ten abstracts and a system handing a hundred candidates to a reranker face different decisions. This example ranks real scientific abstracts three ways, saves the full runs, and follows two claims where fusion changes which evidence the reader sees.

## Run

From the repository root, with Python 3.10+ and Rust:

```sh
python3 examples/scifact/prepare.py target/scifact-data
cargo run --release --no-default-features --features composed-example --example scifact -- target/scifact-data target/scifact-runs
python3 examples/scifact/compare.py target/scifact-data target/scifact-runs
```

Preparation downloads three checksum-verified files from frozen revisions of the BEIR SciFact corpus and judgments on Hugging Face. It works offline when those files are already present. The 300 queries are the published BEIR test split. The retrievers see the claim, title and abstract. Query metadata and labels stay out of indexing and scoring. The judgments are used to select test query IDs and evaluate runs. [BEIR corpus](https://huggingface.co/datasets/BeIR/scifact/tree/984eed826375f18d27936c4a32bf0f8491e3f414) · [Test judgments](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029) · [Original SciFact task](https://aclanthology.org/2020.emnlp-main.609/)

**What is ranked.** Lexir BM25 uses lowercase alphanumeric tokens, no stemming, k1=1.2 and b=0.75. The other source uses the same 256-bin hashed character trigrams as the [HNSW example](../composed_retrieval.rs), with exact cosine over all 5,183 abstracts. Exhaustive search isolates the representation from ANN recall and graph construction. It requires no model. Each retriever supplies up to 100 candidates. Equal-weight RRF uses k=60 and emits its first 100. Equal scores break by ascending numeric document ID. The TREC files encode each resulting order as strictly decreasing integer scores.

## Results

Observed with rankops 0.5.1 and lexir 0.5.2. All 300 queries and 339 judged relevant query–abstract pairs enter the evaluation.

| Run | nDCG@10 | Recall@10 | Judged@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.6634 | 0.7876 | 0.0870 | 0.8826 |
| Character trigrams | 0.3397 | 0.4308 | 0.0477 | 0.6351 |
| RRF | 0.4896 | 0.6586 | 0.0717 | 0.8930 |

For a reader looking at ten results, BM25 is the better baseline on this split. Fusion improved nDCG@10 for 31 claims, tied 153 and lost on 116. For a reranker with room for 100 abstracts, fusion raises recall by 0.0104. Across the **union** of both inputs, 30 judged relevant pairs were absent and 25 claims had no known relevant candidate. Ranking those same candidates differently cannot recover those documents.

| Test claim | Judged abstract | BM25 rank | Trigram rank | Fused rank |
| --- | ---: | ---: | ---: | ---: |
| 51, on ALDH1 expression and breast cancer outcomes | 45638119 | 1 | outside 100 | 35 |
| 1140, on α-tocopheryl acetate and prostate cancer | 12009265 | 9 | 3 | 1 |

These are the largest loss and gain by per-claim nDCG@10, breaking ties by claim ID. They illustrate the mechanism behind RRF: an item supported by both lists can overtake a top item supported by only one. The returned abstracts may support **or refute** a claim. The evidence-retrieval score says nothing about the truth of either claim.

`per-query.tsv` records both cutoff choices and judged coverage for every method and claim. A selector with access to the test labels could pick the better source per claim and reach 0.6896 nDCG@10. That is the ceiling for choosing one of these two source lists on this split. A query router would need disjoint tuning data and held-out evaluation. The trigram encoder is a surface-form baseline, so the results do not establish how RRF behaves with learned scientific embeddings.

### Independent metric check

```sh
git clone --branch v9.0.8 --depth 1 https://github.com/usnistgov/trec_eval.git target/trec_eval
make -C target/trec_eval
python3 examples/scifact/compare.py target/scifact-data target/scifact-runs --trec-eval target/trec_eval/trec_eval
```

The check compares Rankops with NIST on nDCG@10, Recall@10 and Recall@100 for each query and the collection mean. All 2,709 values matched the four decimal places printed by trec_eval. Judged@10 comes from Rankops. This split contains only positive judgments; an unjudged abstract has unknown relevance.

### What to try next

- **Judgment coverage.** SciFact-Open pools results from four systems and annotates evidence over 500K abstracts. A reader deciding whether a genuinely new retriever helps needs judgments on its novel results. Sample the disagreements before treating low Judged@10 as low precision. [SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/)
- **Question length.** The BRIGHT reproducibility study found that query-side BM25 weighting helps on long, reasoning-heavy questions and that the gain transfers unevenly. Measure a long-query collection separately before generalizing from these short claims. [BRIGHT baselines](https://arxiv.org/abs/2509.02558)
- **What evidence is sufficient.** FreshStack breaks community answers into multiple information needs and checks which passages support each. A system can retrieve a relevant document while missing part of an answer. Use nugget coverage when the downstream task needs several pieces of evidence. [FreshStack](https://fresh-stack.github.io/)
