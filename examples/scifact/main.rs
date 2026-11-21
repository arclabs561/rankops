//! Score three first-stage runs on BEIR SciFact. Input preparation, dataset
//! revisions and the independent evaluator command are in README.md.
#[path = "../support/trigrams.rs"]
mod trigrams;

use lexir::bm25::{Bm25Params, InvertedIndex};
use rankops::trec::{evaluate_detailed, parse_qrels, parse_run};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use trigrams::{embed, tokens};

const DEPTH: usize = 100;
const METHODS: [&str; 3] = ["bm25", "trigram", "rrf"];

fn texts(path: &Path) -> Result<Vec<(u32, String)>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let (id, text) = line.split_once('\t').ok_or("expected id and text")?;
        rows.push((id.parse()?, text.to_owned()));
    }
    Ok(rows)
}

fn unit_trigrams(text: &str) -> Vec<f32> {
    let mut v = embed(text);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn cosine_run(vectors: &[(u32, Vec<f32>)], query: &str) -> Vec<(u32, f32)> {
    let q = unit_trigrams(query);
    let mut ranked: Vec<(u32, f32)> = vectors
        .iter()
        .map(|(id, v)| (*id, q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>()))
        .filter(|&(_, score)| score > 0.0)
        .collect();
    rank(&mut ranked);
    ranked.truncate(DEPTH);
    ranked
}

// Lexir already uses document ID ascending for ties. Apply the same tie
// convention to exact cosine and RRF before cutting a run to depth.
fn rank(ranked: &mut [(u32, f32)]) {
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [data, out] = args.as_slice() else {
        return Err("usage: scifact PREPARED_DATA OUTPUT_DIRECTORY".into());
    };
    let (data, out) = (PathBuf::from(data), PathBuf::from(out));
    fs::create_dir_all(&out)?;
    let mut index = InvertedIndex::new();
    let mut vectors = Vec::new();
    for (id, text) in texts(&data.join("corpus.tsv"))? {
        index.add_document(id, &tokens(&text));
        vectors.push((id, unit_trigrams(&text)));
    }
    let qrels = parse_qrels(File::open(data.join("qrels.trec"))?)?;
    let queries = texts(&data.join("queries.tsv"))?;
    let mut files: Vec<_> = METHODS
        .iter()
        .map(|m| File::create(out.join(format!("{m}.run"))).map(BufWriter::new))
        .collect::<Result<_, _>>()?;
    for (qid, query) in &queries {
        let bm25 = index.retrieve(&tokens(query), DEPTH, Bm25Params::default())?;
        let trigram = cosine_run(&vectors, query);
        let mut fused = rankops::rrf(&bm25, &trigram);
        rank(&mut fused);
        for ((method, file), hits) in METHODS
            .iter()
            .zip(&mut files)
            .zip([&bm25, &trigram, &fused])
        {
            for (i, (doc, _)) in hits.iter().take(DEPTH).enumerate() {
                // TREC readers order by score. Strict monotonic scores preserve
                // the retriever's ranking even when its native scores tie.
                writeln!(file, "{qid} Q0 {doc} {} {} {method}", i + 1, DEPTH - i)?;
            }
        }
    }
    for mut file in files {
        file.flush()?;
    }
    let mut per_query = BufWriter::new(File::create(out.join("per-query.tsv"))?);
    writeln!(
        per_query,
        "method\tqid\tndcg10\trecall10\tjudged10\trecall100"
    )?;
    println!(
        "{} documents, {} test queries",
        vectors.len(),
        queries.len()
    );
    println!("method\tnDCG@10\tRecall@10\tJudged@10\tRecall@100");
    for method in METHODS {
        let run = parse_run(File::open(out.join(format!("{method}.run")))?)?;
        let at10 = evaluate_detailed(&run, &qrels, 10);
        let at100 = evaluate_detailed(&run, &qrels, DEPTH);
        println!(
            "{method}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            at10.summary.ndcg_at_k,
            at10.summary.recall_at_k,
            at10.judged_at_k(),
            at100.summary.recall_at_k
        );
        for (a, b) in at10.queries.iter().zip(&at100.queries) {
            writeln!(
                per_query,
                "{method}\t{}\t{}\t{}\t{}\t{}",
                a.query_id, a.ndcg_at_k, a.recall_at_k, a.judged_at_k, b.recall_at_k
            )?;
        }
    }
    per_query.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_keeps_the_lowest_document_ids_when_the_cutoff_ties() {
        let v = unit_trigrams("milk");
        let vectors: Vec<_> = (1..=105).rev().map(|id| (id, v.clone())).collect();
        let ids: Vec<_> = cosine_run(&vectors, "MILK")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, (1..=100).collect::<Vec<_>>());
    }

    #[test]
    fn cosine_uses_normalized_counts_and_skips_empty_vectors() {
        let vectors = [
            (1, unit_trigrams("milk")),
            (2, unit_trigrams("milk milk")),
            (3, unit_trigrams("")),
        ];
        let ranked = cosine_run(&vectors, "milk");
        assert_eq!(ranked.len(), 2);
        for (_, score) in ranked {
            assert!((score - 1.0).abs() < 1e-6);
        }
        assert!(cosine_run(&vectors, "...").is_empty());
    }
}
