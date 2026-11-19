//! Show the judged document moves behind one query's change in nDCG@k.

use rankops::trec::{explain_ndcg_change, parse_qrels, parse_run};
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [qrels_path, before_path, after_path, query_id, k] = args.as_slice() else {
        return Err("usage: trec_ndcg_change QRELS BEFORE.run AFTER.run QUERY K".into());
    };
    let k: usize = k
        .to_str()
        .ok_or("K must be a nonnegative integer")?
        .parse()?;
    let id = query_id.to_str().ok_or("QUERY must be valid UTF-8")?;
    let qrels = parse_qrels(File::open(qrels_path)?)?;
    let before = parse_run(File::open(before_path)?)?;
    let after = parse_run(File::open(after_path)?)?;
    let rows = explain_ndcg_change(&before, &after, &qrels, id, k)
        .ok_or_else(|| format!("no qrels for query {id}"))?;
    let delta: f64 = rows.iter().map(|c| c.ndcg_delta).sum();
    println!("query {id}   nDCG@{k} change {delta:+.7}");
    println!("document\trelevance\tbefore\tafter\tndcg_delta");
    for row in rows {
        let rank = |r: Option<usize>| r.map_or_else(|| "-".to_owned(), |v| v.to_string());
        println!(
            "{}\t{}\t{}\t{}\t{:+.7}",
            row.document_id,
            row.relevance,
            rank(row.before_rank),
            rank(row.after_rank),
            row.ndcg_delta
        );
    }
    Ok(())
}
