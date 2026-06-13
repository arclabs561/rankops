//! TREC-format qrels/run parsing and collection-level evaluation.
//!
//! The crate-root metrics ([`crate::ndcg_at_k`], [`crate::map`], ...) score one
//! query at a time against a [`Qrels`] (a `HashMap<DocId, relevance>`). This
//! module adds the collection layer the standard IR/BEIR workflow needs: parse
//! a TREC qrels file and a TREC run file (both keyed by query id), then average
//! a metric over all judged queries -- the number you report when benchmarking
//! against `trec_eval` / BEIR / ranx.
//!
//! Formats (whitespace-separated, the `trec_eval` convention):
//! - qrels: `query_id  iteration  doc_id  relevance` (iteration ignored;
//!   negative relevance is clamped to 0, i.e. non-relevant).
//! - run:   `query_id  Q0  doc_id  rank  score  run_tag` (results are sorted by
//!   score descending, which is the order the metrics consume).

use crate::{map, mrr, ndcg_at_k, precision_at_k, recall_at_k, Qrels};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

/// Query identifier (TREC `query_id` column).
pub type QueryId = String;
/// Document identifier (TREC `doc_id` column).
pub type DocId = String;
/// Per-query relevance judgments: query id -> (doc id -> relevance).
pub type TrecQrels = HashMap<QueryId, Qrels<DocId>>;
/// Per-query ranked results: query id -> results sorted by score descending.
pub type TrecRun = HashMap<QueryId, Vec<(DocId, f32)>>;

/// Parse a TREC qrels file: `query_id  iter  doc_id  relevance`.
///
/// Lines with fewer than 4 whitespace fields or an unparseable relevance are
/// skipped (so comment/blank lines are tolerated). Negative relevance is
/// clamped to 0, matching `trec_eval`'s non-relevant convention.
pub fn parse_qrels<R: Read>(reader: R) -> std::io::Result<TrecQrels> {
    let mut out: TrecQrels = HashMap::new();
    for line in BufReader::new(reader).lines() {
        let line = line?;
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            continue;
        }
        let Ok(rel) = f[3].parse::<i64>() else {
            continue;
        };
        out.entry(f[0].to_string())
            .or_default()
            .insert(f[2].to_string(), rel.max(0) as u32);
    }
    Ok(out)
}

/// Parse a TREC run file: `query_id  Q0  doc_id  rank  score  run_tag`.
///
/// Each query's results are sorted by score descending (NaN scores sort last),
/// which is the order [`crate::ndcg_at_k`] and the other metrics consume. Lines
/// with fewer than 6 fields or an unparseable score are skipped.
pub fn parse_run<R: Read>(reader: R) -> std::io::Result<TrecRun> {
    let mut out: TrecRun = HashMap::new();
    for line in BufReader::new(reader).lines() {
        let line = line?;
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 6 {
            continue;
        }
        let Ok(score) = f[4].parse::<f32>() else {
            continue;
        };
        out.entry(f[0].to_string())
            .or_default()
            .push((f[2].to_string(), score));
    }
    for results in out.values_mut() {
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    Ok(out)
}

/// Mean metrics of a run against qrels, averaged over every judged query.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecSummary {
    /// Number of queries in the qrels (the averaging denominator).
    pub num_queries: usize,
    /// The `k` used for the @k metrics.
    pub k: usize,
    /// Mean nDCG@k.
    pub ndcg_at_k: f32,
    /// Mean average precision.
    pub map: f32,
    /// Mean reciprocal rank.
    pub mrr: f32,
    /// Mean recall@k.
    pub recall_at_k: f32,
    /// Mean precision@k.
    pub precision_at_k: f32,
}

/// Evaluate `run` against `qrels`, averaging each metric over all judged
/// queries. A query present in `qrels` but absent from `run` contributes 0 to
/// every mean (the `trec_eval` complete-judgments convention); a run query with
/// no judgments is ignored. Returns an all-zero summary for empty qrels.
pub fn evaluate(run: &TrecRun, qrels: &TrecQrels, k: usize) -> TrecSummary {
    let n = qrels.len();
    let mut summary = TrecSummary {
        num_queries: n,
        k,
        ndcg_at_k: 0.0,
        map: 0.0,
        mrr: 0.0,
        recall_at_k: 0.0,
        precision_at_k: 0.0,
    };
    if n == 0 {
        return summary;
    }
    let empty: Vec<(DocId, f32)> = Vec::new();
    for (qid, qr) in qrels {
        let results = run.get(qid).unwrap_or(&empty);
        summary.ndcg_at_k += ndcg_at_k(results, qr, k);
        summary.map += map(results, qr);
        summary.mrr += mrr(results, qr);
        summary.recall_at_k += recall_at_k(results, qr, k);
        summary.precision_at_k += precision_at_k(results, qr, k);
    }
    let n = n as f32;
    summary.ndcg_at_k /= n;
    summary.map /= n;
    summary.mrr /= n;
    summary.recall_at_k /= n;
    summary.precision_at_k /= n;
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    const QRELS: &str = "q1 0 d1 2\nq1 0 d2 0\nq1 0 d3 1\nq2 0 da 1\nq2 0 db 1\n";

    #[test]
    fn parse_qrels_groups_by_query() {
        let q = parse_qrels(QRELS.as_bytes()).unwrap();
        assert_eq!(q.len(), 2);
        assert_eq!(q["q1"]["d1"], 2);
        assert_eq!(q["q1"]["d2"], 0);
        assert_eq!(q["q2"].len(), 2);
    }

    #[test]
    fn parse_run_sorts_by_score_desc() {
        let run_txt = "q1 Q0 d3 1 0.5 tag\nq1 Q0 d1 2 0.9 tag\nq1 Q0 d2 3 0.1 tag\n";
        let r = parse_run(run_txt.as_bytes()).unwrap();
        let ids: Vec<&str> = r["q1"].iter().map(|(d, _)| d.as_str()).collect();
        assert_eq!(ids, vec!["d1", "d3", "d2"], "results must be score-desc");
    }

    #[test]
    fn perfect_run_scores_one() {
        // Each query's relevant docs ranked first, by relevance -> nDCG = MAP = 1.
        let run_txt = "q1 Q0 d1 1 0.9 t\nq1 Q0 d3 2 0.6 t\nq1 Q0 d2 3 0.1 t\n\
                       q2 Q0 da 1 0.8 t\nq2 Q0 db 2 0.7 t\n";
        let qrels = parse_qrels(QRELS.as_bytes()).unwrap();
        let run = parse_run(run_txt.as_bytes()).unwrap();
        let s = evaluate(&run, &qrels, 10);
        assert_eq!(s.num_queries, 2);
        assert!(
            (s.ndcg_at_k - 1.0).abs() < 1e-6,
            "perfect nDCG should be 1.0, got {}",
            s.ndcg_at_k
        );
        assert!(
            (s.map - 1.0).abs() < 1e-6,
            "perfect MAP should be 1.0, got {}",
            s.map
        );
        assert!(
            (s.mrr - 1.0).abs() < 1e-6,
            "perfect MRR should be 1.0, got {}",
            s.mrr
        );
    }

    #[test]
    fn missing_query_contributes_zero() {
        // q2 has no run results -> its metrics are 0, dragging the 2-query mean.
        let run_txt = "q1 Q0 d1 1 0.9 t\nq1 Q0 d3 2 0.6 t\nq1 Q0 d2 3 0.1 t\n";
        let qrels = parse_qrels(QRELS.as_bytes()).unwrap();
        let run = parse_run(run_txt.as_bytes()).unwrap();
        let s = evaluate(&run, &qrels, 10);
        // q1 perfect (nDCG 1.0), q2 absent (0.0) -> mean 0.5.
        assert!((s.ndcg_at_k - 0.5).abs() < 1e-6, "got {}", s.ndcg_at_k);
    }

    #[test]
    fn empty_qrels_is_zero() {
        let s = evaluate(&HashMap::new(), &HashMap::new(), 10);
        assert_eq!(s.num_queries, 0);
        assert_eq!(s.ndcg_at_k, 0.0);
    }
}
