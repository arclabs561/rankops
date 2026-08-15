//! Validated TREC qrels/run parsing and `trec_eval -c`-style evaluation.
//!
//! This module deliberately implements only collection-level nDCG@k, MAP,
//! reciprocal rank, recall@k, and P@k. It ignores the submitted rank, sorts by
//! score, and treats qrels-only queries as zero-score queries.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{BufRead, BufReader, Read};

/// Query identifier (TREC `query_id` column).
pub type QueryId = String;
/// Document identifier (TREC `doc_id` column).
pub type DocId = String;
/// Per-query signed relevance judgments, preserving TREC's negative values.
pub type TrecQrels = HashMap<QueryId, HashMap<DocId, i64>>;
/// Per-query ranked results, sorted by TREC score order.
pub type TrecRun = HashMap<QueryId, Vec<(DocId, f32)>>;

/// Why a TREC data record was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrecRecordError {
    /// The record has the wrong number of columns.
    ColumnCount {
        /// Required column count description.
        expected: &'static str,
        /// Number of columns found.
        found: usize,
    },
    /// A qrels relevance is not a signed integer.
    InvalidRelevance,
    /// A run score is not a finite floating-point number.
    InvalidScore,
    /// The query/document pair appeared earlier in the same file.
    DuplicateDocument {
        /// One-based source line of the first occurrence.
        first_line: usize,
    },
}

impl fmt::Display for TrecRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnCount { expected, found } => {
                write!(f, "expected {expected} columns, found {found}")
            }
            Self::InvalidRelevance => write!(f, "relevance must be a signed integer"),
            Self::InvalidScore => write!(f, "score must be a finite number"),
            Self::DuplicateDocument { first_line } => {
                write!(
                    f,
                    "duplicate query/document pair; first seen at line {first_line}"
                )
            }
        }
    }
}

/// An error while reading or validating a TREC input file.
#[derive(Debug)]
#[non_exhaustive]
pub enum TrecParseError {
    /// The underlying reader failed.
    Io(std::io::Error),
    /// A non-comment data record is invalid.
    Record {
        /// One-based source line.
        line: usize,
        /// Input record type (`qrels` or `run`).
        kind: &'static str,
        /// Specific validation failure.
        error: TrecRecordError,
    },
}

impl fmt::Display for TrecParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "failed to read TREC input: {error}"),
            Self::Record { line, kind, error } => {
                write!(f, "invalid {kind} record at line {line}: {error}")
            }
        }
    }
}
impl std::error::Error for TrecParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Record { .. } => None,
        }
    }
}

type ParseResult<T> = std::result::Result<T, TrecParseError>;

fn record(line: usize, kind: &'static str, error: TrecRecordError) -> TrecParseError {
    TrecParseError::Record { line, kind, error }
}

/// Parse and validate a TREC qrels file. Blank lines and full-line comments are ignored;
/// every other line must contain exactly four whitespace-separated columns.
pub fn parse_qrels<R: Read>(reader: R) -> ParseResult<TrecQrels> {
    let mut out = TrecQrels::new();
    let mut seen: HashMap<(String, String), usize> = HashMap::new();
    for (index, line) in BufReader::new(reader).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(TrecParseError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = trimmed.split_whitespace().collect();
        if fields.len() != 4 {
            return Err(record(
                line_number,
                "qrels",
                TrecRecordError::ColumnCount {
                    expected: "exactly 4",
                    found: fields.len(),
                },
            ));
        }
        let relevance = fields[3]
            .parse::<i64>()
            .map_err(|_| record(line_number, "qrels", TrecRecordError::InvalidRelevance))?;
        let key = (fields[0].to_owned(), fields[2].to_owned());
        if let Some(first_line) = seen.insert(key.clone(), line_number) {
            return Err(record(
                line_number,
                "qrels",
                TrecRecordError::DuplicateDocument { first_line },
            ));
        }
        out.entry(key.0).or_default().insert(key.1, relevance);
    }
    Ok(out)
}

/// Parse and validate a TREC run file. Blank lines and full-line comments are ignored;
/// records need six columns, while trailing columns are tolerated.
pub fn parse_run<R: Read>(reader: R) -> ParseResult<TrecRun> {
    let mut out = TrecRun::new();
    let mut seen: HashMap<(String, String), usize> = HashMap::new();
    for (index, line) in BufReader::new(reader).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(TrecParseError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = trimmed.split_whitespace().collect();
        if fields.len() < 6 {
            return Err(record(
                line_number,
                "run",
                TrecRecordError::ColumnCount {
                    expected: "at least 6",
                    found: fields.len(),
                },
            ));
        }
        let score = fields[4]
            .parse::<f32>()
            .ok()
            .filter(|score| score.is_finite())
            .ok_or_else(|| record(line_number, "run", TrecRecordError::InvalidScore))?;
        let key = (fields[0].to_owned(), fields[2].to_owned());
        if let Some(first_line) = seen.insert(key.clone(), line_number) {
            return Err(record(
                line_number,
                "run",
                TrecRecordError::DuplicateDocument { first_line },
            ));
        }
        out.entry(key.0).or_default().push((key.1, score));
    }
    for results in out.values_mut() {
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
    }
    Ok(out)
}

/// Mean metrics averaged over every qrels query (`trec_eval -c` semantics).
#[derive(Debug, Clone, PartialEq)]
pub struct TrecSummary {
    /// Number of qrels queries in the averaging denominator.
    pub num_queries: usize,
    /// The cutoff used by @k metrics.
    pub k: usize,
    /// Mean linear-gain nDCG@k.
    pub ndcg_at_k: f32,
    /// Mean average precision with relevance threshold 1.
    pub map: f32,
    /// Mean reciprocal rank with relevance threshold 1.
    pub mrr: f32,
    /// Mean recall@k with relevance threshold 1.
    pub recall_at_k: f32,
    /// Mean P@k, padding short runs with non-relevant results.
    pub precision_at_k: f32,
}

fn relevant(rel: i64) -> bool {
    rel >= 1
}
fn gain(rel: i64) -> f32 {
    rel.max(0) as f32
}
fn ndcg(results: &[(DocId, f32)], qrels: &HashMap<DocId, i64>, k: usize) -> f32 {
    let dcg: f32 = results
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, (id, _))| gain(*qrels.get(id).unwrap_or(&0)) / ((i + 2) as f32).log2())
        .sum();
    let mut ideal: Vec<_> = qrels
        .values()
        .copied()
        .map(gain)
        .filter(|gain| *gain > 0.0)
        .collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f32 = ideal
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(i, value)| value / ((i + 2) as f32).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}
fn metrics(
    results: &[(DocId, f32)],
    qrels: &HashMap<DocId, i64>,
    k: usize,
) -> (f32, f32, f32, f32, f32) {
    let total = qrels.values().filter(|&&rel| relevant(rel)).count();
    if total == 0 {
        return (ndcg(results, qrels, k), 0.0, 0.0, 0.0, 0.0);
    }
    let mut seen = HashSet::new();
    let mut hits = 0;
    let mut ap = 0.0;
    let mut reciprocal = 0.0;
    for (index, (id, _)) in results.iter().enumerate() {
        if seen.insert(id) && qrels.get(id).is_some_and(|&rel| relevant(rel)) {
            hits += 1;
            ap += hits as f32 / (index + 1) as f32;
            if reciprocal == 0.0 {
                reciprocal = 1.0 / (index + 1) as f32;
            }
        }
    }
    let top_hits = results
        .iter()
        .take(k)
        .filter(|(id, _)| qrels.get(id).is_some_and(|&rel| relevant(rel)))
        .count();
    (
        ndcg(results, qrels, k),
        ap / total as f32,
        reciprocal,
        top_hits as f32 / total as f32,
        if k == 0 {
            0.0
        } else {
            top_hits as f32 / k as f32
        },
    )
}

/// Evaluate a run against qrels using `trec_eval -c`-style query averaging.
pub fn evaluate(run: &TrecRun, qrels: &TrecQrels, k: usize) -> TrecSummary {
    let mut summary = TrecSummary {
        num_queries: qrels.len(),
        k,
        ndcg_at_k: 0.0,
        map: 0.0,
        mrr: 0.0,
        recall_at_k: 0.0,
        precision_at_k: 0.0,
    };
    if qrels.is_empty() {
        return summary;
    }
    let empty = Vec::new();
    for (qid, judgments) in qrels {
        let (n, a, m, r, p) = metrics(run.get(qid).unwrap_or(&empty), judgments, k);
        summary.ndcg_at_k += n;
        summary.map += a;
        summary.mrr += m;
        summary.recall_at_k += r;
        summary.precision_at_k += p;
    }
    let count = qrels.len() as f32;
    summary.ndcg_at_k /= count;
    summary.map /= count;
    summary.mrr /= count;
    summary.recall_at_k /= count;
    summary.precision_at_k /= count;
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_data_and_duplicates() {
        assert!(parse_qrels(b"q 0 d nope\n".as_slice()).is_err());
        assert!(parse_run(b"q Q0 d 1 NaN tag\n".as_slice()).is_err());
        assert!(parse_run(b"q Q0 d 1 1 tag\nq Q0 d 2 0 tag\n".as_slice()).is_err());
    }
    #[test]
    fn ties_follow_trec_document_order() {
        let run = parse_run(b"q Q0 a 1 1 tag\nq Q0 b 2 1 tag\n".as_slice()).unwrap();
        assert_eq!(run["q"][0].0, "b");
    }
    #[test]
    fn uses_trec_cutoff_and_gain_semantics() {
        let qrels = parse_qrels(b"q 0 a 2\nq 0 b 1\n".as_slice()).unwrap();
        let run = parse_run(b"q Q0 b 1 1 tag\n".as_slice()).unwrap();
        let summary = evaluate(&run, &qrels, 2);
        let expected_ndcg = 1.0 / (2.0 + 1.0 / 3.0_f32.log2());
        assert!((summary.ndcg_at_k - expected_ndcg).abs() < 1e-6);
        assert!((summary.precision_at_k - 0.5).abs() < 1e-6);
    }
    #[test]
    fn missing_qrels_query_contributes_zero() {
        let qrels = parse_qrels(b"q1 0 a 1\nq2 0 b 1\n".as_slice()).unwrap();
        let run = parse_run(b"q1 Q0 a 1 1 tag\n".as_slice()).unwrap();
        assert_eq!(evaluate(&run, &qrels, 10).map, 0.5);
    }
}
