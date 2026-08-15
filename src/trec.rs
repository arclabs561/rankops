//! Validated TREC qrels/run parsing and `trec_eval`-style evaluation.
//!
//! This module implements collection-level nDCG@k, MAP, reciprocal rank,
//! recall@k, and P@k. It ignores the submitted rank, orders by score and then
//! document ID, and uses `trec_eval -c` semantics by default.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{BufRead, BufReader, Read};

/// Query identifier (TREC `query_id` column).
pub type QueryId = String;
/// Document identifier (TREC `doc_id` column).
pub type DocId = String;

/// One in-memory qrels record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrecQrel {
    /// Query identifier.
    pub query_id: QueryId,
    /// Document identifier.
    pub document_id: DocId,
    /// Signed relevance judgment.
    pub relevance: i64,
}

/// One in-memory run record.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecRunEntry {
    /// Query identifier.
    pub query_id: QueryId,
    /// Document identifier.
    pub document_id: DocId,
    /// Retrieval score.
    pub score: f64,
}

/// A violation while constructing validated TREC data in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrecDataError {
    /// A run score is not finite.
    NonFiniteScore {
        /// Query identifier of the invalid result.
        query_id: QueryId,
        /// Document identifier of the invalid result.
        document_id: DocId,
    },
    /// The query/document pair was supplied more than once.
    DuplicateDocument {
        /// Query identifier of the duplicate.
        query_id: QueryId,
        /// Document identifier of the duplicate.
        document_id: DocId,
    },
}

impl fmt::Display for TrecDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteScore {
                query_id,
                document_id,
            } => write!(
                f,
                "non-finite score for query {query_id}, document {document_id}"
            ),
            Self::DuplicateDocument {
                query_id,
                document_id,
            } => write!(f, "duplicate query/document pair: {query_id}/{document_id}"),
        }
    }
}

impl std::error::Error for TrecDataError {}

/// Validated qrels, ordered by query and document ID.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrecQrels(BTreeMap<QueryId, BTreeMap<DocId, i64>>);

impl TrecQrels {
    /// Construct validated qrels from in-memory records.
    pub fn from_records(
        records: impl IntoIterator<Item = TrecQrel>,
    ) -> Result<Self, TrecDataError> {
        let mut judgments: BTreeMap<QueryId, BTreeMap<DocId, i64>> = BTreeMap::new();
        for record in records {
            let query_id = record.query_id;
            let document_id = record.document_id;
            if judgments
                .entry(query_id.clone())
                .or_default()
                .insert(document_id.clone(), record.relevance)
                .is_some()
            {
                return Err(TrecDataError::DuplicateDocument {
                    query_id,
                    document_id,
                });
            }
        }
        Ok(Self(judgments))
    }

    /// Number of judged queries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no judged queries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Relevance judgments for `query_id`, ordered by document ID.
    #[must_use]
    pub fn judgments(&self, query_id: &str) -> Option<&BTreeMap<DocId, i64>> {
        self.0.get(query_id)
    }

    /// Query IDs in deterministic order.
    pub fn query_ids(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

/// One validated result in score order.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecResult {
    document_id: DocId,
    score: f64,
}

impl TrecResult {
    /// Document identifier.
    #[must_use]
    pub fn document_id(&self) -> &str {
        &self.document_id
    }

    /// Retrieval score used to order the result.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.score
    }
}

/// Validated run, ordered by query and TREC score order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrecRun(BTreeMap<QueryId, Vec<TrecResult>>);

impl TrecRun {
    /// Construct a validated run from in-memory records.
    pub fn from_records(
        records: impl IntoIterator<Item = TrecRunEntry>,
    ) -> Result<Self, TrecDataError> {
        let mut results: BTreeMap<QueryId, Vec<TrecResult>> = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for record in records {
            if !record.score.is_finite() {
                return Err(TrecDataError::NonFiniteScore {
                    query_id: record.query_id,
                    document_id: record.document_id,
                });
            }
            let key = (record.query_id, record.document_id);
            if !seen.insert(key.clone()) {
                return Err(TrecDataError::DuplicateDocument {
                    query_id: key.0,
                    document_id: key.1,
                });
            }
            results.entry(key.0).or_default().push(TrecResult {
                document_id: key.1,
                score: record.score,
            });
        }
        for query_results in results.values_mut() {
            sort_results(query_results);
        }
        Ok(Self(results))
    }

    /// Number of queries with submitted results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no submitted results.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Results for `query_id` in deterministic score order.
    #[must_use]
    pub fn results(&self, query_id: &str) -> Option<&[TrecResult]> {
        self.0.get(query_id).map(Vec::as_slice)
    }

    /// Query IDs in deterministic order.
    pub fn query_ids(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

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

fn sort_results(results: &mut [TrecResult]) {
    results.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.document_id.cmp(&a.document_id))
    });
}

/// Parse and validate a TREC qrels file. Blank lines and full-line comments are ignored;
/// every other line must contain exactly four whitespace-separated columns.
pub fn parse_qrels<R: Read>(reader: R) -> ParseResult<TrecQrels> {
    let mut out: BTreeMap<QueryId, BTreeMap<DocId, i64>> = BTreeMap::new();
    let mut seen = BTreeMap::new();
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
    Ok(TrecQrels(out))
}

/// Parse and validate a TREC run file. Blank lines and full-line comments are ignored;
/// records need six columns, while trailing columns are tolerated.
pub fn parse_run<R: Read>(reader: R) -> ParseResult<TrecRun> {
    let mut out: BTreeMap<QueryId, Vec<TrecResult>> = BTreeMap::new();
    let mut seen = BTreeMap::new();
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
            .parse::<f64>()
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
        out.entry(key.0).or_default().push(TrecResult {
            document_id: key.1,
            score,
        });
    }
    for results in out.values_mut() {
        sort_results(results);
    }
    Ok(TrecRun(out))
}

/// Configuration for TREC evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TrecEvalConfig {
    /// The cutoff used by @k metrics.
    pub k: usize,
    /// Minimum qrels relevance considered relevant by binary metrics.
    pub relevance_level: i64,
    /// Whether qrels-only queries contribute zero to collection means.
    pub include_qrels_only: bool,
    /// Maximum number of results used from each query (`None` means all).
    pub max_docs_per_query: Option<usize>,
}

impl TrecEvalConfig {
    /// Create the default `trec_eval -c`-style configuration for cutoff `k`.
    #[must_use]
    pub const fn new(k: usize) -> Self {
        Self {
            k,
            relevance_level: 1,
            include_qrels_only: true,
            max_docs_per_query: None,
        }
    }

    /// Set the minimum relevance for MAP, reciprocal rank, recall, and P@k.
    #[must_use]
    pub const fn with_relevance_level(mut self, relevance_level: i64) -> Self {
        self.relevance_level = relevance_level;
        self
    }

    /// Include or exclude qrels-only queries from collection means.
    #[must_use]
    pub const fn with_qrels_only(mut self, include_qrels_only: bool) -> Self {
        self.include_qrels_only = include_qrels_only;
        self
    }

    /// Limit the number of retrieved documents considered for each query.
    #[must_use]
    pub const fn with_max_docs_per_query(mut self, max_docs_per_query: Option<usize>) -> Self {
        self.max_docs_per_query = max_docs_per_query;
        self
    }
}

impl Default for TrecEvalConfig {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Mean metrics averaged over the configured set of qrels queries.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecSummary {
    /// Number of queries in the averaging denominator.
    pub num_queries: usize,
    /// The cutoff used by @k metrics.
    pub k: usize,
    /// Mean linear-gain nDCG@k.
    pub ndcg_at_k: f64,
    /// Mean average precision with the configured relevance threshold.
    pub map: f64,
    /// Mean reciprocal rank with the configured relevance threshold.
    pub mrr: f64,
    /// Mean recall@k with the configured relevance threshold.
    pub recall_at_k: f64,
    /// Mean P@k, padding short runs with non-relevant results.
    pub precision_at_k: f64,
}

/// Metrics for one qrels query under a [`TrecEvalConfig`].
///
/// The query order in [`TrecEvaluation::queries`] is deterministic. A query is
/// present when it contributes to the collection summary.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecQueryMetrics {
    /// Query identifier from the qrels.
    pub query_id: QueryId,
    /// Number of submitted results considered after the configured document limit.
    pub num_retrieved: usize,
    /// Number of documents meeting the configured relevance threshold.
    pub num_relevant: usize,
    /// Number of top-k results with any qrels judgment, including judgment zero.
    pub num_judged_at_k: usize,
    /// Linear-gain nDCG@k for this query.
    pub ndcg_at_k: f64,
    /// Average precision for this query.
    pub average_precision: f64,
    /// Reciprocal rank for this query.
    pub reciprocal_rank: f64,
    /// Recall@k for this query.
    pub recall_at_k: f64,
    /// P@k for this query, padding short runs with non-relevant results.
    pub precision_at_k: f64,
    /// Judged@k for this query, padding short runs as unjudged.
    pub judged_at_k: f64,
}

/// A collection summary together with the per-query values that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecEvaluation {
    /// Collection-level mean metrics.
    pub summary: TrecSummary,
    /// Per-query metrics in deterministic qrels query order.
    pub queries: Vec<TrecQueryMetrics>,
}

impl TrecEvaluation {
    /// Mean Judged@k over the queries that contributed to this evaluation.
    ///
    /// This is the share of the top-k results with any qrels judgment,
    /// including zero-relevance judgments. It is diagnostic only and does not
    /// affect the other metric denominators.
    #[must_use]
    pub fn judged_at_k(&self) -> f64 {
        if self.queries.is_empty() {
            0.0
        } else {
            self.queries
                .iter()
                .map(|query| query.judged_at_k)
                .sum::<f64>()
                / self.queries.len() as f64
        }
    }
}

fn relevant(rel: i64, config: TrecEvalConfig) -> bool {
    rel >= 0 && rel >= config.relevance_level.max(0)
}

fn gain(rel: i64) -> f64 {
    rel.max(0) as f64
}

fn ndcg(results: &[TrecResult], qrels: &BTreeMap<DocId, i64>, config: TrecEvalConfig) -> f64 {
    let limit = config.max_docs_per_query.unwrap_or(usize::MAX);
    let dcg: f64 = results
        .iter()
        .take(limit)
        .take(config.k)
        .enumerate()
        .map(|(i, result)| {
            gain(*qrels.get(&result.document_id).unwrap_or(&0)) / ((i + 2) as f64).log2()
        })
        .sum();
    let mut ideal: Vec<_> = qrels
        .values()
        .copied()
        .map(gain)
        .filter(|gain| *gain > 0.0)
        .collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal
        .into_iter()
        .take(config.k)
        .enumerate()
        .map(|(i, value)| value / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn metrics(
    query_id: &str,
    results: &[TrecResult],
    qrels: &BTreeMap<DocId, i64>,
    config: TrecEvalConfig,
) -> TrecQueryMetrics {
    let total = qrels.values().filter(|&&rel| relevant(rel, config)).count();
    let limit = config.max_docs_per_query.unwrap_or(usize::MAX);
    let results = &results[..results.len().min(limit)];
    let num_judged_at_k = results
        .iter()
        .take(config.k)
        .filter(|result| qrels.contains_key(&result.document_id))
        .count();
    let judged_at_k = if config.k == 0 {
        0.0
    } else {
        num_judged_at_k as f64 / config.k as f64
    };
    if total == 0 {
        return TrecQueryMetrics {
            query_id: query_id.to_owned(),
            num_retrieved: results.len(),
            num_relevant: 0,
            num_judged_at_k,
            ndcg_at_k: ndcg(results, qrels, config),
            average_precision: 0.0,
            reciprocal_rank: 0.0,
            recall_at_k: 0.0,
            precision_at_k: 0.0,
            judged_at_k,
        };
    }
    let mut seen = BTreeSet::new();
    let mut hits = 0;
    let mut ap = 0.0;
    let mut reciprocal = 0.0;
    for (index, result) in results.iter().enumerate() {
        if seen.insert(&result.document_id)
            && qrels
                .get(&result.document_id)
                .is_some_and(|&rel| relevant(rel, config))
        {
            hits += 1;
            ap += hits as f64 / (index + 1) as f64;
            if reciprocal == 0.0 {
                reciprocal = 1.0 / (index + 1) as f64;
            }
        }
    }
    let top_hits = results
        .iter()
        .take(config.k)
        .filter(|result| {
            qrels
                .get(&result.document_id)
                .is_some_and(|&rel| relevant(rel, config))
        })
        .count();
    TrecQueryMetrics {
        query_id: query_id.to_owned(),
        num_retrieved: results.len(),
        num_relevant: total,
        num_judged_at_k,
        ndcg_at_k: ndcg(results, qrels, config),
        average_precision: ap / total as f64,
        reciprocal_rank: reciprocal,
        recall_at_k: top_hits as f64 / total as f64,
        precision_at_k: if config.k == 0 {
            0.0
        } else {
            top_hits as f64 / config.k as f64
        },
        judged_at_k,
    }
}

/// Evaluate a run against qrels, retaining the values for each contributing query.
#[must_use]
pub fn evaluate_detailed_with_config(
    run: &TrecRun,
    qrels: &TrecQrels,
    config: TrecEvalConfig,
) -> TrecEvaluation {
    let mut summary = TrecSummary {
        num_queries: 0,
        k: config.k,
        ndcg_at_k: 0.0,
        map: 0.0,
        mrr: 0.0,
        recall_at_k: 0.0,
        precision_at_k: 0.0,
    };
    let mut queries = Vec::new();
    let empty = Vec::new();
    for (query_id, judgments) in &qrels.0 {
        let Some(results) = run
            .0
            .get(query_id)
            .or_else(|| config.include_qrels_only.then_some(&empty))
        else {
            continue;
        };
        let query = metrics(query_id, results, judgments, config);
        summary.num_queries += 1;
        summary.ndcg_at_k += query.ndcg_at_k;
        summary.map += query.average_precision;
        summary.mrr += query.reciprocal_rank;
        summary.recall_at_k += query.recall_at_k;
        summary.precision_at_k += query.precision_at_k;
        queries.push(query);
    }
    if summary.num_queries > 0 {
        let count = summary.num_queries as f64;
        summary.ndcg_at_k /= count;
        summary.map /= count;
        summary.mrr /= count;
        summary.recall_at_k /= count;
        summary.precision_at_k /= count;
    }
    TrecEvaluation { summary, queries }
}

/// Evaluate a run against qrels with a configurable TREC evaluation policy.
#[must_use]
pub fn evaluate_with_config(
    run: &TrecRun,
    qrels: &TrecQrels,
    config: TrecEvalConfig,
) -> TrecSummary {
    evaluate_detailed_with_config(run, qrels, config).summary
}

/// Evaluate a run with the default `trec_eval -c`-style configuration,
/// retaining metrics for each contributing query.
#[must_use]
pub fn evaluate_detailed(run: &TrecRun, qrels: &TrecQrels, k: usize) -> TrecEvaluation {
    evaluate_detailed_with_config(run, qrels, TrecEvalConfig::new(k))
}

/// Evaluate a run with the default `trec_eval -c`-style configuration.
#[must_use]
pub fn evaluate(run: &TrecRun, qrels: &TrecQrels, k: usize) -> TrecSummary {
    evaluate_with_config(run, qrels, TrecEvalConfig::new(k))
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
    fn ties_follow_trec_document_order_without_f32_rounding() {
        let run = parse_run(b"q Q0 a 1 1.0000000001 tag\nq Q0 b 2 1 tag\n".as_slice()).unwrap();
        let results = run.results("q").unwrap();
        assert_eq!(results[0].document_id(), "a");
    }

    #[test]
    fn uses_trec_cutoff_and_gain_semantics() {
        let qrels = parse_qrels(b"q 0 a 2\nq 0 b 1\n".as_slice()).unwrap();
        let run = parse_run(b"q Q0 b 1 1 tag\n".as_slice()).unwrap();
        let summary = evaluate(&run, &qrels, 2);
        let expected_ndcg = 1.0 / (2.0 + 1.0 / 3.0_f64.log2());
        assert!((summary.ndcg_at_k - expected_ndcg).abs() < 1e-12);
        assert!((summary.precision_at_k - 0.5).abs() < 1e-12);
    }

    #[test]
    fn config_controls_query_coverage_and_relevance_level() {
        let qrels = parse_qrels(b"q1 0 a 1\nq2 0 b 2\n".as_slice()).unwrap();
        let run = parse_run(b"q1 Q0 a 1 1 tag\n".as_slice()).unwrap();
        let config = TrecEvalConfig::new(10)
            .with_qrels_only(false)
            .with_relevance_level(2);
        let summary = evaluate_with_config(&run, &qrels, config);
        assert_eq!(summary.num_queries, 1);
        assert_eq!(summary.map, 0.0);
    }

    #[test]
    fn detailed_evaluation_exposes_query_diagnostics() {
        let qrels = parse_qrels(b"q1 0 a 1\nq1 0 b 1\nq2 0 c 1\n".as_slice()).unwrap();
        let run = parse_run(b"q1 Q0 a 1 1 tag\nq1 Q0 z 2 0 tag\n".as_slice()).unwrap();
        let evaluation = evaluate_detailed_with_config(&run, &qrels, TrecEvalConfig::new(1));

        assert_eq!(evaluation.summary.num_queries, 2);
        assert_eq!(evaluation.queries.len(), 2);
        assert_eq!(evaluation.queries[0].query_id, "q1");
        assert_eq!(evaluation.queries[0].num_retrieved, 2);
        assert_eq!(evaluation.queries[0].num_relevant, 2);
        assert_eq!(evaluation.queries[0].num_judged_at_k, 1);
        assert_eq!(evaluation.queries[0].judged_at_k, 1.0);
        assert_eq!(evaluation.queries[0].recall_at_k, 0.5);
        assert_eq!(evaluation.queries[1].query_id, "q2");
        assert_eq!(evaluation.queries[1].num_retrieved, 0);
        assert_eq!(evaluation.queries[1].num_relevant, 1);
        assert_eq!(evaluation.queries[1].num_judged_at_k, 0);
        assert_eq!(evaluation.queries[1].judged_at_k, 0.0);
        assert_eq!(evaluation.judged_at_k(), 0.5);
    }

    #[test]
    fn in_memory_records_are_validated_and_sorted() {
        let run = TrecRun::from_records([
            TrecRunEntry {
                query_id: "q".into(),
                document_id: "a".into(),
                score: 1.0,
            },
            TrecRunEntry {
                query_id: "q".into(),
                document_id: "b".into(),
                score: 1.0,
            },
        ])
        .unwrap();
        assert_eq!(run.results("q").unwrap()[0].document_id(), "b");
        assert!(TrecRun::from_records([TrecRunEntry {
            query_id: "q".into(),
            document_id: "a".into(),
            score: f64::NAN,
        }])
        .is_err());
        assert!(TrecQrels::from_records([
            TrecQrel {
                query_id: "q".into(),
                document_id: "a".into(),
                relevance: 1,
            },
            TrecQrel {
                query_id: "q".into(),
                document_id: "a".into(),
                relevance: 1,
            },
        ])
        .is_err());
    }

    #[test]
    fn negative_qrels_are_never_relevant() {
        let qrels = parse_qrels(b"q 0 unjudged -1\nq 0 judged 0\n".as_slice()).unwrap();
        let run = parse_run(b"q Q0 unjudged 1 1 tag\n".as_slice()).unwrap();
        let summary = evaluate_with_config(
            &run,
            &qrels,
            TrecEvalConfig::new(1).with_relevance_level(-1),
        );
        assert_eq!(summary.precision_at_k, 0.0);
    }
}
