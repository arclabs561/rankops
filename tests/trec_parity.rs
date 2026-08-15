use rankops::trec::{evaluate_with_config, parse_qrels, parse_run, TrecEvalConfig, TrecSummary};
use std::collections::BTreeMap;
use std::process::Command;

const QRELS: &str = include_str!("fixtures/trec/parity.qrels");
const RUN: &str = include_str!("fixtures/trec/parity.run");

fn summary() -> TrecSummary {
    let qrels = parse_qrels(QRELS.as_bytes()).unwrap();
    let run = parse_run(RUN.as_bytes()).unwrap();
    evaluate_with_config(&run, &qrels, TrecEvalConfig::new(2))
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn checked_in_trec_parity_corpus() {
    let summary = summary();
    assert_eq!(summary.num_queries, 3);
    assert_close(
        summary.ndcg_at_k,
        (1.0 + 2.0 / 3.0_f64.log2()) / (2.0 + 1.0 / 3.0_f64.log2()) / 3.0
            + 1.0 / 3.0_f64.log2() / 3.0,
    );
    assert_close(summary.map, 0.5);
    assert_close(summary.mrr, 0.5);
    assert_close(summary.recall_at_k, 2.0 / 3.0);
    assert_close(summary.precision_at_k, 0.5);
}

#[test]
fn optional_reference_binary_matches_corpus() {
    let Some(binary) = std::env::var_os("RANKOPS_TREC_EVAL") else {
        return;
    };
    let qrels_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/trec/parity.qrels"
    );
    let run_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/trec/parity.run"
    );
    let output = Command::new(binary)
        .args([
            "-c",
            "-m",
            "map",
            "-m",
            "recip_rank",
            "-m",
            "ndcg_cut.2",
            "-m",
            "recall.2",
            "-m",
            "P.2",
            qrels_path,
            run_path,
        ])
        .output()
        .expect("run trec_eval");
    assert!(
        output.status.success(),
        "trec_eval failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: BTreeMap<_, _> = stdout
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let (Some(measure), Some(topic), Some(value), None) =
                (fields.next(), fields.next(), fields.next(), fields.next())
            else {
                return None;
            };
            (topic == "all").then(|| (measure, value.parse::<f64>().unwrap()))
        })
        .collect();
    let summary = summary();
    assert_close(summary.map, values["map"]);
    assert_close(summary.mrr, values["recip_rank"]);
    assert_close(summary.ndcg_at_k, values["ndcg_cut_2"]);
    assert_close(summary.recall_at_k, values["recall_2"]);
    assert_close(summary.precision_at_k, values["P_2"]);
}
