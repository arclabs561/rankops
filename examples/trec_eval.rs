//! TREC file workflow: parse a qrels file and a run file, then evaluate the
//! run with collection-level (multi-query) mean metrics, the way `trec_eval`,
//! BEIR, and ranx report numbers.
//!
//! Real usage reads these from disk (`File::open(...)`); here the files are
//! inline string literals so the example is self-contained.
//!
//! Run: `cargo run --example trec_eval`

use rankops::trec::{evaluate, parse_qrels, parse_run};

// qrels: query_id  iteration  doc_id  relevance
const QRELS: &str = "\
q1 0 d1 2
q1 0 d2 1
q1 0 d3 0
q1 0 d4 1
q2 0 da 1
q2 0 db 2
q2 0 dc 0
";

// run: query_id  Q0  doc_id  rank  score  run_tag
// (rank column is advisory; results are re-sorted by score on parse)
const RUN: &str = "\
q1 Q0 d1 1 9.5 myrun
q1 Q0 d4 2 8.1 myrun
q1 Q0 d3 3 4.2 myrun
q1 Q0 d2 4 3.0 myrun
q2 Q0 db 1 7.7 myrun
q2 Q0 dc 2 6.0 myrun
q2 Q0 da 3 5.5 myrun
";

fn main() {
    let qrels = parse_qrels(QRELS.as_bytes()).expect("qrels parse");
    let run = parse_run(RUN.as_bytes()).expect("run parse");

    println!(
        "Parsed {} queries from qrels, {} from run.\n",
        qrels.len(),
        run.len()
    );

    for k in [5, 10] {
        let s = evaluate(&run, &qrels, k);
        println!(
            "@{k:<2}  nDCG={:.4}  MAP={:.4}  MRR={:.4}  recall={:.4}  precision={:.4}  (over {} queries)",
            s.ndcg_at_k, s.map, s.mrr, s.recall_at_k, s.precision_at_k, s.num_queries
        );
    }

    println!("\nThis is the BEIR/trec_eval-shaped number: one metric averaged over");
    println!("every judged query, so a rankops run can be compared head-to-head");
    println!("with ranx or pytrec_eval on the same qrels/run files.");
}
