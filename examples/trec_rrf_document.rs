//! Explain one document's equal-weight RRF score from two saved TREC runs.

use rankops::explain::{rrf_explain, RetrieverId};
use rankops::trec::{parse_run, TrecRun};
use rankops::RrfConfig;
use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::Path;

fn explain(
    mut out: impl Write,
    inputs: [(&str, &TrecRun); 2],
    query: &str,
    doc: &str,
    config: RrfConfig,
) -> Result<(), Box<dyn Error>> {
    if inputs.iter().all(|(_, run)| run.results(query).is_none()) {
        return Err(format!("query {query} absent from both runs").into());
    }
    let lists = inputs.map(|(_, run)| {
        run.results(query)
            .unwrap_or_default()
            .iter()
            // parse_run already ordered the f64 scores. RRF uses positions alone.
            .map(|hit| (hit.document_id(), 0.0))
            .collect::<Vec<_>>()
    });
    let ids = [RetrieverId::new("first"), RetrieverId::new("second")];
    let results = rrf_explain(&lists, &ids, config);
    writeln!(out, "query {query}, document {doc}, RRF k={}", config.k)?;
    let Some(hit) = results.iter().find(|hit| hit.id == doc) else {
        writeln!(out, "absent from both runs; no fused rank")?;
        return Ok(());
    };
    writeln!(out, "run\trank\tcontribution")?;
    for ((path, _), id) in inputs.iter().zip(ids) {
        let source = hit
            .explanation
            .sources
            .iter()
            .find(|s| s.retriever_id == id.as_str());
        if let Some((rank, contribution)) =
            source.and_then(|s| s.original_rank.map(|r| (r + 1, s.contribution)))
        {
            writeln!(out, "{path}\t{rank}\t{contribution:.7}")?;
        } else {
            writeln!(out, "{path}\tabsent\t0.0000000")?;
        }
    }
    // Different callers may break fused ties differently. Report their full span.
    let first = results.partition_point(|r| r.score > hit.score) + 1;
    let last = results.partition_point(|r| r.score >= hit.score);
    if first == last {
        writeln!(out, "fused rank {first}, score {:.7}", hit.score)?;
    } else {
        writeln!(
            out,
            "tied for fused ranks {first}..{last}, score {:.7}",
            hit.score
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [a, b, query, doc, rest @ ..] = args.as_slice() else {
        return Err("usage: trec_rrf_document FIRST.run SECOND.run QUERY DOCUMENT [K]".into());
    };
    let config = match rest {
        [] => RrfConfig::default(),
        [k] => {
            let k = k
                .to_str()
                .and_then(|s| s.parse::<NonZeroU32>().ok())
                .ok_or("K must be a positive 32-bit integer")?;
            RrfConfig::new(k.get())
        }
        _ => return Err("expected at most one smoothing constant K".into()),
    };
    let load = |path: &Path| -> Result<TrecRun, Box<dyn Error>> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(parse_run(file).map_err(|e| format!("{}: {e}", path.display()))?)
    };
    let (first, second) = (load(a.as_ref())?, load(b.as_ref())?);
    explain(
        io::stdout().lock(),
        [
            (&a.to_string_lossy(), &first),
            (&b.to_string_lossy(), &second),
        ],
        query.to_str().ok_or("QUERY must be valid UTF-8")?,
        doc.to_str().ok_or("DOCUMENT must be valid UTF-8")?,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(a: &str, b: &str, doc: &str) -> Result<String, Box<dyn Error>> {
        let (a, b) = (parse_run(a.as_bytes())?, parse_run(b.as_bytes())?);
        let mut out = Vec::new();
        explain(
            &mut out,
            [("A", &a), ("B", &b)],
            "q",
            doc,
            RrfConfig::new(1),
        )?;
        Ok(String::from_utf8(out)?)
    }

    #[test]
    fn explains_overlap_and_missing_documents_in_trec_score_order() {
        let a = "q Q0 a 1 7 first\nq Q0 z 2 7 first\n";
        let b = "q Q0 a 1 1 second\n";
        // TREC breaks tied input scores by descending document ID.
        assert!(report(a, b, "a")
            .unwrap()
            .ends_with("A\t2\t0.3333333\nB\t1\t0.5000000\nfused rank 1, score 0.8333334\n"));
        assert!(report(a, b, "z")
            .unwrap()
            .ends_with("A\t1\t0.5000000\nB\tabsent\t0.0000000\nfused rank 2, score 0.5000000\n"));
        assert!(report(a, b, "missing").unwrap().contains("no fused rank"));
    }

    #[test]
    fn equal_fused_scores_have_the_same_rank_range() {
        let (a, b) = ("q Q0 one 1 5 first\n", "q Q0 two 1 5 second\n");
        for doc in ["one", "two"] {
            assert!(report(a, b, doc)
                .unwrap()
                .contains("tied for fused ranks 1..2"));
        }
    }

    #[test]
    fn accepts_a_query_in_one_run_and_flags_a_typo_in_both() {
        assert!(report("", "q Q0 d 1 4 b\n", "d")
            .unwrap()
            .contains("fused rank 1"));
        assert!(report("", "", "d")
            .unwrap_err()
            .to_string()
            .contains("query q absent"));
    }
}
