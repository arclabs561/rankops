use rankops::rerank::{colbert, FdeConfig};

fn main() -> rankops::rerank::Result<()> {
    let query = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let docs = vec![
        ("doc-a", vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]]),
        ("doc-b", vec![vec![0.8, 0.0, 0.2], vec![0.0, 0.1, 0.9]]),
        ("doc-c", vec![vec![0.0, 0.2, 0.8], vec![0.2, 0.0, 0.8]]),
    ];

    let fde = FdeConfig::new();
    let proxy = fde.rank(&query, &docs)?;

    println!("Fixed-dimensional proxy ranking:");
    for (doc, score) in &proxy {
        println!("  {doc}: {score:.3}");
    }

    let shortlist: Vec<_> = proxy
        .iter()
        .take(2)
        .map(|(doc, _score)| {
            let tokens = docs
                .iter()
                .find(|(candidate, _tokens)| candidate == doc)
                .expect("proxy result came from the input corpus")
                .1
                .clone();
            (*doc, tokens)
        })
        .collect();

    let exact = colbert::rank(&query, &shortlist);

    println!("\nExact MaxSim rerank of proxy top 2:");
    for (doc, score) in exact {
        println!("  {doc}: {score:.3}");
    }

    Ok(())
}
