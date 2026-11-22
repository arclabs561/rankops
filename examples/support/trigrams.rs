pub const DIM: usize = 256;

pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Hashed bag of character trigrams, raw term counts before normalisation.
pub fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for token in tokens(text) {
        let padded: Vec<char> = format!("  {token} ").chars().collect();
        for gram in padded.windows(3) {
            // FNV-1a over the trigram.
            let mut h: u64 = 0xcbf29ce484222325;
            for c in gram {
                h = (h ^ *c as u64).wrapping_mul(0x100000001b3);
            }
            v[(h % DIM as u64) as usize] += 1.0;
        }
    }
    v
}
