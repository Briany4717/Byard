//! Small shared utilities (diagnostic hints).

/// The Levenshtein edit distance between two strings, used to suggest a
/// "did you mean …?" correction for unknown views/attributes (RFC-0002 D4).
#[must_use]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Returns the candidate closest to `name` (by edit distance), if one is within
/// a small threshold scaled to the name length — otherwise `None`, so an
/// unrelated typo does not produce a misleading suggestion.
#[must_use]
pub fn closest_match<'a>(
    name: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    let threshold = (name.len() / 2).max(2);
    candidates
        .into_iter()
        .map(|c| (levenshtein(name, c), c))
        .filter(|(d, _)| *d <= threshold)
        // Tie-break by the candidate name so the suggestion is deterministic
        // regardless of iteration order (e.g. `gp` → `gap`, not the equally
        // close `p`).
        .min_by_key(|(d, c)| (*d, *c))
        .map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_basics() {
        assert_eq!(levenshtein("gap", "gap"), 0);
        assert_eq!(levenshtein("gp", "gap"), 1);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn suggests_close_candidate() {
        assert_eq!(
            closest_match("gp", ["gap", "padding", "color"]),
            Some("gap")
        );
    }

    #[test]
    fn no_suggestion_for_unrelated() {
        assert_eq!(closest_match("xyzzy", ["gap", "padding"]), None);
    }
}
