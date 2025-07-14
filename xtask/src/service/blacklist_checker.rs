use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use std::collections::HashSet;

pub struct BlacklistChecker {
    ac: AhoCorasick,
    max_word_length: usize,
}

impl BlacklistChecker {
    pub fn new<I: IntoIterator<Item = String>>(blacklist: I) -> Self {
        let words: HashSet<String> = blacklist.into_iter().collect();
        let max_word_length = words.iter().map(|s| s.len()).max().unwrap_or(0);
        let ac = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(words.iter().map(|s| s.as_str()))
            .expect("Failed to build Aho-Corasick automaton");
        BlacklistChecker {
            ac,
            max_word_length,
        }
    }

    pub fn contains_blacklisted_word(&self, context_suffix: &str) -> bool {
        self.ac.is_match(context_suffix)
    }

    /// Get the maximum length of any blacklisted word (in characters, not bytes)
    pub fn get_max_blacklist_word_length(&self) -> usize {
        self.max_word_length
    }

    /// Get the maximum length of any blacklisted word (alias for compatibility)
    pub fn get_max_word_length(&self) -> usize {
        self.get_max_blacklist_word_length()
    }

    /// Check if text contains any blacklisted words (alias for compatibility)
    pub fn contains_word(&self, text: &str) -> bool {
        self.contains_blacklisted_word(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blacklist_checker_basic() {
        let blacklist = vec![
            "danger".to_string(),
            "leak".to_string(),
            "badword".to_string(),
        ];
        let checker = BlacklistChecker::new(blacklist);

        assert!(checker.contains_blacklisted_word("This is dangerous"));
        assert!(checker.contains_blacklisted_word("Information leak"));
        assert!(!checker.contains_blacklisted_word("This is safe"));
    }

    #[test]
    fn test_blacklist_checker_chinese() {
        let blacklist = vec!["敏感词".to_string(), "违禁词".to_string()];
        let checker = BlacklistChecker::new(blacklist);

        assert!(checker.contains_blacklisted_word("这个句子包含敏感词"));
        assert!(checker.contains_blacklisted_word("违禁词的定义"));
        assert!(!checker.contains_blacklisted_word("正常的中文内容"));
    }

    #[test]
    fn test_blacklist_checker_long_words() {
        let blacklist = vec!["verylongblacklistedword".to_string(), "short".to_string()];
        let checker = BlacklistChecker::new(blacklist);

        assert_eq!(checker.get_max_blacklist_word_length(), 23);
        assert!(checker.contains_blacklisted_word("contains verylongblacklistedword"));
        assert!(checker.contains_blacklisted_word("short word"));
    }
}
