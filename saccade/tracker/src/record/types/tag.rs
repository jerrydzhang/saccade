use serde::{Deserialize, Serialize};

use crate::Reject;

/// What a tag is called: a flat name of letters, digits, dots,
/// underscores and hyphens, led by a letter or digit, at most 64
/// characters. Uniqueness ignores ASCII case and is the store's.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TagName(String);

pub const TAG_NAME_MAX: usize = 64;

impl TagName {
    pub fn new(name: String) -> Result<Self, Reject> {
        let mut chars = name.chars();
        let led = chars.next().is_some_and(|c| c.is_ascii_alphanumeric());
        let rest = chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
        if !led || !rest || name.len() > TAG_NAME_MAX {
            return Err(Reject::InvalidTagName);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TagName {
    type Error = String;

    fn try_from(name: String) -> Result<Self, String> {
        TagName::new(name).map_err(|e| e.to_string())
    }
}

impl From<TagName> for String {
    fn from(name: TagName) -> String {
        name.0
    }
}

/// The one line a tag carries saying what belongs under it: a single
/// non-empty trimmed line without control characters. No length is
/// bound and nothing truncates it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TagLine(String);

impl TagLine {
    pub fn new(text: String) -> Result<Self, Reject> {
        if text.trim().is_empty() || text.trim() != text || text.chars().any(char::is_control) {
            return Err(Reject::InvalidTagLine);
        }
        Ok(Self(text))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TagLine {
    type Error = String;

    fn try_from(text: String) -> Result<Self, String> {
        TagLine::new(text).map_err(|e| e.to_string())
    }
}

impl From<TagLine> for String {
    fn from(line: TagLine) -> String {
        line.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_tag_name_is_a_flat_word_of_at_most_sixty_four_characters() {
        for bad in [
            "",
            "-lead",
            ".lead",
            "_lead",
            "has space",
            "sl/ash",
            "caf\u{e9}",
        ] {
            assert!(TagName::new(bad.into()).is_err(), "{bad:?} must refuse");
        }
        assert!(TagName::new("a".repeat(65)).is_err());
        for bad in ["", "  ", " lead", "trail ", "two\nlines", "tab\there"] {
            assert!(TagLine::new(bad.into()).is_err(), "{bad:?} must refuse");
        }
        assert!(TagLine::new("a".repeat(5000)).is_ok());
        assert!(TagLine::new("what the login flow does".into()).is_ok());
        for good in ["auth", "Auth2", "v1.2_x-y", "9lives", &"a".repeat(64)] {
            assert_eq!(TagName::new(good.into()).unwrap().as_str(), good);
        }
    }
}
