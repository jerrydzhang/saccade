use serde::{Deserialize, Serialize};

use crate::Reject;

/// A non-empty string
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Prose(String);

impl Prose {
    pub fn new(text: String) -> Result<Self, Reject> {
        if text.trim().is_empty() {
            Err(Reject::ReasonRequired)
        } else {
            Ok(Self(text))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Prose {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        if text.trim().is_empty() {
            Err(serde::de::Error::custom(
                "prose cannot be empty or whitespace",
            ))
        } else {
            Ok(Prose(text))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn prose_rejects_empty_and_whitespace() {
        assert!(matches!(
            Prose::new(String::new()),
            Err(Reject::ReasonRequired)
        ));
        assert!(matches!(
            Prose::new("   ".into()),
            Err(Reject::ReasonRequired)
        ));
        assert!(Prose::new("run dead".into()).is_ok());
    }
}
