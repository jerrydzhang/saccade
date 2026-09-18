use serde::{Deserialize, Serialize};

use crate::Reject;

/// A non-empty string
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Prose(String);

impl TryFrom<String> for Prose {
    type Error = String;

    fn try_from(text: String) -> Result<Self, String> {
        Prose::new(text).map_err(|_| "prose cannot be empty or whitespace".into())
    }
}

impl From<Prose> for String {
    fn from(p: Prose) -> String {
        p.0
    }
}

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
