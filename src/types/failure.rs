use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureEvidence {
    pub code: FailureCode,
    pub detail: Option<String>,
}

impl FailureEvidence {
    pub fn new(code: FailureCode, detail: Option<String>) -> Self {
        FailureEvidence {
            code,
            detail: match detail {
                Some(d) if d.is_empty() => None,
                other => other,
            },
        }
    }
}
