use serde::{Deserialize, Serialize};

use crate::Reject;

/// An exact-UTF-8 identity key, not Prose: refuses empty, whitespace-only,
/// and edged whitespace rather than trimming; equality is shared across tiers.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorName(String);

impl ActorName {
    pub fn new(name: String) -> Result<Self, Reject> {
        if name.trim().is_empty() {
            return Err(Reject::InvalidActor);
        }
        if name != name.trim() {
            return Err(Reject::InvalidActor);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn actor_name_refuses_blank_and_edged() {
        assert!(matches!(
            ActorName::new(String::new()),
            Err(Reject::InvalidActor)
        ));
        assert!(matches!(
            ActorName::new("   ".into()),
            Err(Reject::InvalidActor)
        ));
        assert!(matches!(
            ActorName::new(" jerry".into()),
            Err(Reject::InvalidActor)
        ));
        assert!(ActorName::new("saccade bot".into()).is_ok());
    }
}
