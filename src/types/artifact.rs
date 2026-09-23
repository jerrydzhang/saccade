use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::types::prose::Prose;

/// sha256 content identity, lowercase hex
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ContentHash(String);

#[derive(Debug, PartialEq)]
pub enum HashError {
    /// Not 64 lowercase hex characters
    NotHex,
}

impl TryFrom<String> for ContentHash {
    type Error = String;

    fn try_from(value: String) -> Result<Self, String> {
        ContentHash::new(value).map_err(|e| format!("{e:?}"))
    }
}

impl From<ContentHash> for String {
    fn from(h: ContentHash) -> String {
        h.0
    }
}

impl ContentHash {
    pub fn new(value: String) -> Result<Self, HashError> {
        if value.len() == 64
            && value
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        {
            Ok(Self(value))
        } else {
            Err(HashError::NotHex)
        }
    }

    /// The identity of these bytes
    pub fn of(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            hex.push_str(&format!("{byte:02x}"));
        }
        ContentHash(hex)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// An artifact a thread holds: a name and its content's identity. The
/// bytes live in the store, never in the record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub name: Prose,
    pub hash: ContentHash,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_hash_admits_only_sha256_hex_and_names_its_bytes() {
        // the identity of these bytes is the familiar sha256 of "saccade"
        let hash = ContentHash::of(b"saccade");
        assert_eq!(
            hash.as_str(),
            "3941d4453740985f0c363433c74070c2c4aa649d8a73fe52a573831c0f87aa7e"
        );
        // validation admits the hex it produced and refuses the rest
        assert_eq!(ContentHash::new(hash.as_str().into()), Ok(hash.clone()));
        assert!(matches!(
            ContentHash::new(String::new()),
            Err(HashError::NotHex)
        ));
        assert!(matches!(
            ContentHash::new("abc".into()),
            Err(HashError::NotHex)
        ));
        assert!(matches!(
            ContentHash::new(hash.as_str().to_uppercase()),
            Err(HashError::NotHex)
        ));
    }
}
