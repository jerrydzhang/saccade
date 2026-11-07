//! Strict standard-base64 decoding for the core's exec output. The
//! schema promises base64 stdout/stderr; anything else is a protocol
//! error, never a guess.

pub fn decode(input: &str) -> Result<Vec<u8>, String> {
    fn value(byte: u8) -> Option<u8> {
        match byte {
            b'A'..=b'Z' => Some(byte - b'A'),
            b'a'..=b'z' => Some(byte - b'a' + 26),
            b'0'..=b'9' => Some(byte - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("length {} is not a multiple of 4", bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for group in bytes.as_chunks::<4>().0 {
        let mut six = [0u8; 4];
        let mut seen_pad = false;
        for (index, byte) in group.iter().enumerate() {
            if *byte == b'=' {
                if index < 2 {
                    return Err(format!("padding at position {index}"));
                }
                seen_pad = true;
            } else {
                if seen_pad {
                    return Err("data after padding".to_string());
                }
                six[index] =
                    value(*byte).ok_or_else(|| format!("invalid byte {:?}", *byte as char))?;
            }
        }
        let window = ((six[0] as u32) << 18)
            | ((six[1] as u32) << 12)
            | ((six[2] as u32) << 6)
            | (six[3] as u32);
        out.push((window >> 16) as u8);
        if !(group[2] == b'=' && group[3] == b'=') {
            out.push((window >> 8) as u8);
        }
        if group[3] != b'=' {
            out.push(window as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn the_basics_round_trip() {
        assert_eq!(decode("").unwrap(), b"");
        assert_eq!(decode("eA==").unwrap(), b"x");
        assert_eq!(decode("YQ==").unwrap(), b"a");
        assert_eq!(decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode("aGVsbG8h").unwrap(), b"hello!");
    }

    #[test]
    fn lengths_that_are_not_multiples_of_four_refuse() {
        assert!(decode("YQ=").is_err());
        assert!(decode("Y").is_err());
    }

    #[test]
    fn misplaced_or_repeated_padding_refuses() {
        assert!(decode("=AAA").is_err());
        assert!(decode("A=AA").is_err());
        assert!(decode("AA=A").is_err());
        assert!(decode("A===").is_err());
        assert!(decode("AAAA====").is_err());
    }

    #[test]
    fn invalid_characters_refuse() {
        assert!(decode("*AAA").is_err());
        assert!(decode("A\nAA").is_err());
    }
}
