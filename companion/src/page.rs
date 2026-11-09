//! The consumer-side decoder for `/view/v1` pages: the envelope and the
//! six v1 part kinds, exactly the shapes the tracker's page framework
//! emits (t-24's page answers; literals pinned in the tests). Unknown
//! part kinds are ignored, never a refusal: the companion reads one
//! narrow thing - the agents page - out of a generic document.

use serde_json::Value;

/// One page document: the envelope fields plus the parts.
#[derive(Debug, Clone, PartialEq)]
pub struct Page {
    pub view_major: u32,
    pub kind: String,
    pub location: String,
    /// The fold cursor the page was computed at; a refusal whose fold is
    /// unreadable carries none.
    pub seq: Option<u64>,
    /// The root page's tracker name: the serving repo's directory
    /// basename, absent when the server runs `--db` with no repo.
    pub tracker: Option<String>,
    /// The root page's hub identity (the tracker machine's
    /// /etc/machine-id) and repository path, absent on non-root pages
    /// and on roots that cannot name them. Together they form the
    /// tracker identity the companion marks its session with
    /// (t-24@r16).
    pub machine_id: Option<String>,
    pub repo_path: Option<String>,
    pub content: Vec<Part>,
}

/// One part of a page document. Fields the companion never reads keep
/// their raw value; an unparseable part is skipped by the decoders.
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text {
        text: String,
        location: Option<String>,
    },
    Group {
        title: Option<String>,
        content: Vec<Part>,
        location: Option<String>,
    },
    Fact {
        label: String,
        value: String,
        location: Option<String>,
    },
    Time {
        at: Value,
        label: Option<String>,
        location: Option<String>,
    },
    Ref {
        location: String,
        label: String,
    },
    Figure {
        caption: Option<String>,
        file: String,
        location: Option<String>,
    },
}

/// A refusal document: kind refusal, a fact naming the code, a text
/// naming what would make it answerable, plus the HTTP status it rode.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub code: String,
    pub detail: String,
    pub status: u16,
}

/// Decode one HTTP response body as a page document or a refusal
/// document. `status` is the response's own status: refusals are
/// informative-status documents, never manufactured pages.
pub fn decode(body: &str, status: u16) -> Result<Result<Page, Refusal>, String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("page is not JSON: {e}"))?;
    let kind = v
        .get("kind")
        .and_then(Value::as_str)
        .ok_or("page carries no kind")?
        .to_string();
    if kind == "refusal" {
        let code = v
            .get("code")
            .and_then(Value::as_str)
            .ok_or("refusal carries no code")?
            .to_string();
        let detail = v
            .get("content")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .find_map(|p| {
                        let text = p.get("text").and_then(Value::as_str)?;
                        (p.get("kind").and_then(Value::as_str) == Some("text"))
                            .then(|| text.to_string())
                    })
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        return Ok(Err(Refusal {
            code,
            detail,
            status,
        }));
    }
    let view_major = v
        .get("view")
        .and_then(|view| view.get("major"))
        .and_then(Value::as_u64)
        .ok_or("page carries no view major")? as u32;
    let location = v
        .get("location")
        .and_then(Value::as_str)
        .ok_or("page carries no location")?
        .to_string();
    let seq = v.get("seq").and_then(Value::as_u64);
    let tracker = v.get("tracker").and_then(Value::as_str).map(str::to_string);
    let machine_id = v
        .get("machine_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    let repo_path = v
        .get("repo_path")
        .and_then(Value::as_str)
        .map(str::to_string);
    let content = v
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| parts.iter().filter_map(parse_part).collect())
        .unwrap_or_default();
    Ok(Ok(Page {
        view_major,
        kind,
        location,
        seq,
        tracker,
        machine_id,
        repo_path,
        content,
    }))
}

fn parse_part(v: &Value) -> Option<Part> {
    let kind = v.get("kind").and_then(Value::as_str)?;
    let location = |v: &Value| {
        v.get("location")
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    match kind {
        "text" => Some(Part::Text {
            text: v.get("text").and_then(Value::as_str)?.to_string(),
            location: location(v),
        }),
        "group" => Some(Part::Group {
            title: v.get("title").and_then(Value::as_str).map(str::to_string),
            content: v
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| parts.iter().filter_map(parse_part).collect())
                .unwrap_or_default(),
            location: location(v),
        }),
        "fact" => Some(Part::Fact {
            label: v.get("label").and_then(Value::as_str)?.to_string(),
            value: v.get("value").and_then(Value::as_str)?.to_string(),
            location: location(v),
        }),
        "time" => Some(Part::Time {
            at: v.get("at").cloned()?,
            label: v.get("label").and_then(Value::as_str).map(str::to_string),
            location: location(v),
        }),
        "ref" => Some(Part::Ref {
            location: v.get("location").and_then(Value::as_str)?.to_string(),
            label: v.get("label").and_then(Value::as_str)?.to_string(),
        }),
        "figure" => Some(Part::Figure {
            caption: v.get("caption").and_then(Value::as_str).map(str::to_string),
            file: v.get("file").and_then(Value::as_str)?.to_string(),
            location: location(v),
        }),
        // a kind this build does not know renders nothing here; the
        // page still decodes
        _ => None,
    }
}

/// One enrolled session as the agents page carries it: the
/// incarnation, its terminal address, harness, owned tasks (latest
/// claim first) and workspace.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentRow {
    pub incarnation: String,
    pub address: Option<String>,
    pub harness: Option<String>,
    pub tasks: Vec<String>,
    pub workspace: Option<String>,
}

/// Pull the enrolled sessions out of an agents page: one group per
/// session, its location the incarnation (`i-N`), facts inside named by
/// label. Groups that are not session items, and facts without a known
/// label, are ignored.
pub fn agents(page: &Page) -> Vec<AgentRow> {
    let mut rows = Vec::new();
    for part in &page.content {
        let Part::Group {
            content,
            location: Some(location),
            ..
        } = part
        else {
            continue;
        };
        let Some(incarnation) = incarnation_of(location) else {
            continue;
        };
        let mut row = AgentRow {
            incarnation,
            address: None,
            harness: None,
            tasks: Vec::new(),
            workspace: None,
        };
        for part in content {
            let Part::Fact { label, value, .. } = part else {
                continue;
            };
            match label.as_str() {
                "terminal address" => row.address = Some(value.clone()),
                "harness" => row.harness = Some(value.clone()),
                "tasks" => {
                    row.tasks = value.split_whitespace().map(str::to_string).collect();
                }
                "workspace" => row.workspace = Some(value.clone()),
                _ => {}
            }
        }
        rows.push(row);
    }
    rows
}

/// The incarnation a session item's location names: `i-N`.
fn incarnation_of(location: &str) -> Option<String> {
    let rest = location.strip_prefix("i-")?;
    (!rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())).then(|| format!("i-{rest}"))
}

#[cfg(test)]
mod test {
    use super::*;

    // the envelope and part shapes, as the tracker's page framework
    // emits them (the agreed board page literal)
    const BOARD_PAGE: &str = r#"{
      "view": {"major": 1},
      "kind": "document",
      "location": "t-14@r3",
      "seq": 512,
      "content": [
        {"kind": "fact", "label": "task", "value": "t-14"},
        {"kind": "ref", "location": "t-14/history", "label": "history"},
        {"kind": "text", "text": "authored markdown", "location": "t-14@r3:/content/0"},
        {"kind": "group", "title": "Recorded setup", "location": "t-14@r3:/content/1",
         "content": [{"kind": "text", "text": "...", "location": "t-14@r3:/content/1/group/content/0"}]},
        {"kind": "figure", "caption": "Section.", "file": "t-14@r3/file/figures/plinth.png",
         "location": "t-14@r3:/content/2"},
        {"kind": "time", "at": "2026-11-08T12:00:00Z", "label": "opened", "location": "t-14@r3:/content/3"},
        {"kind": "speculative-kind", "of": "a later minor"}
      ]
    }"#;

    // the agents page as t-20's tracker half serves it: one group per
    // enrolled session, location the incarnation, facts by label
    const AGENTS_PAGE: &str = r#"{
      "view": {"major": 1},
      "kind": "document",
      "location": "agents",
      "seq": 540,
      "content": [
        {"kind": "group", "location": "i-12", "content": [
          {"kind": "fact", "label": "incarnation", "value": "i-12"},
          {"kind": "fact", "label": "terminal address", "value": "t68"},
          {"kind": "fact", "label": "harness", "value": "pi"},
          {"kind": "fact", "label": "tasks", "value": "t-20"},
          {"kind": "fact", "label": "workspace", "value": "/home/jerry/Projects/saccade"}
        ]},
        {"kind": "group", "location": "i-10", "content": [
          {"kind": "fact", "label": "terminal address", "value": "t69"},
          {"kind": "fact", "label": "harness", "value": "pi"},
          {"kind": "fact", "label": "tasks", "value": "t-15 t-12"}
        ]},
        {"kind": "group", "location": "i-3", "content": [
          {"kind": "fact", "label": "terminal address", "value": "machine/t6b"},
          {"kind": "fact", "label": "harness", "value": "pi"}
        ]},
        {"kind": "group", "title": "not a session", "location": "t-14@r3:/content/1", "content": [
          {"kind": "fact", "label": "terminal address", "value": "t7"}
        ]}
      ]
    }"#;

    #[test]
    fn the_agreed_envelope_and_kinds_decode() {
        let page = decode(BOARD_PAGE, 200).unwrap().unwrap();
        assert_eq!(page.view_major, 1);
        assert_eq!(page.kind, "document");
        assert_eq!(page.location, "t-14@r3");
        assert_eq!(page.seq, Some(512));
        assert_eq!(page.content.len(), 6, "the unknown kind is skipped");
        assert_eq!(
            page.content[0],
            Part::Fact {
                label: "task".into(),
                value: "t-14".into(),
                location: None,
            }
        );
        assert_eq!(
            page.content[5],
            Part::Time {
                at: serde_json::json!("2026-11-08T12:00:00Z"),
                label: Some("opened".into()),
                location: Some("t-14@r3:/content/3".into()),
            }
        );
    }

    #[test]
    fn the_agents_page_yields_the_enrolled_sessions() {
        let page = decode(AGENTS_PAGE, 200).unwrap().unwrap();
        let rows = agents(&page);
        assert_eq!(rows.len(), 3, "only i- locations are session items");
        assert_eq!(rows[0].incarnation, "i-12");
        assert_eq!(rows[0].address.as_deref(), Some("t68"));
        assert_eq!(rows[0].harness.as_deref(), Some("pi"));
        assert_eq!(rows[0].tasks, vec!["t-20"]);
        assert_eq!(
            rows[0].workspace.as_deref(),
            Some("/home/jerry/Projects/saccade")
        );
        assert_eq!(rows[1].tasks, vec!["t-15", "t-12"], "latest claim first");
        assert_eq!(rows[1].workspace, None, "un derivable workspace is absent");
        assert_eq!(
            rows[2].address.as_deref(),
            Some("machine/t6b"),
            "the record's address verbatim, machine prefix included"
        );
    }

    #[test]
    fn a_refusal_document_decodes_with_its_code_and_detail() {
        let body = r#"{
          "view": {"major": 1},
          "kind": "refusal",
          "location": "no-such-page",
          "seq": null,
          "code": "unknown_location",
          "content": [
            {"kind": "fact", "label": "refusal", "value": "unknown_location"},
            {"kind": "text", "text": "a served location is named at /view/v1/"}
          ]
        }"#;
        let refusal = decode(body, 404).unwrap().unwrap_err();
        assert_eq!(refusal.code, "unknown_location");
        assert_eq!(refusal.detail, "a served location is named at /view/v1/");
        assert_eq!(refusal.status, 404);
    }

    #[test]
    fn the_root_page_names_the_tracker_when_it_serves_one() {
        let page = decode(
            r#"{"view":{"major":1},"kind":"root","location":"","seq":7,"tracker":"saccade","machine_id":"b1f7c2e0","repo_path":"/home/j/saccade","content":[
                 {"kind":"fact","label":"view majors","value":"1"}]}"#,
            200,
        )
        .unwrap()
        .unwrap();
        assert_eq!(page.tracker.as_deref(), Some("saccade"));
        assert_eq!(page.machine_id.as_deref(), Some("b1f7c2e0"));
        assert_eq!(page.repo_path.as_deref(), Some("/home/j/saccade"));
        let page = decode(
            r#"{"view":{"major":1},"kind":"root","location":"","seq":7,"content":[]}"#,
            200,
        )
        .unwrap()
        .unwrap();
        assert_eq!(page.tracker, None, "a --db server names no repo");
    }
}
