use yaml_rust2::parser::{Event, MarkedEventReceiver, Parser};
use yaml_rust2::{ScanError, Yaml, YamlLoader};

use crate::error::{ErrorKind, ParseError, Where};
use crate::select::SelectPath;
use crate::{Board, Component, MAX_NESTING_DEPTH, SUPPORTED_GRAMMAR_VERSIONS, Source};

/// Parse board YAML into a typed board, or refuse with an error that says what to do.
pub fn parse(input: &str) -> Result<Board, ParseError> {
    let docs = YamlLoader::load_from_str(input).map_err(scan_error_to_parse_error)?;
    let root = match docs.as_slice() {
        [] => {
            return Err(ParseError {
                kind: ErrorKind::YamlSyntax,
                at: Where::Board,
                message: "the board text holds no YAML document; write the board mapping with grammar, task and content".into(),
            })
        }
        [one] => one,
        _ => return Err(multiple_documents(&Where::Board)),
    };
    check_stream(input)?;
    build_board(root)
}

fn syntax_error(error: ScanError) -> ParseError {
    let marker = error.marker();
    let at = yaml_at(marker);
    ParseError {
        kind: ErrorKind::YamlSyntax,
        at: at.clone(),
        message: format!(
            "the YAML is malformed at line {} column {}: {error}; fix the syntax so the board mapping parses",
            marker.line(),
            marker.col() + 1
        ),
    }
}

// the loader refuses duplicate keys itself; classify that scan error as its own kind
fn scan_error_to_parse_error(error: ScanError) -> ParseError {
    if error.info().contains("duplicated key") {
        let marker = error.marker();
        return ParseError {
            kind: ErrorKind::DuplicateKey,
            at: yaml_at(marker),
            message: format!("a mapping carries the same key twice ({error}); give each key once"),
        };
    }
    syntax_error(error)
}

fn multiple_documents(at: &Where) -> ParseError {
    ParseError {
        kind: ErrorKind::MultipleDocuments,
        at: at.clone(),
        message: "the file holds several YAML documents; keep exactly one board in the file".into(),
    }
}

fn yaml_at(marker: &yaml_rust2::scanner::Marker) -> Where {
    Where::Yaml {
        line: marker.line(),
        column: marker.col() + 1,
    }
}

fn anchor_or_alias(at: Where, found: &str) -> ParseError {
    ParseError {
        kind: ErrorKind::AnchorOrAlias,
        at,
        message: format!(
            "the board uses {found} in the YAML; write the content out in full instead"
        ),
    }
}

// One board states each piece of content once, so anchors, aliases, and
// repeated keys are refused before the tree is even read.
fn check_stream(input: &str) -> Result<(), ParseError> {
    enum Frame {
        Mapping {
            seen: Vec<String>,
            expecting_key: bool,
        },
        Sequence,
    }
    struct Collected {
        events: Vec<(Event, yaml_rust2::scanner::Marker)>,
    }
    impl MarkedEventReceiver for Collected {
        fn on_event(&mut self, event: Event, marker: yaml_rust2::scanner::Marker) {
            self.events.push((event, marker));
        }
    }
    let mut collected = Collected { events: Vec::new() };
    Parser::new_from_str(input)
        .load(&mut collected, true)
        .map_err(scan_error_to_parse_error)?;
    let mut frames: Vec<Frame> = Vec::new();
    let mut documents = 0usize;
    for (event, marker) in collected.events {
        match event {
            Event::DocumentStart => {
                documents += 1;
                if documents > 1 {
                    return Err(multiple_documents(&yaml_at(&marker)));
                }
            }
            Event::Alias(_) => return Err(anchor_or_alias(yaml_at(&marker), "an alias")),
            Event::Scalar(value, _, anchor, _) => {
                if anchor != 0 {
                    return Err(anchor_or_alias(yaml_at(&marker), "an anchor"));
                }
                if let Some(Frame::Mapping {
                    seen,
                    expecting_key,
                }) = frames.last_mut()
                {
                    if *expecting_key {
                        if seen.iter().any(|key| key == &value) {
                            return Err(ParseError {
                                kind: ErrorKind::DuplicateKey,
                                at: yaml_at(&marker),
                                message: format!(
                                    "the key '{value}' repeats in one mapping; give each key once"
                                ),
                            });
                        }
                        seen.push(value);
                        *expecting_key = false;
                    } else {
                        *expecting_key = true;
                    }
                }
            }
            Event::SequenceStart(anchor, _) | Event::MappingStart(anchor, _) => {
                if anchor != 0 {
                    return Err(anchor_or_alias(yaml_at(&marker), "an anchor"));
                }
                if let Some(Frame::Mapping {
                    expecting_key: true,
                    ..
                }) = frames.last()
                {
                    return Err(ParseError {
                        kind: ErrorKind::WrongType,
                        at: yaml_at(&marker),
                        message: "a mapping key here must be a plain name like 'content'; drop the collection sitting in the key position".into(),
                    });
                }
                frames.push(match event {
                    Event::MappingStart(..) => Frame::Mapping {
                        seen: Vec::new(),
                        expecting_key: true,
                    },
                    _ => Frame::Sequence,
                });
            }
            Event::SequenceEnd | Event::MappingEnd => {
                frames.pop();
                if let Some(Frame::Mapping { expecting_key, .. }) = frames.last_mut() {
                    *expecting_key = true;
                }
            }
            Event::Nothing | Event::StreamStart | Event::StreamEnd | Event::DocumentEnd => {}
        }
    }
    Ok(())
}

fn build_board(root: &Yaml) -> Result<Board, ParseError> {
    let map = root.as_hash().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Board,
        message: format!(
            "the board is a mapping with grammar, task and content at the top; found {}; restructure the file so those three fields sit at the top level",
            describe(root)
        ),
    })?;
    let mut grammar = None;
    let mut task = None;
    let mut content = None;
    let mut unknown = Vec::new();
    for (key, value) in map {
        let name = scalar_key(key).ok_or_else(|| ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Board,
            message: format!(
                "board field names are plain text like 'grammar'; found the key {}; rename it",
                describe(key)
            ),
        })?;
        match name.as_str() {
            "grammar" => grammar = Some(value),
            "task" => task = Some(value),
            "content" => content = Some(value),
            other => unknown.push(other.to_string()),
        }
    }
    if !unknown.is_empty() {
        let listed = unknown.join("', '");
        return Err(ParseError {
            kind: ErrorKind::UnknownField,
            at: Where::Board,
            message: format!(
                "unknown field '{listed}' at the board top; the board mapping takes only grammar, task and content; move that content into a component or drop the field"
            ),
        });
    }
    let grammar = grammar.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Board,
        message: "every board declares its grammar version; add 'grammar: 1' to the top mapping"
            .into(),
    })?;
    check_grammar(grammar)?;
    let task = task.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Board,
        message: "the board does not name its task; add 'task: <task name>' to the top mapping"
            .into(),
    })?;
    let task = plain_string(
        task,
        Where::Board,
        "the board's 'task' is the task's name as text",
    )?;
    let content = content.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Board,
        message: "the board has no 'content' list; add one holding the board's components".into(),
    })?;
    let items = content.as_vec().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Board,
        message: format!(
            "the board's 'content' is a list of components; found {}; write 'content:' with one component per list item",
            describe(content)
        ),
    })?;
    let mut components = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        components.push(build_component(item, &format!("/content/{index}"), 1)?);
    }
    Ok(Board {
        task,
        content: components,
    })
}

fn versions_text() -> String {
    let (low, high) = (
        *SUPPORTED_GRAMMAR_VERSIONS.start(),
        *SUPPORTED_GRAMMAR_VERSIONS.end(),
    );
    if low == high {
        format!("version {low}")
    } else {
        format!("versions {low} through {high}")
    }
}

fn check_grammar(value: &Yaml) -> Result<(), ParseError> {
    let supported = versions_text();
    match value {
        Yaml::Integer(n) if *n >= 0 && SUPPORTED_GRAMMAR_VERSIONS.contains(&(*n as u64)) => Ok(()),
        Yaml::Integer(n) => Err(ParseError {
            kind: ErrorKind::GrammarVersion,
            at: Where::Board,
            message: format!(
                "grammar {n} is outside the versions this validator parses ({supported}); write the board in grammar {}",
                SUPPORTED_GRAMMAR_VERSIONS.start()
            ),
        }),
        other => Err(ParseError {
            kind: ErrorKind::GrammarVersion,
            at: Where::Board,
            message: format!(
                "grammar must be the integer {}; found {}; change the line to 'grammar: {}'",
                SUPPORTED_GRAMMAR_VERSIONS.start(),
                describe(other),
                SUPPORTED_GRAMMAR_VERSIONS.start()
            ),
        }),
    }
}

fn build_component(value: &Yaml, path: &str, depth: usize) -> Result<Component, ParseError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(ParseError {
            kind: ErrorKind::TooDeep,
            at: Where::Path(path.into()),
            message: format!(
                "the components nest more than {MAX_NESTING_DEPTH} levels deep at {path}; flatten the nesting so a reader can follow the board"
            ),
        });
    }
    let map = value.as_hash().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Path(path.into()),
        message: format!(
            "the component at {path} is a mapping naming exactly one kind; found {}; write one of text, group, figure, incorporation, annotation",
            describe(value)
        ),
    })?;
    let mut named = Vec::with_capacity(map.len());
    for (key, body) in map {
        let name = scalar_key(key).ok_or_else(|| ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "component field names are the kind names as plain text; found the key {} at {path}",
                describe(key)
            ),
        })?;
        named.push((name, body));
    }
    if named.is_empty() {
        return Err(ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "the component at {path} names no kind; write exactly one of text, group, figure, incorporation, annotation"
            ),
        });
    }
    if named.len() > 1 {
        let listed = named
            .iter()
            .map(|(name, _)| format!("'{name}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "the component at {path} carries several kinds ({listed}); a component is exactly one kind; split them into separate list items"
            ),
        });
    }
    let (name, body) = &named[0];
    match name.as_str() {
        "text" => Ok(Component::Text {
            text: plain_string(
                body,
                Where::Path(path.into()),
                &format!("the text component at {path} is markdown as text"),
            )?,
        }),
        "annotation" => Ok(Component::Annotation {
            text: plain_string(
                body,
                Where::Path(path.into()),
                &format!("the annotation at {path} is the concern as markdown text"),
            )?,
        }),
        "group" => build_group(body, path, depth),
        "figure" => build_figure(body, path),
        "incorporation" => build_incorporation(body, path, depth),
        other => Err(ParseError {
            kind: ErrorKind::UnknownKind,
            at: Where::Path(path.into()),
            message: format!(
                "unknown component kind '{other}' at {path}; the five kinds are text, group, figure, incorporation, annotation; a new kind is admitted only by ratification"
            ),
        }),
    }
}

fn build_group(value: &Yaml, path: &str, depth: usize) -> Result<Component, ParseError> {
    let map = value.as_hash().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Path(path.into()),
        message: format!(
            "the group at {path} is a mapping with optional 'title' and a 'content' list; found {}",
            describe(value)
        ),
    })?;
    let mut title = None;
    let mut content = None;
    let mut unknown = Vec::new();
    for (key, body) in map {
        let name = scalar_key(key).ok_or_else(|| ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "group field names are plain text; found the key {} at {path}",
                describe(key)
            ),
        })?;
        match name.as_str() {
            "title" => title = Some(body),
            "content" => content = Some(body),
            other => unknown.push(other.to_string()),
        }
    }
    if !unknown.is_empty() {
        let listed = unknown.join("', '");
        return Err(ParseError {
            kind: ErrorKind::UnknownField,
            at: Where::Path(path.into()),
            message: format!(
                "the group at {path} takes optional 'title' and 'content' only; drop the field '{listed}'"
            ),
        });
    }
    let title = match title {
        Some(value) => Some(plain_string(
            value,
            Where::Path(path.into()),
            &format!("the group title at {path} is text"),
        )?),
        None => None,
    };
    let content = nested_content(content, path, "group", depth)?.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the group at {path} needs a 'content' list; add the components it groups"
        ),
    })?;
    Ok(Component::Group { title, content })
}

fn build_figure(value: &Yaml, path: &str) -> Result<Component, ParseError> {
    let map = value.as_hash().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Path(path.into()),
        message: format!(
            "the figure at {path} is a mapping with 'input' and 'caption'; found {}",
            describe(value)
        ),
    })?;
    let mut input = None;
    let mut caption = None;
    let mut unknown = Vec::new();
    for (key, body) in map {
        let name = scalar_key(key).ok_or_else(|| ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "figure field names are plain text; found the key {} at {path}",
                describe(key)
            ),
        })?;
        match name.as_str() {
            "input" => input = Some(body),
            "caption" => caption = Some(body),
            other => unknown.push(other.to_string()),
        }
    }
    if !unknown.is_empty() {
        let listed = unknown.join("', '");
        return Err(ParseError {
            kind: ErrorKind::UnknownField,
            at: Where::Path(path.into()),
            message: format!(
                "the figure at {path} takes only 'input' and 'caption'; drop the field '{listed}'"
            ),
        });
    }
    let input = input.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the figure at {path} needs 'input', the relative path of the file inside the board tree"
        ),
    })?;
    let input = plain_string(
        input,
        Where::Path(path.into()),
        &format!("the figure input at {path} is a relative path as text"),
    )?;
    check_figure_path(&input, path)?;
    let caption = caption.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the figure at {path} needs a 'caption'; write a line saying what the figure shows"
        ),
    })?;
    let caption = plain_string(
        caption,
        Where::Path(path.into()),
        &format!("the figure caption at {path} is text"),
    )?;
    Ok(Component::Figure { input, caption })
}

fn check_figure_path(input: &str, path: &str) -> Result<(), ParseError> {
    let bad = |message: String| ParseError {
        kind: ErrorKind::BadFigurePath,
        at: Where::Path(path.into()),
        message,
    };
    if input.is_empty() {
        return Err(bad(format!(
            "the figure at {path} names no file; give 'input' a relative path inside the board tree, like 'figures/plinth-section.png'"
        )));
    }
    if input.starts_with('/') {
        return Err(bad(format!(
            "the figure at {path} points at '{input}'; figure inputs are relative paths inside the board tree, like 'figures/plinth-section.png'"
        )));
    }
    if input.split('/').any(|segment| segment == "..") {
        return Err(bad(format!(
            "the figure at {path} walks out of the board tree with '..'; reference a file inside the board tree, like 'figures/plinth-section.png'"
        )));
    }
    Ok(())
}

fn build_incorporation(value: &Yaml, path: &str, depth: usize) -> Result<Component, ParseError> {
    let map = value.as_hash().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation at {path} is a mapping with 'source' and 'content'; found {}",
            describe(value)
        ),
    })?;
    let mut source = None;
    let mut content = None;
    let mut unknown = Vec::new();
    for (key, body) in map {
        let name = scalar_key(key).ok_or_else(|| ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "incorporation field names are plain text; found the key {} at {path}",
                describe(key)
            ),
        })?;
        match name.as_str() {
            "source" => source = Some(body),
            "content" => content = Some(body),
            other => unknown.push(other.to_string()),
        }
    }
    if !unknown.is_empty() {
        let listed = unknown.join("', '");
        return Err(ParseError {
            kind: ErrorKind::UnknownField,
            at: Where::Path(path.into()),
            message: format!(
                "the incorporation at {path} takes only 'source' and 'content'; drop the field '{listed}'"
            ),
        });
    }
    let source = source.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation at {path} cites no source; add 'source' with task, revision and select"
        ),
    })?;
    let source = build_source(source, path)?;
    let content = nested_content(content, path, "incorporation", depth)?.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation at {path} needs a 'content' list; add the text receiving what you take from the source"
        ),
    })?;
    if !has_substantive_text(&content) {
        return Err(ParseError {
            kind: ErrorKind::NoReceivingText,
            at: Where::Path(path.into()),
            message: format!(
                "incorporation at {path} has no text component; add a text explaining what you take from the source"
            ),
        });
    }
    Ok(Component::Incorporation { source, content })
}

fn build_source(value: &Yaml, path: &str) -> Result<Source, ParseError> {
    let map = value.as_hash().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation source at {path} is a mapping with task, revision and select; found {}",
            describe(value)
        ),
    })?;
    let mut task = None;
    let mut revision = None;
    let mut select = None;
    let mut unknown = Vec::new();
    for (key, body) in map {
        let name = scalar_key(key).ok_or_else(|| ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "source field names are plain text; found the key {} at {path}",
                describe(key)
            ),
        })?;
        match name.as_str() {
            "task" => task = Some(body),
            "revision" => revision = Some(body),
            "select" => select = Some(body),
            other => unknown.push(other.to_string()),
        }
    }
    if !unknown.is_empty() {
        let listed = unknown.join("', '");
        return Err(ParseError {
            kind: ErrorKind::UnknownField,
            at: Where::Path(path.into()),
            message: format!(
                "the incorporation source at {path} takes only task, revision and select; drop the field '{listed}'"
            ),
        });
    }
    let task = task.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation source at {path} does not name its task; add 'task: <task name>'"
        ),
    })?;
    let task = plain_string(
        task,
        Where::Path(path.into()),
        &format!("the source 'task' at {path} is the source task's name as text"),
    )?;
    let revision = revision.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation source at {path} pins no revision; add 'revision: <full commit hash>'"
        ),
    })?;
    let revision = plain_string(
        revision,
        Where::Path(path.into()),
        &format!("the source 'revision' at {path} is the commit hash as text"),
    )?;
    check_revision(&revision, path)?;
    let select = select.ok_or_else(|| ParseError {
        kind: ErrorKind::MissingField,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation source at {path} selects nothing; add 'select' listing the paths it takes, like [/content/0]"
        ),
    })?;
    let items = select.as_vec().ok_or_else(|| ParseError {
        kind: ErrorKind::WrongType,
        at: Where::Path(path.into()),
        message: format!(
            "the source 'select' at {path} is a list of paths; found {}",
            describe(select)
        ),
    })?;
    if items.is_empty() {
        return Err(ParseError {
            kind: ErrorKind::EmptySelect,
            at: Where::Path(path.into()),
            message: format!(
                "the incorporation at {path} selects nothing; list at least one path into the source board, like [/content/0]"
            ),
        });
    }
    let mut paths = Vec::with_capacity(items.len());
    for item in items {
        let written = plain_string(
            item,
            Where::Path(path.into()),
            "each 'select' entry is a path as text",
        )?;
        let selected = SelectPath::parse(&written).map_err(|error| ParseError {
            kind: ErrorKind::BadSelectPath,
            at: Where::Path(path.into()),
            message: format!("at {}: {}", path, error.message),
        })?;
        paths.push(selected);
    }
    Ok(Source {
        task,
        revision,
        select: paths,
    })
}

fn check_revision(revision: &str, path: &str) -> Result<(), ParseError> {
    let full = revision.len() == 40
        && revision
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    if full {
        return Ok(());
    }
    Err(ParseError {
        kind: ErrorKind::BadRevision,
        at: Where::Path(path.into()),
        message: format!(
            "the incorporation at {path} cites revision '{revision}'; cite the full 40-character lowercase hex commit of the source revision, as 'git rev-parse' prints it"
        ),
    })
}

// None means the field is absent; the caller turns that into its own refusal.
fn nested_content(
    field: Option<&Yaml>,
    path: &str,
    kind: &str,
    depth: usize,
) -> Result<Option<Vec<Component>>, ParseError> {
    let Some(value) = field else {
        return Ok(None);
    };
    let Some(items) = value.as_vec() else {
        return Err(ParseError {
            kind: ErrorKind::WrongType,
            at: Where::Path(path.into()),
            message: format!(
                "the {kind}'s 'content' at {path} is a list of components; found {}",
                describe(value)
            ),
        });
    };
    let mut content = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        content.push(build_component(
            item,
            &format!("{path}/{kind}/content/{index}"),
            depth + 1,
        )?);
    }
    Ok(Some(content))
}

fn has_substantive_text(content: &[Component]) -> bool {
    for component in content {
        match component {
            Component::Text { text } if !text.trim().is_empty() => return true,
            Component::Group { content, .. } | Component::Incorporation { content, .. }
                if has_substantive_text(content) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn plain_string(value: &Yaml, at: Where, what: &str) -> Result<String, ParseError> {
    let message = match value {
        Yaml::String(text) => return Ok(text.clone()),
        Yaml::Null => format!("{what} but is empty; write it as text"),
        other => format!("{what}; found {}", describe(other)),
    };
    Err(ParseError {
        kind: ErrorKind::WrongType,
        at,
        message,
    })
}

fn scalar_key(key: &Yaml) -> Option<String> {
    match key {
        Yaml::String(text) => Some(text.clone()),
        _ => None,
    }
}

fn describe(value: &Yaml) -> String {
    match value {
        Yaml::Null => "nothing (an empty value)".into(),
        Yaml::BadValue => "an unreadable value".into(),
        Yaml::Real(_) => "a decimal number".into(),
        Yaml::Integer(_) => "a number".into(),
        Yaml::String(text) => format!("the text \"{text}\""),
        Yaml::Boolean(_) => "a boolean".into(),
        Yaml::Array(_) => "a list".into(),
        Yaml::Hash(_) => "a mapping".into(),
        Yaml::Alias(_) => "a YAML alias".into(),
    }
}
