//! The console: task rail, focused thread, docked compose.
//! Every section is a renderer
//! over view types; fragments are sections, so the compose fetch swaps
//! what the page itself renders. The @-compiler is the composer's input
//! syntax over Commented events - the parse lives here, never in the
//! record. So does the prose renderer: bodies and explanations parse
//! as markdown with math on the way out, never in.

use crate::views::RefTarget;
use math_core::{LatexToMathML, MathDisplay};
use pulldown_cmark::{Alignment, Event as MdEvent, Options, Parser, Tag, TagEnd};
use std::io::Read;
use std::path::PathBuf;

/// The artifact store's render-side face: where the bytes live, whether
/// they are present, and whether they sniff as an image. The one sniff
/// definition - the read door serves the same content types it inlines.
#[derive(Clone, Default)]
pub struct ArtifactStore {
    dir: Option<PathBuf>,
}

impl ArtifactStore {
    pub fn at(dir: PathBuf) -> Self {
        ArtifactStore { dir: Some(dir) }
    }

    /// One artifact's bytes in the store.
    pub fn path(&self, hash: &str) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join(hash))
    }

    /// Some(true): present and an image. Some(false): present, other
    /// content. None: absent - the honest miss.
    pub fn probe(&self, hash: &str) -> Option<bool> {
        let mut prefix = [0u8; 16];
        let read = std::fs::File::open(self.path(hash)?)
            .ok()?
            .read(&mut prefix)
            .ok()?;
        Some(sniff_image(&prefix[..read]).is_some())
    }
}

/// Content type by magic bytes - the formats the console inlines.
pub fn sniff_image(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

/// The task state's chip color, one per state out of the c-522 palette:
/// open whispers, claimed is gold like the work it holds, delivered is
/// the link tone like the hand-off it awaits, done is green like a
/// settled run, dropped is rose like the human act it was.
fn mention_card(seq: usize, task: usize, name: &str, hash: &str, store: &ArtifactStore) -> String {
    let home = format!("/t/{task}#a-{seq}");
    let caption = esc(name);
    let short: String = hash.chars().take(12).collect();
    match store.probe(hash) {
        Some(true) => format!(
            "<a class=\"acard\" href=\"{home}\"><img src=\"/a/{hash}\" alt=\"{caption}\"><span class=\"acap\">{caption}</span></a>"
        ),
        Some(false) => format!(
            "<a class=\"acard file\" href=\"{home}\"><span class=\"acap\">{caption} | {short}</span></a>"
        ),
        None => format!(
            "<a class=\"acard miss\" href=\"{home}\"><span class=\"acap\">{caption} | unavailable</span></a>"
        ),
    }
}

/// The reading-side resolver over escaped text: a `[^k]` marker whose
/// comment carries a reference at that position resolves - a comment
/// to its home anchor, an artifact to its card, a task or incarnation
/// birth to its thread; every other token stays the words it was.
/// Bare c-N tokens are prose: navigation follows the mapping, never
/// the mention. The parse lives here, never in the record.
fn markerize(escaped: &str, refs: &[RefTarget], store: &ArtifactStore) -> String {
    let mut out = String::with_capacity(escaped.len());
    let mut at = 0usize;
    for token in crate::markers::tokens_in_run(escaped) {
        // a number the mapping cannot take stays the words it was
        let resolved = token
            .number
            .and_then(|k| refs.get(k - 1))
            .filter(|_| token.number.is_some_and(|k| k >= 1));
        match resolved {
            Some(target) => {
                out.push_str(&escaped[at..token.start]);
                out.push_str(&marker_link(token.number.unwrap_or(0), target, store));
                at = token.end;
            }
            None => continue,
        }
    }
    out.push_str(&escaped[at..]);
    out
}

/// One resolved marker's render: a superscript link for records, the
/// card for an artifact.
fn marker_link(k: usize, target: &RefTarget, store: &ArtifactStore) -> String {
    match target {
        RefTarget::Comment { num, task } => {
            format!("<a class=\"mref\" href=\"/t/{task}#c-{num}\">[^{k}]</a>")
        }
        RefTarget::Task { task } => {
            format!("<a class=\"mref\" href=\"/t/{task}\">[^{k}]</a>")
        }
        RefTarget::Incarnation { num, task } => match task {
            Some(task) => {
                format!("<a class=\"mref\" href=\"/t/{task}\" title=\"i-{num}\">[^{k}]</a>")
            }
            None => format!("<span class=\"mref dead\" title=\"i-{num}\">[^{k}]</span>"),
        },
        RefTarget::Message { num } => {
            format!("<span class=\"mref dead\" title=\"m-{num}\">[^{k}]</span>")
        }
        RefTarget::Artifact {
            seq,
            task,
            name,
            hash,
        } => format!(
            "<span class=\"mrefa\">{}<sup><a class=\"mref\" href=\"/t/{task}#a-{seq}\">[^{k}]</a></sup></span>",
            mention_card(*seq, *task, name, hash, store)
        ),
        RefTarget::Record { seq, task } => match task {
            Some(task) => format!("<a class=\"mref\" href=\"/t/{task}\">[^{k}]</a>"),
            None => format!("<span class=\"mref dead\">[^{k}] | seq {seq}</span>"),
        },
    }
}

/// The one scheme door for a markdown link: http or https renders as a
/// link; every other scheme - and no scheme - renders as words.
fn http_s(dest: &str) -> bool {
    let d = dest.as_bytes();
    let head = |p: &[u8]| d.len() >= p.len() && d[..p.len()].eq_ignore_ascii_case(p);
    head(b"http://") || head(b"https://")
}

/// A conversion that fails - or markup beyond the crate's own
/// vocabulary - renders no math at all; the fence renderers fall back
/// to the literal source. The conversion is unwound by catch: a parser
/// panic on legal latex is a render miss, never a dead request.
fn math_mml(src: &str, display: MathDisplay) -> Option<String> {
    let converted = LatexToMathML::new(Default::default()).ok().and_then(|c| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            c.convert_with_local_state(src, display)
                .ok()
                .map(|r| r.mathml)
        }))
        .ok()
        .flatten()
    });
    let m = converted.filter(|m| crate_authored(m))?;
    // the return path: a click on the formula copies these bytes
    let raw = esc(src);
    Some(m.replacen(
        "<math",
        &format!("<math data-raw=\"{raw}\" title=\"{raw}\""),
        1,
    ))
}

fn math_html(src: &str, fence: &str, display: MathDisplay) -> String {
    match math_mml(src, display) {
        Some(m) => m,
        None => format!("{fence}{}{fence}", esc(src)),
    }
}

/// The converter renders author bytes into markup; only its own tag
/// vocabulary may pass, so any other tag shape is a miss, never markup.
fn crate_authored(mathml: &str) -> bool {
    const TAGS: &[&str] = &[
        "math",
        "mi",
        "mn",
        "mo",
        "mtext",
        "mspace",
        "mpadded",
        "mrow",
        "mfrac",
        "msqrt",
        "mroot",
        "msub",
        "msup",
        "msubsup",
        "mmultiscripts",
        "mprescripts",
        "mover",
        "munder",
        "munderover",
        "menclose",
        "merror",
        "mtable",
        "mtr",
        "mtd",
    ];
    let b = mathml.as_bytes();
    let mut i = 0;
    while let Some(at) = b[i..].iter().position(|&c| c == b'<') {
        let open = i + at + 1;
        let open = if open < b.len() && b[open] == b'/' {
            open + 1
        } else {
            open
        };
        let end = b[open..]
            .iter()
            .position(|&c| !c.is_ascii_alphanumeric())
            .map_or(b.len(), |p| open + p);
        let name = std::str::from_utf8(&b[open..end]).unwrap_or("");
        if !TAGS.contains(&name) {
            return false;
        }
        i = end;
    }
    true
}

/// The prose renderer over raw bytes: CommonMark with tables and
/// math, the same law as the @-compiler and the marker resolver - the
/// parse lives here, never in the record. Text events escape and
/// resolve `[^k]` markers through the given mapping; code spans and
/// code blocks are verbatim; raw HTML never passes, rendering as the
/// words it was; a link renders only on the http(s) schemes with
/// rel=noopener; soft breaks are whitespace and hard breaks the
/// CommonMark kinds.
pub(crate) fn prose_html(raw: &str, refs: &[RefTarget], store: &ArtifactStore) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES | Options::ENABLE_MATH);
    let mut out = String::with_capacity(raw.len());
    let mut in_code = false;
    let mut code_buf = String::new();
    let mut in_head = false;
    let mut link_open = false;
    let mut aligns: Vec<Alignment> = Vec::new();
    let mut col = 0;
    // text accumulates until the next event: a marker's brackets can
    // span the parser's own text splits, so the resolver reads whole runs
    let mut text_buf = String::new();
    let flush_text = |buf: &mut String, out: &mut String| {
        if !buf.is_empty() {
            out.push_str(&markerize(buf, refs, store));
            buf.clear();
        }
    };
    for ev in Parser::new_ext(raw, opts) {
        if !matches!(ev, MdEvent::Text(_)) {
            flush_text(&mut text_buf, &mut out);
        }
        match ev {
            MdEvent::Start(tag) => match tag {
                Tag::Paragraph => out.push_str("<p>"),
                Tag::Heading { level, .. } => out.push_str(&format!("<h{}>", level as u8)),
                Tag::BlockQuote(_) => out.push_str("<blockquote>\n"),
                Tag::CodeBlock(_) => {
                    in_code = true;
                    code_buf.clear();
                }
                Tag::List(None) => out.push_str("<ul>\n"),
                Tag::List(Some(1)) => out.push_str("<ol>\n"),
                Tag::List(Some(start)) => out.push_str(&format!("<ol start=\"{start}\">\n")),
                Tag::Item => out.push_str("<li>"),
                Tag::Table(align) => {
                    aligns = align;
                    out.push_str("<table>\n");
                }
                Tag::TableHead => {
                    in_head = true;
                    col = 0;
                    out.push_str("<thead><tr>");
                }
                Tag::TableRow => {
                    col = 0;
                    out.push_str("<tr>");
                }
                Tag::TableCell => {
                    let style = match aligns.get(col).copied().unwrap_or(Alignment::None) {
                        Alignment::None => "",
                        Alignment::Left => " style=\"text-align:left\"",
                        Alignment::Center => " style=\"text-align:center\"",
                        Alignment::Right => " style=\"text-align:right\"",
                    };
                    let cell = if in_head {
                        format!("<th{style}>")
                    } else {
                        format!("<td{style}>")
                    };
                    out.push_str(&cell);
                    col += 1;
                }
                Tag::Emphasis => out.push_str("<em>"),
                Tag::Strong => out.push_str("<strong>"),
                Tag::Link {
                    dest_url, title, ..
                } if http_s(&dest_url) => {
                    link_open = true;
                    let t = if title.is_empty() {
                        String::new()
                    } else {
                        format!(" title=\"{}\"", esc(&title))
                    };
                    out.push_str(&format!(
                        "<a href=\"{}\" rel=\"noopener\"{t}>",
                        esc(&dest_url)
                    ));
                }
                _ => {}
            },
            MdEvent::End(end) => match end {
                TagEnd::Paragraph => out.push_str("</p>\n"),
                TagEnd::Heading(level) => out.push_str(&format!("</h{}>\n", level as u8)),
                TagEnd::BlockQuote(_) => out.push_str("</blockquote>\n"),
                TagEnd::CodeBlock => {
                    in_code = false;
                    // the whole block's bytes ride the pre for the copy button
                    out.push_str(&format!(
                        "<pre data-raw=\"{}\"><code>{}</code></pre>\n",
                        esc(&code_buf),
                        esc(&code_buf)
                    ));
                }
                TagEnd::List(ordered) => out.push_str(if ordered { "</ol>\n" } else { "</ul>\n" }),
                TagEnd::Item => out.push_str("</li>\n"),
                TagEnd::Table => {
                    aligns.clear();
                    out.push_str("</table>\n");
                }
                TagEnd::TableHead => {
                    in_head = false;
                    out.push_str("</tr></thead>\n");
                }
                TagEnd::TableRow => out.push_str("</tr>\n"),
                TagEnd::TableCell => {
                    out.push_str(if in_head { "</th>" } else { "</td>" });
                }
                TagEnd::Emphasis => out.push_str("</em>"),
                TagEnd::Strong => out.push_str("</strong>"),
                TagEnd::Link if link_open => {
                    link_open = false;
                    out.push_str("</a>");
                }
                _ => {}
            },
            MdEvent::Text(t) => {
                if in_code {
                    code_buf.push_str(&t);
                } else {
                    text_buf.push_str(&esc(&t));
                }
            }
            MdEvent::Code(c) => out.push_str(&format!("<code>{}</code>", esc(&c))),
            MdEvent::InlineMath(src) => out.push_str(&math_html(&src, "$", MathDisplay::Inline)),
            MdEvent::DisplayMath(src) => match math_mml(&src, MathDisplay::Block) {
                Some(m) => out.push_str(&format!("<span class=\"dmath\">{m}</span>")),
                // a miss is literal bytes, not a broken block
                None => out.push_str(&format!("$${}$$", esc(&src))),
            },
            MdEvent::Html(h) | MdEvent::InlineHtml(h) => out.push_str(&esc(&h)),
            MdEvent::SoftBreak => out.push('\n'),
            MdEvent::HardBreak => out.push_str("<br />\n"),
            MdEvent::Rule => out.push_str("<hr>\n"),
            _ => {}
        }
    }
    flush_text(&mut text_buf, &mut out);
    out
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommentId;
    use crate::CommentKind;
    use crate::Prose;
    use crate::events::Event;
    use crate::store::{Context, Record, Tier, World};
    use crate::types::actor::ActorName;
    use crate::{RecordId, TaskId};
    use std::sync::OnceLock;

    fn human() -> &'static Context {
        static H: OnceLock<Context> = OnceLock::new();
        H.get_or_init(|| Context {
            actor: ActorName::new("jerry".into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        })
    }

    fn agent() -> &'static Context {
        static A: OnceLock<Context> = OnceLock::new();
        A.get_or_init(|| Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope: None,
        })
    }

    fn record(seq: usize, at: u64, ctx: &Context, event: Event) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: at,
            context: ctx.clone(),
            event,
        }
    }

    fn note(title: &str, body: &str) -> CommentKind {
        CommentKind::Note {
            title: Prose::new(title.into()).unwrap(),
            body: Prose::new(body.into()).unwrap(),
        }
    }

    fn note_on(
        seq: usize,
        at: u64,
        ctx: &Context,
        home: TaskId,
        title: &str,
        body: &str,
        refs: Vec<RecordId>,
    ) -> Record {
        record(
            seq,
            at,
            ctx,
            Event::Commented {
                home,
                kind: note(title, body),
                references: refs,
            },
        )
    }

    fn offer_kind(title: &str, body: &str) -> CommentKind {
        CommentKind::Offer {
            title: Prose::new(title.into()).unwrap(),
            body: Prose::new(body.into()).unwrap(),
        }
    }

    // ---- the @-compiler's equivalence classes ----

    const PNG_HEAD: &[u8] = b"\x89PNG\r\n\x1a\nrest-of-the-bytes";

    fn store_of(contents: &[&[u8]], tag: &str) -> (ArtifactStore, std::path::PathBuf) {
        let dir = std::env::temp_dir()
            .join(format!("sac-web-store-{}", std::process::id()))
            .join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for bytes in contents {
            std::fs::write(dir.join(crate::ContentHash::of(bytes).as_str()), bytes).unwrap();
        }
        (ArtifactStore::at(dir.clone()), dir)
    }

    /// t-0 holds the figure, then an offer citing it through a marker,
    /// then its accept; t-1's note mentions ids in prose only.
    fn resolver_world() -> World {
        World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            record(
                1,
                1,
                human(),
                Event::ArtifactAdded {
                    root: TaskId(0),
                    artifact: crate::types::artifact::Artifact {
                        name: Prose::new("sweep figure".into()).unwrap(),
                        hash: crate::ContentHash::of(PNG_HEAD),
                    },
                },
            ),
            record(
                2,
                2,
                human(),
                Event::Commented {
                    home: TaskId(0),
                    kind: offer_kind("suite green", "suite green, per the figure [^1]"),
                    references: vec![RecordId(1)],
                },
            ),
            record(
                3,
                3,
                human(),
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Accept {
                        offer: CommentId(RecordId(2)),
                    },
                    references: Vec::new(),
                },
            ),
            record(
                5,
                5,
                human(),
                Event::TaskCreated {
                    name: Prose::new("other work".into()).unwrap(),
                },
            ),
            note_on(
                6,
                6,
                human(),
                TaskId(1),
                "bare ids stay words",
                "see c-0 and the figure - no links, just words; c-99 unresolved",
                Vec::new(),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn markers_resolve_to_links_and_cards_or_stay_text() {
        let world = resolver_world();
        let (store, dir) = store_of(&[PNG_HEAD], "png");
        let hash = crate::ContentHash::of(PNG_HEAD).as_str().to_string();

        // the citing offer: a marker names the artifact, the card rides
        let html = note_html(&world, 0, &store);
        assert!(
            html.contains("<a class=\"mref\" href=\"/t/0#a-1\">[^1]</a>"),
            "{html}"
        );
        assert!(
            html.contains(&format!(
                "<a class=\"acard\" href=\"/t/0#a-1\"><img src=\"/a/{hash}\" alt=\"sweep figure\"><span class=\"acap\">sweep figure</span></a>"
            )),
            "{html}"
        );

        // the foreign body: bare ids are prose, no edges, no links
        let html = note_html(&world, 1, &store);
        assert!(html.contains("see c-0 and the figure"), "{html}");
        assert!(!html.contains("c-0</a>"), "{html}");
        assert!(!html.contains("c-99</a>"), "{html}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A marker naming a comment links its home anchor; one naming a
    /// task birth links the thread; a marker with no mapping stays the
    /// words it was.
    #[test]
    fn markers_name_comments_tasks_and_stay_words_without_a_mapping() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(
                1,
                1,
                human(),
                TaskId(0),
                "the finding",
                "the finding",
                Vec::new(),
            ),
            note_on(
                2,
                2,
                human(),
                TaskId(0),
                "the citations",
                "the comment [^1], the task [^2], the void [^3]",
                vec![RecordId(1), RecordId(0)],
            ),
        ])
        .unwrap();
        let (store, dir) = store_of(&[], "markers");
        let html = note_html(&world, 0, &store);
        assert!(
            html.contains("<a class=\"mref\" href=\"/t/0#c-0\">[^1]</a>"),
            "{html}"
        );
        assert!(
            html.contains("<a class=\"mref\" href=\"/t/0\">[^2]</a>"),
            "{html}"
        );
        // a marker the mapping cannot resolve stays literal words
        assert!(html.contains("the void [^3]"), "{html}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_resolver_escapes_before_it_links() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(
                1,
                1,
                human(),
                TaskId(0),
                "the bold verdict",
                "the <b>bold</b> verdict & [^1]",
                vec![RecordId(0)],
            ),
        ])
        .unwrap();
        let (store, dir) = store_of(&[], "escape");
        let html = note_html(&world, 0, &store);
        // the words escaped, and the marker still resolved
        assert!(
            html.contains(
                "the &lt;b&gt;bold&lt;/b&gt; verdict &amp; <a class=\"mref\" href=\"/t/0\">[^1]</a>"
            ),
            "{html}"
        );
        assert!(!html.contains("<b>"), "{html}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn prose_world(body: &str) -> World {
        World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(1, 1, human(), TaskId(0), "the finding", body, Vec::new()),
        ])
        .unwrap()
    }

    fn prose_html_of(world: &World) -> String {
        note_html(world, 0, &ArtifactStore::default())
    }

    /// Every comment on one home rendered whole: the canonical renderer
    /// over each body and mapping, the reading the marker tests pin.
    fn note_html(world: &World, home: usize, store: &ArtifactStore) -> String {
        use crate::views::{ThreadItem, thread_view};
        let thread = thread_view(world, TaskId(home)).expect("the home folds");
        thread
            .items
            .into_iter()
            .filter_map(|item| match item {
                ThreadItem::Note(line) => Some(prose_html(&line.body, &line.references, store)),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn math_renders_to_inline_mathml_and_a_miss_keeps_the_source() {
        let html = prose_html_of(&prose_world(
            "the loss $x^2+1$ and a miss $\\notacommand{x}$ stay honest",
        ));
        assert!(html.contains("<math"), "{html}");
        assert!(html.contains("msup"), "{html}");
        // a parse that fails keeps its source bytes
        assert!(html.contains("$\\notacommand{x}$"), "{html}");
    }

    #[test]
    fn display_math_renders_a_block_and_a_miss_keeps_its_fences() {
        let html = prose_html_of(&prose_world("sets it out: $$a^2+b^2=c^2$$ done"));
        // display math is its own block, distinct from inline
        assert!(
            html.contains(
                "<span class=\"dmath\"><math data-raw=\"a^2+b^2=c^2\" title=\"a^2+b^2=c^2\" display=\"block\">"
            ),
            "{html}"
        );
        // a display miss keeps both fences and renders no block
        let html = prose_html_of(&prose_world("$$\\notacommand{x}$$"));
        assert!(html.contains("$$\\notacommand{x}$$"), "{html}");
        assert!(!html.contains("dmath"), "{html}");
    }

    #[test]
    fn aligned_derivations_render_inside_display_math() {
        let body = "$$\\begin{aligned}\\sigma \\odot w_S &= 0\\\\G &= X^\\top X\\end{aligned}$$";
        let html = prose_html_of(&prose_world(body));
        assert!(html.contains("<mtable displaystyle=\"true\""), "{html}");
        assert_eq!(html.matches("<mtr>").count(), 2, "{html}");
        assert!(html.contains("<mtd"), "{html}");
    }

    #[test]
    fn negative_space_after_a_scripted_group_renders() {
        let html = prose_html_of(&prose_world("$G_{SS}^{-1}\\!x$ holds"));
        assert!(
            html.contains("<mspace style=\"margin-left:-0.16666667em;\"/>"),
            "{html}"
        );
        // the author bytes ride the data-raw copy hook, not the text
        assert!(html.contains("title=\"G_{SS}^{-1}\\!x\""), "{html}");
        assert!(!html.contains("<p>$G"), "{html}");
    }

    #[test]
    fn markup_beyond_the_crate_vocabulary_stays_out() {
        // the allow-list, not the converter, decides what passes
        assert!(crate_authored("<math><mi>x</mi></math>"));
        assert!(!crate_authored(
            "<math><mi>x</mi><script>alert(1)</script></math>"
        ));
        assert!(!crate_authored("<math><foreignobject/></math>"));
    }

    #[test]
    fn code_spans_stay_verbatim() {
        let html = prose_html_of(&prose_world("run `sac show c-1 [^1]` now"));
        assert!(html.contains("<code>sac show c-1 [^1]</code>"), "{html}");
        assert!(!html.contains("<code>sac show <a"), "{html}");
    }

    #[test]
    fn code_blocks_carry_their_bytes_for_copy() {
        let html = prose_html_of(&prose_world("```rust\nlet x = 1;\n```"));
        assert!(
            html.contains("<pre data-raw=\"let x = 1;\n\"><code>let x = 1;\n</code></pre>"),
            "{html}"
        );
    }

    #[test]
    fn raw_html_never_passes() {
        let html = prose_html_of(&prose_world("<script>alert(1)</script> after"));
        assert!(html.contains("&lt;script&gt;"), "{html}");
        assert!(!html.contains("<script>"), "{html}");
    }

    #[test]
    fn links_render_only_on_http_s() {
        let html = prose_html_of(&prose_world(
            "[safe](https://example.com) and [evil](javascript:alert(1))",
        ));
        assert!(
            html.contains("<a href=\"https://example.com\" rel=\"noopener\">safe</a>"),
            "{html}"
        );
        assert!(!html.contains("href=\"javascript"), "{html}");
    }

    #[test]
    fn tables_render_aligned() {
        let html = prose_html_of(&prose_world(
            "| left | right |\n|-----|------:|\n| a   | 1    |",
        ));
        assert!(html.contains("<table>"), "{html}");
        assert!(html.contains("text-align:right"), "{html}");
    }
}
