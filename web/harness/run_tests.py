import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from playwright.sync_api import sync_playwright
from server import Server
from editor_checks import run_editor_checks

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
RUNS.mkdir(exist_ok=True)
RUN = Path(tempfile.mkdtemp(prefix='browser-', dir=RUNS))
ENV = {'PATH': os.environ['PATH'] + ':/etc/profiles/per-user/jerry/bin:/run/current-system/sw/bin', 'HOME': str(RUN / 'scratch/home'),
       'TMPDIR': str(RUN / 'scratch/tmp'),
       'PLAYWRIGHT_BROWSERS_PATH': '/nix/store/9ibf73nllmrcqx4w7x5jayfxanv2gyfq-playwright-browsers'}
checks = []


def check(ok, name):
    checks.append({'name': name, 'passed': bool(ok)})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)


def request(url, data=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def state(page):
    return page.evaluate('window.workbenchUI.state()')


def draft(page):
    data = state(page)
    selected = next(p for p in data['panes'] if p['id'] == data['focus'])['bufferId']
    return next(b for b in data['buffers'] if b['id'] == selected)


def command(page, name):
    page.keyboard.press('Escape')
    page.keyboard.press(':')
    page.wait_for_function('document.querySelector("#picker").open')
    page.keyboard.type(name)
    page.keyboard.press('Enter')


def choose_id(page, name, beside=False):
    page.wait_for_function('document.querySelector("#picker").open')
    page.keyboard.type(name)
    page.keyboard.press('Shift+Enter' if beside else 'Enter')
    page.wait_for_function('!document.querySelector("#picker").open')


def leader(page, keys):
    page.keyboard.press('Escape')
    page.keyboard.press('Space')
    page.keyboard.type(keys)


def local(page, key):
    page.keyboard.press('Escape')
    page.keyboard.press(',')
    page.keyboard.type(key)


def finder(page, query):
    page.keyboard.press('Escape')
    page.keyboard.press('Space')
    page.keyboard.press('Space')
    page.wait_for_function('document.querySelector("#picker").open')
    page.keyboard.type(query)


def ready(page, base):
    page.goto(base)
    page.wait_for_function('window.workbenchUI !== undefined')


def prompt_ready(page):
    page.wait_for_function('document.querySelector("#picker").open')


def answer_prompt(page, text):
    prompt_ready(page)
    page.fill('#query', text)
    page.keyboard.press('Enter')
    page.wait_for_function('!document.querySelector("#picker").open')


def wait_submission(page, word):
    page.wait_for_function('(word) => document.querySelector("#message").textContent.includes(word)', arg=word)


def main():
    server = Server(RUN / 'scratch')
    browser = None
    page = None
    failed = None
    try:
        server.start()
        base = server.url
        status, original = request(base + '/api/v1/workbench')
        check(status == 200, 'canonical snapshot is served by the real core')
        check(original['comments'][1]['revised'] and 'Refernces' not in original['comments'][1]['body'], 'canonical revision projection replaces presentation text')
        withdrawn = next(c for c in original['comments'] if c['id'] == 'c-3')
        check(withdrawn['withdrawn'] and not withdrawn['body'] and withdrawn['title'] is None, 'withdrawn prose and title stay out of current content')
        check(next(c for c in original['comments'] if c['id'] == 'c-0')['cited_by'] == [2], 'backlinks reverse born explicit mappings')
        before = original['cursor']
        # the write door refuses attribution drift and mismatched cookies
        mismatch, body = request(base + '/api/v1/workbench/command',
                                 {'actor': 'jerry', 'command': {'create_task': {'name': 'not allowed'}}},
                                 headers={'Cookie': 'actor=someone-else'})
        check(mismatch == 409 and body['error']['code'] == 'actor_changed', 'the write door binds the submission to the claimed cookie')
        malformed, body = request(base + '/api/v1/workbench/command', {'command': {'create_task': {'name': 'no actor'}}})
        check(malformed == 400, 'a submission without an actor refuses')
        check(request(base + '/api/v1/workbench')[1]['cursor'] == before, 'door refusals change no records')
        check(next(c for c in original['comments'] if c['id'] == 'c-0')['title'] == 'The insertion bookmark', 'authored titles ride the projection')
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            ready(page, base)
            check(len(state(page)['panes']) == 1 and state(page)['mode'] == 'View', 'one home pane opens in rendered View')
            check(page.locator('.pane.active .home-activity .home-entry').count() > 0
                  and page.locator('.pane.active .home-tasks .home-entry').count() > 0,
                  'home renders recent activity and recently active tasks')
            page.screenshot(path=str(RUN / '01-home.png'))
            for axis, binding in [('beside', 'wv'), ('below', 'ws')]:
                for close_index in (0, 1):
                    leader(page, binding)
                    divider = page.locator('.divider'); rect = divider.bounding_box()
                    x, y = rect['x'] + rect['width'] / 2, rect['y'] + rect['height'] / 2
                    page.mouse.move(x, y); page.mouse.down()
                    page.mouse.move(x + (90 if axis == 'beside' else 0), y + (70 if axis == 'below' else 0)); page.mouse.up()
                    page.locator('.pane').nth(close_index).locator('.buffer-title').click()
                    command(page, 'close')
                    full = page.evaluate('() => { const p = document.querySelector(".pane").getBoundingClientRect(), w = document.querySelector("#work").getBoundingClientRect(); return Math.abs(p.width-w.width)<2 && Math.abs(p.height-w.height)<2; }')
                    check(len(state(page)['panes']) == 1 and full, f'closing resized {axis} pane {close_index} expands survivor to full work area')
            leader(page, 'wv'); leader(page, 'ws')
            command(page, 'close'); command(page, 'close')
            check(page.evaluate('Math.abs(document.querySelector(".pane").getBoundingClientRect().width-document.querySelector("#work").getBoundingClientRect().width)<2'), 'nested split collapse also removes inherited leaf sizing')
            check(page.locator('#switch-buffer').count() == 0, 'navigation does not expose a buffer-management picker')
            finder(page, 'cttn')
            first_result = page.locator('#results button').first.inner_text()
            check(first_result.startswith('Task t-0') and 'Preserve' in first_result, 'task names narrow fuzzily with the best match first')
            page.keyboard.press('Enter')
            check(draft(page)['id'] == 't-0', 'finder selection opens task buffer')
            leader(page, 'sg')
            page.keyboard.type('c-6')
            check(page.locator('#picker-preview math').count() > 0, 'content picker previews canonical MathML')
            check(page.locator('#picker-preview table').count() == 1, 'content picker previews a rendered table')
            check(page.locator('#picker-preview img[src="x"]').count() == 0, 'core renderer keeps hostile HTML as text')
            page.screenshot(path=str(RUN / '02-rich-picker.png'))
            page.keyboard.press('Enter')
            check(draft(page)['id'] == 'c-6', 'content selection opens focused contribution text')
            check(state(page)['mode'] == 'View' and page.locator('.pane.active .reading math').count() == 2 and page.locator('.pane.active .reading table').count() == 1, 'published contributions open in fully rendered View mode')
            page.keyboard.press(','); page.keyboard.press('a')
            page.wait_for_timeout(250)
            check(not page.evaluate('() => document.querySelector("#picker").open'), 'accept is unavailable on a contribution that is not the standing offer')
            page.keyboard.press('Escape')
            check(page.evaluate('getComputedStyle(document.querySelector(".pane.active .reading .dmath")).display === "block"'), 'display math is a block even when its source shares a prose line')
            page.locator('.pane.active .reading table').hover()
            check(state(page)['mode'] == 'View' and page.locator('.pane.active .reading table').count() == 1, 'mouse interaction does not switch rendered tables to source')
            page.set_viewport_size({'width': 1500, 'height': 480})
            page.locator('.pane.active .reading').focus()
            page.keyboard.press('j')
            check(page.locator('.pane.active .reading').evaluate('(node) => node.scrollTop > 0'), 'View j scrolls pixels without moving a text cursor across blocks')
            page.keyboard.press('k')
            check(page.locator('.pane.active .reading').evaluate('(node) => node.scrollTop === 0'), 'View k scrolls upward through the same document')
            page.set_viewport_size({'width': 1500, 'height': 980})
            page.screenshot(path=str(RUN / '02-view-contribution.png'))
            leader(page, 'uv')
            check(state(page)['mode'] == 'Normal' and page.locator('.pane.active .reading').is_hidden() and '$x^2+1$' in draft(page)['text'], 'Space uv opens original read-only Source with no replacements')
            page.keyboard.press('/'); page.keyboard.type('Action'); page.keyboard.press('Enter'); page.keyboard.press('0')
            source_head = next(p['position']['head'] for p in state(page)['panes'] if p['id'] == state(page)['focus'])
            page.keyboard.press('j')
            next_head = next(p['position']['head'] for p in state(page)['panes'] if p['id'] == state(page)['focus'])
            check(draft(page)['text'][:next_head].count('\n') == draft(page)['text'][:source_head].count('\n') + 1, 'Source j visits the next raw table line without skipping a rendered block')
            page.keyboard.press('V'); page.keyboard.press('y')
            check(state(page)['mode'] == 'Normal', 'Source Visual line selection and yank retain Vim interaction')
            leader(page, 'uv')
            check(state(page)['mode'] == 'View', 'the same Space uv binding returns from Source to View')
            command(page, 'rendering')
            check(state(page)['mode'] == 'Normal', 'the palette toggle provides the same View to Source action')
            leader(page, 'wv'); leader(page, 'uv')
            check(page.locator('.reading:visible').count() == 1 and page.locator('.cm-editor:visible').count() == 1, 'two panes of one contribution can independently show View and Source')
            page.set_viewport_size({'width': 1500, 'height': 480})
            page.keyboard.press('G')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            page.wait_for_function('document.querySelector(".pane.active .reading").scrollTop > 0')
            check(page.locator('.reading:visible').count() == 1 and page.locator('.cm-editor:visible').count() == 1, 'presentation choices and rendered reading position survive reload')
            leader(page, 'wo'); leader(page, 'uv')
            page.set_viewport_size({'width': 1500, 'height': 980})
            before_text = draft(page)['text']
            page.keyboard.press('i'); page.keyboard.type('do not rewrite published text'); page.keyboard.press('Escape')
            check(draft(page)['text'] == before_text, 'Vim editing cannot mutate published buffers')
            page.keyboard.type('ggdd')
            check(draft(page)['text'] == before_text, 'Vim delete operators also respect published read-only state')
            page.keyboard.press('/'); page.keyboard.type('Action'); page.keyboard.press('Enter'); page.keyboard.type('0j')
            at = next(p['position']['head'] for p in state(page)['panes'] if p['id'] == state(page)['focus'])
            expected_yank = before_text[at:before_text.index('\n', at)]
            page.keyboard.type('Vy')
            command(page, 'new')
            page.keyboard.press('p')
            check(draft(page)['text'] == '\n' + expected_yank, 'raw Source line yank pastes exactly after the empty draft line')
            page.keyboard.press('u')
            page.keyboard.press('i'); page.keyboard.type('Check: the cursor.'); page.keyboard.press('Escape')
            note_id = draft(page)['id']
            check(draft(page)['text'] == 'Check: the cursor.' and state(page)['mode'] == 'Normal', 'draft Insert mode types colon and spaces literally and Escape returns Normal')
            page.keyboard.type('ggvl')
            check(state(page)['mode'] == 'Visual' and 'VISUAL' in page.locator('#bar').inner_text(), 'Vim Visual mode selects document text and the status bar names it')
            page.keyboard.type('yp')
            check(draft(page)['text'] != 'Check: the cursor.', 'Vim yank and paste use the selected text')
            page.keyboard.press('u'); page.keyboard.press('$')
            check(draft(page)['text'] == 'Check: the cursor.', 'Vim undo restores text after visual paste')
            command(page, 'w')
            check('[-]' not in page.locator('#bar').inner_text() and '[+]' not in page.locator('#bar').inner_text(), 'colon w checkpoints the savepoint and clears the [+] flag')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == before, 'colon w saves locally without publication')
            local(page, 'i'); choose_id(page, 'c-0')
            first = draft(page)
            source_seq = next(c for c in original['comments'] if c['id'] == 'c-0')['seq']
            check(first['text'] == 'Check: the cursor[^1].' and first['references'] == [source_seq], 'draft-first exact-id citation inserts at saved cursor with birth-record mapping')
            local(page, 'i'); choose_id(page, 'c-0')
            check(draft(page)['text'].count('[^1]') == 2 and draft(page)['references'] == [source_seq], 'repeated source reuses its marker without duplicating mappings')
            saved = draft(page)
            local(page, 'i'); page.keyboard.type('c-1'); page.keyboard.press('Escape')
            check(draft(page)['text'] == saved['text'] and draft(page)['references'] == saved['references'], 'cancelling a citation picker leaves draft intact')
            leader(page, 'wv')
            page.keyboard.type('gg')
            data = state(page)
            positions = [p['position']['head'] for p in data['panes']]
            check(len(data['panes']) == 2 and positions[0] != positions[1], 'two panes of one draft have independent cursor positions')
            right_pane = state(page)['focus']
            leader(page, 'wh'); check(state(page)['focus'] != right_pane, 'leader focuses the pane on the left')
            leader(page, 'wl'); check(state(page)['focus'] == right_pane, 'leader focuses the pane on the right')
            page.keyboard.press('i'); page.keyboard.type('Prefix '); page.keyboard.press('Escape')
            check(draft(page)['text'].startswith('Prefix '), 'editing a split draft updates the shared document')
            check(page.locator('.pane .cm-content').all_text_contents()[0] == page.locator('.pane .cm-content').all_text_contents()[1], 'both draft views display the same edited document')
            page.keyboard.press('u')
            check(not draft(page)['text'].startswith('Prefix '), 'Vim undo operates on the shared draft')
            command(page, 'redo'); check(draft(page)['text'].startswith('Prefix '), 'palette redo reaches Vim history without stealing browser reload')
            page.keyboard.press('u')
            leader(page, 'ws'); lower_pane = state(page)['focus']
            leader(page, 'wk'); check(state(page)['focus'] != lower_pane, 'leader focuses the pane above')
            leader(page, 'wj'); check(state(page)['focus'] == lower_pane, 'leader focuses the pane below')
            leader(page, 'wo')
            check(len(state(page)['panes']) == 1 and any(b['id'] == note_id for b in state(page)['buffers']), 'closing other panes keeps the draft buffer')
            leader(page, 'sg'); choose_id(page, 'c-2', beside=True)
            check(len(state(page)['panes']) == 2 and any(p['bufferId'] == note_id for p in state(page)['panes']), 'opening a source beside retains the draft pane')
            page.screenshot(path=str(RUN / '03-sources-and-draft.png'))
            leader(page, 'uv')
            page.keyboard.press('/')
            page.keyboard.type('\\[\\^1\\]')
            page.keyboard.press('Enter')
            page.keyboard.type('gd')
            check(draft(page)['id'] == 'c-0', 'Vim search and gd follow an explicit citation under the cursor')
            page.keyboard.press('Control+o')
            check(draft(page)['id'] == 'c-2', 'Ctrl+O restores the prior contribution location')
            page.keyboard.press('Control+i')
            check(draft(page)['id'] == 'c-0', 'Ctrl+I traverses forward through the jump list')
            page.keyboard.type('gr')
            check(page.locator('#results button').count() == 1 and 'c-2' in page.locator('#results button').inner_text(), 'gr shows explicit backlinks with source context')
            check('picker starts from the draft' in page.locator('#picker-preview').inner_text(), 'backlink preview exposes unchanged citing prose')
            page.keyboard.press('Enter')
            check(draft(page)['id'] == 'c-2', 'backlink selection opens the citing contribution')
            finder(page, 'Check:')
            results = page.locator('#results button').all_inner_texts()
            check(any(label.startswith('Local draft') for label in results), 'recorded titles beside tasks still leave the local draft findable')
            page.keyboard.press('Escape')
            finder(page, 'Check: the cursor.')
            page.keyboard.press('Enter')
            check(draft(page)['id'] == note_id and draft(page)['text'] == saved['text'], 'main finder searches draft prose and restores its existing document')
            page.screenshot(path=str(RUN / '03-shared-draft-buffers.png'))
            check(state(page)['mode'] == 'View', 'opening an existing draft defaults to rendered View')
            page.wait_for_function('document.querySelectorAll(".pane.active .reading .mref").length === 2')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == before, 'draft View renders current repeated citations without publication')
            # the authored title is required and distinct from the display name
            check(draft(page)['authoredTitle'] == '', 'a new draft starts without an authored title')
            local(page, 't'); prompt_ready(page)
            check(page.locator('#picker').evaluate('node => node.classList.contains("prompt")'), 'comma t opens the shared title prompt')
            page.fill('#query', 'The cursor rule under review')
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(draft(page)['authoredTitle'] == 'The cursor rule under review', 'comma t authors the distinct title field')
            title_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'p'); prompt_ready(page)
            check('name' in page.locator('#picker-title').inner_text().lower(), 'publishing without a claimed name prompts for it, not a review form')
            answer_prompt(page, 'jerry')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == title_before, 'a prompt answer writes nothing by itself')
            check(draft(page)['id'] == note_id, 'the name prompt returns to the draft without publishing')
            local(page, 'p'); wait_submission(page, 'Recorded as')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == before + 1, 'keyboard-only flow publishes exactly one real note')
            landed = request(base + '/api/v1/workbench')[1]['comments'][-1]
            check(landed['title'] == 'The cursor rule under review' and landed['kind'] == 'note', 'the landed note carries the authored title')
            check(landed['actor'] == 'jerry', 'the claimed name attributed the landed note')
            page.screenshot(path=str(RUN / '04-keyboard-publication.png'))
            page.keyboard.press('Escape')
            repeat_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'p'); wait_submission(page, 'Recorded as')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == repeat_before, 'confirmed publication refuses a duplicate submission')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            check(any(b['id'] == note_id and b['text'] == saved['text'] for b in state(page)['buffers']), 'local draft survives browser reload')
            if errors:
                print('PAGE-ERRORS:', errors, flush=True)
            check(not errors, 'keyboard flow has no browser exceptions')
            context.close()

            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); page.on('pageerror', lambda error: errors.append(str(error)))
            ready(page, base)
            # mouse flow through the palette: the same commands the keys run
            page.locator('#open-commands').click()
            page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('find task'); page.keyboard.press('Enter')
            page.wait_for_function('(t) => document.querySelector("#picker-title").textContent.includes(t)', arg="Find task")
            page.keyboard.type('Keep buffers'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            page.locator('#open-commands').click()
            page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('new note'); page.keyboard.press('Enter')
            page.keyboard.press('i')
            page.keyboard.type('Mouse draft with $x^2$ and a source ')
            # the touch/mouse route: the palette runs the same cite command without exiting Insert
            page.locator('#open-commands').click()
            page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('insert citation'); page.keyboard.press('Enter')
            choose_id(page, 'c-7')
            page.wait_for_function('document.querySelector("#picker-preview img")?.naturalWidth > 0')
            check(page.locator('#picker-preview img').count() == 1, 'citation picker loads server-held artifact bytes')
            check(draft(page)['text'].endswith('[^1]') and state(page)['mode'] == 'Insert', 'mouse citation insertion preserves Insert mode and cursor')
            page.keyboard.press('Escape'); leader(page, 'uv')
            page.wait_for_function('document.querySelector(".pane.active .reading math") !== null')
            check(page.locator('.pane.active .reading math').count() > 0, 'draft View renders math through the canonical renderer')
            check('t-1' in page.locator('.pane.active .reading').inner_text(), 'the draft View names its home task while authoring')
            local(page, 'p'); answer_prompt(page, 'jerry')
            local(page, 'p'); prompt_ready(page)
            check('title' in page.locator('#picker-title').inner_text().lower(), 'the missing title prompts from the shared surface')
            page.fill('#query', 'The mouse-authored finding')
            page.locator('#choose').click()
            page.wait_for_function('!document.querySelector("#picker").open')
            mouse_before = request(base + '/api/v1/workbench')[1]['cursor']
            command(page, 'publish')
            wait_submission(page, 'Recorded as')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == mouse_before + 1, 'mouse-driven flow publishes exactly one note')
            leader(page, 'wv')
            divider = page.locator('.divider').first
            rect = divider.bounding_box()
            old_ratio = state(page)['layout']['ratio']
            page.mouse.move(rect['x'] + 3, rect['y'] + rect['height'] / 2); page.mouse.down()
            page.mouse.move(rect['x'] + 100, rect['y'] + rect['height'] / 2); page.mouse.up()
            check(state(page)['layout']['ratio'] != old_ratio, 'mouse dragging resizes pane arrangement')
            leader(page, 'sg'); choose_id(page, 'c-6', beside=True)
            source_title = page.locator('.pane.active .buffer-title')
            target_pane = page.locator('.pane:not(.active)').first
            source_title.drag_to(target_pane)
            check({p['bufferId'] for p in state(page)['panes']} == {'c-6', note_id} and state(page)['layout']['children'][0]['view'] == state(page)['focus'], 'dragging a pane swaps locations without replacing either document')
            check(any(b['kind'] == 'draft' for b in state(page)['buffers']), 'dragging pane contents retains unpublished buffer data')
            page.screenshot(path=str(RUN / '05-mouse-panes.png'))
            leader(page, 'sg'); choose_id(page, 'c-7')
            page.wait_for_function('document.querySelector(".pane.active .reading img")?.naturalWidth > 0')
            check(page.locator('.pane.active .reading img').count() == 1, 'published artifact figure is inspectable inside its contribution buffer')
            page.locator('.pane.active .reading [data-record-target="t-2"]').click()
            check(draft(page)['id'] == 't-2', 'structured metadata links can also be followed with the mouse')
            page.keyboard.press('Control+o')
            check(draft(page)['id'] == 'c-7', 'mouse link navigation records a jump-list return location')
            context.close()

            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            finder(page, 'cursor')
            page.keyboard.press('Enter')
            command(page, 'new')
            page.keyboard.press('i'); page.keyboard.type('Unresolved marker [^9]')
            rejected_text = draft(page)['text']
            refusal_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'p'); answer_prompt(page, 'jerry')
            local(page, 'p'); answer_prompt(page, 'The refused marker note')
            local(page, 'p'); wait_submission(page, 'Not sent')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == refusal_before, 'actual core marker refusal lands no record')
            check(draft(page)['text'] == rejected_text, 'core refusal preserves the draft')
            # a refused submission can be corrected and retried
            page.locator('.pane.active').click()
            page.keyboard.type('ggdG'); page.keyboard.press('i'); page.keyboard.type('A resolved body for the corrected retry.')
            page.keyboard.press('Escape')
            check('[+]' in page.locator('#bar').inner_text(), 'a corrected refusal shows the modified flag')
            local(page, 'p'); wait_submission(page, 'Recorded as')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == refusal_before + 1, 'the corrected retry lands exactly one note')
            page.keyboard.press('Escape')
            command(page, 'new'); page.keyboard.press('i'); page.keyboard.type('A response can be lost after the note lands.')
            local(page, 'p'); answer_prompt(page, 'The lost response note')
            unknown_before = request(base + '/api/v1/workbench')[1]['cursor']

            def lose_response(route):
                route.fetch()
                route.abort('failed')
            page.route('**/api/v1/workbench/command', lose_response)
            local(page, 'p'); wait_submission(page, 'outcome unknown')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_before + 1, 'response-loss scenario really lands the note upstream')
            local(page, 'p'); wait_submission(page, 'outcome unknown')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_before + 1, 'unknown outcome blocks retry')
            check(draft(page)['text'] == 'A response can be lost after the note lands.', 'lost response preserves draft text')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            local(page, 'p'); wait_submission(page, 'outcome unknown')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_before + 1, 'inspection and reload do not retry the landed request')
            page.screenshot(path=str(RUN / '06-unknown-outcome.png'))
            check(page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'desktop workbench has no page overflow')
            context.close()
            context = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
            page = context.new_page(); ready(page, base)
            leader(page, 'sg')
            page.keyboard.type('c-6')
            check(page.locator('#picker-preview math').count() > 0 and page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'narrow touch viewport retains rich picker without page overflow')
            page.screenshot(path=str(RUN / '07-narrow-picker.png'))
            page.locator('#picker-close').click()
            page.locator('#open-commands').tap()
            page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('new note'); page.keyboard.press('Enter')
            check(page.locator('#open-commands').bounding_box()['height'] >= 40, 'the touch opener is a comfortable hit area')
            local(page, 'p'); prompt_ready(page)
            check('name' in page.locator('#picker-title').inner_text().lower(), 'a touch draft is prompted for the claimed name first')
            answer_prompt(page, 'jerry')
            local(page, 'p'); prompt_ready(page)
            check('home' in page.locator('#picker-title').inner_text().lower(), 'a draft without a home is prompted for it, not a review form')
            page.keyboard.type('t-0'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            local(page, 'p'); prompt_ready(page)
            check('title' in page.locator('#picker-title').inner_text().lower(), 'a home alone cannot replace the authored title')
            page.keyboard.press('Escape')
            local(page, 't'); prompt_ready(page)
            page.fill('#query', 'The narrow view note'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            check('t-0' in page.locator('#bar-info').inner_text() and 'The narrow view note' in page.locator('#bar-info').inner_text(), 'home task and authored title stay visible in the bar while authoring')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_before + 1, 'narrow-view prompts and home selection write nothing')
            context.close()

            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            finder(page, 'cursor')
            page.keyboard.press('Enter')
            command(page, 'new')
            page.keyboard.press('i'); page.keyboard.type('Old preview.')
            page.keyboard.press('Escape')
            held_preview = []
            def delay_first_preview(route):
                if not held_preview:
                    response = route.fetch()
                    held_preview.append((route, response))
                    route.request.frame.evaluate('window.workbenchPreviewHeld = true')
                else:
                    route.continue_()
            page.route('**/workbench/preview', delay_first_preview)
            leader(page, 'uv')
            page.wait_for_function('window.workbenchPreviewHeld === true')
            check(state(page)['mode'] == 'View', 'draft View can wait for its renderer without starting publication')
            leader(page, 'uv')
            page.keyboard.type('ggdG'); page.keyboard.press('i'); page.keyboard.type('New preview.'); page.keyboard.press('Escape')
            leader(page, 'uv')
            page.wait_for_function('document.querySelector(".pane.active .reading").textContent.includes("New preview.")')
            held_preview[0][0].fulfill(response=held_preview[0][1])
            page.evaluate('new Promise(requestAnimationFrame)')
            check('New preview.' in page.locator('.pane.active .reading').inner_text() and 'Old preview.' not in page.locator('.pane.active .reading').inner_text(), 'late old draft preview cannot replace the current draft View')
            page.unroute('**/workbench/preview', delay_first_preview)
            leader(page, 'uv')
            local(page, 't'); prompt_ready(page)
            page.fill('#query', 'The preview note'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            prompted_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'p'); prompt_ready(page)
            check('name' in page.locator('#picker-title').inner_text().lower(), 'publishing from rendered View prompts for the claimed name first')
            page.keyboard.press('Escape')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == prompted_before, 'an escaped prompt writes nothing')
            saved_point = next(p['position'] for p in state(page)['panes'] if p['id'] == state(page)['focus'])
            leader(page, 'uv'); leader(page, 'uv')
            check(next(p['position']['head'] for p in state(page)['panes'] if p['id'] == state(page)['focus']) == saved_point['head'], 'draft View Source round trip retains the editing cursor')
            local_before = request(base + '/api/v1/workbench')[1]['cursor']
            leader(page, 'wv'); leader(page, 'uv'); leader(page, 'wh')
            page.keyboard.press('A'); page.keyboard.type(' More context.'); page.keyboard.press('Escape')
            page.wait_for_function('document.querySelector(".pane:not(.active) .reading").textContent.includes("More context.")')
            check('More context.' in page.locator('.pane:not(.active) .reading').inner_text(), 'editing Source refreshes another pane showing the same draft View')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == local_before, 'draft editing and View refresh add no records')
            leader(page, 'wo')
            held_note = []
            def delay_note_answer(route):
                route.fetch()
                held_note.append(route)
                route.request.frame.evaluate('window.workbenchNoteHeld = true')
            page.route('**/workbench/command', delay_note_answer)
            local(page, 'p'); answer_prompt(page, 'jerry')
            inflight_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'p')
            page.wait_for_function('window.workbenchNoteHeld === true')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == inflight_before + 1, 'in-flight reload scenario lands one real upstream note')
            check(page.evaluate('JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k => k.startsWith("saccade-workbench:")))).buffers.find(b => b.kind === "draft").submission.kind') == 'unknown', 'unconfirmed submission state is saved before its answer arrives')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            local(page, 'p'); wait_submission(page, 'no final answer')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == inflight_before + 1, 'in-flight reload creates no second record')
            held_note[0].abort('failed')
            page.unroute('**/workbench/command', delay_note_answer)
            command(page, 'new')
            page.keyboard.press('i'); page.keyboard.type('Keep this draft when local persistence fails.')
            # the in-flight note's held response carried the claim cookie, so
            # only the title prompt stands between this draft and submission
            local(page, 'p'); answer_prompt(page, 'The persistence-failure note')
            blocked_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.evaluate('() => { Storage.prototype.setItem = function() { throw new DOMException("local quota", "QuotaExceededError") }; }')
            local(page, 'p'); wait_submission(page, 'Submission not attempted')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == blocked_before, 'failed local submission-state save prevents forwarding')
            check(draft(page)['text'] == 'Keep this draft when local persistence fails.', 'persistence failure keeps the in-memory draft')
            context.close()

            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            finder(page, 'cursor')
            page.keyboard.press('Enter')
            command(page, 'new')
            page.keyboard.press('i'); page.keyboard.type('Keep my unfinished migration draft.')
            page.keyboard.press('Escape')
            local(page, 'i'); choose_id(page, 'c-0')
            saved_v1 = draft(page)
            migration_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            restored_current = draft(page)
            check(restored_current['text'] == saved_v1['text'] and restored_current['references'] == saved_v1['references'] and restored_current['home'] == saved_v1['home'], 'client-local draft text, home and references survive a reload')
            check(restored_current['authoredTitle'] == saved_v1['authoredTitle'], 'draft migration keeps the authored-title field distinct')
            finder(page, 'unfinished migration')
            check(draft(page)['id'] == saved_v1['id'], 'main finder retrieves the draft by prose')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == migration_before, 'client draft migration and retrieval write no records')
            context.close()

            # targeted decision laws through the real doors: accept and
            # reject on standing offers, refusal kept drafts, unknown
            # outcomes, and the repair baseline separation
            server.cli('offer', 't-2', '--title', 'The standing account',
                       'The suite is green; accept or reject this account with grounds.')
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            finder(page, 'standing account'); page.keyboard.press('Enter')
            check(draft(page)['id'].startswith('c-'), 'the seeded standing offer opens by its authored title')
            local(page, 'a'); prompt_ready(page)
            check('name' in page.locator('#picker-title').inner_text().lower(), 'the accept prompts for its claimed name, not a review form')
            answer_prompt(page, 'jerry')
            accept_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'a'); wait_submission(page, 'Recorded')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == accept_before + 1, 'the accept landed exactly one record through the browser door')
            concluded = next(t for t in request(base + '/api/v1/workbench')[1]['tasks'] if t['id'] == 2)
            check(concluded['state'] == 'concluded', 'the accepted account concluded its task')
            page.keyboard.press('Escape')

            server.cli('offer', 't-1', '--title', 'The second account', 'Accept or reject this other account.')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            finder(page, 'second account'); page.keyboard.press('Enter')
            second_id = draft(page)['id']
            local(page, 'x')
            check(draft(page)['eventual'] and draft(page)['eventual']['offer'] == second_id, 'reject drafts grounds against the named offer')
            reject_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'p'); wait_submission(page, 'Grounds are required')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == reject_before, 'the empty-grounds refusal wrote nothing')
            page.keyboard.press('i'); page.keyboard.type('The edge case still fails on the empty grounds door.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.eventual && b.eventual.kind === 'reject' && b.text.includes('edge case still fails'))")
            local(page, 'p'); wait_submission(page, 'Recorded')
            after_reject = request(base + '/api/v1/workbench')[1]
            check(next(t for t in after_reject['tasks'] if t['id'] == 1)['state'] == 'open', 'the rejection reopened the task')
            landed_reject = next(c for c in after_reject['comments'] if c['kind'] == 'reject')
            check(landed_reject['body'] == 'The edge case still fails on the empty grounds door.', 'grounds land verbatim')
            page.keyboard.press('Escape')

            server.cli('offer', 't-0', '--title', 'The lost-reply account', 'One more account to decide.')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            finder(page, 'lost-reply'); page.keyboard.press('Enter')
            local(page, 'x')
            page.keyboard.press('i'); page.keyboard.type('Losing this reply must not resend it.')
            unknown_cursor = request(base + '/api/v1/workbench')[1]['cursor']
            page.route('**/api/v1/workbench/command', lambda route: (route.fetch(), route.abort('failed')))
            local(page, 'p'); wait_submission(page, 'outcome unknown')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_cursor + 1, 'the lost reject reply really landed upstream')
            local(page, 'p'); wait_submission(page, 'outcome unknown')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_cursor + 1, 'an unknown reject outcome blocks retry')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_cursor + 1, 'inspection and reload do not resend the decided reject')
            page.keyboard.press('Escape')
            page.unroute('**/api/v1/workbench/command')

            # repair laws: immutable record baselines, local :w savepoint,
            # pending body across reload, concurrent fields untouched
            finder(page, 'insertion bookmark'); page.keyboard.press('Enter')
            original_body = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['id'] == 'c-0')['body']
            local(page, 'e')
            check(draft(page)['revisionPending'] is not None, 'comma e starts a pending repair')
            check(draft(page)['revisionPending']['originalBody'] == original_body, 'the repair pins the original record body as its patch baseline')
            page.keyboard.press('i'); page.keyboard.type(' The retry lands both fields.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.id === 'c-0' && b.text.includes('The retry lands both fields.'))")
            command(page, 'write')
            check(page.locator('#bar-flags').inner_text() == '', 'a saved repair shows no modified flag')
            check(draft(page)['revisionPending']['savedBody'] == draft(page)['text'], 'colon w advances only the local savepoint')
            page.keyboard.press('i'); page.keyboard.type(' Unsaved tail.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.id === 'c-0' && b.text.includes('Unsaved tail.'))")
            page.evaluate('() => { Storage.prototype.setItem = function() { throw new DOMException("local quota", "QuotaExceededError") }; }')
            command(page, 'write')
            page.wait_for_function('() => document.querySelector("#message").textContent.includes("Local save failed")')
            check(draft(page)['revisionPending']['savedBody'] != draft(page)['text'], 'a failed local save does not advance the savepoint')
            check(page.locator('#bar-flags').inner_text() == '[+]', 'the modified flag survives a failed save')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            check('The retry lands both fields' in draft(page)['text'] and 'Unsaved tail.' in draft(page)['text'], 'the persisted pending body survives the reload')
            check('Unsaved tail.' not in draft(page)['revisionPending']['savedBody'], 'the reloaded savepoint is the successful save, not the failed one')
            check(draft(page)['revisionPending']['originalBody'] == original_body, 'the reload keeps the original patch baseline intact')
            local(page, 'p')
            server.cli('revise', 'c-0', '--title', 'The concurrently retitled bookmark')
            pending_body = draft(page)['text']
            wait_submission(page, 'Recorded')
            after_revive = request(base + '/api/v1/workbench')[1]
            revised_now = next(c for c in after_revive['comments'] if c['id'] == 'c-0')
            check(revised_now['title'] == 'The concurrently retitled bookmark', 'a concurrently revised title stands after the body patch lands')
            check('The retry lands both fields' in revised_now['body'] and 'Unsaved tail.' in revised_now['body'], 'the whole pending body lands after the reload')
            page.keyboard.press('Escape')
            local(page, 'e')
            check(draft(page)['revisionPending'] is not None, 'a fresh repair starts after the last landed')
            local(page, 'e')
            check(draft(page)['revisionPending'] is None, 'comma e again discards the pending repair as the status promises')

            # in-flight completion must not discard local work: hold the
            # answer, keep editing the repair, then let the answer land
            leader(page, 'sg'); choose_id(page, 'c-0')
            local(page, 'e')
            page.keyboard.press('i'); page.keyboard.type(' Held-answer base.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.revisionPending && b.text.includes('Held-answer base.'))")
            sent_body = draft(page)['text']
            held_revision = []
            def hold_revision(route):
                response = route.fetch()
                held_revision.append((route, response))
                route.request.frame.evaluate('window.workbenchRevisionHeld = true')
            page.route('**/workbench/command', hold_revision)
            page.keyboard.press('Escape')
            local(page, 'p')
            page.wait_for_function('window.workbenchRevisionHeld === true')
            page.keyboard.press('i'); page.keyboard.type(' Unsent while held.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.revisionPending && b.text.includes('Unsent while held.'))")
            held_revision[0][0].fulfill(response=held_revision[0][1])
            page.wait_for_function('document.querySelector("#message").textContent.includes("continues on the recorded bytes")')
            still = draft(page)
            check(still['revisionPending'] is not None and 'Unsent while held.' in still['text'], 'a landed answer keeps repair work typed while it was outstanding')
            check(still['revisionPending']['originalBody'] == sent_body and still['revisionPending']['savedBody'] == sent_body, 'the surviving repair rebases on the recorded bytes')
            check(still['revisionPending']['submission'] is None, 'the rebased repair is a fresh unsent patch')
            landed_body = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['id'] == 'c-0')['body']
            check('Held-answer base.' in landed_body and 'Unsent while held.' not in landed_body, 'the record carries exactly the sent bytes')
            page.unroute('**/workbench/command', hold_revision)
            page.keyboard.press('Escape')
            local(page, 'p'); wait_submission(page, 'Recorded as')
            repaired = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['id'] == 'c-0')['body']
            check('Unsent while held.' in repaired, 'the rebased repair publishes its remaining diff')
            check(draft(page)['revisionPending'] is None, 'the fully published repair completes')

            # discarding a repair while its answer is outstanding leaves the
            # buffer on the recorded bytes and touches no other pending
            local(page, 'e')
            page.keyboard.press('i'); page.keyboard.type(' Discard candidate.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.revisionPending && b.text.includes('Discard candidate.'))")
            held_discard = []
            def hold_discard(route):
                response = route.fetch()
                held_discard.append((route, response))
                route.request.frame.evaluate('window.workbenchDiscardHeld = true')
            page.route('**/workbench/command', hold_discard)
            page.keyboard.press('Escape')
            local(page, 'p')
            page.wait_for_function('window.workbenchDiscardHeld === true')
            local(page, 'e')
            check(draft(page)['revisionPending'] is None, 'the repair discards while its answer is outstanding')
            held_discard[0][0].fulfill(response=held_discard[0][1])
            page.wait_for_function('document.querySelector("#message").textContent.includes("the record stands")')
            check(draft(page)['revisionPending'] is None, 'the outstanding answer completes without reviving the discarded repair')
            check('Discard candidate.' in draft(page)['text'], 'the landed record shows through the discarded local repair')
            page.unroute('**/workbench/command', hold_discard)

            # continuation must not manufacture an edit to an untouched field:
            # body-only repair, held answer, another actor retitles meanwhile
            leader(page, 'sg'); choose_id(page, 'c-0')
            local(page, 'e')
            page.keyboard.press('G'); page.keyboard.press('A'); page.keyboard.type(' Body-only continuation.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.revisionPending && b.text.includes('Body-only continuation.'))")
            cont_sent = draft(page)['text']
            server.cli('revise', 'c-0', '--title', 'The concurrent continuation title')
            held_cont = []
            def hold_cont(route):
                response = route.fetch()
                held_cont.append((route, response))
                route.request.frame.evaluate('window.workbenchContHeld = true')
            page.route('**/workbench/command', hold_cont)
            page.keyboard.press('Escape')
            local(page, 'p')
            page.wait_for_function('window.workbenchContHeld === true')
            page.keyboard.press('G'); page.keyboard.press('A'); page.keyboard.type(' Continuation tail.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.revisionPending && b.text.includes('Continuation tail.'))")
            held_cont[0][0].fulfill(response=held_cont[0][1])
            page.wait_for_function('document.querySelector("#message").textContent.includes("continues on the recorded bytes")')
            cont = draft(page)
            check(cont['revisionPending']['title'] == 'The concurrent continuation title', 'an untouched title adopts the concurrent title on continuation')
            check(cont['revisionPending']['originalBody'] == cont_sent, 'the touched body advances only its baseline to the recorded bytes')
            check(cont['text'].endswith('Continuation tail.'), 'the authored tail survives continuation')
            page.unroute('**/workbench/command', hold_cont)
            page.keyboard.press('Escape')
            local(page, 'p'); wait_submission(page, 'Recorded as')
            cont_record = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['id'] == 'c-0')
            check(cont_record['title'] == 'The concurrent continuation title', 'publishing the tail does not revert the concurrent title')
            check(cont_record['body'].endswith('Continuation tail.'), 'the continuation publishes the authored tail')
            check(draft(page)['revisionPending'] is None, 'the continuation repair completes after its tail lands')

            # the symmetric case: a body-only repair in flight, the title
            # edited while held, another actor revises the body meanwhile -
            # the untouched body adopts the concurrent revise, the authored
            # title keeps its intent
            leader(page, 'sg'); choose_id(page, 'c-0')
            local(page, 'e')
            page.keyboard.press('G'); page.keyboard.press('A'); page.keyboard.type(' Symbase.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.revisionPending && b.text.includes('Symbase.'))")
            held_sym = []
            def hold_sym(route):
                response = route.fetch()
                held_sym.append((route, response))
                route.request.frame.evaluate('window.workbenchSymHeld = true')
            page.route('**/workbench/command', hold_sym)
            page.keyboard.press('Escape')
            local(page, 'p')
            page.wait_for_function('window.workbenchSymHeld === true')
            local(page, 't')
            page.wait_for_function('document.querySelector("#picker").open')
            page.fill('#query', 'The in-flight retitle'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            server.cli('revise', 'c-0', 'Concurrent body stands.')
            held_sym[0][0].fulfill(response=held_sym[0][1])
            page.wait_for_function('document.querySelector("#message").textContent.includes("continues on the recorded bytes")')
            sym = draft(page)
            check(sym['text'] == 'Concurrent body stands.', 'an untouched body adopts the concurrent body on continuation')
            check(sym['revisionPending']['title'] == 'The in-flight retitle', 'the authored in-flight title survives continuation')
            page.unroute('**/workbench/command', hold_sym)
            local(page, 'p'); wait_submission(page, 'Recorded as')
            sym_record = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['id'] == 'c-0')
            check(sym_record['title'] == 'The in-flight retitle', 'the authored title publishes')
            check(sym_record['body'] == 'Concurrent body stands.', 'publishing the retitle does not revert the concurrent body')
            check(draft(page)['revisionPending'] is None, 'the symmetric continuation repair completes')

            # authoring metadata renders live, without leaving rendered View
            command(page, 'home')
            command(page, 'new')
            page.keyboard.press('i'); page.keyboard.type('Metadata renders as it is authored.')
            page.keyboard.press('Escape')
            leader(page, 'uv')
            page.wait_for_function('document.querySelector(".pane.active .reading").textContent.includes("home: not selected")')
            local(page, 'h')
            page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('t-1'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            page.wait_for_function('document.querySelector(".pane.active .reading").textContent.includes("home: t-1")')
            local(page, 't')
            page.wait_for_function('document.querySelector("#picker").open')
            page.fill('#query', 'The live metadata title'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            page.wait_for_function('document.querySelector(".pane.active .reading").textContent.includes("title: The live metadata title")')
            check('not authored' not in page.locator('.pane.active .reading').inner_text() and 'home: t-1' in page.locator('.pane.active .reading').inner_text(), 'title and home changes render live in the draft View')
            leader(page, 'sg'); choose_id(page, 'c-0')
            local(page, 'e')
            leader(page, 'uv')
            page.wait_for_function('document.querySelector(".pane.active .draft-title") !== null')
            local(page, 't')
            page.wait_for_function('document.querySelector("#picker").open')
            page.fill('#query', 'The live repair title'); page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            page.wait_for_function('document.querySelector(".pane.active .draft-title").textContent.includes("The live repair title")')
            check('The live repair title' in page.locator('.pane.active .reading').inner_text(), 'a repair title prompt refreshes the rendered repair in View')
            local(page, 'e')
            check(draft(page)['revisionPending'] is None, 'the probe repair discards cleanly after the render checks')
            context.close()

            # the bar names the tracker kind across record kinds and
            # draft intents; seeds land through the CLI, not the browser
            server.cli('create', 'task', 'The kind survey')
            snapshot = request(base + '/api/v1/workbench')[1]
            survey = next(t for t in snapshot['tasks'] if t['name'] == 'The kind survey')
            survey_id = survey['id']
            server.cli('offer', f't-{survey_id}', '--title', 'The kind offer', 'Accept or reject this survey.')
            snapshot = request(base + '/api/v1/workbench')[1]
            offer1_id = next(c for c in snapshot['comments'] if c['kind'] == 'offer' and c['home'] == survey_id)['id']
            server.cli('reject', offer1_id, '--grounds', 'The survey rejects this first account.')
            reject_id = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['kind'] == 'reject')['id']
            server.cli('offer', f't-{survey_id}', '--title', 'The kind second offer', 'One more account to decide.')
            snapshot = request(base + '/api/v1/workbench')[1]
            offer2_id = next(c for c in sorted(snapshot['comments'], key=lambda c: c['seq']) if c['kind'] == 'offer' and c['home'] == survey_id and c['id'] != offer1_id)['id']
            server.cli('accept', offer2_id)
            accept_id = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['kind'] == 'accept')['id']
            server.cli('use', f't-{survey_id}', '--source', 't-0', '--reason', 'This survey builds on the cursor note.')
            use_id = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['kind'] == 'use')['id']
            server.cli('create', 'task', 'The kind standing task')
            standing_id = next(t for t in request(base + '/api/v1/workbench')[1]['tasks'] if t['name'] == 'The kind standing task')['id']
            server.cli('offer', f't-{standing_id}', '--title', 'The kind standing offer', 'A standing account for the grounds draft.')
            standing_offer_id = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['kind'] == 'offer' and c['home'] == standing_id)['id']
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            # textContent, not inner_text: the renderer collapses the bar's spacing runs
            bar = lambda: [page.locator('#bar-mode').text_content(), page.locator('#bar-kind').text_content(), page.locator('#bar-ident').text_content(), page.locator('#bar-info').text_content(), page.locator('#bar-flags').text_content()]
            check(bar() == ['RENDERED', 'Home', 'home', '', '[-]'], 'home opens with the exact mode-kind-ident-info-flags bar tuple')
            page.goto(base + f'/t/{survey_id}'); page.wait_for_function('window.workbenchUI !== undefined')
            check(bar()[:3] == ['RENDERED', 'Task', f't-{survey_id}'], 'a task deep link opens its thread as rendered Task cards')
            def deep(comment_id):
                page.goto(base + '/' + comment_id.replace('-', '/', 1))
                page.wait_for_function('window.workbenchUI !== undefined')
            deep(offer1_id)
            check(bar()[:3] == ['RENDERED', 'Offer', offer1_id], 'an offer shows its Offer kind rendered')
            deep(accept_id)
            check(bar()[:3] == ['RENDERED', 'Accept', accept_id], 'an accept shows its Accept kind')
            deep(use_id)
            check(bar()[:3] == ['RENDERED', 'Use', use_id], 'a use shows its Use kind')
            reject_id = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['kind'] == 'reject')['id']
            deep(reject_id)
            check(bar()[:3] == ['RENDERED', 'Reject', reject_id], 'a landed reject shows its Reject kind')
            page.goto(base); page.wait_for_function('window.workbenchUI !== undefined')
            leader(page, 'sg'); choose_id(page, 'c-0')
            check(bar()[:3] == ['RENDERED', 'Note', 'c-0'], 'a note shows its Note kind')
            local(page, 'e')
            # c-0 carries the concurrent repair title from the repair laws above
            check(bar() == ['NORMAL', 'Note', 'c-0', 'revising  The in-flight retitle', ''], 'a repair keeps the record kind and names the title it revises')
            local(page, 'e')
            finder(page, 'kind standing offer'); page.keyboard.press('Enter')
            check(draft(page)['id'] == standing_offer_id, 'the standing survey offer opens for the draft-intent tuple')
            local(page, 'x')
            tuple_now = bar()
            check(tuple_now[1] == 'Reject' and tuple_now[2].startswith('draft:') and tuple_now[3].startswith('rejecting c-'), 'a grounds draft shows the Reject intent kind with its draft ident')
            context.close()

            # thread cards: the ratified read-only thread interaction - every
            # contribution a rich card, j/k selection kept visible, contextual
            # actions on the selection, and /t/N#c-M selected-comment routing
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            thread_bar = lambda: [page.locator('#bar-mode').text_content(), page.locator('#bar-kind').text_content(), page.locator('#bar-ident').text_content(), page.locator('#bar-info').text_content()]
            page.goto(base + '/t/0'); page.wait_for_function('window.workbenchUI !== undefined')
            check(thread_bar()[:3] == ['RENDERED', 'Task', 't-0'], 'a task deep link opens its thread as rendered cards')
            t0_cards = [c['id'] for c in request(base + '/api/v1/workbench')[1]['comments'] if c['home'] == 0]
            check(page.locator('.pane.active .thread-card').count() == len(t0_cards), 'every contribution of the thread renders as its own card')
            check(page.locator('.pane.active .thread-card[data-comment="c-2"] .rich .mref').count() >= 1, 'thread cards carry the canonical rich rendering with citation links')
            check('withdrawn' in page.locator('.pane.active .thread-card[data-comment="c-3"]').inner_text(), 'a withdrawn contribution renders its tombstone card')
            page.locator('.pane.active .reading').focus()
            check(page.locator('.pane.active .thread-card.selected').count() == 0, 'a thread opens with no card selected')
            page.keyboard.press('j')
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-0"]').count() == 1, 'j selects the first card from no selection')
            check(page.locator('#bar-info').text_content().startswith('selected c-0'), 'the bar names the selected contribution')
            local(page, 'e')
            check(draft(page)['id'] == 'c-0' and draft(page)['revisionPending'] is not None, 'comma e on the selected card opens that contribution repair in its standalone buffer')
            check(thread_bar()[:3] == ['NORMAL', 'Note', 'c-0'] and thread_bar()[3].startswith('revising'), 'the repair keeps the record kind and names the revision')
            local(page, 'e')
            check(draft(page)['revisionPending'] is None, 'comma e again discards the probe repair')
            page.goto(base + '/t/0#c-0'); page.wait_for_function('window.workbenchUI !== undefined')
            page.locator('.pane.active .reading').focus()
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-0"]').count() == 1, 'a deep link reselects its card in a restored session')
            page.keyboard.press('j')
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-1"]').count() == 1, 'j steps to the next card')
            page.keyboard.press('k')
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-0"]').count() == 1, 'k steps back to the previous card')
            page.keyboard.press('k')
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-0"]').count() == 1, 'k clamps at the first card without wrapping')
            page.set_viewport_size({'width': 1500, 'height': 420})
            page.keyboard.press('j'); page.keyboard.press('j'); page.keyboard.press('j')
            page.wait_for_timeout(150)
            in_view = page.evaluate('''() => {
                const reader = document.querySelector(".pane.active .reading");
                const card = document.querySelector(".pane.active .thread-card.selected");
                const a = card.getBoundingClientRect(), r = reader.getBoundingClientRect();
                return a.top >= r.top - 1 && a.bottom <= r.bottom + 1;
            }''')
            check(in_view and page.locator('.pane.active .thread-card.selected[data-comment="c-3"]').count() == 1, 'a newly selected card is scrolled fully into view')
            page.set_viewport_size({'width': 1500, 'height': 980})
            # the location already carries #c-0, so this is a fresh full
            # load onto the deep link, not a same-document hash change
            page.goto(base + '/t/0#c-0')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-0"]').count() == 1, 'a /t/N#c-M deep link selects that card in the thread')
            page.keyboard.press('g'); page.keyboard.press('r')
            page.wait_for_function('document.querySelector("#picker").open')
            check('Backlinks to c-0' in page.locator('#picker-title').text_content(), 'gr backlinks the selected card')
            check('c-2' in page.locator('#results button').first.inner_text(), 'the backlinks list the citing contribution')
            page.keyboard.press('Escape')
            page.evaluate("location.hash = '#c-2'")
            page.wait_for_timeout(250)
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-2"]').count() == 1, 'a hash change reselects within the open thread')
            # a wrong-thread selected-comment link: same membership contract
            # as deepLink, no card selected and no hidden contextual target
            page.evaluate('''() => {
                const a = document.createElement("a");
                a.href = "/t/0#c-5";
                a.textContent = "wrong thread";
                document.querySelector(".pane.active .thread-card .rich").append(a);
            }''')
            page.locator('.pane.active .thread-card a', has_text='wrong thread').click()
            page.wait_for_timeout(200)
            check(draft(page)['id'] == 't-0' and page.locator('.pane.active .thread-card.selected').count() == 0, 'a wrong-thread selected-comment link stays on the named thread and selects no card')
            check(page.locator('#bar-info').text_content() == '', 'the bar names no selection after a wrong-thread link')
            local(page, 'e')
            check(draft(page)['id'] == 't-0', 'comma e after a wrong-thread link leaves no hidden cross-thread repair target')
            page.evaluate("location.hash = '#c-4'")
            page.wait_for_timeout(250)
            check(page.locator('.pane.active .thread-card.selected').count() == 0, 'a hash naming a contribution of another thread selects nothing here')
            server.cli('create', 'task', 'The thread selection account')
            fresh_task = next(t for t in request(base + '/api/v1/workbench')[1]['tasks'] if t['name'] == 'The thread selection account')['id']
            server.cli('offer', f't-{fresh_task}', '--title', 'The thread standing account', 'Accept or reject this account from its thread card.')
            fresh_offer = next(c for c in request(base + '/api/v1/workbench')[1]['comments'] if c['kind'] == 'offer' and c['home'] == fresh_task)['id']
            page.goto(base + f'/t/{fresh_task}'); page.wait_for_function('window.workbenchUI !== undefined')
            page.locator('.pane.active .reading').focus()
            page.keyboard.press('j')
            check(page.locator(f'.pane.active .thread-card.selected[data-comment="{fresh_offer}"]').count() == 1, 'the standing offer card selects on its thread')
            local(page, 'a'); prompt_ready(page)
            check('name' in page.locator('#picker-title').text_content().lower(), 'the accept from the thread still prompts for its claimed name')
            answer_prompt(page, 'jerry')
            accept_before = request(base + '/api/v1/workbench')[1]['cursor']
            local(page, 'a'); wait_submission(page, 'Recorded')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == accept_before + 1, 'the accept through the selected card landed exactly one record')
            check(next(t for t in request(base + '/api/v1/workbench')[1]['tasks'] if t['id'] == fresh_task)['state'] == 'concluded', 'the selected card accept concluded its task')
            page.goto(base + f'/t/{standing_id}'); page.wait_for_function('window.workbenchUI !== undefined')
            page.locator('.pane.active .reading').focus()
            page.keyboard.press('j')
            check(page.locator(f'.pane.active .thread-card.selected[data-comment="{standing_offer_id}"]').count() == 1, 'the survey standing offer card selects on its thread')
            local(page, 'x')
            check(draft(page)['eventual'] and draft(page)['eventual']['offer'] == standing_offer_id, 'comma x on the selected card drafts grounds against that offer')
            page.goto(base + '/c/2'); page.wait_for_function('window.workbenchUI !== undefined')
            check(draft(page)['id'] == 'c-2' and state(page)['mode'] == 'View', 'the standalone contribution view stays available for deeper reading')
            page.locator('.pane.active .reading a.mref').first.click()
            check(draft(page)['id'] == 't-0' and page.locator('.pane.active .thread-card.selected[data-comment="c-0"]').count() == 1, 'a citation marker opens the referenced comment in its thread with it selected')
            page.keyboard.press('Control+o')
            check(draft(page)['id'] == 'c-2', 'the jump list returns to the standalone contribution')
            page.goto(base + '/t/1'); page.wait_for_function('window.workbenchUI !== undefined')
            page.locator('.pane.active .reading').focus()
            page.keyboard.press('j'); page.keyboard.press('j')
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-5"]').count() == 1, 'j clamps at the last card')
            leader(page, 'uv')
            check(state(page)['mode'] == 'Normal' and page.locator('.pane.active .reading').is_hidden(), 'Space uv opens the read-only recorded listing as Source')
            check('c-5' in draft(page)['text'], 'the Source listing still carries the recorded contributions')
            leader(page, 'uv')
            check(state(page)['mode'] == 'View' and page.locator('.pane.active .thread-card.selected[data-comment="c-5"]').count() == 1, 'returning to View restores the cards with the selection')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            check(page.locator('.pane.active .thread-card.selected[data-comment="c-5"]').count() == 1, 'the card view and its selection survive a reload')
            page.goto(base + '/t/2'); page.wait_for_function('window.workbenchUI !== undefined')
            check(page.locator('.pane.active .thread-card .rich math').count() >= 2 and page.locator('.pane.active .thread-card .rich table').count() >= 1, 'thread cards carry the canonical math and table rendering')
            check(page.locator('.pane.active .thread-card .rich img').count() >= 1, 'thread cards render artifact pointers through the canonical renderer')
            page.set_viewport_size({'width': 1500, 'height': 420})
            # the survey's own accept: earlier contexts landed accepts on other tasks
            survey_accept = next(c['id'] for c in request(base + '/api/v1/workbench')[1]['comments'] if c['kind'] == 'accept' and c['home'] == survey_id)
            page.goto(base + f'/t/{survey_id}#{survey_accept}'); page.wait_for_function('window.workbenchUI !== undefined')
            # the scroll to the selected card rides a requestAnimationFrame;
            # a card near the thread end stops at the maximum scroll position
            page.wait_for_function('''() => {
                const reader = document.querySelector(".pane.active .reading");
                const card = document.querySelector(".pane.active .thread-card.selected");
                if (!card) return false;
                const a = card.getBoundingClientRect(), r = reader.getBoundingClientRect();
                return reader.scrollTop > 0 && a.top >= r.top - 1 && a.bottom <= r.bottom + 1;
            }''')
            check(page.locator(f'.pane.active .thread-card.selected[data-comment="{survey_accept}"]').count() == 1, 'a deep link scrolls its selected card into view beside the surrounding thread')
            page.set_viewport_size({'width': 1500, 'height': 980})
            survey_text = page.locator('.pane.active .reading').inner_text()
            check(page.locator('.pane.active .thread-card').count() == 5 and all(word in survey_text for word in ['offer', 'reject', 'accept', 'use']), 'a thread renders every recorded kind as a card')
            # shift-clicking a valid selected-comment link splits the thread
            # into a new pane, selects there, and scrolls the card into view
            tail_card = [c['id'] for c in request(base + '/api/v1/workbench')[1]['comments'] if c['home'] == 0][-1]
            page.set_viewport_size({'width': 1500, 'height': 420})
            page.goto(base + '/c/2'); page.wait_for_function('window.workbenchUI !== undefined')
            page.evaluate(f'''() => {{
                const a = document.createElement("a");
                a.href = "/t/0#{tail_card}";
                a.textContent = "deep tail";
                document.querySelector(".pane.active .reading").append(a);
            }}''')
            page.locator('.pane.active .reading a', has_text='deep tail').click(modifiers=['Shift'])
            check(len(state(page)['panes']) == 2 and draft(page)['id'] == 't-0', 'a shift-clicked selected-comment link opens the thread beside the source')
            page.wait_for_function('''() => {
                const reader = document.querySelector(".pane.active .reading");
                const card = document.querySelector(".pane.active .thread-card.selected");
                if (!card) return false;
                const a = card.getBoundingClientRect(), r = reader.getBoundingClientRect();
                return reader.scrollTop > 0 && a.top >= r.top - 1 && a.bottom <= r.bottom + 1;
            }''')
            check(page.locator(f'.pane.active .thread-card.selected[data-comment="{tail_card}"]').count() == 1, 'the shift-clicked selected card is scrolled into view in its new pane')
            page.set_viewport_size({'width': 1500, 'height': 980})
            page.keyboard.press('Escape')
            page.keyboard.press(':')
            page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('contribution')
            thread_labels = page.locator('#results button').all_inner_texts()
            page.keyboard.press('Escape')
            check(sum('Select next contribution in the thread' in label for label in thread_labels) == 1 and sum('Select previous contribution in the thread' in label for label in thread_labels) == 1, 'the palette lists the thread selection commands on a thread')
            server.cli('create', 'task', 'The empty thread survey')
            empty_id = next(t for t in request(base + '/api/v1/workbench')[1]['tasks'] if t['name'] == 'The empty thread survey')['id']
            page.goto(base + f'/t/{empty_id}'); page.wait_for_function('window.workbenchUI !== undefined')
            check(page.locator('.pane.active .thread-card').count() == 0, 'a thread without contributions shows only its task meta')
            page.locator('.pane.active .reading').focus()
            page.keyboard.press('j')
            wait_submission(page, 'no contributions')
            context.close()

            # home: the ratified activity surfaces - inclusion, ordering,
            # first-pass bounds with Show more, selected-card navigation,
            # and layout that reads the pane's own space
            server.cli('claim', 't-0')
            for room in range(6):
                server.cli('create', 'task', f'The wide room {room}')
            # a rebinding history: two bindings of one use to different
            # recorded offers, so no row may borrow the other's target
            server.cli('create', 'task', 'The binding survey')
            binding_task = next(t for t in request(base + '/api/v1/workbench')[1]['tasks'] if t['name'] == 'The binding survey')['id']
            def latest_comment():
                return request(base + '/api/v1/workbench')[1]['comments'][-1]
            server.cli('offer', f't-{binding_task}', '--title', 'The first account', 'The first usable source account.')
            first_offer = latest_comment()['id']
            server.cli('use', 't-0', '--source', f't-{binding_task}', '--reason', 'Rides the binding survey.')
            binding_use = latest_comment()['id']
            server.cli('bind', binding_use, first_offer, '--grounds', 'Select the first account.')
            server.cli('reject', first_offer, '--grounds', 'Replace the source account.')
            server.cli('offer', f't-{binding_task}', '--title', 'The second account', 'The replacement usable source account.')
            second_offer = latest_comment()['id']
            server.cli('bind', binding_use, second_offer, '--grounds', 'Select the replacement account.')
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            snap = request(base + '/api/v1/workbench')[1]
            activity = snap['recent_activity']
            machinery = {'workspace_established', 'task_workspace_checkpointed', 'workspace_created',
                         'workspace_changed', 'incarnation_registered', 'message_sent', 'message_submitted'}
            included = [r['seq'] for r in request(base + '/api/v1/records')[1]['records'] if r['kind'] not in machinery]
            seqs = [e['seq'] for e in activity]
            check(seqs == list(reversed(included)), 'recent activity is exactly the included records, newest first')
            check(all(e['event'] not in machinery for e in activity), 'no session, workspace, or message bookkeeping is home activity')
            comments_by_seq = {c['seq']: c for c in snap['comments']}
            claimed = next(e for e in activity if e['event'] == 'task_claimed')
            check(claimed['home'] == 0 and claimed['comment'] is None, 'a claim is activity on its task with no card anchor')
            offered = next(e for e in activity if e['event'] == 'commented' and comments_by_seq[e['seq']]['kind'] == 'offer')
            check(offered['comment'] is not None and offered['home'] == comments_by_seq[offered['seq']]['home'], 'an offer birth anchors its card on its home thread')
            artifact_entry = next(e for e in activity if e['event'] == 'artifact_added')
            check(artifact_entry['artifact'] and artifact_entry['comment'] is None, 'an artifact entry names its pointer, not a card')
            # each use_bound row carries the offer its own record named
            bindings = [e for e in activity if e['event'] == 'use_bound']
            nums = {c['id']: c['num'] for c in snap['comments']}
            check(len(bindings) == 2 and {e['offer'] for e in bindings} == {nums[first_offer], nums[second_offer]},
                  'use_bound entries carry their own recorded targets')
            active = snap['recently_active_tasks']
            latest = {}
            for e in reversed(activity):  # oldest first: the newest write wins
                latest[e['home']] = e['seq']
            check([t['task'] for t in active] == sorted(latest, key=lambda t: -latest[t])
                  and [t['seq'] for t in active] == [latest[t['task']] for t in active],
                  'recently active tasks order by latest included activity')
            check(len(active) == len(snap['tasks']), 'every task appears exactly once')
            # the rendered first-pass bounds: 20 entries, 10 tasks, Show
            # more in the same increments, no totals anywhere
            pane = '.pane.active '
            check(page.locator(pane + '.home-activity .home-entry').count() == 20, 'first pass renders twenty activity entries')
            check(page.locator(pane + '.home-tasks .home-entry').count() == 10, 'first pass renders ten task entries')
            check(page.locator(pane + '.home-activity .home-more').count() == 1 and page.locator(pane + '.home-tasks .home-more').count() == 1, 'both surfaces offer Show more')
            check(page.evaluate('!/\\b\\d+ of \\d+\\b/.test(document.querySelector(".pane.active .reading").innerText)'), 'no totals or badges on home')
            # rendered rows label the target each binding recorded
            rows = [text for text in page.locator(pane + '.home-activity .home-entry').all_text_contents() if 'use bound' in text]
            check(len(rows) == 2 and any(f'-> c-{nums[first_offer]}' in r for r in rows) and any(f'-> c-{nums[second_offer]}' in r for r in rows),
                  'each binding row labels its own recorded offer')
            # Show more reveals older entries in the same increments, to
            # the full included log, however long it is
            shown = 20
            while page.locator(pane + '.home-activity .home-more').count() == 1:
                page.locator(pane + '.home-activity .home-more').click()
                shown += 20
                page.wait_for_function(f'document.querySelectorAll(".pane.active .home-activity .home-entry").length === {min(shown, len(activity))}')
            check(page.locator(pane + '.home-activity .home-entry').count() == len(activity), 'activity Show more reveals older entries in twenty-entry increments to the full log')
            check(page.locator(pane + '.home-tasks .home-entry').count() == 10, 'Show more on one surface leaves the other bounded')
            shown = 10
            while page.locator(pane + '.home-tasks .home-more').count() == 1:
                page.locator(pane + '.home-tasks .home-more').click()
                shown += 10
                page.wait_for_function(f'document.querySelectorAll(".pane.active .home-tasks .home-entry").length === {min(shown, len(active))}')
            check(page.locator(pane + '.home-tasks .home-entry').count() == len(active), 'task Show more reveals older entries in ten-entry increments to the full board')
            check(page.locator(pane + '.home-more').count() == 0, 'no Show more remains once both surfaces are exhausted')
            page.screenshot(path=str(RUN / '30-home-wide.png'))
            # wide pane: tasks sit beside activity
            columns = page.locator(pane + '.home-surfaces').evaluate('node => getComputedStyle(node).gridTemplateColumns.split(" ").length')
            act_rect = page.locator(pane + '.home-activity').bounding_box()
            task_rect = page.locator(pane + '.home-tasks').bounding_box()
            check(columns == 2 and act_rect['x'] < task_rect['x'], 'a wide pane puts tasks beside activity')
            # narrow window: the surfaces stack - the pane decides
            page.set_viewport_size({'width': 620, 'height': 980})
            page.wait_for_function('getComputedStyle(document.querySelector(".pane.active .home-surfaces")).gridTemplateColumns.split(" ").length === 1')
            act_rect = page.locator(pane + '.home-activity').bounding_box()
            task_rect = page.locator(pane + '.home-tasks').bounding_box()
            check(task_rect['y'] > act_rect['y'] + act_rect['height'] - 2 and task_rect['x'] == act_rect['x'], 'a narrow pane stacks tasks below activity')
            page.screenshot(path=str(RUN / '31-home-narrow.png'))
            page.set_viewport_size({'width': 1500, 'height': 980})
            # a split keeps home narrow while the window stays wide: same
            # stacking, because the container query reads the pane
            leader(page, 'wv')
            page.wait_for_function('getComputedStyle(document.querySelector(".pane.active .home-surfaces")).gridTemplateColumns.split(" ").length === 1')
            check(page.evaluate('window.innerWidth') == 1500, 'the window is still wide inside the split')
            check(page.locator('.pane.active .home-surfaces').bounding_box()['width'] < 760, 'the home pane itself is narrow')
            page.screenshot(path=str(RUN / '32-home-split-narrow.png'))
            # the Source/View affordance reaches home like every reading kind
            page.keyboard.press('Escape')
            page.keyboard.press('Space'); page.keyboard.type('uv')
            page.wait_for_function('window.workbenchUI.state().presentation === "source"')
            check('Saccade' in page.locator('.pane.active .cm-content').text_content(), 'home Source shows the key reference')
            page.keyboard.press('Space'); page.keyboard.type('uv')
            page.wait_for_function('window.workbenchUI.state().presentation === "view"')
            check(page.locator(pane + '.home-activity').count() == 1, 'Space uv returns home to its surfaces')
            context.close()
            # navigation: an activity entry lands on its card selected in
            # the thread; an artifact entry opens the thread unselected
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            while page.locator('.pane.active .home-activity .home-more').count() == 1:
                page.locator('.pane.active .home-activity .home-more').click()
                page.wait_for_timeout(120)
            card_entry = next(e for e in snap['recent_activity'] if e['comment'] is not None)
            card_href = f"/t/{card_entry['home']}#c-{card_entry['comment']}"
            check(page.locator(f'.home-activity a[href="{card_href}"]').count() >= 1, 'a contribution birth links its card in the home thread')
            page.locator(f'.home-activity a[href="{card_href}"]').first.click()
            page.wait_for_function('document.querySelector(".pane.active .thread-card.selected") !== null')
            check(page.locator(f'.pane.active .thread-card.selected[data-comment="c-{card_entry["comment"]}"]').count() == 1, 'an activity link lands on its card selected in the thread')
            check(page.locator('#bar-info').text_content().startswith(f'selected c-{card_entry["comment"]}'), 'the bar names the arrived selection')
            check(page.locator('.pane.active .thread-card').count() > 1, 'the thread keeps its context around the selected card')
            command(page, 'home')
            task_href = f"/t/{artifact_entry['home']}"
            page.locator(f'.home-activity a[href="{task_href}"]').first.click()
            page.wait_for_function('document.querySelector(".pane.active .thread-card") !== null')
            check(page.locator('.pane.active .thread-card.selected').count() == 0, 'an artifact entry opens the thread with no card selected')
            check(page.locator('.pane.active .reading .thread-meta').first.inner_text().startswith('t-'), 'standalone thread reading is unchanged beside it')
            context.close()

            # registry parity: folds, reader scroll, and pane focus are
            # palette commands, and the keys dispatch through them
            server.cli('comment', 't-1', '--title', 'The foldable report',
                       '# First section\n\nbody one\n\n## Nested part\n\nbody two under the nested heading.')
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); ready(page, base)
            finder(page, 'foldable report'); page.keyboard.press('Enter')
            leader(page, 'uv')  # source mode
            page.keyboard.press('G')  # into the body's last section
            page.keyboard.press('z'); page.keyboard.press('a')
            page.wait_for_timeout(200)
            check(page.locator('.pane.active .cm-foldPlaceholder').count() > 0, 'za folds the section under the cursor through the fold command')
            command(page, 'Open all Markdown sections')
            page.wait_for_timeout(200)
            check(page.locator('.pane.active .cm-foldPlaceholder').count() == 0, 'the palette fold command opens every section')
            page.keyboard.press(':'); page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('fold')
            labels = page.locator('#results button').all_inner_texts()
            page.keyboard.press('Escape')
            check(sum('Markdown section' in label for label in labels) >= 5, 'the palette lists the five Markdown fold commands')
            page.keyboard.press(':'); page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.type('Focus pane')
            focus_labels = page.locator('#results button').all_inner_texts()
            page.keyboard.press('Escape')
            check(sum('Focus pane' in label for label in focus_labels) == 4, 'the palette lists the four pane focus commands')
            finder(page, 'foldable report'); page.keyboard.press('Enter')
            leader(page, 'wv')
            page.locator('.pane').nth(0).locator('.buffer-title').click()
            focus_before = state(page)['focus']
            page.keyboard.press('Escape')
            page.keyboard.down('Control'); page.keyboard.press('w'); page.keyboard.up('Control')
            page.keyboard.press('l')
            page.wait_for_timeout(150)
            check(state(page)['focus'] != focus_before, 'ctrl-w l focuses the pane to the right through its command')
            page.keyboard.press('Escape')
            command(page, 'close')
            # rendered scroll rides its registry command
            page.set_viewport_size({'width': 1500, 'height': 420})
            page.goto(base + '/c/6'); page.wait_for_function('window.workbenchUI !== undefined')
            before_scroll = page.evaluate('() => document.querySelector(".pane.active .reading").scrollTop')
            page.locator('.pane.active .reading').focus()
            page.keyboard.press('j'); page.keyboard.press('j')
            page.wait_for_timeout(150)
            after_scroll = page.evaluate('() => document.querySelector(".pane.active .reading").scrollTop')
            check(after_scroll > before_scroll, 'j scrolls the rendered pane through the reader command')
            # the transient status line sits above the one bottom bar
            order = page.evaluate('() => [document.querySelector("#message").nextElementSibling === document.querySelector("#bar"), document.querySelector("#message:empty") === null]')
            page.locator('#open-commands').click()
            page.wait_for_function('document.querySelector("#picker").open')
            page.keyboard.press('Escape')
            check(order[0], 'the status line renders directly above the single bottom bar')
            context.close()

            # task creation and incarnation launch: separate explicit
            # workbench commands on the shared prompt surface; a picker or
            # prompt Enter only stages, the command re-run acts, and the
            # launch goes through the fake host, never the production one
            code, body = request(base + '/api/v1/workspace/create', {'label': 'harness-seeded', 'client': 'workbench'})
            check(code == 200 and 'harness-seeded' in body['workspace']['branch'], 'the seeded workspace provisions through the unchanged create door')
            seeded = next(w for w in request(base + '/api/v1/workbench')[1]['workspaces'] if 'harness-seeded' in w['branch'])
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page(); page.on('pageerror', lambda error: errors.append(str(error)))
            ready(page, base)
            orca_log = server.fakebin / 'orca.log'
            orca_lines = lambda: len(orca_log.read_text().splitlines()) if orca_log.exists() else 0
            cursor_now = lambda: request(base + '/api/v1/workbench')[1]['cursor']

            # staging safety: the workspace picker writes nothing and asks
            # the host for nothing
            launch_before = cursor_now()
            host_lines = orca_lines()
            leader(page, 'nl')
            prompt_ready(page)
            check('Launch: choose the workspace' in page.locator('#picker-title').text_content(), 'the launch picker names its choice')
            check(page.locator('#results button').first.inner_text().startswith('Provision a new workspace'), 'the new-workspace choice stands first')
            page.keyboard.type('harness-seeded')
            page.wait_for_timeout(120)
            check(seeded['path'] in page.locator('#results button').first.inner_text(), 'the recorded workspace lists by its path')
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(state(page)['staged'] == {'task': None, 'launch': {'existing': seeded['path']}}, 'the picker Enter stages exactly the chosen workspace')
            check('Run Launch again' in page.locator('#message').text_content(), 'the staged choice says what completes it')
            check(cursor_now() == launch_before and orca_lines() == host_lines, 'staging wrote no record and never contacted the host')

            # the explicit re-run acts on the staged choice: honest
            # feedback, no record
            leader(page, 'nl')
            wait_submission(page, 'Submitted to the host')
            message = page.locator('#message').text_content()
            check('terminal term_fake_1' in message and seeded['path'] in message and 'registers itself' in message, f'the submission names the terminal and the workspace: {message}')
            check(cursor_now() == launch_before, 'a launch lands no record')
            check(state(page)['staged']['launch'] is None, 'the consumed choice clears')
            host_lines = orca_lines()
            check(host_lines > 0, 'the fake host saw the terminal create')
            typed = orca_log.read_text().splitlines()
            create_line = next(l for l in typed if '--command' in l)
            check(f'path:{seeded["path"]}' in create_line and '--title sac launch' in create_line, 'the submission went to the staged workspace')

            # the explicit existing-directory choice: the CLI contract on
            # the shared prompt foundation, with the recorded list as
            # suggestions only
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('directory')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            prompt_ready(page)
            check('existing directory' in page.locator('#picker-title').text_content().lower(), 'the directory action opens the naming prompt')
            page.keyboard.press('Escape')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(state(page)['staged'] == {'task': None, 'launch': None}, 'an escaped directory prompt stages nothing')
            check(cursor_now() == launch_before, 'an escaped directory prompt writes nothing')
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('directory')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            prompt_ready(page)
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            wait_submission(page, 'No directory named')
            check(state(page)['staged'] == {'task': None, 'launch': None} and cursor_now() == launch_before, 'an empty directory stages nothing')
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('directory')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            prompt_ready(page)
            page.fill('#query', str(server.repo))
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(state(page)['staged'] == {'task': None, 'launch': {'existing': str(server.repo)}}, 'a named existing directory stages exactly its path')
            check(cursor_now() == launch_before and orca_lines() == host_lines, 'staging a named directory writes nothing and contacts no host')
            leader(page, 'nl')
            wait_submission(page, 'Submitted to the host')
            check(str(server.repo) in page.locator('#message').text_content(), 'the named directory reached the launch door')
            check(any(f'path:{server.repo}' in line for line in orca_log.read_text().splitlines()), 'the fake host saw the named directory')
            check(cursor_now() == launch_before, 'the named-directory launch lands no record')
            host_lines = orca_lines()

            # a lost launch reply is an unknown outcome: the host may have
            # accepted, so the wording never claims non-submission
            lost_launch = []
            def lose_launch(route):
                lost_launch.append(route.fetch().json())
                route.abort('failed')
            page.route('**/api/v1/workbench/launch', lose_launch)
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('harness-seeded')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            leader(page, 'nl')
            wait_submission(page, 'Launch outcome unknown')
            message = page.locator('#message').text_content()
            check('may have accepted' in message and 'inspect the host' in message, f'the lost reply says outcome unknown: {message}')
            check('not submitted' not in message, 'the lost reply never claims non-submission')
            check(len(lost_launch) == 1 and lost_launch[0]['submitted'] is True, 'the lost reply really carried an accepted submission')
            check(orca_lines() > host_lines and cursor_now() == launch_before, 'the unknown outcome still landed no record')
            page.unroute('**/api/v1/workbench/launch', lose_launch)
            host_lines = orca_lines()

            # the same honesty for provisioning: a lost create reply is
            # unknown, the launch provably never started
            lost_provision = []
            def lose_provision(route):
                lost_provision.append(route.fetch().json())
                route.abort('failed')
            page.route('**/api/v1/workspace/create', lose_provision)
            provision_lost_before = cursor_now()
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('provision')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            prompt_ready(page)
            page.fill('#query', 'lost-provision')
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            leader(page, 'nl')
            wait_submission(page, 'Provisioning outcome unknown')
            message = page.locator('#message').text_content()
            check('may have provisioned' in message and 'launch was not submitted' in message, f'the lost provision names both facts: {message}')
            check(cursor_now() == provision_lost_before + 1 and lost_provision and 'lost-provision' in lost_provision[0]['workspace']['branch'], 'the lost provision really landed upstream exactly once')
            page.unroute('**/api/v1/workspace/create', lose_provision)

            # a reply that lands but names no provisioned record fails the
            # dependent launch closed: unknown wording, no fabricated facts,
            # no host contact
            def misshape_provision(route):
                response = route.fetch()
                data = response.json()
                del data['workspace']['seq']
                route.fulfill(status=200, content_type='application/json', body=json.dumps(data))
            misshape_before = cursor_now()
            host_lines = orca_lines()
            page.route('**/api/v1/workspace/create', misshape_provision)
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('provision')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            prompt_ready(page)
            page.fill('#query', 'misshape-provision')
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            leader(page, 'nl')
            wait_submission(page, 'Provisioning outcome unknown')
            message = page.locator('#message').text_content()
            check('record undefined' not in message and 'Submitted' not in message and 'may have provisioned' in message and 'launch was not submitted' in message, f'the misshapen provision fails closed: {message}')
            check(cursor_now() == misshape_before + 1, 'the misshapen reply really landed its provision upstream')
            check(orca_lines() == host_lines, 'the misshapen provision reached no host')
            page.unroute('**/api/v1/workspace/create', misshape_provision)

            # a created terminal the door cannot report: submitted, with
            # no handle to name
            (server.fakebin / 'orca-mode').write_text('no-handle')
            host_lines = orca_lines()
            nohandle_before = cursor_now()
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('harness-seeded')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            leader(page, 'nl')
            wait_submission(page, 'created a terminal but named no handle')
            message = page.locator('#message').text_content()
            check('went through' in message and 'not submitted' not in message, f'the unreportable terminal is a submission: {message}')
            check(orca_lines() > host_lines and cursor_now() == nohandle_before, 'the no-handle submission wrote no record')
            (server.fakebin / 'orca-mode').unlink()
            refuse_baseline = cursor_now()

            # an honest host refusal: the fake host fails the create
            (server.fakebin / 'orca-mode').write_text('fail')
            leader(page, 'nl')
            page.keyboard.type('harness-seeded')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            leader(page, 'nl')
            wait_submission(page, 'Launch not submitted')
            message = page.locator('#message').text_content()
            check('the host refused the create' in message, f'the host refusal is named: {message}')
            check(cursor_now() == refuse_baseline, 'a refused launch wrote no record')
            (server.fakebin / 'orca-mode').unlink()

            # a vanished directory refuses before the host is asked
            moved = seeded['path'] + '.moved'
            os.rename(seeded['path'], moved)
            host_lines = orca_lines()
            leader(page, 'nl')
            page.keyboard.type('harness-seeded')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            leader(page, 'nl')
            wait_submission(page, 'Launch not submitted')
            check(cursor_now() == refuse_baseline and orca_lines() == host_lines, 'a missing workspace refuses without host contact')
            os.rename(moved, seeded['path'])

            # with nothing staged the command re-opens the picker; the new
            # choice walks through the label prompt, and Escape there
            # stages nothing at all
            leader(page, 'nl')
            prompt_ready(page)
            page.keyboard.type('provision')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            prompt_ready(page)
            check('label' in page.locator('#picker-title').text_content().lower(), 'the new choice prompts for its optional label')
            page.keyboard.press('Escape')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(state(page)['staged'] == {'task': None, 'launch': None} and cursor_now() == refuse_baseline, 'an escaped label prompt stages nothing and writes nothing')

            # the door's origin law, through the wire face
            code, body = request(base + '/api/v1/workbench/launch', {'workspace': {'existing': seeded['path']}}, headers={'Sec-Fetch-Site': 'cross-site'})
            check(code == 403 and body['error']['code'] == 'cross_site_refused', 'a cross-site launch POST refuses')

            # task creation: the claimed name prompts first, then the
            # staged name, then the explicit re-run creates one record
            task_before = cursor_now()
            leader(page, 'nt')
            prompt_ready(page)
            check('name' in page.locator('#picker-title').text_content().lower(), 'a task creation from a fresh browser prompts for the claimed name first')
            answer_prompt(page, 'jerry')
            check(cursor_now() == task_before, 'the name prompt wrote nothing')
            leader(page, 'nt')
            prompt_ready(page)
            check('task name' in page.locator('#picker-title').text_content().lower(), 'the task prompt asks for the name')
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            wait_submission(page, 'No name given')
            check(state(page)['staged']['task'] is None and cursor_now() == task_before, 'an empty name stages nothing')
            leader(page, 'nt')
            prompt_ready(page)
            page.fill('#query', 'The browser-authored task')
            page.keyboard.press('Escape')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(state(page)['staged']['task'] is None and cursor_now() == task_before, 'an escaped name stages nothing')
            leader(page, 'nt')
            prompt_ready(page)
            page.fill('#query', 'The browser-authored task')
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(state(page)['staged']['task'] == 'The browser-authored task', 'the confirmed name is staged')
            check(cursor_now() == task_before, 'staging a name wrote no record')
            leader(page, 'nt')
            wait_submission(page, 'Created t-')
            created = next(t for t in request(base + '/api/v1/workbench')[1]['tasks'] if t['name'] == 'The browser-authored task')
            check(draft(page)['id'] == f"t-{created['id']}" and state(page)['staged']['task'] is None, 'the born thread opens and the stage clears')
            check(page.locator('#bar-kind').text_content() == 'Task', 'the bar names the created task thread')
            check(cursor_now() == task_before + 1, 'the create landed exactly one record')
            records = request(base + '/api/v1/records')[1]['records']
            born = next(r for r in records if r['kind'] == 'task_created' and r['payload']['name'] == 'The browser-authored task')
            check(born['actor'] == 'jerry' and born['tier'] == 'human', 'the create attributes the claimed name at human tier')

            # the new-workspace choice: provision through the unchanged
            # create door, then the launch, in one honest status
            provision_before = cursor_now()
            leader(page, 'nl')
            page.keyboard.type('provision')
            page.wait_for_timeout(120)
            page.keyboard.press('Enter')
            prompt_ready(page)
            page.fill('#query', 'browser-made')
            page.keyboard.press('Enter')
            page.wait_for_function('!document.querySelector("#picker").open')
            check(state(page)['staged']['launch'] == {'new': {'label': 'browser-made'}}, 'the label stages the new-workspace choice')
            check(cursor_now() == provision_before, 'staging the label wrote no record')
            leader(page, 'nl')
            wait_submission(page, 'Submitted to the host')
            message = page.locator('#message').text_content()
            check('Workspace provisioned - record' in message and 'browser-made' in message and 'terminal term_fake_1' in message, f'provision and launch report together: {message}')
            provisioned = next(w for w in request(base + '/api/v1/workbench')[1]['workspaces'] if 'browser-made' in w['branch'])
            check(cursor_now() == provision_before + 1, 'the provision landed exactly one workspace birth')
            check(Path(provisioned['path']).is_dir(), 'the provisioned worktree exists on disk')
            check(not errors, 'the create and launch flows raise no browser exceptions')
            context.close()

            # legacy prototype storage migrates by the exact db-bound key,
            # in the prototype's own serialization shape
            context = browser.new_context(viewport={'width': 1500, 'height': 980})
            page = context.new_page()
            instance = request(base + '/api/v1/workbench/identity')[1]['legacy_instance']
            import hashlib
            other_instance = hashlib.sha256(b'/tmp/another-tracker/workbench.db').hexdigest()[:16]
            legacy = {'buffers': [
                {'id': 'draft:1', 'kind': 'draft', 'title': 'Note 1', 'text': 'the legacy cursor note',
                 'references': [3, 1], 'home': 0, 'submission': None,
                 'lastPosition': {'anchor': 5, 'head': 5, 'scrollTop': 12}, 'inline': False},
                {'id': 'draft:2', 'kind': 'draft', 'title': 'Note 2', 'text': 'the unknown legacy draft',
                 'references': [], 'home': 1,
                 'submission': {'kind': 'unknown', 'seq': 9, 'message': 'Submission outcome unknown: lost. No retry will be sent; inspect the task record.'},
                 'lastPosition': {'anchor': 0, 'head': 0, 'scrollTop': 0}, 'inline': True},
                {'id': 'draft:3', 'kind': 'draft', 'title': 'Note 3', 'text': 'the landed legacy draft',
                 'references': [1, 2], 'home': 0,
                 'submission': {'kind': 'landed', 'seq': 4, 'message': 'Recorded as c-0.'},
                 'lastPosition': {'anchor': 7, 'head': 7, 'scrollTop': 3}, 'inline': False}],
                'layout': {'axis': 'row', 'ratio': 0.6, 'children': [{'view': 1}, {'view': 2}]},
                'focus': 1, 'nextView': 3, 'nextDraft': 4,
                'views': [{'id': 1, 'bufferId': 'draft:1', 'positions': [['draft:1', {'anchor': 9, 'head': 9, 'scrollTop': 40}]]},
                          {'id': 2, 'bufferId': 'draft:2', 'positions': []}]}
            other = {'buffers': [{'id': 'draft:9', 'kind': 'draft', 'title': 'Note 9', 'text': 'another tracker draft',
                                  'references': [], 'home': 0, 'submission': None}],
                     'layout': {'view': 1}, 'focus': 1, 'nextView': 2, 'nextDraft': 10,
                     'views': [{'id': 1, 'bufferId': 'draft:9', 'positions': []}]}
            cursor_before = request(base + '/api/v1/workbench')[1]['cursor']
            # seed at document-start so no earlier page load can persist
            # a fresh-format key ahead of the migration
            seed = json.dumps({'instance': instance, 'legacy': json.dumps(legacy),
                               'otherInstance': other_instance, 'other': json.dumps(other)})
            context.add_init_script(
                "const seed = " + seed + ";"
                " localStorage.setItem('saccade-web-vim:' + seed.instance, seed.legacy);"
                " localStorage.setItem('saccade-web-vim:' + seed.otherInstance, seed.other);")
            ready(page, base)
            buffers = state(page)['buffers']
            check(request(base + '/api/v1/workbench')[1]['cursor'] == cursor_before, 'legacy migration writes no records')
            by_id = {b['id']: b for b in buffers}
            check('draft:9' not in by_id, "another tracker's legacy key is not imported")
            check(by_id['draft:1']['text'] == 'the legacy cursor note' and by_id['draft:1']['home'] == 0, 'the legacy draft migrates with its text and home')
            check(by_id['draft:1']['references'] == [3, 1], 'legacy citation mappings keep their order')
            check(by_id['draft:2']['submission']['kind'] == 'unknown' and by_id['draft:3']['submission']['kind'] == 'landed', 'unknown and landed submission states ride the migration')
            check(all(b['authoredTitle'] == '' for b in buffers if b['id'].startswith('draft:')), 'generated prototype nicknames never become authored titles')
            check(state(page)['layout'] == legacy['layout'] and len(state(page)['panes']) == 2, 'the legacy split layout survives the migration')
            pane_positions = {p['id']: p['position'] for p in state(page)['panes']}
            check(pane_positions.get(1, {}).get('head') == 9 and pane_positions.get(2) is not None, 'legacy reading and cursor positions survive')
            check(page.evaluate('() => localStorage.getItem("saccade-web-vim:' + instance + '")') == json.dumps(legacy), 'the legacy key bytes remain in place after migration')
            check(page.evaluate('() => localStorage.getItem("saccade-web-vim:' + other_instance + '")') == json.dumps(other), "another tracker's legacy bytes stay untouched")
            finder(page, 'legacy cursor note')
            check(draft(page)['id'] == 'draft:1', 'the migrated draft is findable by its prose')
            command(page, 'new')
            check(draft(page)['id'] == 'draft:4', 'new drafts continue after the migrated numbering')
            # deep links apply to a restored session, navigate by hash,
            # and the startup hint never buries a missing-target naming
            page.goto(base + '/t/1'); page.wait_for_function('window.workbenchUI !== undefined')
            check(draft(page)['id'] == 't-1', 'a deep link opens its task after a restored session')
            check('names nothing recorded' not in page.locator('#message').inner_text(), 'a valid deep link leaves the default hint standing')
            page.goto(base + '/t/99'); page.wait_for_function('window.workbenchUI !== undefined')
            check('names nothing recorded' in page.locator('#message').inner_text(), 'the initial URL names its missing target instead of the hint')
            page.goto(base); page.wait_for_function('window.workbenchUI !== undefined')
            page.evaluate("location.hash = '#c-6'")
            page.wait_for_timeout(250)
            check(draft(page)['id'] == 'c-6', 'a hash link navigates the restored session to the standalone contribution')
            context.close()
            run_editor_checks(browser, base, check, request, state, draft, leader, command, RUN)
            browser.close(); browser = None
    except Exception as error:
        failed = str(error)
        traceback.print_exc()
        if page:
            try:
                page.screenshot(path=str(RUN / 'failure.png'))
                (RUN / 'failure.html').write_text(page.content())
            except Exception:
                pass
    finally:
        if browser:
            try:
                browser.close()
            except Exception as error:
                print(f'Browser connection already closed during context teardown: {error}', flush=True)
        server.close()
        (RUN / 'checks.json').write_text(json.dumps({'checks': checks, 'failure': failed}, indent=2))
        print(f'RUN {RUN}; {len(checks)} checks; failure={failed}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
