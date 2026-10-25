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


def review_ready(page):
    page.wait_for_function('document.querySelector("#publication").open')


def wait_submission(page, word):
    page.wait_for_function('(word) => document.querySelector("#publication-status").textContent.includes(word)', arg=word)


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
            check(len(state(page)['panes']) == 1 and state(page)['mode'] == 'Normal', 'one text-first home pane opens in Normal mode')
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
            check(not page.evaluate("() => document.querySelector('#publication').open"), 'accept is unavailable on a contribution that is not the standing offer')
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
            local(page, 'r'); review_ready(page)
            check(page.locator('#publish').is_disabled() and 'unset' in page.locator('#review-title').inner_text(), 'publication refuses a note without an authored title')
            page.locator('#edit-note').click()
            local(page, 't')
            page.wait_for_function('document.querySelector("#title-dialog").open')
            page.fill('#title-input', 'The cursor rule under review')
            page.locator('#title-apply').click()
            check(draft(page)['authoredTitle'] == 'The cursor rule under review', 'comma t authors the distinct title field')
            local(page, 'r'); review_ready(page)
            check('The cursor rule under review' in page.locator('#review-title').inner_text(), 'review shows the authored title it will record')

            check(page.locator('#publish').is_disabled() and page.locator('#review-actor-name').is_visible(), 'a missing claimed name keeps publication disabled until the act names it')
            page.fill('#review-actor-name', 'jerry')
            check(page.locator('#review-body').count() == 0 and page.locator('#publish').is_enabled(), 'Review confirms publication details without duplicating or requiring preview')
            for _ in range(20):
                if page.evaluate('document.activeElement.id') == 'publish':
                    break
                page.keyboard.press('Tab')
            check(page.evaluate('document.activeElement.id') == 'publish', 'publish is reachable through ordinary keyboard focus')
            page.keyboard.press('Enter'); wait_submission(page, 'Recorded as')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == before + 1, 'keyboard-only flow publishes exactly one real note')
            landed = request(base + '/api/v1/workbench')[1]['comments'][-1]
            check(landed['title'] == 'The cursor rule under review' and landed['kind'] == 'note', 'the landed note carries the authored title')
            check(page.locator('#publish').is_disabled(), 'confirmed publication prevents a duplicate submission')
            page.screenshot(path=str(RUN / '04-keyboard-publication.png'))
            page.keyboard.press('Escape')
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
            local(page, 't')
            page.wait_for_function('document.querySelector("#title-dialog").open')
            page.fill('#title-input', 'The mouse-authored finding'); page.locator('#title-apply').click()
            local(page, 'r'); review_ready(page)
            check('t-1' in page.locator('#review-home option:checked').inner_text(), 'draft home stays explicit in publication review')
            page.fill('#review-actor-name', 'jerry')
            mouse_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.locator('#publish').click(); wait_submission(page, 'Recorded as')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == mouse_before + 1, 'mouse-driven flow publishes exactly one note')
            page.locator('#edit-note').click()
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
            local(page, 't')
            page.wait_for_function('document.querySelector("#title-dialog").open')
            page.fill('#title-input', 'The refused marker note'); page.locator('#title-apply').click()
            local(page, 'r'); review_ready(page)
            page.fill('#review-actor-name', 'jerry')
            refusal_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.locator('#publish').click(); wait_submission(page, 'Not sent')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == refusal_before, 'actual core marker refusal lands no record')
            page.locator('#edit-note').click()
            check(draft(page)['text'] == rejected_text, 'core refusal preserves the draft')
            # a refused submission can be corrected and retried
            page.locator('.pane.active').click()
            page.keyboard.type('ggdG'); page.keyboard.press('i'); page.keyboard.type('A resolved body for the corrected retry.')
            page.keyboard.press('Escape')
            check('[+]' in page.locator('#bar').inner_text(), 'a corrected refusal shows the modified flag')
            local(page, 'r'); review_ready(page)
            check(page.locator('#publish').is_enabled(), 'a proved core refusal enables the corrected retry')
            page.locator('#publish').click(); wait_submission(page, 'Recorded as')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == refusal_before + 1, 'the corrected retry lands exactly one note')
            page.keyboard.press('Escape')
            command(page, 'new'); page.keyboard.press('i'); page.keyboard.type('A response can be lost after the note lands.')
            local(page, 't')
            page.wait_for_function('document.querySelector("#title-dialog").open')
            page.fill('#title-input', 'The lost response note'); page.locator('#title-apply').click()
            local(page, 'r'); review_ready(page)
            page.fill('#review-actor-name', 'jerry')
            unknown_before = request(base + '/api/v1/workbench')[1]['cursor']

            def lose_response(route):
                route.fetch()
                route.abort('failed')
            page.route('**/api/v1/workbench/command', lose_response)
            page.locator('#publish').click(); wait_submission(page, 'outcome unknown')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_before + 1, 'response-loss scenario really lands the note upstream')
            check(page.locator('#publish').is_disabled(), 'unknown outcome blocks retry')
            page.locator('#edit-note').click()
            check(draft(page)['text'] == 'A response can be lost after the note lands.', 'lost response preserves draft text')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            local(page, 'r'); review_ready(page)
            check(page.locator('#publish').is_disabled() and 'unknown' in page.locator('#publication-status').inner_text(), 'unknown state survives reload without resending')
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
            local(page, 'r'); review_ready(page)
            check(page.locator('#publish').is_disabled(), 'draft without a home or title cannot be published')
            page.locator('#review-home').select_option('0')
            check(page.locator('#publish').is_disabled(), 'a home alone cannot replace the authored title')
            page.locator('#edit-note').click()
            local(page, 't')
            page.wait_for_function('document.querySelector("#title-dialog").open')
            page.fill('#title-input', 'The narrow view note'); page.locator('#title-apply').click()
            local(page, 'r'); review_ready(page)
            page.fill('#review-actor-name', 'jerry')
            check(page.locator('#publish').is_enabled(), 'explicit home title and claimed name enable the reviewed draft action')
            page.keyboard.press('Escape')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_before + 1, 'narrow-view review and home selection write nothing')
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
            local(page, 't')
            page.wait_for_function('document.querySelector("#title-dialog").open')
            page.fill('#title-input', 'The preview note'); page.locator('#title-apply').click()
            local(page, 'r'); review_ready(page)
            page.fill('#review-actor-name', 'jerry')
            check(page.locator('#publish').is_enabled() and page.locator('#review-body').count() == 0, 'publication confirmation is independent of rendered preview')
            page.locator('.pane.active .cm-content').evaluate('(node) => { node.textContent = "Changed after confirmation."; node.dispatchEvent(new InputEvent("input", {bubbles: true, inputType: "insertText", data: "Changed after confirmation."})); }')
            page.wait_for_function('window.workbenchUI.state().buffers.some(b => b.text === "Changed after confirmation.")')
            check(page.locator('#publish').is_disabled(), 'editing a draft invalidates its open publication confirmation')
            page.locator('#review-home').select_option('1')
            check(page.locator('#publish').is_disabled(), 'changing home cannot revalidate a changed draft body')
            page.locator('#edit-note').click(); local(page, 'r'); review_ready(page)
            check(page.locator('#publish').is_enabled(), 'reviewing the current draft creates a fresh publication confirmation')
            page.locator('#edit-note').click()
            page.keyboard.press('Escape')
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
            local(page, 'r'); review_ready(page)
            if page.locator('#review-actor-name').is_visible():
                page.fill('#review-actor-name', 'jerry')
            inflight_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.locator('#publish').click()
            page.wait_for_function('window.workbenchNoteHeld === true')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == inflight_before + 1, 'in-flight reload scenario lands one real upstream note')
            check(page.evaluate('JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k => k.startsWith("saccade-workbench:")))).buffers.find(b => b.kind === "draft").submission.kind') == 'unknown', 'unconfirmed submission state is saved before its answer arrives')
            page.reload(); page.wait_for_function('window.workbenchUI !== undefined')
            local(page, 'r'); review_ready(page)
            check(page.locator('#publish').is_disabled(), 'reload during a pending answer cannot silently enable resubmission')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == inflight_before + 1, 'in-flight reload creates no second record')
            held_note[0].abort('failed')
            page.unroute('**/workbench/command', delay_note_answer)
            page.locator('#edit-note').click()
            command(page, 'new')
            page.keyboard.press('i'); page.keyboard.type('Keep this draft when local persistence fails.')
            local(page, 't')
            page.wait_for_function('document.querySelector("#title-dialog").open')
            page.fill('#title-input', 'The persistence-failure note'); page.locator('#title-apply').click()
            local(page, 'r'); review_ready(page)
            if page.locator('#review-actor-name').is_visible():
                page.fill('#review-actor-name', 'jerry')
            blocked_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.evaluate('() => { Storage.prototype.setItem = function() { throw new DOMException("local quota", "QuotaExceededError") }; }')
            page.locator('#publish').click(); wait_submission(page, 'Submission not attempted')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == blocked_before, 'failed local submission-state save prevents forwarding')
            page.locator('#edit-note').click()
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
            local(page, 'a'); review_ready(page)
            page.fill('#review-actor-name', 'jerry')
            accept_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.locator('#publish').click(); wait_submission(page, 'Recorded')
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
            local(page, 'r'); review_ready(page)
            if page.locator('#review-actor-name').is_visible():
                page.fill('#review-actor-name', 'jerry')
            reject_before = request(base + '/api/v1/workbench')[1]['cursor']
            page.locator('#publish').click(); wait_submission(page, 'Not sent')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == reject_before, 'the empty-grounds refusal wrote nothing')
            page.locator('#edit-note').click()
            page.keyboard.press('i'); page.keyboard.type('The edge case still fails on the empty grounds door.')
            page.wait_for_function("() => window.workbenchUI.state().buffers.some((b) => b.eventual && b.eventual.kind === 'reject' && b.text.includes('edge case still fails'))")
            local(page, 'r'); review_ready(page)
            check(page.locator('#publish').is_enabled(), 'the corrected grounds enable publication')
            page.locator('#publish').click(); wait_submission(page, 'Recorded')
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
            local(page, 'r'); review_ready(page)
            unknown_cursor = request(base + '/api/v1/workbench')[1]['cursor']
            page.route('**/api/v1/workbench/command', lambda route: (route.fetch(), route.abort('failed')))
            page.locator('#publish').click(); wait_submission(page, 'outcome unknown')
            check(request(base + '/api/v1/workbench')[1]['cursor'] == unknown_cursor + 1, 'the lost reject reply really landed upstream')
            check(page.locator('#publish').is_disabled(), 'an unknown reject outcome blocks retry')
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
            local(page, 'r'); review_ready(page)
            if page.locator('#review-actor-name').is_visible():
                page.fill('#review-actor-name', 'jerry')
            server.cli('revise', 'c-0', '--title', 'The concurrently retitled bookmark')
            pending_body = draft(page)['text']
            page.locator('#publish').click(); wait_submission(page, 'Recorded')
            after_revive = request(base + '/api/v1/workbench')[1]
            revised_now = next(c for c in after_revive['comments'] if c['id'] == 'c-0')
            check(revised_now['title'] == 'The concurrently retitled bookmark', 'a concurrently revised title stands after the body patch lands')
            check('The retry lands both fields' in revised_now['body'] and 'Unsaved tail.' in revised_now['body'], 'the whole pending body lands after the reload')
            page.keyboard.press('Escape')
            local(page, 'e')
            check(draft(page)['revisionPending'] is not None, 'a fresh repair starts after the last landed')
            local(page, 'e')
            check(draft(page)['revisionPending'] is None, 'comma e again discards the pending repair as the status promises')
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
            bar = lambda: [page.locator('#bar-mode').inner_text(), page.locator('#bar-kind').inner_text(), page.locator('#bar-ident').inner_text(), page.locator('#bar-info').inner_text(), page.locator('#bar-flags').inner_text()]
            check(bar() == ['NORMAL', 'Home', 'home', '', '[-]'], 'home opens with the exact mode-kind-ident-info-flags bar tuple')
            page.goto(base + f'/t/{survey_id}'); page.wait_for_function('window.workbenchUI !== undefined')
            check(bar()[:3] == ['NORMAL', 'Task', f't-{survey_id}'], 'a task buffer shows the Task kind')
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
            finder(page, 'insertion bookmark'); page.keyboard.press('Enter')
            check(bar()[:3] == ['RENDERED', 'Note', 'c-0'], 'a note shows its Note kind')
            local(page, 'e')
            check(bar() == ['NORMAL', 'Note', 'c-0', 'revising', '[+]'] or bar()[:4] == ['NORMAL', 'Note', 'c-0', 'revising'], 'a repair keeps the record kind with the revising info')
            local(page, 'e')
            finder(page, 'kind standing offer'); page.keyboard.press('Enter')
            check(draft(page)['id'] == standing_offer_id, 'the standing survey offer opens for the draft-intent tuple')
            local(page, 'x')
            tuple_now = bar()
            check(tuple_now[1] == 'Reject' and tuple_now[2].startswith('draft:') and tuple_now[3].startswith('rejecting c-'), 'a grounds draft shows the Reject intent kind with its draft ident')
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
            page.evaluate("location.hash = '#c-6'")
            page.wait_for_timeout(250)
            check(draft(page)['id'] == 'c-6', 'a hash link navigates the restored session')
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
