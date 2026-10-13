# Native submission architecture

[Open the standalone source demo](submission-architecture.html). It also opens
directly from a checkout, without a build step or web server.

The architecture is native HTML, CSS and JavaScript rather than a screenshot.
It uses the artifact explorer's ivory/plum light theme and charcoal dark theme,
with purple for pairwise, teal for evidence and terracotta for combined.

## What is included

- **Overview:** pairwise, evidence and 50/50 combined branches, each applied
  separately to RAG and RAGTIME.
- **Evidence flow:** saved claim–citation judgments, one shared reference and
  checklist, coverage-only grading, exact segment-ID joins and track aggregation.
- **Scoring and submission:** track-specific evidence formulas, partial support
  credit, independent rankings and the validation → package → upload → receipt
  handoff.

These are implementation proxies, not official organizer weights. This page
explains the shared-track implementation as of 12 October 2026; adding the
viewer does not add or change the judge implementation on `main`.

## Public demo versus private explorer

The public demo contains six **synthetic** package identifiers, clearly labeled
on the page. Its buttons demonstrate the artifact handoff; they are not actual
TIRA receipts or download links. It contains no saved answer text, evidence,
judgments, competition scores, team identities, credentials or receipt files.

The existing private explorer retains the real saved examples, receipt-backed
packages and optional original image diagrams. This PR does not publish that
explorer or its data, and it does not change the private deployment, perform
judging, or submit anything to TIRA.

## Integrating the component

The component uses three files:

- `submission-architecture.html`: semantic native panels and tabs.
- `assets/submission-architecture.css`: responsive component styles using the
  explorer's theme tokens. `architecture-theme.css` supplies the standalone shell.
- `assets/submission-architecture.js`: tab navigation and package activation.

Supply an `architecture-data` JSON block with `demo`, `packages`, and, for a
receipt-backed view, an explicit `snapshot`. A package needs `track`, `method`
and `group`; `code` is optional. Populate it only from authorized, verified
metadata. The component displays supplied metadata; it does not verify receipts
against TIRA.

```json
{"demo":true,"packages":[{"track":"rag","method":"pairwise","group":"DEMO-P-RAG"}]}
```

Package activation emits a bubbling event with the exact track and method:

```javascript
document.getElementById('viewer').addEventListener('architecture:package', event => {
  const {track, method} = event.detail;
  // Select that package in the host's existing artifact panel.
});
```

`architecture-demo.js` is only the synthetic adapter. Replace it in the private
explorer with its existing package selection handler; keep actual receipt data
and allowed examples private. No model calls are needed for the viewer.

## Verification

With Node.js, Playwright and its Chromium browser installed:

```bash
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node scripts/check_native_architecture.cjs
```

The browser check loads the real static demo, exercises all three tabs at
1416, 390 and 320 pixels in both themes, checks overflow and runtime errors,
tests keyboard navigation and disclosures, and checks every package activation.
It uses only the synthetic metadata, not evaluation data. `BROWSER_EXECUTABLE`
can select an installed Chromium binary; `QA_SCREENSHOTS` optionally captures
screenshots outside the repository. An optional URL argument tests a served copy.

Run the repository's existing judge tests separately:

```bash
python -m pytest tests -q
```
