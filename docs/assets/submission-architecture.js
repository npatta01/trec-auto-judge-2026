/* Presentation only: native tabs and explicit, supplied package metadata. */
(() => {
  const root = document.getElementById('viewer');
  if (!root) return;
  const tabs = [...root.querySelectorAll('[data-figure]')];
  const panels = [...root.querySelectorAll('.architecture-panel')];
  function select(index, focus = false) {
    tabs.forEach((tab, position) => {
      const selected = position === index;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[position].hidden = !selected;
    });
    if (focus) tabs[index].focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(index));
    tab.addEventListener('keydown', event => {
      const next = {ArrowRight:(index + 1) % tabs.length, ArrowLeft:(index + tabs.length - 1) % tabs.length, Home:0, End:tabs.length - 1}[event.key];
      if (next !== undefined) {event.preventDefault(); select(next, true);}
    });
  });
  root.querySelectorAll('[data-architecture-next]').forEach(button => {
    button.addEventListener('click', () => select(Number(button.dataset.architectureNext), true));
  });
  const data = JSON.parse(document.getElementById('architecture-data').textContent);
  const methods = ['pairwise', 'evidence', 'combined'];
  const packages = [...data.packages].sort((a, b) => methods.indexOf(a.method) - methods.indexOf(b.method) || a.track.localeCompare(b.track));
  for (const pkg of packages) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.architecturePackage = `${pkg.track}:${pkg.method}`;
    const label = document.createElement('strong');
    label.textContent = `${pkg.track.toUpperCase()} · ${pkg.method[0].toUpperCase() + pkg.method.slice(1)}`;
    const proof = document.createElement('span');
    proof.textContent = data.demo ? `Synthetic package ${pkg.group}` : `Data receipt ${pkg.group} · ${pkg.code ? `code ${pkg.code} confirmed` : 'code registration pending'}`;
    button.append(label, proof);
    document.getElementById('architecture-submissions').append(button);
  }
  const confirmed = packages.filter(pkg => pkg.code).length;
  document.getElementById('architecture-receipt-summary').textContent = data.demo
    ? `${packages.length} synthetic examples · no actual submissions or receipts in this demo.`
    : `${packages.length} data packages with supplied receipts. Code registrations: ${confirmed} confirmed; ${packages.length - confirmed} pending. Saved snapshot: ${data.snapshot}.`;
  root.addEventListener('click', event => {
    const button = event.target.closest('[data-architecture-package]');
    if (!button || !root.contains(button)) return;
    const [track, method] = button.dataset.architecturePackage.split(':');
    root.dispatchEvent(new CustomEvent('architecture:package', {bubbles:true, detail:{track, method}}));
  });
})();
