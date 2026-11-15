/* Synthetic adapter. Does not load judging data, contact TIRA, or upload files. */
(() => {
  if (new URLSearchParams(location.search).get('embed') === '1') {
    const root = document.documentElement;
    root.classList.add('embedded');

    const prefersDark = matchMedia('(prefers-color-scheme: dark)');
    const syncTheme = () => { root.dataset.theme = prefersDark.matches ? 'dark' : 'light'; };
    syncTheme();
    prefersDark.addEventListener('change', syncTheme);

    const frame = window.frameElement;
    if (frame && typeof ResizeObserver === 'function') {
      const hub = document.getElementById('hub');
      // Measure the content, so the frame can shrink when a panel is closed.
      new ResizeObserver(() => {
        frame.style.height = Math.ceil(hub.getBoundingClientRect().height) + 'px';
      }).observe(hub);
    }
  }

  document.getElementById('theme').addEventListener('change', event => {
    document.documentElement.dataset.theme = event.target.value;
  });
  document.getElementById('viewer').addEventListener('architecture:package', event => {
    const {track, method} = event.detail;
    const result = document.getElementById('demo-package');
    result.hidden = false;
    result.dataset.selected = `${track}:${method}`;
    const title = document.getElementById('demo-package-title');
    title.textContent = `${track.toUpperCase()} · ${method[0].toUpperCase() + method.slice(1)} example package`;
    document.getElementById('demo-package-description').textContent = 'A production artifact panel links the selected numeric leaderboard, config, measure definitions, checksums and recorded receipt. This synthetic example illustrates that linkage without publishing saved competition content.';
    result.scrollIntoView({block:'nearest'});
    title.focus({preventScroll:true});
  });
})();
