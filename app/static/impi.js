// Shared helpers for the IMPI pages (scanner, market, research)
(function () {
  try {
    const t = localStorage.getItem('dl-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.dataset.theme = t;
  } catch (e) { document.documentElement.dataset.theme = 'light'; }
})();
const $ = id => document.getElementById(id);
const fmt = (v, d = 2) => v == null || Number.isNaN(v) ? '–' : Number(v).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => v == null ? '–' : (v * 100).toFixed(d) + '%';
const sgnPct = (v, d = 1) => v == null ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v * 100).toFixed(d) + '%';
const cls = v => v > 0 ? 'up' : v < 0 ? 'down' : '';
const dateStr = s => s ? new Date(s + 'T00:00:00').toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) : '–';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const ICONS = { light: '<path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/>',
  dark: '<circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>' };

function topBar(active) {
  const links = [['Terminal', '/terminal'], ['Scanner', '/scanner'], ['Market', '/market'], ['Research', '/research']];
  return `<header class="top"><a class="brand" href="/"><i></i>Delivery Lens <span>IMPI</span></a>
    <nav class="nav">${links.map(([n, h]) => `<a href="${h}" class="${n === active ? 'on' : ''}">${n}</a>`).join('')}</nav>
    <div class="right"><span>Data as of <b class="num" id="asof">–</b></span>
    <button class="icon-btn" id="themeBtn" aria-label="Light / dark theme"><svg id="themeIco" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"></svg></button></div></header>`;
}
function initTheme(onChange) {
  const ico = () => { $('themeIco').innerHTML = ICONS[document.documentElement.dataset.theme]; };
  ico();
  $('themeBtn').onclick = () => {
    const t = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem('dl-theme', t); } catch (e) {}
    ico(); if (onChange) onChange(t);
  };
  document.addEventListener('keydown', e => { if (e.key.toLowerCase() === 't' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) $('themeBtn').click(); });
}
async function getJSON(url) {
  const r = await fetch(url);
  let body = null;
  try { body = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((body && body.detail) || `HTTP ${r.status}`);
  return body;
}
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

// every IMPI page refreshes itself when the nightly job publishes a newer trading day (freshness.js)
window.addEventListener('load', () => { if (window.watchData) watchData({ url: '/api/impi/meta', read: j => j.available ? j.as_of : null }); });
