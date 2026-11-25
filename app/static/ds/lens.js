// Lens — shared shell for every Holdfast page.
// Loaded in <head> (sync, tiny): sets the Simple/Pro mode before first paint, then exposes helpers on window.
(function () {
  const root = document.documentElement;
  root.classList.add('js');
  let mode = 'simple';
  try { mode = localStorage.getItem('dl-mode') || 'simple'; } catch (e) {}
  // a link can open a page in a given view: ?mode=pro|simple, ?theme=light|dark (not saved)
  const qs = new URLSearchParams(location.search);
  if (['pro', 'simple'].includes(qs.get('mode'))) mode = qs.get('mode');
  root.dataset.mode = mode === 'pro' ? 'pro' : 'simple';
  // theme: saved choice, else the device setting; set before first paint so there is no flash
  let theme = null;
  try { theme = localStorage.getItem('dl-theme'); } catch (e) {}
  if (['light', 'dark'].includes(qs.get('theme'))) theme = qs.get('theme');
  root.dataset.theme = theme === 'dark' || theme === 'light' ? theme : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
})();

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 2) => v == null || Number.isNaN(+v) ? '–' : Number(v).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
/** signed value with a minus sign (−) and optional suffix. v already in display units */
const sgn = (v, d = 1, suf = '%') => v == null || Number.isNaN(+v) ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d) + suf;
/** fraction (0.12) → "12.0%" */
const pct = (v, d = 1) => v == null ? '–' : (v * 100).toFixed(d) + '%';
const sgnPct = (v, d = 1) => v == null ? '–' : sgn(v * 100, d);
const cls = v => v > 0 ? 'up' : v < 0 ? 'down' : '';
/** price change with arrow, never colour alone */
const arrow = (v, d = 2, suf = '%') => v == null ? '<span class="muted">–</span>'
  : `<span class="${cls(v)}">${v > 0 ? '▲' : v < 0 ? '▼' : ''}&#8202;${Math.abs(v).toFixed(d)}${suf}</span>`;
const dstr = (s, y = 'numeric') => {
  if (!s) return '–';
  const d = /^\d{4}-\d{2}-\d{2}/.test(s) ? new Date(s.slice(0, 10) + 'T00:00:00') : new Date(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: y });
};
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  let body = null;
  try { body = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((body && body.detail) || `HTTP ${r.status}`);
  return body;
}
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const Lens = (() => {
  // ------------------------------------------------------------ verdict language
  const capScore = v => v == null ? null : v >= 95 ? Math.min(99.9, Math.round(v * 10) / 10).toFixed(1) : Math.round(v);
  const VERDICT = {
    High: { cls: 'strong', word: 'Strong', head: 'Buyers are holding', line: 'More shares than usual are being delivered — taken home, not flipped the same day.' },
    Neutral: { cls: 'normal', word: 'Normal', head: 'Usual activity', line: 'Delivery is close to what this stock normally sees.' },
    Low: { cls: 'weak', word: 'Weak', head: 'Mostly day-trading', line: 'Fewer shares than usual are being kept — most are bought and sold the same day.' },
  };
  const verdict = band => VERDICT[band] || { cls: 'plain', word: '—', head: 'Not enough history', line: 'This stock needs more trading days before it can be scored.' };
  const pill = (band, score, sm = false) => {
    const v = verdict(band);
    return `<span class="pill ${v.cls}${sm ? ' sm' : ''}">${v.word}${score != null ? ` <span class="n">${capScore(score)}</span>` : ''}</span>`;
  };

  // ------------------------------------------------------------ glossary (plain English)
  const G = {
    delivery: ['Delivery %', 'Out of all shares traded today, the share that buyers actually took home. The rest were bought and sold the same day.'],
    normal: ['Usual delivery', 'What this stock normally sees: the middle value of its delivery % over the previous 60 trading days.'],
    gap: ['Above / below usual', 'How far delivery has been above or below usual, averaged over the last 5 days. In percentage points.'],
    score: ['Conviction score', 'Today’s delivery compared with this stock’s own last 250 days, from 0 to 100. 70 or more = Strong. Under 30 = Weak. It is a description, not a prediction.'],
    zone: ['Buyer zone', 'A stretch of days when smoothed delivery stayed clearly above usual — buyers kept holding. Shaded on the chart.'],
    rise: ['Sharp rise', 'A stock that gained 20% or more within 20 trading days. We study what happened after each one.'],
    p: ['p-value', 'How likely a gap this big would appear by pure chance. Under 0.05 is the usual bar for “proven”. Above it means “maybe, not proven yet”.'],
    auc: ['AUC', 'How well a model ranks winners above losers. 0.50 is a coin flip, 1.00 is perfect. Above 0.70 is useful.'],
    breakout: ['Breakout chance', 'Model estimate that the price rises 2 ATR (a normal-size move) before it falls 1 ATR, within 10 days. This model is weak — treat it as a hint.'],
    vol: ['Big-move chance', 'Model estimate that the next 10 days will be 1.5× more volatile than recent days. Says nothing about direction. This is the model that works best (AUC 0.77).'],
    rank: ['Rank vs F&O', 'Where today’s model score sits among all F&O stocks, 0–100. Higher = more likely to beat the typical stock over 10 days. Weak signal.'],
    delz: ['Delivery vs its year', 'Recent delivery compared with this stock’s past year, in standard deviations (σ). +1 σ = clearly above normal.'],
    bbw: ['Squeeze', 'How tight the price range is compared with the past year (Bollinger width percentile). Low = coiled, quiet price.'],
    oi: ['Futures open interest', 'How many futures contracts are open. Rising OI = new positions being built.'],
    buildup: ['OI build-up', 'Price and open interest together. Long build-up: price ↑ OI ↑. Short covering: price ↑ OI ↓. Short build-up: price ↓ OI ↑. Long unwinding: price ↓ OI ↓.'],
    vcp: ['VCP', 'Volatility contraction pattern: a base where each pull-back is smaller than the last, before a breakout.'],
    cot: ['Positioning range', 'Where a group’s futures position sits within its own past 52 weeks: 0 = most bearish of the year, 100 = most bullish.'],
    fii: ['FII', 'Foreign institutional investors — overseas funds trading on the NSE.'],
    dii: ['DII', 'Domestic institutional investors — Indian mutual funds, insurers and banks.'],
    pro: ['Pro', 'Brokers’ proprietary desks trading their own money.'],
    client: ['Client', 'Everyone else: retail traders and other non-institutional clients.'],
    vix: ['India VIX', 'The market’s expected volatility for the next 30 days. High = fear.'],
    breadth: ['Breadth', 'Of all stocks, how many rose minus how many fell over the last 10 days. Positive = most stocks rising.'],
    base: ['Base rate', 'How often the outcome happened for an average stock. A model is only useful if it beats this.'],
    pre: ['Pre-registered', 'The rules were written and saved before looking at the test data, so the result could not be tuned to look good.'],
    posthoc: ['Post-hoc', 'Decided after seeing the data. Interesting, but needs a fresh test before anyone should trust it.'],
    impi: ['IMPI', 'Our research model for F&O stocks. It combines delivery, futures positioning and price compression.'],
    acc: ['Accumulation / distribution', 'High delivery on an up day = accumulation (buyers keeping shares). High delivery on a down day = distribution (holders letting go).'],
  };
  const qm = key => G[key] ? `<button type="button" class="qm" data-x="${key}" aria-label="What is ${esc(G[key][0])}?">?</button>` : '';

  let pop = null, popFor = null;
  function closePop() { if (pop) pop.classList.remove('open'); if (popFor) popFor.classList.remove('open'); popFor = null; }
  function openPop(btn) {
    if (!pop) { pop = document.createElement('div'); pop.className = 'pop'; pop.setAttribute('role', 'tooltip'); document.body.appendChild(pop); }
    const [t, body] = G[btn.dataset.x] || ['', ''];
    pop.innerHTML = `<b>${esc(t)}</b>${esc(body)}`;
    if (popFor) popFor.classList.remove('open');
    popFor = btn; btn.classList.add('open');
    const r = btn.getBoundingClientRect(), w = Math.min(300, innerWidth - 24);
    pop.style.maxWidth = w + 'px';
    pop.classList.add('open');
    const ph = pop.offsetHeight, pw = pop.offsetWidth;
    let x = Math.min(innerWidth - pw - 12, Math.max(12, r.left + r.width / 2 - pw / 2));
    let y = r.bottom + 10; if (y + ph > innerHeight - 12) y = r.top - ph - 10;
    pop.style.left = x + 'px'; pop.style.top = y + 'px';
  }
  document.addEventListener('click', e => {
    const b = e.target.closest('.qm');
    if (b) { e.preventDefault(); e.stopPropagation(); popFor === b ? closePop() : openPop(b); return; }
    if (pop && !e.target.closest('.pop')) closePop();
  });
  document.addEventListener('mouseover', e => { const b = e.target.closest?.('.qm'); if (b && matchMedia('(hover:hover)').matches) openPop(b); });
  document.addEventListener('mouseout', e => { const b = e.target.closest?.('.qm'); if (b && b === popFor && matchMedia('(hover:hover)').matches && !e.relatedTarget?.closest?.('.pop')) closePop(); });
  addEventListener('scroll', closePop, { passive: true, capture: true });

  // ------------------------------------------------------------ mode
  const modeSubs = [];
  const getMode = () => document.documentElement.dataset.mode;
  function setMode(m) {
    if (m === getMode()) return;
    const go = () => {
      document.documentElement.dataset.mode = m;
      try { localStorage.setItem('dl-mode', m); } catch (e) {}
      document.querySelectorAll('.seg.mode').forEach(s => s._set && s._set(m, false));
      modeSubs.forEach(f => f(m));
    };
    if (document.startViewTransition && !reduceMotion) document.startViewTransition(go); else go();
  }
  const onMode = f => modeSubs.push(f);

  // ------------------------------------------------------------ theme (light / dark) with a circular reveal from the toggle
  const themeSubs = [];
  const getTheme = () => document.documentElement.dataset.theme;
  function setTheme(t, from) {
    if (t === getTheme()) return;
    const root = document.documentElement;
    const go = () => {
      root.dataset.theme = t;
      try { localStorage.setItem('dl-theme', t); } catch (e) {}
      themeSubs.forEach(f => { try { f(t); } catch (e) { console.error(e); } });
    };
    if (document.startViewTransition && !reduceMotion) {
      if (from) { const r = from.getBoundingClientRect(); root.style.setProperty('--tx', r.left + r.width / 2 + 'px'); root.style.setProperty('--ty', r.top + r.height / 2 + 'px'); }
      root.classList.add('theme-swap');
      const vt = document.startViewTransition(go);
      vt.finished.finally(() => root.classList.remove('theme-swap'));
    } else go();
  }
  const onTheme = f => themeSubs.push(f);

  // ------------------------------------------------------------ segmented control with sliding thumb
  function seg(el, onChange, attr) {
    attr = attr || Object.keys(el.querySelector('button')?.dataset || {})[0];
    let thumb = el.querySelector('.thumb');
    if (!thumb) { thumb = document.createElement('span'); thumb.className = 'thumb'; el.prepend(thumb); }
    const place = () => {
      const on = el.querySelector('button.on'); if (!on) { thumb.style.width = '0'; return; }
      thumb.style.width = on.offsetWidth + 'px'; thumb.style.transform = `translateX(${on.offsetLeft}px)`;
    };
    el._set = (val, fire = true) => {
      el.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset[attr] === String(val)));
      place(); if (fire && onChange) onChange(val);
    };
    el.addEventListener('click', e => { const b = e.target.closest('button'); if (b && b.dataset[attr] != null && !b.classList.contains('on')) el._set(b.dataset[attr]); });
    new ResizeObserver(place).observe(el);
    if (document.fonts) document.fonts.ready.then(place);
    requestAnimationFrame(place);
    return el;
  }

  // ------------------------------------------------------------ nav
  const LOGO = `<svg width="26" height="26" viewBox="0 0 28 28" aria-hidden="true"><rect x="3" y="3" width="22" height="22" rx="7" fill="none" stroke="currentColor" stroke-width="2.6"/><rect x="8" y="13" width="12" height="7.5" rx="2.4" fill="#0071E3"/></svg>`;
  const LINKS = [['Home', '/'], ['Terminal', '/terminal'], ['Scanner', '/scanner'], ['Market', '/market'], ['Research', '/research'], ['About', '/about']];
  function nav(active, opts = {}) {
    const host = document.getElementById('nav') || document.body.insertBefore(document.createElement('header'), document.body.firstChild);
    host.id = 'nav'; host.className = 'nav' + (opts.full ? ' full solid' : '');
    host.innerHTML = `<div class="nav-in">
      <a class="brand" href="/" aria-label="Holdfast home">${LOGO}<span>Holdfast</span></a>
      <nav class="links" aria-label="Main"><span class="hover-pill"></span>${LINKS.map(([n, h]) => `<a href="${h}" class="${n === active ? 'on' : ''}"${n === active ? ' aria-current="page"' : ''}>${n}</a>`).join('')}</nav>
      <div class="nav-r">
        ${opts.asof === false ? '' : '<span class="asof">Data <b class="num" id="asof">–</b></span>'}
        ${opts.mode ? `<div class="seg mode sm" id="modeSeg" role="group" aria-label="View mode"><button data-m="simple" title="Plain-English view">Simple</button><button data-m="pro" title="Full trading terminal">Pro</button></div>` : ''}
        <button class="icon-btn theme-btn" id="themeBtn" aria-label="Switch light / dark (T)" title="Light / dark (T)"><svg class="moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg><svg class="sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2M12 19.5v2M4.6 4.6 6 6M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4"/></svg></button>
        <button class="search-btn" id="kOpen" aria-label="Search stocks"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg><span>Search a stock</span><kbd>/</kbd></button>
        <button class="menu-btn" id="menuBtn" aria-label="Menu" aria-expanded="false"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 8h16M4 16h16"/></svg></button>
      </div></div>`;
    const sheet = document.createElement('div'); sheet.className = 'mob-sheet'; sheet.id = 'mobSheet';
    sheet.innerHTML = LINKS.map(([n, h]) => `<a href="${h}" class="${n === active ? 'on' : ''}">${n}</a>`).join('');
    document.body.appendChild(sheet);
    $('menuBtn').onclick = () => { const o = sheet.classList.toggle('open'); $('menuBtn').setAttribute('aria-expanded', o); };
    // hover pill follows the pointer between links
    const links = host.querySelector('.links'), hp = links.querySelector('.hover-pill');
    links.querySelectorAll('a').forEach(a => a.addEventListener('mouseenter', () => {
      hp.style.width = a.offsetWidth + 'px'; hp.style.transform = `translateX(${a.offsetLeft}px)`; hp.style.opacity = 1;
    }));
    links.addEventListener('mouseleave', () => { hp.style.opacity = 0; });
    if (opts.mode) { const s = seg($('modeSeg'), m => setMode(m), 'm'); s._set(getMode(), false); }
    $('kOpen').onclick = () => palette.open();
    $('themeBtn').onclick = e => setTheme(getTheme() === 'dark' ? 'light' : 'dark', e.currentTarget);
    if (!opts.full) {
      const onS = () => host.classList.toggle('scrolled', scrollY > 4);
      addEventListener('scroll', onS, { passive: true }); onS();
    }
    return host;
  }

  // ------------------------------------------------------------ search palette ("/" or Ctrl+K)
  const palette = (() => {
    let bg, box, input, list, rows = null, hits = [], sel = 0, onPick = null;
    function build() {
      bg = document.createElement('div'); bg.className = 'palette-bg';
      box = document.createElement('div'); box.className = 'palette'; box.setAttribute('role', 'dialog'); box.setAttribute('aria-label', 'Search stocks');
      box.innerHTML = `<div class="pin"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" style="color:var(--ink-3)" stroke-width="2.2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>
        <input placeholder="Type a company or symbol — e.g. Tata, INFY" autocomplete="off" spellcheck="false" aria-label="Search"><kbd>Esc</kbd></div>
        <div class="plist" role="listbox"></div>
        <div class="pfoot"><span><kbd>↑</kbd> <kbd>↓</kbd> move</span><span><kbd>Enter</kbd> open</span><span>Verdict = how much buying is real, today</span></div>`;
      document.body.append(bg, box);
      input = box.querySelector('input'); list = box.querySelector('.plist');
      bg.onclick = close;
      input.oninput = search;
      input.onkeydown = e => {
        if (e.key === 'ArrowDown') { sel = Math.min(hits.length - 1, sel + 1); draw(); e.preventDefault(); }
        else if (e.key === 'ArrowUp') { sel = Math.max(0, sel - 1); draw(); e.preventDefault(); }
        else if (e.key === 'Enter' && hits[sel]) go(hits[sel].symbol);
        else if (e.key === 'Escape') close();
      };
      list.onmousedown = e => { const it = e.target.closest('[data-s]'); if (it) { e.preventDefault(); go(it.dataset.s); } };
      list.onmousemove = e => { const it = e.target.closest('[data-i]'); if (it && +it.dataset.i !== sel) { sel = +it.dataset.i; draw(); } };
    }
    async function load() {
      if (rows) return rows;
      try { rows = (await getJSON('/api/watchlist')).rows; } catch (e) { rows = []; }
      return rows;
    }
    function search() {
      const q = input.value.trim().toUpperCase();
      if (!rows) { list.innerHTML = '<div class="phint">Loading stocks…</div>'; return; }
      if (!q) {
        hits = rows.filter(r => r.band === 'High').sort((a, b) => b.score - a.score).slice(0, 8);
        sel = 0; draw('Strongest delivery today'); return;
      }
      const starts = rows.filter(r => r.symbol.startsWith(q)), name = rows.filter(r => !r.symbol.startsWith(q) && (r.name || '').toUpperCase().includes(q));
      hits = starts.concat(name).slice(0, 12); sel = 0; draw();
    }
    function draw(title) {
      if (!hits.length) { list.innerHTML = `<div class="phint">No stock matches “${esc(input.value)}”. Try the company name or NSE symbol.</div>`; return; }
      list.innerHTML = (title ? `<div class="phint" style="padding:10px 14px 6px">${title}</div>` : '') + hits.map((r, i) => `<div class="pitem ${i === sel ? 'sel' : ''}" role="option" data-s="${esc(r.symbol)}" data-i="${i}">
        <span><b>${esc(r.symbol)}</b><small>${esc(r.name)}</small></span>
        <span class="num small">${fmt(r.close)} ${arrow(r.chg_1d)}</span>${pill(r.band, r.score, true)}</div>`).join('');
      list.querySelector('.sel')?.scrollIntoView({ block: 'nearest' });
    }
    function go(sym) { close(); if (onPick) onPick(sym); else location.href = '/terminal#' + encodeURIComponent(sym); }
    async function open() {
      if (!box) build();
      bg.classList.add('open'); box.classList.add('open'); input.value = ''; hits = [];
      setTimeout(() => input.focus(), 30);
      search(); await load(); search();
    }
    function close() { if (!box) return; bg.classList.remove('open'); box.classList.remove('open'); input.blur(); }
    const isOpen = () => !!box && box.classList.contains('open');
    return { open, close, isOpen, setRows: r => { rows = r; }, onPick: f => { onPick = f; } };
  })();
  document.addEventListener('keydown', e => {
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
    if ((e.key === 'k' && (e.ctrlKey || e.metaKey)) || (e.key === '/' && !typing)) { e.preventDefault(); palette.open(); }
    if (e.key.toLowerCase() === 't' && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) { const b = document.getElementById('themeBtn'); setTheme(getTheme() === 'dark' ? 'light' : 'dark', b); }
    if (e.key === 'Escape') { closePop(); document.getElementById('mobSheet')?.classList.remove('open'); }
  });

  // ------------------------------------------------------------ motion helpers
  function reveal(scope = document) {
    const els = scope.querySelectorAll('.rv:not(.in)');
    if (reduceMotion || !('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('in')); return; }
    const io = new IntersectionObserver(es => es.forEach(en => {
      if (!en.isIntersecting) return; en.target.classList.add('in'); io.unobserve(en.target);
      en.target.dispatchEvent(new CustomEvent('reveal'));
    }), { threshold: .12, rootMargin: '0px 0px -6% 0px' });
    els.forEach(e => io.observe(e));
  }
  /** roll a number from its previous value to `to` */
  function tween(el, to, f = v => Math.round(v), dur = 900) {
    if (!el) return;
    const from = parseFloat(el.dataset.v ?? 0); el.dataset.v = to;
    if (reduceMotion || to == null || Number.isNaN(+to) || Number.isNaN(from) || !dur) { el.textContent = to == null ? '–' : f(+to); return; }
    const t0 = performance.now(), tk = ++el._tk || (el._tk = 1);
    (function step(t) {
      if (el._tk !== tk) return;
      const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 4);
      el.textContent = f(from + (to - from) * e); if (p < 1) requestAnimationFrame(step);
    })(t0);
  }
  /** FLIP: animate children (with data-key) from old to new positions around a DOM update */
  function flip(container, update, dur = 520) {
    if (reduceMotion) return update();
    const first = new Map();
    container.querySelectorAll('[data-key]').forEach(n => first.set(n.dataset.key, n.getBoundingClientRect().top));
    update();
    const vh = innerHeight;
    container.querySelectorAll('[data-key]').forEach(n => {
      const r = n.getBoundingClientRect(); if (r.top > vh + 200 || r.bottom < -200) return;
      const was = first.get(n.dataset.key);
      if (was == null) { n.animate([{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { duration: dur * .8, easing: 'cubic-bezier(.16,1,.3,1)' }); return; }
      const dy = was - r.top; if (Math.abs(dy) < 1) return;
      n.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: dur, easing: 'cubic-bezier(.16,1,.3,1)' });
    });
  }
  /** stagger children in */
  function stagger(container, sel = ':scope > *', step = 28, max = 18) {
    if (reduceMotion) return;
    [...container.querySelectorAll(sel)].slice(0, max).forEach((n, i) =>
      n.animate([{ opacity: 0, transform: 'translateY(10px)' }, { opacity: 1, transform: 'none' }], { duration: 560, delay: i * step, easing: 'cubic-bezier(.16,1,.3,1)', fill: 'backwards' }));
  }
  /** ring gauge */
  function ring(el, size = 132, stroke = 11) {
    const r = (size - stroke) / 2, c = 2 * Math.PI * r;
    el.classList.add('ring'); el.style.width = el.style.height = size + 'px';
    el.innerHTML = `<svg viewBox="0 0 ${size} ${size}"><circle class="bg" cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke-width="${stroke}"/>
      <circle class="fg" cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke-width="${stroke}" stroke-linecap="round" stroke-dasharray="0 ${c}"/></svg>
      <div class="val"><b class="num">–</b><small>of 100</small></div>`;
    const fg = el.querySelector('.fg'), b = el.querySelector('b');
    return { set(v, color) {
      const x = v == null ? 0 : Math.max(0, Math.min(100, +v));
      fg.setAttribute('stroke-dasharray', `${(x / 100 * c).toFixed(1)} ${c}`); if (color) fg.style.stroke = color;
      tween(b, v == null ? null : +v, n => +v >= 95 ? n.toFixed(1) : Math.round(n));
    } };
  }
  /** give an element a view-transition name right before navigating, so it morphs into the next page */
  function morph(el, name = 'sym-hero') { if (el) el.style.viewTransitionName = name; }
  function toast(msg, ms = 3200) {
    const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg; t.setAttribute('role', 'status');
    document.body.appendChild(t); requestAnimationFrame(() => t.classList.add('open'));
    setTimeout(() => { t.classList.remove('open'); setTimeout(() => t.remove(), 400); }, ms);
  }
  function footer() {
    const f = document.createElement('footer'); f.className = 'foot';
    f.innerHTML = `<div class="wrap"><div class="row" style="gap:10px">${LOGO.replace('width="26" height="26"', 'width="20" height="20"')}<span><b style="color:var(--ink)">Holdfast</b> · Research and education tool, not investment advice.</span></div>
      <nav>${LINKS.map(([n, h]) => `<a href="${h}">${n}</a>`).join('')}</nav></div>`;
    document.body.appendChild(f);
  }
  // restore scroll-based reveal on back/forward cache
  addEventListener('pageshow', e => { if (e.persisted) document.querySelectorAll('.rv').forEach(x => x.classList.add('in')); });

  return { G, qm, verdict, pill, capScore, nav, seg, getMode, setMode, onMode, getTheme, setTheme, onTheme, palette, reveal, tween, flip, stagger, ring, morph, toast, footer, LOGO };
})();
