// Data freshness for every page: shows when the next nightly update is due, flags an overdue update, and refreshes
// the page's data by itself when the nightly job (Mon-Fri 7:30 PM, after NSE publishes) has produced a newer day.
(function () {
  const UPDATE_H = 19, UPDATE_M = 30, DONE_AFTER_MIN = 75;   // the job usually finishes 30-45 min after it starts
  const POLL_MS = 5 * 60 * 1000;
  const fmtDay = d => d.toLocaleDateString('en-GB', { weekday: 'short', day: '2-digit', month: 'short' });
  const isWeekday = d => d.getDay() >= 1 && d.getDay() <= 5;
  const iso = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

  function nextUpdate(now) {
    const d = new Date(now);
    d.setHours(UPDATE_H, UPDATE_M, 0, 0);
    if (d <= now || !isWeekday(d)) {
      do { d.setDate(d.getDate() + 1); } while (!isWeekday(d));
    }
    return d;
  }
  // most recent trading day whose update should already have finished (weekends skipped; NSE holidays are not known here)
  function expectedLatest(now) {
    const d = new Date(now);
    const done = new Date(now); done.setHours(UPDATE_H, UPDATE_M + DONE_AFTER_MIN, 0, 0);
    if (now < done || !isWeekday(d)) d.setDate(d.getDate() - 1);
    while (!isWeekday(d)) d.setDate(d.getDate() - 1);
    return iso(d);
  }

  // the APIs return either "2026-10-09" or "09 Oct 2026"
  function toIso(s) {
    if (!s) return null;
    if (/^\d{4}-\d{2}-\d{2}/.test(s)) return s.slice(0, 10);
    const d = new Date(s);
    return Number.isNaN(d.getTime()) ? null : iso(d);
  }

  function status(asOfRaw, now = new Date()) {
    const asOf = toIso(asOfRaw);
    const nxt = nextUpdate(now);
    const when = `${fmtDay(nxt)}, 7:30 PM`;
    const exp = expectedLatest(now);
    const start = new Date(now); start.setHours(UPDATE_H, UPDATE_M, 0, 0);
    const end = new Date(start.getTime() + DONE_AFTER_MIN * 60000);
    if (asOf && isWeekday(now) && now >= start && now < end && asOf < iso(now)) {
      return { overdue: false, text: 'updating now (7:30 PM job)', title: `The nightly job started at 7:30 PM and usually takes 30–45 minutes. ` +
        `This page reloads by itself when today's data arrives.` };
    }
    if (asOf && asOf < exp) {
      return { overdue: true, text: `update overdue`, title: `The newest NSE trading day should be ${exp} (unless it was an NSE holiday). ` +
        `The nightly job may not have run: it needs the laptop on at 7:30 PM on weekdays (it also catches up when the laptop is next switched on). ` +
        `Next scheduled run: ${when}.` };
    }
    return { overdue: false, text: `next update ${when}`, title: `Up to date: ${asOf || asOfRaw} is the latest NSE trading day available. ` +
      `NSE publishes each evening; the nightly job runs Mon–Fri at 7:30 PM. Next run: ${when}.` };
  }

  function paint(el, asOf) {
    if (!el) return;
    const s = status(asOf);
    let tag = document.getElementById('freshTag');
    if (!tag) {
      tag = document.createElement('span');
      tag.id = 'freshTag';
      tag.style.cssText = 'margin-left:6px;font-size:11px;font-weight:400;white-space:nowrap';
      el.insertAdjacentElement('afterend', tag);
    }
    tag.textContent = `· ${s.text}`;
    tag.style.color = s.overdue ? 'var(--down, #e5484d)' : 'var(--muted, #888)';
    tag.title = s.title;
    el.title = s.title;
  }

  function toast(msg) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.setAttribute('role', 'status');
    t.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:9999;padding:9px 14px;border-radius:10px;' +
      'font-size:13px;background:var(--text,#111);color:var(--bg,#fff);box-shadow:0 6px 24px rgba(0,0,0,.25)';
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 6000);
  }

  // watchData({ url, read(json) -> as-of ISO date, el, onUpdate(newAsOf) })  - onUpdate defaults to a page reload
  window.watchData = function ({ url = '/api/meta', read = j => j.as_of, el = null, onUpdate = null, show = true } = {}) {
    let current = null;
    const check = async (first) => {
      try {
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) return;
        const asOf = read(await r.json());
        if (!asOf) return;
        if (current && asOf !== current) {
          toast(`New NSE data loaded: ${asOf}`);
          if (onUpdate) await onUpdate(asOf); else setTimeout(() => location.reload(), 1200);
        }
        current = asOf;
        if (show) paint(el || document.getElementById('asof'), asOf);
      } catch (e) { /* offline or server restarting: try again next time */ }
    };
    check(true);
    setInterval(check, POLL_MS);
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') check(false); });
    setInterval(() => show && current && paint(el || document.getElementById('asof'), current), 60 * 1000);   // keeps "overdue" honest
    return () => check(false);
  };
  window._freshnessStatus = status;   // for tests
})();
