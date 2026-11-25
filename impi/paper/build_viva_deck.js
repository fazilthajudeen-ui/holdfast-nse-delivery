// Phase I viva deck, exact college format: 15 slides, required slide names as 32pt titles,
// Arial, body text >= 20pt, white / light-blue backgrounds, one blue accent, no grey, real numbers only.
const path = require('path');
const pptxgen = require('pptxgenjs');
const { applyTheme } = require(process.env.SKILL + '/scripts/apply_theme.js');

const OUT = process.argv[2];
const SH = path.join(__dirname, 'shots');
const INK = '1D1D1F', SOFT = '0066CC', STAGE = 'EEF5FD', BLUE = '0071E3', BAR = 'A9CDF4', TINT = 'D6E8FB', RED = 'D70015';
const FONT = 'Arial';
const BODY = 20;                                    // guideline: body text 20-24 pt
const THEME = { name: 'Viva Light', headFontFace: FONT, bodyFontFace: FONT,
  colors: { dk1: '000000', lt1: 'FFFFFF', dk2: INK, lt2: STAGE, accent1: BLUE, accent2: INK, accent3: BAR, accent4: SOFT, accent5: RED, accent6: TINT, hlink: BLUE, folHlink: SOFT } };

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';                        // 13.333 x 7.5 in
pres.title = 'Do Delivery and Futures-Positioning Footprints Predict Breakout Success? A Pre-Registered Test on Survivorship-Free NSE Data';
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const W = 13.333, X = 0.7, CW = W - 2 * X, TOP = 1.5;

// fresh objects per layout: pptxgenjs converts option objects in place
const footer = () => [{ text: { text: 'Do Delivery and Futures-Positioning Footprints Predict Breakout Success?  ·  Team 8', options: { x: X, y: 7.0, w: 7, h: 0.28, fontFace: FONT, fontSize: 11, color: SOFT, margin: 0 } } }];
const num = () => ({ x: W - X - 0.6, y: 7.0, w: 0.6, h: 0.28, fontFace: FONT, fontSize: 11, color: SOFT, align: 'right' });
const titlePh = () => ({ placeholder: { options: { name: 'title', type: 'title', x: X, y: 0.5, w: CW, h: 0.7, fontFace: FONT, fontSize: 32, bold: true, color: INK, align: 'left', valign: 'middle', margin: 0 }, text: '' } });
pres.defineSlideMaster({ title: 'Cover', background: { color: 'FFFFFF' }, objects: [] });
pres.defineSlideMaster({ title: 'Light', background: { color: 'FFFFFF' }, objects: [titlePh(), ...footer()], slideNumber: num() });
pres.defineSlideMaster({ title: 'Stage', background: { color: STAGE }, objects: [titlePh(), ...footer()], slideNumber: num() });

const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: FONT, color: INK, margin: 0, valign: 'top', fontSize: BODY, ...o });
const shadow = () => ({ type: 'outer', color: '000000', opacity: 0.13, blur: 22, offset: 6, angle: 90 });
const softShadow = () => ({ type: 'outer', color: '000000', opacity: 0.07, blur: 14, offset: 3, angle: 90 });
const card = (s, x, y, w, h, o = {}) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.14, fill: { color: o.fill || 'FFFFFF' },
  line: { color: o.fill || 'FFFFFF', width: 0 }, shadow: o.flat ? undefined : softShadow(), objectName: o.name });
const slide = (master, title, section) => { const s = pres.addSlide({ masterName: master, sectionTitle: section }); s.addText(title, { placeholder: 'title' }); return s; };
const list = (items, size = BODY, gap = 14) => items.map((t, i) => ({ text: t, options: { bullet: { indent: 24 }, breakLine: i < items.length - 1, paraSpaceAfter: gap, fontSize: size } }));
const frame = (s, file, x, y, w, alt) => {           // screenshot in a soft-shadow frame (images are 16:10)
  const h = w * 0.625;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: 'FFFFFF' }, line: { color: TINT, width: 0.75 }, shadow: shadow() });
  s.addImage({ path: path.join(SH, file), x: x + 0.04, y: y + 0.04, w: w - 0.08, h: h - 0.08, altText: alt });
  return h;
};
const pill = (s, x, y, cap) => { const w = 0.4 + cap.length * 0.112;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.44, rectRadius: 0.22, fill: { color: INK }, line: { color: INK, width: 0 } });
  T(s, cap, { x, y, w, h: 0.44, fontSize: 16, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' }); };
const arrow = (s, x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
  flipH: x2 < x1, flipV: y2 < y1, line: { color: BAR, width: 2, endArrowType: 'triangle' } });
const dot = (s, x, y, d, n, fill = BLUE, size = 18) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  T(s, String(n), { x, y, w: d, h: d, fontSize: size, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
};

// ================================================================== 1. Title Slide
pres.addSection({ title: 'Opening' });
{
  const s = pres.addSlide({ masterName: 'Cover', sectionTitle: 'Opening' });
  T(s, 'PROJECT PHASE I  ·  TEAM 8', { x: X, y: 0.6, w: 7, h: 0.35, fontSize: 16, bold: true, color: BLUE, charSpacing: 1 });
  T(s, 'Do Delivery and Futures-Positioning Footprints Predict Breakout Success? A Pre-Registered Test on Survivorship-Free NSE Data', { x: X, y: 1.05, w: 6.1, h: 2.2, fontSize: 28, bold: true });
  T(s, 'Team members', { x: X, y: 3.72, w: 4, h: 0.3, fontSize: 16, bold: true, color: SOFT });
  const team = [['Fazil T', '23CU0330013'], ['Faizur Rahman S', '23CU0330012'], ['Suhail Ahmed S', '23CU0330039']];
  T(s, team.flatMap(([n, r], i) => [{ text: n + '   ', options: { bold: true } }, { text: r, options: { color: BLUE, breakLine: i < 2 } }]),
    { x: X, y: 4.04, w: 6, h: 1.15, paraSpaceAfter: 3 });
  T(s, 'Under the guidance of', { x: X, y: 5.3, w: 4, h: 0.3, fontSize: 16, bold: true, color: SOFT });
  T(s, [{ text: 'Ms. Stefin Thomas', options: { bold: true } }, { text: ', Assistant Professor' }], { x: X, y: 5.6, w: 6, h: 0.4 });
  T(s, 'Department of CSE (AI & ML), Hindustan Institute of Technology and Science, Chennai', { x: X, y: 6.1, w: 6.0, h: 0.8 });
  frame(s, 'terminal_simple.png', 7.0, 1.3, 5.75, 'Holdfast terminal showing a stock verdict');
  T(s, 'Holdfast: the web app built in this project', { x: 7.0, y: 5.05, w: 5.75, h: 0.4, fontSize: 16, color: SOFT, align: 'center' });
  s.addNotes('Good morning. We are Team 8. Our project asks: do delivery and futures-positioning footprints predict breakout success? Put simply: when a stock goes up, is that move real? We answer it with NSE delivery data and a web app called Holdfast.');
}

// ================================================================== 2. Abstract
pres.addSection({ title: 'Background' });
{
  const s = slide('Light', 'Abstract', 'Background');
  const cols = [
    ['Problem', 'A price chart cannot show if buyers kept the shares or sold them the same day.'],
    ['Solution', 'A 0–100 score from NSE delivery data, comparing each stock with its own past.'],
    ['Implementation', 'A nightly pipeline on 12 years of NSE data and the Holdfast web app.'],
    ['Outcome', 'Strong-delivery rallies held gains better; big moves are predictable (0.77).'],
  ];
  const gap = 0.35, cw = (CW - 3 * gap) / 4;
  cols.forEach(([h, d], i) => {
    const x = X + i * (cw + gap);
    T(s, `0${i + 1}`, { x, y: TOP + 0.2, w: cw, h: 0.6, fontSize: 32, bold: true, color: BLUE });
    T(s, h, { x, y: TOP + 0.95, w: cw, h: 0.45, fontSize: 22, bold: true });
    T(s, d, { x, y: TOP + 1.55, w: cw - 0.05, h: 2.6 });
  });
  s.addNotes('In four parts. The problem: a price chart cannot show whether buyers kept the shares. Our solution: a 0 to 100 score built from NSE delivery data. We implemented it as a nightly pipeline and a web app. The outcome: rallies with strong delivery held their gains better, and big moves can be predicted with accuracy 0.77.');
}

// ================================================================== 3. Introduction
{
  const s = slide('Light', 'Introduction', 'Background');
  T(s, list([
    'Every evening NSE shows how many traded shares were delivered to buyers.',
    'The rest are bought and sold on the same day.',
    'Delivered shares mean a buyer chose to keep them.',
    'On a normal day, about half of all traded shares are delivered.',
  ], BODY, 16), { x: X, y: TOP + 0.1, w: 6.1, h: 4.8 });
  const gx = 7.95, gy = TOP + 0.1, step = 0.43, d = 0.31, n = 48;
  for (let i = 0; i < 100; i++) {
    const r = Math.floor(i / 10), c = i % 10;
    s.addShape(pres.shapes.OVAL, { x: gx + c * step, y: gy + r * step, w: d, h: d, fill: { color: i < n ? BLUE : TINT }, line: { color: i < n ? BLUE : TINT, width: 0 } });
  }
  T(s, [{ text: '48', options: { bold: true, color: BLUE, fontSize: 30 } }, { text: '  of every 100 shares delivered' }],
    { x: 7.4, y: 6.0, w: 5.3, h: 0.55, align: 'center', valign: 'middle' });
  s.addNotes('Delivery is the share of traded quantity actually transferred to the buyer. In our data the typical stock has about 48 percent delivery. The rest is same-day trading. A price chart treats both exactly the same.');
}

// ================================================================== 4. Problem Statement
{
  const s = slide('Light', 'Problem Statement', 'Background');
  card(s, X, TOP, CW, 1.6, { fill: STAGE, flat: true, name: 'statement' });
  T(s, 'Can NSE delivery data show whether a price move is backed by real buying, and help predict what happens next?',
    { x: X + 0.45, y: TOP, w: CW - 0.9, h: 1.6, fontSize: 24, bold: true, valign: 'middle' });
  T(s, 'Limitations of current approaches', { x: X, y: 3.5, w: CW, h: 0.45, fontSize: 22, bold: true, color: BLUE });
  const lim = [['Looks the same', 'Real buying and same-day trading look identical on a chart.'],
    ['Not comparable', 'Raw delivery % differs by stock, so one rule fails.'],
    ['Not tested', 'Most tools never test their signals on new data.']];
  const gap = 0.4, cw = (CW - 2 * gap) / 3;
  lim.forEach(([h, d], i) => {
    const x = X + i * (cw + gap);
    T(s, h, { x, y: 4.1, w: cw, h: 0.45, fontSize: 22, bold: true });
    T(s, d, { x, y: 4.65, w: cw, h: 1.6 });
  });
  s.addNotes('Price and volume cannot separate real buying from churn. Delivery percentages differ by stock, so a fixed threshold fails. And most trading tools never test their signals on unseen data.');
}

// ================================================================== 5. Objectives
{
  const s = slide('Light', 'Objectives', 'Background');
  const obj = [
    'Build a daily pipeline for official NSE data (equity, delivery, F&O).',
    'Score each stock against its own history, from 0 to 100.',
    'Check what happens after sharp rises with high vs low delivery.',
    'Test popular trading beliefs fairly, with rules written down first.',
    'Build a clear web app that anyone can understand.',
  ];
  obj.forEach((t, i) => {
    const y = TOP + 0.1 + i * 1.0;
    dot(s, X, y, 0.58, i + 1);
    T(s, t, { x: X + 0.95, y, w: CW - 1, h: 0.58, fontSize: 22, valign: 'middle' });
  });
  s.addNotes('Five objectives: a reliable data pipeline, a fair score, an evidence study, a fair test of popular beliefs, and an app anyone can read. The conclusion slide answers each one by number.');
}

// ================================================================== 6. Literature Survey
{
  const s = slide('Light', 'Literature Survey', 'Background');
  const HD = { bold: true, color: BLUE, fill: { color: STAGE } };
  const rows = [
    ['Paper', 'Key idea', 'Gap for our work'],
    ['Easley et al. (1996)', 'Trading reveals informed buyers', 'Needs trade-level data'],
    ['Easley et al. (2012)', 'Measures “toxic” order flow', 'Needs tick data'],
    ['Harvey et al. (2016)', 'Many trading signals are false', 'Not tested in India'],
    ['Azevedo & Hoegner (2023)', 'ML can improve signals', 'Rules not fixed first'],
    ['Thapa et al. (2026)', 'FII trading in India', 'No stock-level delivery'],
  ].map((r, i) => r.map(c => i === 0 ? { text: c, options: HD } : { text: c }));
  s.addTable(rows, { x: X, y: TOP, w: CW, colW: [3.75, 4.45, 3.733], fontFace: FONT, fontSize: BODY, color: INK,
    border: [{ type: 'none' }, { type: 'none' }, { type: 'solid', color: TINT, pt: 0.75 }, { type: 'none' }], valign: 'middle',
    rowH: [0.55, 0.62, 0.62, 0.62, 0.62, 0.62], margin: [4, 10, 4, 10] });
  T(s, [{ text: 'Research gap:  ', options: { bold: true, color: BLUE } },
    { text: 'no fair test of NSE delivery and futures data on breakouts, and no tool that explains delivery in plain words.' }],
    { x: X, y: 5.45, w: CW, h: 0.95, valign: 'middle' });
  s.addNotes('Earlier work shows trading activity can carry information, but it needs expensive trade-level data. Other work shows most backtested signals are false. Nobody had tested NSE delivery data with the rules fixed in advance. That is our gap.');
}

// ================================================================== 7. Proposed Methodology
pres.addSection({ title: 'System' });
{
  const s = slide('Light', 'Proposed Methodology', 'System');
  const steps = [['Delivery %', 'Delivered ÷ traded shares'], ['Usual level', 'Middle of last 60 days'], ['Gap', 'Today vs usual (5-day)'], ['Score 0–100', 'Gap vs the stock’s year']];
  const cw = 2.65, gap = (CW - 4 * cw) / 3, y = TOP;
  steps.forEach(([h, d], i) => {
    const x = X + i * (cw + gap);
    dot(s, x, y, 0.62, i + 1, i === 3 ? BLUE : INK);
    T(s, h, { x, y: y + 0.82, w: cw, h: 0.45, fontSize: 22, bold: true });
    T(s, d, { x, y: y + 1.32, w: cw, h: 0.75 });
    if (i < 3) arrow(s, x + 0.8, y + 0.31, x + cw + gap - 0.15, y + 0.31);
  });
  card(s, X, 3.85, CW, 2.75, { fill: STAGE, flat: true });
  T(s, 'How the test was kept fair', { x: X + 0.4, y: 4.05, w: 7, h: 0.4, fontSize: 22, bold: true });
  const fair = [['Rules first', 'Ideas written down before testing'], ['Opened once', 'Test data (2023–26) used one time'], ['Real costs', 'All Indian trading costs included']];
  const fw = (CW - 0.8 - 2 * 0.3) / 3;
  fair.forEach(([h, d], i) => {
    const x = X + 0.4 + i * (fw + 0.3);
    T(s, h, { x, y: 4.65, w: fw, h: 0.4, bold: true, color: BLUE });
    T(s, d, { x, y: 5.1, w: fw, h: 1.3 });
  });
  s.addNotes('Four steps make the score. It is relative: each stock is compared only with its own history, using only past data. For the research part, we wrote the rules down before testing, opened the test data once, and included real costs.');
}

// ================================================================== 8. System Architecture
{
  const s = slide('Stage', 'System Architecture', 'System');
  const bw = 3.5, bh = 1.3, gx = (CW - 3 * bw) / 2, y1 = TOP, y2 = 3.25;
  const node = (x, y, h, d, n) => { card(s, x, y, bw, bh, { name: h });
    T(s, n, { x: x + 0.3, y: y + 0.16, w: 0.6, h: 0.3, fontSize: 16, bold: true, color: BLUE });
    T(s, h, { x: x + 0.3, y: y + 0.45, w: bw - 0.45, h: 0.38, bold: true });
    T(s, d, { x: x + 0.3, y: y + 0.85, w: bw - 0.45, h: 0.38 }); };
  const xs = [X, X + bw + gx, X + 2 * (bw + gx)];
  node(xs[0], y1, 'NSE public files', 'Equity, delivery, F&O', '01');
  node(xs[1], y1, 'Nightly pipeline', 'Download and check', '02');
  node(xs[2], y1, 'Point-in-time store', 'Only past data', '03');
  node(xs[2], y2, 'Analysis', 'Scores and models', '04');
  node(xs[1], y2, 'API server', 'FastAPI (JSON)', '05');
  node(xs[0], y2, 'Holdfast web app', 'Charts and verdicts', '06');
  arrow(s, xs[0] + bw + 0.08, y1 + bh / 2, xs[1] - 0.08, y1 + bh / 2);
  arrow(s, xs[1] + bw + 0.08, y1 + bh / 2, xs[2] - 0.08, y1 + bh / 2);
  arrow(s, xs[2] + bw / 2, y1 + bh + 0.06, xs[2] + bw / 2, y2 - 0.06);
  arrow(s, xs[2] - 0.08, y2 + bh / 2, xs[1] + bw + 0.08, y2 + bh / 2);
  arrow(s, xs[1] - 0.08, y2 + bh / 2, xs[0] + bw + 0.08, y2 + bh / 2);
  const stats = [['12 years', '2014 – 2026'], ['3,748', 'stocks'], ['5.84 M', 'daily rows'], ['0', 'data mismatches']];
  const sw = CW / 4;
  stats.forEach(([v, k], i) => {
    const x = X + i * sw;
    T(s, v, { x, y: 5.0, w: sw - 0.2, h: 0.75, fontSize: 36, bold: true, color: i === 0 ? BLUE : INK });
    T(s, k, { x, y: 5.8, w: sw - 0.2, h: 0.4 });
  });
  s.addNotes('Every evening the pipeline downloads NSE files, checks them and fixes splits and bonuses. It stores only what was known on each day, so there is no look-ahead. The analysis layer computes scores and models, a FastAPI server serves them, and Holdfast shows them. The dataset covers 12 years and 3,748 stocks, including delisted ones, with zero mismatches in the rebuilt data.');
}

// ================================================================== 9. Modules Description
{
  const s = slide('Stage', 'Modules Description', 'System');
  const mods = [
    ['Data collection', 'Downloads NSE files every evening'],
    ['Data cleaning', 'Fixes splits; rejects bad files'],
    ['Conviction score', 'Ranks delivery vs the stock’s year'],
    ['IMPI models', 'Tests breakouts and big moves'],
    ['Holdfast web app', 'Plain-word verdicts, Simple & Pro'],
    ['Assistant & updates', 'Answers questions; updates at 7:30 PM'],
  ];
  const gap = 0.35, cw = (CW - 2 * gap) / 3, ch = 2.2;
  mods.forEach(([h, d], i) => {
    const x = X + (i % 3) * (cw + gap), y = TOP + Math.floor(i / 3) * (ch + 0.3);
    card(s, x, y, cw, ch, { name: h });
    T(s, `0${i + 1}`, { x: x + 0.35, y: y + 0.25, w: 1, h: 0.35, fontSize: 16, bold: true, color: BLUE });
    T(s, h, { x: x + 0.35, y: y + 0.62, w: cw - 0.6, h: 0.45, fontSize: 22, bold: true });
    T(s, d, { x: x + 0.35, y: y + 1.15, w: cw - 0.6, h: 0.9 });
  });
  s.addNotes('Six modules: collection, cleaning, the conviction score, the IMPI research models, the web app, and the assistant with the nightly update.');
}

// ================================================================== 10. Technologies Used
{
  const s = slide('Light', 'Technologies Used', 'System');
  const rows = [['Languages', ['Python 3.12', 'JavaScript', 'HTML', 'CSS']], ['Data & database', ['pandas', 'NumPy', 'Parquet', 'DuckDB']],
    ['Machine learning', ['scikit-learn', 'XGBoost', 'SciPy']], ['Web framework', ['FastAPI', 'Uvicorn', 'Lightweight Charts']],
    ['Tools', ['pytest', 'Git', 'VS Code']], ['Platform', ['Windows 11', 'Web browser', 'NSE public data']]];
  rows.forEach(([k, chips], i) => {
    const y = TOP + 0.05 + i * 0.8;
    T(s, k, { x: X, y, w: 3.3, h: 0.55, bold: true, valign: 'middle' });
    let x = 4.2;
    chips.forEach(c => {
      const w = 0.45 + c.length * 0.14;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.55, rectRadius: 0.27, fill: { color: STAGE }, line: { color: STAGE, width: 0 } });
      T(s, c, { x, y, w, h: 0.55, align: 'center', valign: 'middle' });
      x += w + 0.2;
    });
  });
  s.addNotes('Python for data and models, DuckDB and Parquet files for storage, scikit-learn and XGBoost for machine learning, FastAPI for the server, and HTML, CSS and JavaScript for the web app. It runs on Windows 11 in any web browser, using NSE’s free public data.');
}

// ================================================================== 11. Implementation
pres.addSection({ title: 'Implementation' });
{
  const s = slide('Stage', 'Implementation', 'Implementation');
  const lw = 7.35, ly = TOP - 0.05, lh = frame(s, 'terminal_simple.png', X, ly, lw, 'Holdfast terminal in Simple view');
  pill(s, X + 0.2, ly + lh - 0.62, 'Simple view: plain-word verdict');
  const rw = 4.1, rx = W - X - rw, rh = rw * 0.625;
  frame(s, 'terminal_pro.png', rx, ly, rw, 'Pro view with candles, tools and model signals');
  pill(s, rx + 0.15, ly + rh - 0.58, 'Pro view: trading tools');
  const y2 = ly + rh + 0.3;
  frame(s, 'scanner.png', rx, y2, rw, 'Scanner listing every F&O stock');
  pill(s, rx + 0.15, y2 + rh - 0.58, 'Scanner: all F&O stocks');
  T(s, 'Also: Market and Research pages', { x: X, y: ly + lh + 0.2, w: lw, h: 0.45, bold: true, color: BLUE });
  s.addNotes('This is Holdfast, the working app. In Simple view, any stock gets a verdict in plain words, a score from 0 to 100 and delivery bars under the chart. Pro view adds candles, drawing tools and model signals. The scanner covers every F&O stock, and there are also Market and Research pages. Everything updates by itself every evening.');
}

// ================================================================== 12. Results and Analysis
pres.addSection({ title: 'Results' });
{
  const s = slide('Light', 'Results and Analysis', 'Results');
  const stats = [['+2.2 pts', 'more return when a sharp rise had strong delivery', BLUE], ['0.77', 'accuracy at spotting big moves (0.5 = guess)', BLUE],
    ['0 of 4', 'popular breakout beliefs held up on new data', INK]];
  const gap = 0.35, cw = (CW - 2 * gap) / 3;
  stats.forEach(([v, k, c], i) => {
    const x = X + i * (cw + gap);
    card(s, x, TOP, cw, 1.7, { fill: STAGE, flat: true });
    T(s, v, { x: x + 0.3, y: TOP + 0.12, w: cw - 0.5, h: 0.7, fontSize: 36, bold: true, color: c });
    T(s, k, { x: x + 0.3, y: TOP + 0.85, w: cw - 0.5, h: 0.8 });
  });
  // bottom row: graph | output screenshot | comparison table
  const ry = 3.45, rh = 3.3, gp = 0.3;
  const c1 = 4.3, c2 = 2.82, c3 = CW - c1 - c2 - 2 * gp;
  s.addChart(pres.charts.BAR, [{ name: 'Extra 20-day return', labels: ['Weak', 'Normal', 'Strong'], values: [0.89, 0.82, 3.08] }], {
    x: X, y: ry, w: c1, h: rh, barDir: 'col', chartColors: [BAR, BAR, BLUE], barGapWidthPct: 45,
    showTitle: true, title: 'Return after a sharp rise (%)', titleFontFace: FONT, titleFontSize: 16, titleColor: INK,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '+0.0"%"', dataLabelFontFace: FONT, dataLabelFontSize: 18, dataLabelColor: INK, dataLabelFontBold: true,
    catAxisLabelFontFace: FONT, catAxisLabelFontSize: 18, catAxisLabelColor: INK, catAxisLineShow: false, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 3.9,
    valGridLine: { style: 'none' }, catGridLine: { style: 'none' }, showLegend: false, objectName: 'excess-chart' });
  // output screenshot: the verdict Holdfast produced for TCS
  const ox = X + c1 + gp, oh = rh, ow = Math.min(c2, oh * 0.8538);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ox, y: ry, w: ow, h: oh, rectRadius: 0.1, fill: { color: 'FFFFFF' }, line: { color: TINT, width: 0.75 }, shadow: shadow() });
  s.addImage({ path: path.join(SH, 'output_verdict.png'), x: ox + 0.05, y: ry + 0.05, w: ow - 0.1, h: oh - 0.1, altText: 'Output: Holdfast verdict for TCS, score 93, buyers are holding' });
  pill(s, ox + (ow - (0.4 + 'Output: TCS verdict'.length * 0.112)) / 2, ry + oh - 0.6, 'Output: TCS verdict');
  // comparison table: costed backtests on the 2023-26 test set (paper Table V)
  const HD = { bold: true, color: BLUE, fill: { color: STAGE } };
  const tx = ox + ow + gp, tw = W - X - tx;
  const rows = [[{ text: 'After costs', options: HD }, { text: 'Per year', options: { ...HD, align: 'right' } }],
    ['Best strategy', '+6.0%'], ['ML top picks', '+4.3%'], ['All breakouts', '−26.8%'],
    [{ text: 'Nifty 500', options: { bold: true } }, { text: '+9.7%', options: { bold: true, color: BLUE } }]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => typeof c === 'string' ? { text: c, options: j ? { align: 'right', color: c.startsWith('−') ? RED : INK } : {} } : (j ? { ...c, options: { ...c.options, align: 'right' } } : c)));
  s.addTable(rows, { x: tx, y: ry + 0.15, w: tw, colW: [tw - 1.45, 1.45], fontFace: FONT, fontSize: BODY, color: INK,
    border: [{ type: 'none' }, { type: 'none' }, { type: 'solid', color: TINT, pt: 0.75 }, { type: 'none' }], valign: 'middle',
    rowH: [0.55, 0.55, 0.55, 0.55, 0.55], margin: [4, 10, 4, 10] });
  T(s, 'Test data 2023–26 (paper)', { x: tx, y: ry + 3.0, w: tw, h: 0.3, fontSize: 14, color: SOFT });
  s.addNotes('Three results. First, after 744 sharp rises, stocks with strong delivery beat the Nifty 500 by about 3.1 percent over 20 days, against under 1 percent for weak delivery: 2.2 points more. Second, spotting stocks about to make a big move works well, with accuracy 0.77 on new data. Third, we wrote four popular breakout beliefs down before testing; none held up, and after all costs no strategy beat the Nifty 500. In the middle is a real output: Holdfast’s verdict for TCS, score 93, buyers are holding. Delivery shows how strongly a stock may move, not which way.');
}

// ================================================================== 13. Conclusion
pres.addSection({ title: 'Closing' });
{
  const s = slide('Light', 'Conclusion', 'Closing');
  T(s, 'Each objective, achieved', { x: X, y: TOP - 0.05, w: 7.3, h: 0.4, fontSize: 22, bold: true, color: BLUE });
  const done = ['Pipeline built: 12 years, 3,748 stocks', 'Score live for 484 stocks every evening', '744 sharp rises studied: strong delivery held',
    'Beliefs tested fairly: honest answer', 'Holdfast web app working, Simple & Pro'];
  done.forEach((t, i) => {
    const y = TOP + 0.55 + i * 0.9;
    dot(s, X, y, 0.55, i + 1);
    T(s, t, { x: X + 0.85, y, w: 6.6, h: 0.55, valign: 'middle' });
  });
  card(s, 8.55, TOP, CW - 7.85, 4.95, { fill: STAGE, flat: true });
  T(s, 'Future work (Phase II)', { x: 8.9, y: TOP + 0.3, w: CW - 8.5, h: 0.45, fontSize: 22, bold: true });
  T(s, list(['Live forward-test results (Feb 2027)', 'More years of delivery data', 'Full paper submission', 'Testing with real users'], BODY, 14),
    { x: 8.9, y: TOP + 0.95, w: CW - 8.55, h: 3.8 });
  s.addNotes('Each objective is met; the numbers match the objectives slide. One: the pipeline covers 12 years and 3,748 stocks. Two: the score runs for 484 stocks every evening. Three: 744 sharp rises studied. Four: popular beliefs tested fairly, with an honest answer. Five: the Holdfast web app works in Simple and Pro views. In Phase II we add live forward-test results, more data and the full paper.');
}

// ================================================================== 14. Journal Paper Status
{
  const s = slide('Stage', 'Journal Paper Status', 'Closing');
  const rows = [['Title', 'Do Delivery and Futures-Positioning Footprints Predict Breakout Success? A Pre-Registered Test on Survivorship-Free NSE Data', 0.75],
    ['Authors', 'Fazil T, Faizur Rahman S, Suhail Ahmed S, Ms. Stefin Thomas', 0.4],
    ['Conference', 'NEXA Research international conference (IFERP)', 0.4],
    ['Status', 'Abstract submitted  ·  Submission ID 102160807', 0.4]];
  let y = TOP - 0.05;
  rows.forEach(([k, v, h]) => {
    T(s, k, { x: X, y: y + 0.03, w: 1.7, h: 0.4, fontSize: 16, bold: true, color: SOFT });
    T(s, v, { x: X + 1.8, y, w: CW - 1.8, h, bold: k === 'Title' || k === 'Status', color: k === 'Status' ? BLUE : INK });
    y += h + 0.15;
  });
  // proof of submission, large and centred
  T(s, 'Proof of submission', { x: X, y: 3.98, w: CW, h: 0.4, fontSize: 20, bold: true, color: BLUE, align: 'center' });
  const fh = 2.4, fw = fh * 3.276, fx = (W - fw) / 2, fy = 4.45;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: fx, y: fy, w: fw, h: fh, rectRadius: 0.1, fill: { color: 'FFFFFF' }, line: { color: TINT, width: 0.75 }, shadow: shadow() });
  s.addImage({ path: path.join(SH, 'proof.png'), x: fx + 0.06, y: fy + 0.06, w: fw - 0.12, h: fh - 0.12, altText: 'Submission confirmation page: thank you for submitting the article' });
  s.addNotes('Our paper reports the pre-registered study. The abstract has been submitted to the NEXA Research international conference through the IFERP system, submission ID 102160807; the review result is expected within 24 to 48 hours.');
}

// ================================================================== 15. References
{
  const s = slide('Light', 'References', 'Closing');
  const refs = [
    'D. Easley et al., “Liquidity, information, and infrequently traded stocks,” J. Finance, vol. 51, no. 4, 1996.',
    'D. Easley, M. López de Prado and M. O’Hara, “Flow toxicity and liquidity in a high-frequency world,” Rev. Financ. Stud., vol. 25, no. 5, 2012.',
    'C. R. Harvey, Y. Liu and H. Zhu, “…and the cross-section of expected returns,” Rev. Financ. Stud., vol. 29, no. 1, 2016.',
    'V. Azevedo and C. Hoegner, “Enhancing stock market anomalies with machine learning,” Rev. Quant. Finance Account., vol. 60, 2023.',
    'C. Thapa et al., “Policy information uncertainty and FII trading behavior: Evidence from India,” Rev. Quant. Finance Account., vol. 67, 2026.',
    'B. A. Nosek et al., “The preregistration revolution,” Proc. Natl. Acad. Sci. USA, vol. 115, no. 11, 2018.',
    'M. López de Prado, Advances in Financial Machine Learning. Hoboken, NJ, USA: Wiley, 2018.',
    'NSE, daily bhavcopy, delivery and F&O files, nseindia.com, 2026.',
  ];
  T(s, refs.map((r, i) => ({ text: `[${i + 1}]  ${r}`, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 4 } })),
    { x: X, y: TOP - 0.15, w: CW, h: 5.45 });
  s.addNotes('These are our main references. Thank you. We are happy to take questions.');
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log('wrote', OUT);
})();
