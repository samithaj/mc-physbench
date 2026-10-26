'use strict';

// ---------------------------------------------------------------- basics

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const view = $('#view');
const LEVEL_COLORS = ['var(--l1)', 'var(--l2)', 'var(--l3)', 'var(--l4)'];
const ACTIVE = ['queued', 'provisioning', 'running', 'uploading'];

const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let token = null;
try { token = localStorage.getItem('hjs_token'); } catch (_) { /* storage blocked */ }
let me = null;
let catalog = null;
let timers = [];

function setToken(t) {
  token = t;
  try { t ? localStorage.setItem('hjs_token', t) : localStorage.removeItem('hjs_token'); } catch (_) { /* ignore */ }
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  let body = opts.body;
  if (body !== undefined && !(body instanceof Blob) && !(body instanceof ArrayBuffer)) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(body);
  }
  const res = await fetch(path, { method: opts.method || (body ? 'POST' : 'GET'), headers, body });
  if (res.status === 401 && !path.startsWith('/api/auth/')) {
    setToken(null);
    location.hash = '#/login';
    throw new Error('please sign in');
  }
  const data = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) {
    let msg = data?.detail ?? res.statusText;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    if (typeof msg === 'object') msg = msg.message || JSON.stringify(msg);
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function toast(msg, bad = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'show' + (bad ? ' bad' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = ''; }, 4200);
}

let routeSeq = 0;

// Poll while the current page is shown; late results from a previous page are dropped.
function every(ms, fn) {
  const seq = routeSeq;
  const run = async () => {
    if (seq !== routeSeq) return;
    try { await fn(); } catch (err) { if (seq === routeSeq) toast(err.message, true); }
  };
  run();
  timers.push(setInterval(run, ms));
}

const fmtBytes = (n) => {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
};
const fmtUsd = (n) => `$${(n || 0).toFixed(n >= 10 ? 1 : 2)}`;
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : '—');
const ago = (iso) => {
  if (!iso) return '—';
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return `${Math.max(0, Math.round(s))}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
};

function statusPill(status) {
  const cls = { succeeded: 'good', ready: 'good', failed: 'bad', invalid: 'bad', cancelled: 'warn',
    stopped: 'warn', running: 'info', uploading: 'info', provisioning: 'info', queued: 'info',
    starting: 'info', processing: 'info' }[status] || '';
  const live = ['running', 'uploading', 'provisioning', 'queued', 'starting', 'processing'].includes(status);
  return `<span class="pill ${cls} ${live ? 'live' : ''}"><span class="dot"></span>${esc(status)}</span>`;
}
const visPill = (v) => `<span class="pill ${v === 'public' ? 'info' : ''}">${esc(v)}</span>`;
const kindLabel = { train: 'Training', evaluate: 'Planning eval', simulate: 'Dream rollout' };

function modal(html, onMount) {
  const back = document.createElement('div');
  back.className = 'modal-back';
  back.innerHTML = `<div class="modal" role="dialog" aria-modal="true">${html}</div>`;
  const close = () => back.remove();
  back.addEventListener('click', (e) => { if (e.target === back) close(); });
  document.addEventListener('keydown', function onKey(e) {
    if (e.key === 'Escape') { close(); document.removeEventListener('keydown', onKey); }
  });
  document.body.appendChild(back);
  $$('[data-close]', back).forEach((b) => b.addEventListener('click', close));
  onMount?.(back, close);
  $('input, select, textarea', back)?.focus();
  return close;
}

function formData(form) {
  const out = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    if (el.type === 'checkbox') out[el.name] = el.checked;
    else if (el.type === 'number') out[el.name] = el.value === '' ? null : Number(el.value);
    else out[el.name] = el.value;
  }
  return out;
}

// ---------------------------------------------------------------- charts

function lineChart(series, { height = 240, logY = false } = {}) {
  const names = Object.keys(series).filter((k) => series[k].length);
  if (!names.length) return '<div class="empty">No metrics yet.</div>';
  const W = 760; const H = height; const P = { l: 52, r: 12, t: 10, b: 24 };
  const tf = (v) => (logY ? Math.log10(Math.max(v, 1e-8)) : v);
  let xMin = Infinity; let xMax = -Infinity; let yMin = Infinity; let yMax = -Infinity;
  for (const n of names) for (const [x, y] of series[n]) {
    xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
    yMin = Math.min(yMin, tf(y)); yMax = Math.max(yMax, tf(y));
  }
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) { yMax += 1; yMin -= 1; }
  const sx = (x) => P.l + ((x - xMin) / (xMax - xMin)) * (W - P.l - P.r);
  const sy = (y) => P.t + (1 - (tf(y) - yMin) / (yMax - yMin)) * (H - P.t - P.b);
  const palette = ['#7cc4ff', '#c49bff', '#ffb26b', '#6fe0c8', '#ff8fb1', '#d7e36b', '#9aa7ff', '#ff7a7a'];
  const paths = names.map((n, i) => {
    const pts = series[n];
    const step = Math.max(1, Math.floor(pts.length / 500));
    const d = pts.filter((_, k) => k % step === 0 || k === pts.length - 1)
      .map(([x, y], k) => `${k ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('');
    return `<path d="${d}" fill="none" stroke="${palette[i % palette.length]}" stroke-width="1.8"/>`;
  }).join('');
  const ticks = [0, 0.5, 1].map((f) => {
    const v = yMin + f * (yMax - yMin);
    const label = logY ? (10 ** v).toPrecision(2) : v.toPrecision(3);
    const y = P.t + (1 - f) * (H - P.t - P.b);
    return `<line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" stroke-dasharray="2 4"/>
      <text x="${P.l - 6}" y="${y + 4}" text-anchor="end">${label}</text>`;
  }).join('');
  const legend = names.map((n, i) => `<span style="--c:${palette[i % palette.length]}">${esc(n)}</span>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="metrics chart">
    ${ticks}<text x="${P.l}" y="${H - 6}">${xMin}</text><text x="${W - P.r}" y="${H - 6}" text-anchor="end">step ${xMax}</text>${paths}</svg>
    <div class="legend">${legend}</div>`;
}

function sparkline(values, color = 'var(--accent)') {
  if (!values.length) return '';
  const W = 200; const H = 36;
  const max = Math.max(...values, 1e-6);
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${(i / Math.max(1, values.length - 1)) * W},${H - (v / max) * (H - 4) - 2}`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:36px"><path d="${d}" fill="none" stroke="${color}" stroke-width="1.6"/></svg>`;
}

// ---------------------------------------------------------------- router

const routes = {
  login: renderAuth, home: renderHome, datasets: renderDatasets, dataset: renderDataset,
  models: renderModels, model: renderModel, train: renderTrain, jobs: renderJobs, job: renderJob,
  simulate: renderSimulate, session: renderSession, settings: renderSettings,
};

async function route() {
  timers.forEach(clearInterval);
  timers = [];
  routeSeq += 1;
  const [path, query = ''] = location.hash.replace(/^#\/?/, '').split('?');
  const [name = '', id, ...rest] = path.split('/');
  const params = new URLSearchParams(query);
  if (!token && name !== 'login') { location.hash = '#/login'; return; }
  if (token && !me) {
    try { me = await api('/api/auth/me'); } catch (_) { return; }
  }
  if (!catalog) catalog = await api('/api/meta/catalog');
  const key = { '': 'home', datasets: id ? 'dataset' : 'datasets', models: id ? 'model' : 'models',
    jobs: id ? 'job' : 'jobs', sessions: 'session' }[name] ?? name;
  const handler = routes[key] || renderHome;
  const authed = key !== 'login';
  $('#nav').hidden = !authed;
  $('#app').classList.toggle('no-nav', !authed);
  if (authed) {
    $('#whoami').textContent = me ? `@${me.username}` : '';
    const navKey = { dataset: 'datasets', model: 'models', job: 'jobs', session: 'simulate' }[key] || key;
    $$('#nav nav a').forEach((a) => a.classList.toggle('active', a.dataset.route === navKey));
  }
  view.innerHTML = '<div class="hint">Loading…</div>';
  try {
    await handler(id || '', params, rest);
  } catch (err) {
    view.innerHTML = `<div class="panel error">${esc(err.message)}</div>`;
  }
}

window.addEventListener('hashchange', route);
$('#logout').addEventListener('click', () => { setToken(null); me = null; location.hash = '#/login'; });

// ---------------------------------------------------------------- auth

async function renderAuth() {
  if (token && me) { location.hash = '#/'; return; }
  let mode = 'login';
  const draw = () => {
    view.innerHTML = `<div class="auth">
      <div class="brand"><svg viewBox="0 0 32 32"><path d="M7 23h18M10 17h12M13 11h6" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>H-JEPA Studio</div>
      <div class="panel">
        <div class="tabs"><button data-m="login" class="${mode === 'login' ? 'on' : ''}">Sign in</button>
          ${catalog.allow_registration ? `<button data-m="register" class="${mode === 'register' ? 'on' : ''}">Create account</button>` : ''}</div>
        <form class="stack" id="authform">
          ${mode === 'register' ? `
            <label>Email<input name="email" type="email" required autocomplete="email"></label>
            <label>Username<input name="username" required pattern="[A-Za-z0-9_\\-]{3,32}" autocomplete="username"></label>
            <label>Password<input name="password" type="password" minlength="8" required autocomplete="new-password"></label>`
    : `<label>Email or username<input name="login" required autocomplete="username"></label>
            <label>Password<input name="password" type="password" required autocomplete="current-password"></label>`}
          <button class="primary" type="submit">${mode === 'login' ? 'Sign in' : 'Create account'}</button>
          <p class="error" id="autherr"></p>
        </form>
      </div>
      <p class="hint" style="margin-top:14px;text-align:center">Train and share hierarchical JEPA world models on rented RunPod GPUs.</p>
    </div>`;
    $$('[data-m]').forEach((b) => b.addEventListener('click', () => { mode = b.dataset.m; draw(); }));
    $('#authform').addEventListener('submit', async (e) => {
      e.preventDefault();
      try {
        const res = await api(`/api/auth/${mode}`, { body: formData(e.target) });
        setToken(res.access_token);
        me = res.user;
        location.hash = '#/';
      } catch (err) { $('#autherr').textContent = err.message; }
    });
  };
  draw();
}

// ---------------------------------------------------------------- dashboard

async function renderHome() {
  const [jobs, myModels, myData] = await Promise.all([
    api('/api/jobs?limit=50'), api('/api/models?scope=mine'), api('/api/datasets?scope=mine')]);
  const active = jobs.filter((j) => ACTIVE.includes(j.status));
  const spend = jobs.reduce((s, j) => s + (j.cost_usd || 0), 0);
  view.innerHTML = `
    <div class="page-head"><div><h1>Welcome back, ${esc(me.username)}</h1>
      <p>Hierarchical JEPA world models: each level predicts farther ahead in its own latent space.</p></div>
      <div class="row"><a class="btn" href="#/datasets?upload=1">Upload data</a><a class="btn primary" href="#/train">Train a model</a></div></div>
    <div class="grid cols-4">
      <div class="panel"><div class="stat">${active.length}</div><div class="stat-label">active GPU jobs</div></div>
      <div class="panel"><div class="stat">${myModels.length}</div><div class="stat-label">your world models</div></div>
      <div class="panel"><div class="stat">${myData.length}</div><div class="stat-label">your datasets</div></div>
      <div class="panel"><div class="stat">${fmtUsd(spend)}</div><div class="stat-label">GPU spend (last 50 jobs)</div></div>
    </div>
    <div class="grid cols-3" style="margin-top:14px">
      <a class="card" href="#/train?preset=pusht&variant=hjepa_l2"><div class="title">Train H-JEPA on Push-T</div>
        <div class="sub">Official data, 2 levels, one RTX 5090. Includes planning eval and pixel decoders.</div></a>
      <a class="card" href="#/simulate"><div class="title">Dream with a released checkpoint</div>
        <div class="sub">28 official models from the paper: roll every level forward and compare to reality.</div></a>
      <a class="card" href="#/datasets?upload=1"><div class="title">Share your own world-model data</div>
        <div class="sub">Upload an HDF5 of frames + actions; train on it as custom data.</div></a>
    </div>
    <div class="panel" style="margin-top:14px"><div class="row spread"><h2>Recent jobs</h2><a href="#/jobs">All jobs →</a></div>
      ${jobsTable(jobs.slice(0, 8))}</div>`;
  bindJobRows();
}

function jobsTable(jobs) {
  if (!jobs.length) return '<div class="empty">No jobs yet. Start one from Train or Simulate.</div>';
  return `<div class="table-wrap"><table><thead><tr><th>Job</th><th>Kind</th><th>Status</th><th>GPU</th><th>Cost</th><th>Created</th></tr></thead><tbody>
    ${jobs.map((j) => `<tr class="click" data-job="${j.id}"><td>${esc(j.name)}<div class="hint">${esc(j.message).slice(0, 90)}</div></td>
      <td>${kindLabel[j.kind] || j.kind}</td><td>${statusPill(j.status)}</td>
      <td class="hint">${esc((j.spec.gpu || {}).type || '').replace('NVIDIA ', '').replace('GeForce ', '')}</td>
      <td>${fmtUsd(j.cost_usd)}</td><td class="hint">${ago(j.created_at)}</td></tr>`).join('')}</tbody></table></div>`;
}
function bindJobRows() {
  $$('[data-job]').forEach((r) => r.addEventListener('click', () => { location.hash = `#/jobs/${r.dataset.job}`; }));
}

// ---------------------------------------------------------------- datasets

async function renderDatasets(_, params) {
  let scope = params.get('scope') || 'all';
  let preset = '';
  const draw = async () => {
    const rows = await api(`/api/datasets?scope=${scope}${preset ? `&preset=${preset}` : ''}`);
    $('#dslist').innerHTML = rows.length ? rows.map(datasetCard).join('')
      : '<div class="empty">No datasets match.</div>';
  };
  view.innerHTML = `
    <div class="page-head"><div><h1>Datasets</h1><p>Shared world-model data: episodes of frames and actions in H-JEPA's HDF5 layout.</p></div>
      <div class="row"><button id="hfbtn">Import from Hugging Face</button><button class="primary" id="upbtn">Upload HDF5</button></div></div>
    <div class="row spread" style="margin-bottom:12px">
      <div class="tabs" style="margin:0;border:0">${['all', 'mine', 'public'].map((s) => `<button data-scope="${s}" class="${s === scope ? 'on' : ''}">${s}</button>`).join('')}</div>
      <select id="presetf" style="width:auto"><option value="">All environments</option>
        ${Object.entries(catalog.presets).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join('')}</select></div>
    <div class="grid cards" id="dslist"></div>`;
  $$('[data-scope]').forEach((b) => b.addEventListener('click', () => {
    scope = b.dataset.scope;
    $$('[data-scope]').forEach((x) => x.classList.toggle('on', x === b));
    draw();
  }));
  $('#presetf').addEventListener('change', (e) => { preset = e.target.value; draw(); });
  $('#upbtn').addEventListener('click', () => uploadModal());
  $('#hfbtn').addEventListener('click', () => hfModal());
  await draw();
  if (params.get('upload')) uploadModal();
}

function datasetCard(d) {
  const ins = d.inspection || {};
  const facts = [ins.episodes != null && `${ins.episodes.toLocaleString()} episodes`,
    ins.img_size && `${ins.img_size}px`, ins.action_dim && `action ${ins.action_dim}d`,
    d.size_bytes && fmtBytes(d.size_bytes)].filter(Boolean).join(' · ');
  return `<a class="card" href="#/datasets/${d.id}">
    ${d.preview_url ? `<img class="preview" src="${esc(d.preview_url)}" alt="">` : ''}
    <div class="row spread"><span class="title">${esc(d.name)}</span>${statusPill(d.status)}</div>
    <div class="sub">${esc(d.env_label)} · ${esc(d.source)} · by @${esc(d.owner?.username)}</div>
    <div class="sub">${esc(facts)}</div>
    <div class="row">${visPill(d.visibility)}${(d.tags || []).map((t) => `<span class="pill">${esc(t)}</span>`).join('')}</div></a>`;
}

function presetOptions(selected = 'custom') {
  return Object.entries(catalog.presets).map(([k, p]) =>
    `<option value="${k}" ${k === selected ? 'selected' : ''}>${esc(p.label)}</option>`).join('');
}

function uploadModal() {
  modal(`<h2>Upload an HDF5 dataset</h2>
    <p class="hint">Columns <code>ep_len</code>, <code>ep_offset</code>, <code>pixels</code> (uint8 N×H×W×3, square)
      and <code>action</code> (N×A). Uploads are chunked and resume if interrupted.</p>
    <form class="stack" id="upform">
      <label>File<input type="file" name="file" accept=".h5,.hdf5" required></label>
      <label>Name<input name="name" required maxlength="128"></label>
      <div class="form-grid">
        <label>Environment<select name="env_preset">${presetOptions('custom')}</select></label>
        <label>Visibility<select name="visibility"><option value="private">private</option><option value="public">public</option></select></label>
        <label>License<input name="license" placeholder="e.g. MIT, CC-BY-4.0"></label>
      </div>
      <label>Description<textarea name="description" style="font-family:inherit"></textarea></label>
      <div class="progress" hidden><div style="width:0"></div></div><div class="hint" id="upstat"></div>
      <div class="row"><button class="primary" type="submit">Upload</button><button type="button" data-close>Cancel</button></div>
    </form>`, (root, close) => {
    const form = $('#upform', root);
    form.file.addEventListener('change', () => { if (!form.name.value) form.name.value = form.file.files[0]?.name.replace(/\.(h5|hdf5)$/, '') || ''; });
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const file = form.file.files[0];
      const meta = formData(form);
      delete meta.file;
      form.querySelector('button.primary').disabled = true;
      const bar = $('.progress', root);
      bar.hidden = false;
      try {
        const ds = await api('/api/datasets', { body: { ...meta, size_bytes: file.size } });
        await chunkedUpload(ds.id, file, (frac) => {
          bar.firstElementChild.style.width = `${(frac * 100).toFixed(1)}%`;
          $('#upstat', root).textContent = `${fmtBytes(file.size * frac)} of ${fmtBytes(file.size)}`;
        });
        $('#upstat', root).textContent = 'Validating file…';
        const done = await api(`/api/datasets/${ds.id}/complete`, { method: 'POST' });
        close();
        toast(done.status === 'ready' ? 'Dataset ready' : 'Upload finished, but the file failed validation', done.status !== 'ready');
        location.hash = `#/datasets/${ds.id}`;
      } catch (err) {
        $('#upstat', root).innerHTML = `<span class="error">${esc(err.message)}</span>`;
        form.querySelector('button.primary').disabled = false;
      }
    });
  });
}

async function chunkedUpload(id, file, onProgress) {
  const CHUNK = 32 * 1024 * 1024;
  let { received } = await api(`/api/datasets/${id}/upload`);
  while (received < file.size) {
    const blob = file.slice(received, Math.min(file.size, received + CHUNK));
    let attempt = 0;
    for (;;) {
      try {
        const res = await api(`/api/datasets/${id}/upload?offset=${received}`, { method: 'PUT', body: blob,
          headers: { 'Content-Type': 'application/octet-stream' } });
        received = res.received;
        break;
      } catch (err) {
        if (err.status === 409 && err.data?.detail?.received != null) { received = err.data.detail.received; break; }
        if (++attempt > 4) throw err;
        await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
    onProgress(received / file.size);
  }
}

function hfModal() {
  modal(`<h2>Import from Hugging Face</h2>
    <p class="hint">The file is downloaded straight onto the GPU worker when a job uses it.</p>
    <form class="stack" id="hfform">
      <div class="form-grid">
        <label>Repository<input name="repo" required placeholder="jepa-world-models/h-jepa"></label>
        <label>Repo type<select name="repo_type"><option>dataset</option><option>model</option></select></label>
      </div>
      <label>File (.h5)<input name="filename" required placeholder="fourroom_tp35_d1_probing.h5"></label>
      <label>Name<input name="name" required></label>
      <div class="form-grid">
        <label>Environment<select name="env_preset">${presetOptions('custom')}</select></label>
        <label>Visibility<select name="visibility"><option>private</option><option>public</option></select></label>
      </div>
      <p class="error" id="hferr"></p>
      <div class="row"><button class="primary" type="submit">Import</button><button type="button" data-close>Cancel</button></div>
    </form>`, (root, close) => {
    $('#hfform', root).addEventListener('submit', async (e) => {
      e.preventDefault();
      try {
        const ds = await api('/api/datasets/import-hf', { body: formData(e.target) });
        close();
        location.hash = `#/datasets/${ds.id}`;
      } catch (err) { $('#hferr', root).textContent = err.message; }
    });
  });
}

async function renderDataset(id) {
  const d = await api(`/api/datasets/${id}`);
  const ins = d.inspection || {};
  const cols = Object.entries(ins.columns || {});
  view.innerHTML = `
    <div class="page-head"><div><h1>${esc(d.name)}</h1>
      <p>${esc(d.env_label)} · ${esc(d.source)} · shared by @${esc(d.owner?.username)} · ${ago(d.created_at)}</p></div>
      <div class="row">${statusPill(d.status)}${visPill(d.visibility)}
        ${d.status === 'ready' ? `<a class="btn primary" href="#/train?dataset=${d.id}&preset=${d.env_preset}">Train on this</a>` : ''}</div></div>
    ${d.preview_url ? `<img class="preview" style="margin-bottom:14px" src="${esc(d.preview_url)}" alt="first episode frames">` : ''}
    <div class="grid cols-2">
      <div class="panel"><h3>About</h3><p style="white-space:pre-wrap">${esc(d.description) || '<span class="hint">No description.</span>'}</p>
        <dl class="kv"><dt>License</dt><dd>${esc(d.license) || '—'}</dd><dt>Used by jobs</dt><dd>${d.use_count}</dd>
        ${d.size_bytes ? `<dt>Size</dt><dd>${fmtBytes(d.size_bytes)}</dd>` : ''}
        ${d.source_ref?.repo ? `<dt>Hugging Face</dt><dd><code>${esc(d.source_ref.repo)}/${esc(d.source_ref.filename || '')}</code></dd>` : ''}
        ${d.source_ref?.train ? `<dt>Splits</dt><dd><code>${esc(d.source_ref.train)}</code> / <code>${esc(d.source_ref.val)}</code></dd>` : ''}</dl></div>
      <div class="panel"><h3>Validation</h3>
        ${ins.valid === false ? `<p class="error">${(ins.errors || []).map(esc).join('<br>')}</p>` : ''}
        ${ins.valid === null ? `<p class="hint">${esc(ins.note)}</p>` : ''}
        ${ins.episodes != null ? `<dl class="kv"><dt>Episodes</dt><dd>${ins.episodes.toLocaleString()}</dd>
          <dt>Steps</dt><dd>${ins.steps.toLocaleString()}</dd>
          <dt>Episode length</dt><dd>${ins.min_episode_len}–${ins.max_episode_len} (mean ${ins.mean_episode_len.toFixed(1)})</dd>
          <dt>Frames</dt><dd>${ins.img_size}×${ins.img_size}</dd><dt>Action dim</dt><dd>${ins.action_dim}</dd>
          <dt>Trains as</dt><dd>${(ins.compatible_presets || []).map((p) => esc(catalog.presets[p]?.label || p)).join(', ')}</dd></dl>`
    : (ins.img_size ? `<dl class="kv"><dt>Frames</dt><dd>${ins.img_size}px</dd><dt>Action dim</dt><dd>${ins.action_dim}</dd></dl>` : '')}
        ${(ins.warnings || []).map((w) => `<p class="hint">⚠ ${esc(w)}</p>`).join('')}</div>
    </div>
    ${cols.length ? `<div class="panel" style="margin-top:14px"><h3>Columns</h3><div class="table-wrap"><table>
      <thead><tr><th>Name</th><th>Shape</th><th>dtype</th><th>Compression</th></tr></thead><tbody>
      ${cols.map(([k, c]) => `<tr><td><code>${esc(k)}</code></td><td class="mono">${esc(JSON.stringify(c.shape))}</td><td>${esc(c.dtype)}</td><td>${esc(c.compression || '—')}</td></tr>`).join('')}
      </tbody></table></div></div>` : ''}
    ${d.is_owner ? `<div class="panel" style="margin-top:14px"><h3>Manage</h3><div class="row">
      <button id="togglevis">Make ${d.visibility === 'public' ? 'private' : 'public'}</button>
      ${d.source === 'upload' && d.status === 'ready' ? '<button id="dl">Download</button>' : ''}
      ${d.source === 'upload' && d.status === 'uploading' ? '<button id="resume">Resume upload</button>' : ''}
      <button class="danger" id="del">Delete</button></div></div>`
    : (d.source === 'upload' && d.status === 'ready' ? '<div class="row" style="margin-top:14px"><button id="dl">Download</button></div>' : '')}`;
  $('#togglevis')?.addEventListener('click', async () => {
    await api(`/api/datasets/${id}`, { method: 'PATCH', body: { visibility: d.visibility === 'public' ? 'private' : 'public' } });
    renderDataset(id);
  });
  $('#dl')?.addEventListener('click', async () => { location.href = (await api(`/api/datasets/${id}/download`)).url; });
  $('#del')?.addEventListener('click', async () => {
    if (!confirm(`Delete dataset "${d.name}"? This removes the file.`)) return;
    await api(`/api/datasets/${id}`, { method: 'DELETE' });
    location.hash = '#/datasets';
  });
  $('#resume')?.addEventListener('click', () => modal(`<h2>Resume upload</h2><p class="hint">Pick the same file again.</p>
    <input type="file" id="rf" accept=".h5,.hdf5"><div class="progress" style="margin-top:12px"><div style="width:0"></div></div>
    <p class="hint" id="rs"></p>`, (root, close) => {
    $('#rf', root).addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (file.size !== d.expected_bytes) { $('#rs', root).textContent = 'That file has a different size.'; return; }
      try {
        await chunkedUpload(id, file, (f) => { $('.progress div', root).style.width = `${f * 100}%`; });
        await api(`/api/datasets/${id}/complete`, { method: 'POST' });
        close();
        renderDataset(id);
      } catch (err) { $('#rs', root).textContent = err.message; }
    });
  }));
}

// ---------------------------------------------------------------- models

async function renderModels(_, params) {
  let scope = params.get('scope') || 'all';
  let env = params.get('env') || '';
  let tab = 'list';
  const draw = async () => {
    if (tab === 'board') {
      const e = env || 'pusht';
      const rows = await api(`/api/models/leaderboard?env=${e}`);
      $('#mlist').innerHTML = `<div class="panel" style="grid-column:1/-1"><h2>${esc(catalog.presets[e].label)}: planning success</h2>
        ${rows.length ? `<table><thead><tr><th>#</th><th>Model</th><th>Variant</th><th>Planner</th><th>Success</th></tr></thead><tbody>
        ${rows.map((r, i) => `<tr class="click" data-model="${r.model.id}"><td>${i + 1}</td><td>${esc(r.model.name)}<div class="hint">@${esc(r.model.owner?.username)}</div></td>
          <td>${esc(r.model.variant_label)}</td><td><code>${esc(r.planner)}</code></td><td><b>${r.success_rate.toFixed(1)}%</b></td></tr>`).join('')}</tbody></table>`
    : '<div class="empty">No evaluated models for this environment yet. Run a planning eval from a model page.</div>'}</div>`;
      $$('[data-model]').forEach((r) => r.addEventListener('click', () => { location.hash = `#/models/${r.dataset.model}`; }));
      return;
    }
    const rows = await api(`/api/models?scope=${scope}${env ? `&env=${env}` : ''}`);
    $('#mlist').innerHTML = rows.length ? rows.map(modelCard).join('') : '<div class="empty">No models match.</div>';
  };
  view.innerHTML = `
    <div class="page-head"><div><h1>World models</h1><p>Trained H-JEPA checkpoints: yours, shared by others, and the paper's released models.</p></div>
      <a class="btn primary" href="#/train">Train a model</a></div>
    <div class="row spread" style="margin-bottom:12px">
      <div class="tabs" style="margin:0;border:0">${['all', 'mine', 'public'].map((s) => `<button data-scope="${s}" class="${s === scope ? 'on' : ''}">${s}</button>`).join('')}
        <button data-board>leaderboard</button></div>
      <select id="envf" style="width:auto"><option value="">All environments</option>
        ${Object.entries(catalog.presets).map(([k, p]) => `<option value="${k}" ${k === env ? 'selected' : ''}>${esc(p.label)}</option>`).join('')}</select></div>
    <div class="grid cards" id="mlist"></div>`;
  $$('[data-scope]').forEach((b) => b.addEventListener('click', () => {
    scope = b.dataset.scope; tab = 'list';
    $$('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
    draw();
  }));
  $('[data-board]').addEventListener('click', (e) => {
    tab = 'board';
    $$('.tabs button').forEach((x) => x.classList.toggle('on', x === e.target));
    draw();
  });
  $('#envf').addEventListener('change', (e) => { env = e.target.value; draw(); });
  await draw();
}

function bestPlanning(m) {
  const p = Object.entries(m.metrics?.planning || {}).filter(([, v]) => v?.success_rate != null);
  if (!p.length) return null;
  return p.reduce((a, b) => (b[1].success_rate > a[1].success_rate ? b : a));
}

function modelCard(m) {
  const best = bestPlanning(m);
  const bars = Array.from({ length: m.num_levels }, (_, i) =>
    `<span style="display:inline-block;width:${18 + i * 10}px;height:4px;border-radius:2px;background:${LEVEL_COLORS[i]}"></span>`).join(' ');
  return `<a class="card" href="#/models/${m.id}">
    <div class="title">${esc(m.name)}</div>
    <div class="sub">${esc(m.env_label)} · ${esc(m.variant_label)}</div>
    <div class="row">${bars}</div>
    <div class="row spread"><span class="sub">@${esc(m.owner?.username)} · ${ago(m.created_at)}</span>
      ${best ? `<span class="pill good">${best[1].success_rate.toFixed(1)}% · ${esc(best[0])}</span>` : visPill(m.visibility)}</div></a>`;
}

async function renderModel(id) {
  const m = await api(`/api/models/${id}`);
  const variant = catalog.variants[m.variant] || { planners: ['flat'] };
  const sim = catalog.presets[m.env_preset]?.simulator;
  const planning = Object.entries(m.metrics?.planning || {});
  const probing = Object.entries(m.metrics?.probing || {});
  const final = Object.entries(m.metrics?.final || {});
  view.innerHTML = `
    <div class="page-head"><div><h1>${esc(m.name)}</h1>
      <p>${esc(m.env_label)} · ${esc(m.variant_label)} · @${esc(m.owner?.username)} · ${ago(m.created_at)}</p></div>
      <div class="row">${visPill(m.visibility)}
        <a class="btn" href="#/simulate?model=${m.id}">Dream rollout</a>
        <a class="btn primary" href="#/simulate?model=${m.id}&mode=interactive">Interactive simulator</a></div></div>
    <div class="grid cols-2">
      <div class="panel"><h3>Hierarchy</h3>
        ${Array.from({ length: m.num_levels }, (_, i) => `<div class="row" style="margin-bottom:6px">
          <span class="lvl-tag" style="--c:${LEVEL_COLORS[i]}">L${i + 1}</span>
          <span class="hint">${i === 0 ? 'encodes pixels; actions are raw blocks of frameskip steps' : `encodes chunks of L${i} latents; plans with learned macro-actions`}</span></div>`).reverse().join('')}
        <p class="hint" style="margin-top:10px">${esc(m.description)}</p>
        <dl class="kv"><dt>Source</dt><dd>${m.source === 'huggingface' ? `<code>${esc(m.source_ref.repo)}/${esc(m.source_ref.prefix)}</code>` : (m.job_id ? `<a href="#/jobs/${m.job_id}">training job</a>` : '—')}</dd>
          <dt>Files</dt><dd>${m.files.length ? m.files.map((f) => `<button class="link" data-file="${esc(f)}"><code>${esc(f)}</code></button>`).join(' ') : (m.source === 'huggingface' ? 'on Hugging Face' : '—')}</dd></dl></div>
      <div class="panel"><h3>Planning success</h3>
        ${planning.length ? `<table><thead><tr><th>Planner</th><th>Success</th><th></th></tr></thead><tbody>${planning.map(([p, v]) => `<tr><td><code>${esc(p)}</code></td>
          <td><b>${v.success_rate != null ? `${v.success_rate.toFixed(1)}%` : '—'}</b></td><td>${v.job_id ? `<a href="#/jobs/${v.job_id}">job</a>` : ''}</td></tr>`).join('')}</tbody></table>`
    : '<p class="hint">Not evaluated yet.</p>'}
        ${sim ? `<form class="row" id="evalform" style="margin-top:12px">
          <select name="planner" style="width:auto">${variant.planners.map((p) => `<option ${p === (m.num_levels > 1 ? `l${m.num_levels}` : 'flat') ? 'selected' : ''}>${p}</option>`).join('')}</select>
          <select name="gpu" style="width:auto">${gpuOptions()}</select>
          <button class="primary" type="submit">Run planning eval</button></form>
          <p class="hint">50 fixed start/goal tasks in the ${esc(m.env_label)} simulator (upstream <code>eval.py</code>).</p>`
    : '<p class="hint">Custom-data models have no simulator to plan in; use dream rollouts instead.</p>'}</div>
    </div>
    ${probing.length || final.length ? `<div class="grid cols-2" style="margin-top:14px">
      ${final.length ? `<div class="panel"><h3>Final training losses</h3><dl class="kv">${final.map(([k, v]) => `<dt><code>${esc(k)}</code></dt><dd>${v.toPrecision(4)}</dd>`).join('')}</dl></div>` : ''}
      ${probing.length ? `<div class="panel"><h3>Probing / decoding</h3><dl class="kv">${probing.slice(0, 30).map(([k, v]) => `<dt><code>${esc(k)}</code></dt><dd>${Number(v).toPrecision(4)}</dd>`).join('')}</dl></div>` : ''}</div>` : ''}
    ${m.is_owner ? `<div class="panel" style="margin-top:14px"><h3>Manage</h3><div class="row">
      <button id="togglevis">Make ${m.visibility === 'public' ? 'private' : 'public'}</button><button class="danger" id="del">Delete</button></div></div>` : ''}`;
  $('#evalform')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = formData(e.target);
    try {
      const job = await api('/api/jobs/evaluate', { body: { model_id: id, planner: f.planner, gpu: { type: f.gpu } } });
      location.hash = `#/jobs/${job.id}`;
    } catch (err) { toast(err.message, true); }
  });
  $$('[data-file]').forEach((b) => b.addEventListener('click', async () => {
    location.href = (await api(`/api/models/${id}/files/${b.dataset.file}`)).url;
  }));
  $('#togglevis')?.addEventListener('click', async () => {
    await api(`/api/models/${id}`, { method: 'PATCH', body: { visibility: m.visibility === 'public' ? 'private' : 'public' } });
    renderModel(id);
  });
  $('#del')?.addEventListener('click', async () => {
    if (!confirm('Delete this model entry?')) return;
    await api(`/api/models/${id}`, { method: 'DELETE' });
    location.hash = '#/models';
  });
}

// ---------------------------------------------------------------- GPUs

let gpuCache = null;
async function loadGpus() {
  if (!gpuCache) gpuCache = await api('/api/meta/gpus');
  return gpuCache;
}
function gpuOptions(selected) {
  const list = gpuCache || [];
  return list.map((g) => `<option value="${esc(g.id)}" ${g.id === selected || (!selected && g.recommended) ? 'selected' : ''}>${esc(g.label)} · ${g.vram_gb} GB</option>`).join('');
}
function gpuPicker(selected) {
  return `<div class="choices" id="gpus">${gpuCache.map((g) => {
    const price = g.live ? (g.secure_usd_hr ?? g.community_usd_hr) : g.est_usd_hr;
    return `<button type="button" class="choice ${g.id === selected ? 'on' : ''}" data-gpu="${esc(g.id)}">
      <span class="t">${esc(g.label)}${g.recommended ? ' ★' : ''}</span>
      <span class="d">${g.vram_gb} GB · ${price != null ? `${fmtUsd(price)}/h${g.live ? '' : ' est.'}` : 'price n/a'}</span>
      ${g.stock ? `<span class="d">stock: ${esc(g.stock)}</span>` : ''}</button>`;
  }).join('')}</div>`;
}

// ---------------------------------------------------------------- train

async function renderTrain(_, params) {
  await loadGpus();
  const datasets = (await api('/api/datasets?scope=all')).filter((d) => d.status === 'ready');
  const state = {
    preset: params.get('preset') || 'pusht', variant: params.get('variant') || 'hjepa_l2',
    gpu: (gpuCache.find((g) => g.recommended) || gpuCache[0]).id, dataset: params.get('dataset') || '',
  };
  const fits = (d) => (d.source === 'official' ? d.env_preset === state.preset
    : state.preset === 'custom' || d.env_preset === state.preset);
  const draw = () => {
    const preset = catalog.presets[state.preset];
    const usable = datasets.filter(fits);
    if (!usable.find((d) => d.id === state.dataset)) state.dataset = usable[0]?.id || '';
    const chosen = usable.find((d) => d.id === state.dataset);
    view.innerHTML = `
      <div class="page-head"><div><h1>Train a world model</h1>
        <p>Runs upstream <code>main_hjepa.py</code> (H-JEPA @ ${esc(catalog.hjepa.commit.slice(0, 7))}) on a rented GPU, then trains pixel decoders${preset.simulator ? ' and runs the planning eval' : ''}.</p></div></div>
      <form class="stack" id="trainform">
        <div class="panel"><h3>1 · Environment</h3><div class="choices">
          ${Object.entries(catalog.presets).map(([k, p]) => `<button type="button" class="choice ${k === state.preset ? 'on' : ''}" data-preset="${k}">
            <span class="t">${esc(p.label)}</span><span class="d">${esc(p.description).slice(0, 90)}</span></button>`).join('')}</div></div>
        <div class="panel"><h3>2 · Model</h3><div class="choices">
          ${Object.entries(catalog.variants).map(([k, v]) => `<button type="button" class="choice ${k === state.variant ? 'on' : ''}" data-variant="${k}">
            <span class="t">${esc(v.label)}</span><span class="d">${v.family === 'hjepa' ? 'end-to-end latent hierarchy' : v.family === 'hwm' ? 'baseline: identity upper encoders' : 'single-level baseline'}</span></button>`).join('')}</div></div>
        <div class="panel"><h3>3 · Data</h3><div class="form-grid">
          <label>Training dataset<select name="train_dataset_id" id="dsel">${usable.length ? usable.map((d) => `<option value="${d.id}" ${d.id === state.dataset ? 'selected' : ''}>${esc(d.name)} · @${esc(d.owner?.username)}</option>`).join('') : '<option value="">no compatible datasets</option>'}</select></label>
          ${chosen && chosen.source !== 'official' ? `<label>Validation dataset<select name="val_dataset_id"><option value="">hold out 5% of training episodes</option>
            ${usable.filter((d) => d.id !== chosen.id && d.source !== 'official').map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join('')}</select></label>` : ''}
          ${state.preset === 'custom' ? `<label>Frameskip<input name="frameskip" type="number" min="1" max="50" value="${preset.frameskip}"></label>` : ''}
        </div>${!usable.length ? '<p class="hint">Upload a dataset tagged for this environment, or pick "Custom data".</p>' : ''}</div>
        <div class="panel"><h3>4 · GPU</h3>${gpuPicker(state.gpu)}
          <div class="form-grid" style="margin-top:12px">
            <label>Cloud<select name="cloud_type"><option value="SECURE">Secure cloud</option><option value="COMMUNITY">Community cloud (cheaper)</option></select></label>
            <label>GPUs<select name="gpu_count"><option>1</option><option>2</option></select></label>
            <label>Time limit (hours)<input name="max_hours" type="number" min="0.5" step="0.5" max="${catalog.limits.max_hours}" value="6"></label>
            <label class="check" style="align-self:end"><input type="checkbox" name="allow_fallback"> fall back to other small GPUs</label>
          </div></div>
        <div class="panel"><h3>5 · Run</h3><div class="form-grid">
          <label>Epochs<input name="epochs" type="number" min="1" max="200" value="${state.preset === 'pusht' ? 9 : 5}"></label>
          <label>Batch size<input name="batch_size" type="number" min="4" max="1024" value="128"></label>
          <label>Seed<input name="seed" type="number" min="0" value="42"></label>
          <label>Result visibility<select name="output_visibility"><option value="private">private</option><option value="public">public</option></select></label>
          <label>Model name<input name="model_name" placeholder="optional"></label>
        </div>
        <div class="row" style="margin-top:10px">
          <label class="check"><input type="checkbox" name="run_planning_eval" ${preset.simulator ? 'checked' : 'disabled'}> planning eval after training</label>
          <label class="check"><input type="checkbox" name="train_decoders" checked> train pixel decoders (for simulation)</label></div>
        <details style="margin-top:10px"><summary class="hint">Advanced: Hydra overrides</summary>
          <textarea name="extra_overrides" placeholder="level2.lr=2e-5&#10;optimizer.weight_decay=0.01"></textarea></details></div>
        <div class="panel row spread"><div id="estimate" class="hint">Estimating cost…</div>
          <button class="primary" type="submit" ${usable.length ? '' : 'disabled'}>Launch training</button></div>
      </form>`;
    $$('[data-preset]').forEach((b) => b.addEventListener('click', () => { state.preset = b.dataset.preset; draw(); }));
    $$('[data-variant]').forEach((b) => b.addEventListener('click', () => { state.variant = b.dataset.variant; draw(); }));
    $$('[data-gpu]').forEach((b) => b.addEventListener('click', () => {
      state.gpu = b.dataset.gpu;
      $$('[data-gpu]').forEach((x) => x.classList.toggle('on', x === b));
      estimate();
    }));
    $('#dsel').addEventListener('change', (e) => { state.dataset = e.target.value; draw(); });
    const form = $('#trainform');
    form.addEventListener('change', estimate);
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = form.querySelector('button.primary');
      btn.disabled = true;
      try {
        const job = await api('/api/jobs/train', { body: trainBody() });
        toast('Training job launched');
        location.hash = `#/jobs/${job.id}`;
      } catch (err) { toast(err.message, true); btn.disabled = false; }
    });
    estimate();
  };
  const trainBody = () => {
    const f = formData($('#trainform'));
    return {
      preset: state.preset, variant: state.variant, train_dataset_id: f.train_dataset_id,
      val_dataset_id: f.val_dataset_id || null, epochs: f.epochs, batch_size: f.batch_size, seed: f.seed,
      frameskip: f.frameskip ?? null, run_planning_eval: !!f.run_planning_eval, train_decoders: !!f.train_decoders,
      output_visibility: f.output_visibility, model_name: f.model_name, max_hours: f.max_hours,
      extra_overrides: (f.extra_overrides || '').split('\n').map((s) => s.trim()).filter(Boolean),
      gpu: { type: state.gpu, count: Number(f.gpu_count), cloud_type: f.cloud_type, allow_fallback: !!f.allow_fallback },
    };
  };
  let estSeq = 0;
  const estimate = async () => {
    const seq = ++estSeq;
    const body = trainBody();
    if (!body.train_dataset_id) { $('#estimate').textContent = 'Choose a dataset.'; return; }
    try {
      const est = await api('/api/jobs/estimate', { body });
      if (seq === estSeq) $('#estimate').innerHTML = `≈ <b>${est.hours} h</b> at ${fmtUsd(est.usd_per_hr)}/h → <b>${fmtUsd(est.usd)}</b> <span class="hint">(${esc(est.note)})</span>`;
    } catch (err) { if (seq === estSeq) $('#estimate').innerHTML = `<span class="error">${esc(err.message)}</span>`; }
  };
  draw();
}

// ---------------------------------------------------------------- jobs

async function renderJobs() {
  view.innerHTML = `<div class="page-head"><div><h1>Jobs</h1><p>Training, planning evaluation and dream rollouts on rented GPUs.</p></div></div>
    <div class="panel" id="jl"></div>`;
  every(5000, async () => {
    const jobs = await api('/api/jobs?limit=200');
    $('#jl').innerHTML = jobsTable(jobs);
    bindJobRows();
  });
}

async function renderJob(id) {
  let lastLog = 0;
  let metricKey = null;
  view.innerHTML = '<div id="jhead"></div><div id="jbody" class="stack"></div>';
  const body = $('#jbody');
  body.innerHTML = `
    <div class="panel" id="jchart" hidden><div class="row spread"><h3>Training curves</h3>
      <div class="row"><select id="mfilter" style="width:auto"></select><label class="check"><input type="checkbox" id="logy" checked> log scale</label></div></div><div id="chart"></div></div>
    <div class="panel" id="jresult" hidden></div>
    <div class="panel" id="jarts" hidden><h3>Outputs</h3><div class="gallery" id="gallery"></div><div id="files" class="row" style="margin-top:10px"></div></div>
    <div class="panel"><div class="row spread"><h3>Logs</h3><label class="check"><input type="checkbox" id="follow" checked> follow</label></div>
      <div class="console" id="console"></div></div>`;
  let job = null;
  let artsLoaded = false;
  const drawHead = () => {
    const spec = job.spec;
    const what = job.kind === 'train' ? `${catalog.presets[spec.preset]?.label} · ${catalog.variants[spec.variant]?.label} · ${spec.epochs} epochs · seed ${spec.seed}`
      : job.kind === 'evaluate' ? `planner ${spec.planner} · ${spec.num_eval} tasks` : `episodes ${spec.episodes.join(', ')} · horizon ${spec.horizon} · ${spec.action_mode}`;
    $('#jhead').innerHTML = `<div class="page-head"><div><h1>${esc(job.name)}</h1><p>${esc(kindLabel[job.kind])} · ${esc(what)}</p></div>
      <div class="row">${statusPill(job.status)}${ACTIVE.includes(job.status) ? '<button class="danger" id="cancel">Cancel</button>' : ''}
        ${job.output_model_id ? `<a class="btn primary" href="#/models/${job.output_model_id}">Open model</a>` : ''}</div></div>
      <div class="grid cols-4" style="margin-bottom:14px">
        <div class="panel"><div class="stat-label">Progress</div><div class="progress" style="margin:10px 0 6px"><div style="width:${(job.progress * 100).toFixed(0)}%"></div></div><div class="hint">${esc(job.message)}</div></div>
        <div class="panel"><div class="stat-label">GPU</div><div><b>${esc((spec.gpu?.type || '').replace('NVIDIA ', ''))}</b>${spec.gpu?.count > 1 ? ` ×${spec.gpu.count}` : ''}</div><div class="hint">${esc(job.backend)}${job.pod_id ? ` · ${esc(job.pod_id)}` : ''}</div></div>
        <div class="panel"><div class="stat-label">Cost so far</div><div class="stat">${fmtUsd(job.cost_usd)}</div><div class="hint">${fmtUsd(job.cost_per_hr)}/h · ${job.runtime_hours.toFixed(2)} h${spec.estimate ? ` · est. ${fmtUsd(spec.estimate.usd)}` : ''}</div></div>
        <div class="panel"><div class="stat-label">Timeline</div><div class="hint">created ${fmtTime(job.created_at)}<br>${job.finished_at ? `finished ${fmtTime(job.finished_at)}` : `heartbeat ${ago(job.last_heartbeat)}`}</div></div>
      </div>`;
    $('#cancel')?.addEventListener('click', async () => {
      if (!confirm('Cancel this job and release the GPU?')) return;
      try { job = await api(`/api/jobs/${id}/cancel`, { method: 'POST' }); drawHead(); } catch (err) { toast(err.message, true); }
    });
  };
  const drawResult = () => {
    const r = job.result || {};
    const metrics = r.metrics || {};
    const lines = [];
    if (job.kind === 'evaluate' && metrics.success_rate != null) lines.push(`<div class="stat">${metrics.success_rate.toFixed(1)}%</div><div class="stat-label">planning success over ${metrics.num_eval ?? job.spec.num_eval} tasks</div>`);
    for (const [planner, v] of Object.entries(metrics.planning || {})) lines.push(`<p>Planning success (<code>${esc(planner)}</code>): <b>${v.success_rate?.toFixed(1)}%</b></p>`);
    for (const [lv, v] of Object.entries(r.summary || {})) lines.push(`<p><span class="lvl-tag" style="--c:${LEVEL_COLORS[Number(lv.replace('level', '')) - 1]}">${esc(lv)}</span> normalized latent error: first step ${v.first.toFixed(3)}, last step ${v.last.toFixed(3)}</p>`);
    if (job.status === 'failed') lines.push(`<p class="error">${esc(job.message)}</p>`);
    $('#jresult').hidden = !lines.length;
    $('#jresult').innerHTML = `<h3>Result</h3>${lines.join('')}`;
  };
  const loadArtifacts = async () => {
    const arts = await api(`/api/jobs/${id}/artifacts`);
    if (!arts.length) return;
    $('#jarts').hidden = false;
    const media = arts.filter((a) => /^image\//.test(a.content_type));
    $('#gallery').innerHTML = media.map((a) => `<figure><img src="${esc(a.url)}" alt="${esc(a.name)}" loading="lazy"><figcaption>${esc(a.name)}</figcaption></figure>`).join('');
    $('#files').innerHTML = arts.filter((a) => !media.includes(a)).map((a) => `<a class="btn" href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.name)} · ${fmtBytes(a.size_bytes)}</a>`).join('');
    const simJson = arts.find((a) => a.name === 'simulation.json');
    if (simJson) {
      const sim = await (await fetch(simJson.url)).json();
      const series = {};
      for (const ep of sim.episodes) for (const [lv, r] of Object.entries(ep.levels)) series[`ep${ep.episode} L${lv}`] = r.latent_error.map((v, i) => [i + 1, v]);
      $('#jchart').hidden = false;
      $('#jchart h3').textContent = 'Dream error: normalized latent MSE vs. reality, per predicted step';
      $('#mfilter').hidden = true;
      $('#chart').innerHTML = lineChart(series, { logY: $('#logy').checked });
    }
  };
  const tick = async () => {
    job = await api(`/api/jobs/${id}`);
    drawHead();
    drawResult();
    const logs = await api(`/api/jobs/${id}/logs?after=${lastLog}`);
    if (logs.lines.length) {
      const c = $('#console');
      c.insertAdjacentHTML('beforeend', logs.lines.map((l) => (l.line.startsWith('[status]') ? `<span class="st">${esc(l.line)}</span>` : esc(l.line))).join('\n') + '\n');
      lastLog = logs.last_id;
      if ($('#follow').checked) c.scrollTop = c.scrollHeight;
    }
    if (job.kind === 'train') {
      const metrics = await api(`/api/jobs/${id}/metrics`);
      const names = Object.keys(metrics).sort();
      if (names.length) {
        $('#jchart').hidden = false;
        const groups = [...new Set(names.map((n) => n.replace(/_level\d+$/, '').replace(/^(fit|validate|val|train)\//, '')))];
        const sel = $('#mfilter');
        if (sel.options.length !== groups.length) {
          sel.innerHTML = groups.map((g) => `<option>${esc(g)}</option>`).join('');
          metricKey = metricKey || groups.find((g) => /pred_loss|loss$/.test(g)) || groups[0];
          sel.value = metricKey;
        }
        const pick = Object.fromEntries(names.filter((n) => n.replace(/_level\d+$/, '').replace(/^(fit|validate|val|train)\//, '') === metricKey).map((n) => [n, metrics[n]]));
        $('#chart').innerHTML = lineChart(pick, { logY: $('#logy').checked });
      }
    }
    if (!ACTIVE.includes(job.status) && !artsLoaded) { artsLoaded = true; await loadArtifacts(); }
  };
  $('#mfilter').addEventListener('change', (e) => { metricKey = e.target.value; tick(); });
  $('#logy').addEventListener('change', () => { artsLoaded = false; tick(); });
  every(4000, () => ((!job || ACTIVE.includes(job.status) || !artsLoaded) ? tick() : null));
}

// ---------------------------------------------------------------- simulate

async function renderSimulate(_, params) {
  await loadGpus();
  const models = await api('/api/models?scope=all');
  const datasets = (await api('/api/datasets?scope=all')).filter((d) => d.status === 'ready');
  const sessions = await api('/api/sessions');
  let modelId = params.get('model') || models.find((m) => m.source === 'job')?.id || models[0]?.id;
  let mode = params.get('mode') || 'dream';
  const draw = () => {
    const m = models.find((x) => x.id === modelId);
    const dsFor = datasets.filter((d) => (d.source === 'official' ? d.env_preset === m?.env_preset
      : d.env_preset === m?.env_preset || d.env_preset === 'custom' || m?.env_preset === 'custom'));
    view.innerHTML = `
      <div class="page-head"><div><h1>Simulate a world model</h1>
        <p>Start from real frames, then let the model imagine the future at every level of its hierarchy. Predictions are decoded to pixels with the decoders trained after training (or shown as the nearest real frame).</p></div></div>
      <div class="stack">
        <div class="panel"><div class="form-grid">
          <label>World model<select id="msel">${models.map((x) => `<option value="${x.id}" ${x.id === modelId ? 'selected' : ''}>${esc(x.name)}</option>`).join('')}</select></label>
          <label>Start frames from<select id="dsel">${dsFor.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join('')}</select></label>
          <label>GPU<select id="gsel">${gpuOptions()}</select></label></div>
          <div class="tabs" style="margin:14px 0 0"><button data-mode="dream" class="${mode === 'dream' ? 'on' : ''}">Dream rollout (batch)</button>
            <button data-mode="interactive" class="${mode === 'interactive' ? 'on' : ''}">Interactive simulator</button></div>
          ${mode === 'dream' ? `<form class="stack" id="dreamform" style="margin-top:12px"><div class="form-grid">
            <label>Episodes (comma separated)<input name="episodes" value="0, 1, 2"></label>
            <label>Start raw step<input name="start" type="number" min="0" value="0"></label>
            <label>Context latents<input name="context" type="number" min="1" max="16" value="3"></label>
            <label>Horizon (steps per level)<input name="horizon" type="number" min="1" max="256" value="24"></label>
            <label>Actions<select name="action_mode"><option value="ground_truth">recorded (ground truth)</option><option value="zero">zero (counterfactual)</option>
              <option value="random">random (counterfactual)</option><option value="reversed">reversed (counterfactual)</option></select></label>
            <label>Action scale<input name="action_scale" type="number" step="0.1" value="1"></label></div>
            <div class="row">${Array.from({ length: m?.num_levels || 1 }, (_, i) => `<label class="check"><input type="checkbox" name="lv${i + 1}" checked> <span class="lvl-tag" style="--c:${LEVEL_COLORS[i]}">L${i + 1}</span></label>`).join('')}</div>
            <div><button class="primary" type="submit">Launch dream job</button></div>
            <p class="hint">Produces side-by-side real-vs-dream GIFs per level and the latent error curve.</p></form>`
    : `<div style="margin-top:12px"><p class="hint">Rents a GPU that keeps the model in memory; step each level yourself and steer level 1 with your own actions. Stops automatically after idling.</p>
            <button class="primary" id="startsess">Start interactive session</button></div>`}
        </div>
        ${sessions.length ? `<div class="panel"><h3>Your sessions</h3><table><tbody>${sessions.slice(0, 8).map((s) => `<tr class="click" data-sess="${s.id}">
          <td>${esc(s.spec.model_name || s.model_id)}</td><td>${statusPill(s.status)}</td><td>${fmtUsd(s.cost_usd)}</td><td class="hint">${ago(s.created_at)}</td></tr>`).join('')}</tbody></table></div>` : ''}
      </div>`;
    $('#msel').addEventListener('change', (e) => { modelId = e.target.value; draw(); });
    $$('[data-mode]').forEach((b) => b.addEventListener('click', () => { mode = b.dataset.mode; draw(); }));
    $$('[data-sess]').forEach((r) => r.addEventListener('click', () => { location.hash = `#/sessions/${r.dataset.sess}`; }));
    $('#dreamform')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const f = formData(e.target);
      const levels = Array.from({ length: m.num_levels }, (_, i) => i + 1).filter((lv) => f[`lv${lv}`]);
      try {
        const job = await api('/api/jobs/simulate', { body: {
          model_id: modelId, dataset_id: $('#dsel').value || null, start: f.start, context: f.context, horizon: f.horizon,
          action_mode: f.action_mode, action_scale: f.action_scale, levels,
          episodes: f.episodes.split(',').map((s) => parseInt(s, 10)).filter((n) => n >= 0),
          gpu: { type: $('#gsel').value } } });
        location.hash = `#/jobs/${job.id}`;
      } catch (err) { toast(err.message, true); }
    });
    $('#startsess')?.addEventListener('click', async () => {
      try {
        const s = await api('/api/sessions', { body: { model_id: modelId, dataset_id: $('#dsel').value || null, gpu: { type: $('#gsel').value } } });
        location.hash = `#/sessions/${s.id}`;
      } catch (err) { toast(err.message, true); }
    });
  };
  if (!models.length) { view.innerHTML = '<div class="empty">No models yet.</div>'; return; }
  draw();
}

async function renderSession(id) {
  let sess = await api(`/api/sessions/${id}`);
  let info = null;
  const errs = {};
  const call = (action, payload = {}) => api(`/api/sessions/${id}/call/${action}`, { body: payload });
  const head = () => `<div class="page-head"><div><h1>Interactive simulator</h1><p>${esc(sess.spec.model_name || '')}</p></div>
    <div class="row">${statusPill(sess.status)}<span class="pill">${fmtUsd(sess.cost_usd)} · ${fmtUsd(sess.cost_per_hr)}/h</span>
    ${['starting', 'ready'].includes(sess.status) ? '<button class="danger" id="stopsess">Stop &amp; release GPU</button>' : ''}</div></div>`;
  const bindStop = () => $('#stopsess')?.addEventListener('click', async () => {
    sess = await api(`/api/sessions/${id}/stop`, { method: 'POST' });
    route();
  });
  if (sess.status !== 'ready') {
    view.innerHTML = `${head()}<div class="panel"><p>${esc(sess.message)}</p>${sess.status === 'starting' ? '<p class="hint">Renting a GPU, pulling the worker image and loading the model. This usually takes a few minutes.</p>' : ''}</div>`;
    bindStop();
    if (sess.status === 'starting') {
      const t = setInterval(async () => {
        const s = await api(`/api/sessions/${id}`);
        if (s.status !== 'starting') { clearInterval(t); route(); }
      }, 3000);
      timers.push(t);
    }
    return;
  }
  info = await call('info');
  const L = info.num_levels;
  view.innerHTML = `${head()}
    <div class="panel"><div class="form-grid">
      <label>Episode (0–${info.episodes - 1})<input id="ep" type="number" min="0" max="${info.episodes - 1}" value="${info.episode ?? 0}"></label>
      <label>Start raw step<input id="st" type="number" min="0" value="0"></label>
      <div style="align-self:end"><button class="primary" id="reset">Reset to real frame</button></div></div>
      <p class="hint" id="epinfo">Frameskip ${info.frameskip}; decoders for levels ${(info.decoders || []).join(', ') || 'none (nearest real frame shown)'}.</p></div>
    <div class="panel" id="actpanel" hidden><h3>Level-1 action (held for ${info.frameskip} env steps)</h3><div class="form-grid" id="sliders"></div>
      <div class="row" style="margin-top:8px"><button id="randact">Random action</button></div></div>
    <div class="sim-levels" id="levels">${Array.from({ length: L }, (_, i) => L - i).map((lv) => `
      <div class="panel" data-lv="${lv}"><div class="row spread"><span class="lvl-tag" style="--c:${LEVEL_COLORS[lv - 1]}">Level ${lv}</span><span class="hint" data-meta></span></div>
        <div class="frames" style="margin:10px 0"><figure><img data-real alt="real frame"><figcaption>reality</figcaption></figure>
          <figure><img data-dream alt="dreamed frame"><figcaption>dream</figcaption></figure></div>
        <div data-spark></div>
        <div class="row"><select data-mode style="width:auto">
          <option value="ground_truth">recorded ${lv === 1 ? 'action' : 'macro-action'}</option>
          ${lv === 1 ? '<option value="user">my action (sliders)</option><option value="zero">zero action</option>' : ''}
          <option value="random">${lv === 1 ? 'random action' : 'sampled macro-action'}</option></select>
          <button data-step>Step</button><button data-roll>Rollout ×8</button></div></div>`).join('')}</div>`;
  bindStop();
  const setLevel = (lv, frame, real, meta) => {
    const p = $(`[data-lv="${lv}"]`);
    if (frame) $('[data-dream]', p).src = `data:image/png;base64,${frame}`;
    if (real) $('[data-real]', p).src = `data:image/png;base64,${real}`;
    if (meta) $('[data-meta]', p).textContent = meta;
    $('[data-spark]', p).innerHTML = sparkline(errs[lv] || [], LEVEL_COLORS[lv - 1]);
  };
  const drawSliders = () => {
    if (!info.action_dim) return;
    $('#actpanel').hidden = false;
    $('#sliders').innerHTML = Array.from({ length: info.action_dim }, (_, i) => {
      const lo = info.action_low[i]; const hi = info.action_high[i];
      return `<label>a[${i}] <span data-v="${i}">0</span><input type="range" data-a="${i}" min="${lo}" max="${hi}" step="${(hi - lo) / 100 || 0.01}" value="${Math.min(hi, Math.max(lo, 0))}"></label>`;
    }).join('');
    $$('[data-a]').forEach((s) => s.addEventListener('input', () => { $(`[data-v="${s.dataset.a}"]`).textContent = Number(s.value).toFixed(3); }));
  };
  const action = () => $$('[data-a]').map((s) => Number(s.value));
  const doReset = async () => {
    const r = await call('reset', { episode: Number($('#ep').value), start: Number($('#st').value) });
    info = { ...info, ...r };
    for (let lv = 1; lv <= L; lv++) { errs[lv] = []; setLevel(lv, r.levels[lv], r.real, 'context loaded'); }
    $('#epinfo').textContent = `Episode ${r.episode}: ${r.episode_len} raw steps. Frameskip ${info.frameskip}.`;
    drawSliders();
  };
  const applyStep = (lv, s) => {
    if (s.latent_error != null) (errs[lv] = errs[lv] || []).push(s.latent_error);
    setLevel(lv, s.frame, s.real, `t+${s.dreamed}${s.raw_step != null ? ` · raw step ${s.raw_step}` : ' · past episode end'}${s.latent_error != null ? ` · err ${s.latent_error.toFixed(3)}` : ''}`);
  };
  $('#reset').addEventListener('click', () => doReset().catch((e) => toast(e.message, true)));
  $('#randact').addEventListener('click', async () => {
    const { action: a } = await call('sample_action');
    a.forEach((v, i) => { const s = $(`[data-a="${i}"]`); if (s) { s.value = v; s.dispatchEvent(new Event('input')); } });
  });
  $$('[data-lv]').forEach((p) => {
    const lv = Number(p.dataset.lv);
    const payload = () => ({ level: lv, mode: $('[data-mode]', p).value, action: action() });
    $('[data-step]', p).addEventListener('click', async () => {
      try { applyStep(lv, await call('step', payload())); } catch (e) { toast(e.message, true); }
    });
    $('[data-roll]', p).addEventListener('click', async () => {
      try {
        const r = await call('rollout', { ...payload(), horizon: 8 });
        for (const s of r.steps) { applyStep(lv, s); await new Promise((res) => setTimeout(res, 120)); }
      } catch (e) { toast(e.message, true); }
    });
  });
  await doReset();
  const t = setInterval(async () => {
    sess = await api(`/api/sessions/${id}`);
    if (sess.status !== 'ready') { clearInterval(t); route(); }
  }, 15000);
  timers.push(t);
}

// ---------------------------------------------------------------- settings

async function renderSettings() {
  me = await api('/api/auth/me');
  view.innerHTML = `
    <div class="page-head"><div><h1>Settings</h1></div></div>
    <div class="grid cols-2">
      <div class="panel"><h3>Account</h3><dl class="kv"><dt>Username</dt><dd>@${esc(me.username)}</dd><dt>Email</dt><dd>${esc(me.email)}</dd>
        <dt>Role</dt><dd>${me.is_admin ? 'admin' : 'member'}</dd><dt>Joined</dt><dd>${fmtTime(me.created_at)}</dd></dl></div>
      <div class="panel"><h3>RunPod billing</h3>
        <p>${me.has_runpod_key ? 'GPU jobs bill to <b>your RunPod account</b>.' : 'GPU jobs bill to the platform account (if the admin configured one).'}</p>
        <form class="stack" id="keyform"><label>RunPod API key<input name="runpod_api_key" type="password" autocomplete="off" placeholder="${me.has_runpod_key ? '•••••••• (saved, encrypted)' : 'rpa_…'}"></label>
          <div class="row"><button class="primary" type="submit">Save key</button>${me.has_runpod_key ? '<button type="button" id="clearkey">Remove key</button>' : ''}</div></form>
        <p class="hint">Create one at runpod.io → Settings → API Keys. Stored encrypted; only used to start and stop your pods.</p></div>
      <div class="panel"><h3>Instance</h3><dl class="kv"><dt>GPU launcher</dt><dd>${esc(catalog.launcher)}</dd>
        <dt>H-JEPA</dt><dd><a href="${esc(catalog.hjepa.repo)}" target="_blank" rel="noopener">${esc(catalog.hjepa.repo.replace('https://github.com/', ''))}</a> @ <code>${esc(catalog.hjepa.commit.slice(0, 10))}</code></dd>
        <dt>Max job length</dt><dd>${catalog.limits.max_hours} h</dd><dt>Concurrent GPU jobs</dt><dd>${catalog.limits.max_concurrent_jobs}</dd></dl></div>
    </div>`;
  $('#keyform').addEventListener('submit', async (e) => {
    e.preventDefault();
    try { me = await api('/api/auth/settings', { method: 'PUT', body: formData(e.target) }); gpuCache = null; toast('Key saved'); renderSettings(); } catch (err) { toast(err.message, true); }
  });
  $('#clearkey')?.addEventListener('click', async () => {
    me = await api('/api/auth/settings', { method: 'PUT', body: { clear_runpod_api_key: true } });
    gpuCache = null;
    renderSettings();
  });
}

route();
