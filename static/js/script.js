/**
 * script.js
 * =========
 * Client-side logic for the Lifestyle Telemetry & Overthinking Risk Predictor.
 *
 * Covers:
 *  - Slider live-value updates
 *  - Form submission with loading animation
 *  - Dark mode toggle + localStorage persistence
 *  - Demo data fill + Reset
 *  - Local analysis history (localStorage)
 *  - Model metrics fetch (About section)
 *  - Navbar scroll behaviour + mobile menu
 *  - Result page: risk circle animation, factor bars, charts
 *  - Dashboard page: metric cards, charts, data explorer with pagination/sort/search
 */

/* ── Constants ──────────────────────────────────────────────── */
const HISTORY_KEY  = 'lifestyleRiskHistory';
const THEME_KEY    = 'lifestyleTheme';
const MAX_HISTORY  = 20;

const DEMO_VALUES = {
  sleep_hours:               5.5,
  caffeine_mg:               350,
  screen_time_hours:         9,
  workload_level:            8,
  physical_activity_minutes: 20,
  study_hours:               9,
  break_frequency:           3,
  mood_score:                4,
  hydration_glasses:         3,
  meditation_minutes:        0,
  social_interaction_hours:  0.5,
};

const DEFAULT_VALUES = {
  sleep_hours:               7.0,
  caffeine_mg:               200,
  screen_time_hours:         6.0,
  workload_level:            5,
  physical_activity_minutes: 45,
  study_hours:               6.0,
  break_frequency:           8,
  mood_score:                6,
  hydration_glasses:         6,
  meditation_minutes:        0,
  social_interaction_hours:  2.0,
};

/* Slider display suffix map */
const SLIDER_SUFFIX = {
  sleep_hours:               ' hrs',
  caffeine_mg:               ' mg',
  screen_time_hours:         ' hrs',
  workload_level:            '/10',
  physical_activity_minutes: ' min',
  study_hours:               ' hrs',
  break_frequency:           '',
  mood_score:                '/10',
  hydration_glasses:         ' gl',
  meditation_minutes:        ' min',
  social_interaction_hours:  ' hrs',
};

/* ── Utility helpers ────────────────────────────────────────── */
function $(sel, ctx) { return (ctx || document).querySelector(sel); }
function $$(sel, ctx) { return Array.from((ctx || document).querySelectorAll(sel)); }

function formatDate(ts) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

/* ── Dark mode ──────────────────────────────────────────────── */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const icon = $('#themeIcon');
  if (icon) {
    icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
  }
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || 'light';
  applyTheme(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
}

/* ── Navbar ─────────────────────────────────────────────────── */
function initNavbar() {
  const navbar = $('#navbar');
  const toggle = $('#navToggle');
  const links  = $('#navLinks');

  if (!navbar) return;

  // Scroll shadow
  window.addEventListener('scroll', () => {
    navbar.style.boxShadow = window.scrollY > 10 ? '0 2px 16px rgba(0,0,0,0.08)' : '';
  });

  // Mobile menu
  if (toggle && links) {
    toggle.addEventListener('click', () => links.classList.toggle('open'));
    // Close when a link is clicked
    links.querySelectorAll('.nav-link').forEach(a => {
      a.addEventListener('click', () => links.classList.remove('open'));
    });
  }

  // Theme toggle
  const themeBtn = $('#themeToggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);
}

/* ── Slider live updates ────────────────────────────────────── */
function initSliders() {
  const sliders = $$('input.slider');
  sliders.forEach(slider => {
    updateSliderDisplay(slider);
    slider.addEventListener('input', () => updateSliderDisplay(slider));
  });
}

function updateSliderDisplay(slider) {
  const id     = slider.id;
  const suffix = SLIDER_SUFFIX[id] || '';
  const val    = parseFloat(slider.value);
  const label  = $(`#${id.replace('_', '_').replace(/_(hours|mg|level|minutes|frequency|score)$/, function(m){
    const map = {
      '_hours': '_val',
      '_mg': '_val',
      '_level': '_val',
      '_minutes': '_val',
      '_frequency': '_val',
      '_score': '_val',
    };
    return map[m] || '_val';
  })}`);

  // Simpler mapping
  const valIdMap = {
    sleep_hours:               'sleep_val',
    caffeine_mg:               'caffeine_val',
    screen_time_hours:         'screen_val',
    workload_level:            'workload_val',
    physical_activity_minutes: 'activity_val',
    study_hours:               'study_val',
    break_frequency:           'breaks_val',
    mood_score:                'mood_val',
  };
  const valEl = document.getElementById(valIdMap[id]);
  if (valEl) {
    valEl.textContent = val % 1 === 0 ? val + suffix : val.toFixed(1) + suffix;
  }

  // Update track fill
  const min  = parseFloat(slider.min);
  const max  = parseFloat(slider.max);
  const pct  = ((val - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(to right, var(--primary) ${pct}%, var(--bg-alt) ${pct}%)`;
}

/* ── Form submit with loading states ───────────────────────── */
function initForm() {
  const form    = $('#predictForm');
  if (!form) return;

  const btn          = $('#analyzeBtn');
  const btnText      = btn && btn.querySelector('.btn-text');
  const btnLoading   = btn && btn.querySelector('.btn-loading');
  const overlay      = $('#loadingOverlay');
  const loadingText  = $('#loadingText');

  const loadingMessages = [
    'Analyzing your lifestyle telemetry...',
    'Finding patterns in your data...',
    'Preparing your insights...',
  ];

  form.addEventListener('submit', (e) => {
    // Let the form POST naturally — just show loading UI
    if (btn) btn.disabled = true;
    if (btnText)    btnText.classList.add('hidden');
    if (btnLoading) btnLoading.classList.remove('hidden');
    if (overlay)    overlay.classList.remove('hidden');

    let msgIdx = 0;
    const cycle = setInterval(() => {
      msgIdx = (msgIdx + 1) % loadingMessages.length;
      if (loadingText) loadingText.textContent = loadingMessages[msgIdx];
    }, 1400);

    // Clean up after 10s in case of server error
    setTimeout(() => {
      clearInterval(cycle);
      if (btn) btn.disabled = false;
      if (btnText)    btnText.classList.remove('hidden');
      if (btnLoading) btnLoading.classList.add('hidden');
      if (overlay)    overlay.classList.add('hidden');
    }, 10000);
  });
}

/* ── Demo data ──────────────────────────────────────────────── */
function fillForm(values) {
  Object.entries(values).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) {
      el.value = val;
      updateSliderDisplay(el);
    }
  });
}

function initDemoReset() {
  const demoBtn  = $('#demoBtn');
  const resetBtn = $('#resetBtn');
  if (demoBtn)  demoBtn.addEventListener('click',  () => fillForm(DEMO_VALUES));
  if (resetBtn) resetBtn.addEventListener('click', () => fillForm(DEFAULT_VALUES));
}

/* ── History ────────────────────────────────────────────────── */
function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function addToHistory(result) {
  const history = getHistory();
  history.unshift({ ...result, ts: Date.now() });
  if (history.length > MAX_HISTORY) history.pop();
  saveHistory(history);
}

function getBadgeClass(level) {
  if (!level) return 'badge-moderate';
  const l = level.toLowerCase();
  if (l.includes('low'))  return 'badge-low';
  if (l.includes('high')) return 'badge-high';
  return 'badge-moderate';
}

function renderHistory() {
  const list    = $('#historyList');
  const empty   = $('#historyEmpty');
  const actions = $('#historyActions');
  if (!list) return;

  const history = getHistory();

  if (history.length === 0) {
    if (empty)   empty.style.display = '';
    if (actions) actions.style.display = 'none';
    return;
  }

  if (empty)   empty.style.display = 'none';
  if (actions) actions.style.display = '';

  // Remove existing items (keep empty placeholder)
  list.querySelectorAll('.history-item').forEach(el => el.remove());

  history.forEach((entry, i) => {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.setAttribute('role', 'button');
    item.setAttribute('tabindex', '0');
    item.setAttribute('aria-label', `View analysis from ${formatDate(entry.ts)}`);
    item.innerHTML = `
      <div class="history-info">
        <span class="history-date"><i class="fas fa-calendar-alt"></i> ${formatDate(entry.ts)}</span>
        <span class="history-score" style="color:${entry.color||'var(--primary)'}">${entry.risk_score} / 100</span>
      </div>
      <span class="history-badge ${getBadgeClass(entry.risk_level)}">${entry.risk_level || 'Unknown'}</span>
    `;
    item.addEventListener('click', () => showHistoryModal(entry));
    item.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') showHistoryModal(entry); });
    list.appendChild(item);
  });
}

function showHistoryModal(entry) {
  // Simple approach: fill the form with the entry's inputs and scroll to form
  if (entry.inputs) {
    fillForm(entry.inputs);
    const form = $('#analyze-section') || $('#predictForm');
    if (form) form.scrollIntoView({ behavior: 'smooth' });
  }
}

function initHistory() {
  renderHistory();
  const clearBtn = $('#clearHistoryBtn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (confirm('Clear all previous analyses?')) {
        localStorage.removeItem(HISTORY_KEY);
        renderHistory();
      }
    });
  }
}

/* ── Model metrics (About section) ─────────────────────────── */
function loadModelMetrics() {
  const maeEl = $('#metricMAE strong');
  const r2El  = $('#metricR2 strong');
  if (!maeEl && !r2El) return;

  fetch('/api/model-performance')
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      if (maeEl) maeEl.textContent = data.mae !== undefined ? data.mae.toFixed(3) : '—';
      if (r2El)  r2El.textContent  = data.r2  !== undefined ? data.r2.toFixed(4)  : '—';
    })
    .catch(() => {});
}

/* ── Smooth scroll for anchor links ─────────────────────────── */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
}

/* ══════════════════════════════════════════════════════════════
   RESULT PAGE
   ══════════════════════════════════════════════════════════════ */

/**
 * Animate the circular risk score gauge.
 * @param {number} score  0–100
 * @param {string} color  stroke colour
 */
function animateRiskCircle(score, color) {
  const fg = document.querySelector('.risk-circle-fg');
  const numEl = document.querySelector('.risk-score-num');
  if (!fg) return;

  const circumference = 565;  // ~2π × 90
  const target = circumference - (score / 100) * circumference;

  fg.style.stroke = color;
  setTimeout(() => { fg.style.strokeDashoffset = target; }, 100);

  // Count-up animation
  if (numEl) {
    let current = 0;
    const duration = 1500;
    const step = 16;
    const increment = score / (duration / step);
    const timer = setInterval(() => {
      current = Math.min(current + increment, score);
      numEl.textContent = Math.round(current);
      if (current >= score) clearInterval(timer);
    }, step);
  }
}

/**
 * Animate factor progress bars.
 */
function animateFactorBars() {
  document.querySelectorAll('.factor-bar-fill').forEach(fill => {
    const pct = parseFloat(fill.dataset.pct || 0);
    setTimeout(() => { fill.style.width = pct + '%'; }, 200);
  });
}

/**
 * Create the Radar chart (Lifestyle Overview).
 */
function createRadarChart(canvasId, inputs) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;

  const labels = ['Sleep', 'Caffeine', 'Screen Time', 'Workload', 'Activity', 'Mood'];

  // Normalise each input to 0–100 for display
  const data = [
    (inputs.sleep_hours               / 12)  * 100,
    (inputs.caffeine_mg               / 600) * 100,
    (inputs.screen_time_hours         / 16)  * 100,
    ((inputs.workload_level - 1)      / 9)   * 100,
    (inputs.physical_activity_minutes / 180) * 100,
    ((inputs.mood_score - 1)          / 9)   * 100,
  ].map(v => Math.round(v * 10) / 10);

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const labelColor = isDark ? '#94a3b8' : '#64748b';

  new Chart(canvas, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label: 'Your Lifestyle',
        data,
        backgroundColor: 'rgba(99,102,241,0.15)',
        borderColor: 'rgba(99,102,241,0.8)',
        pointBackgroundColor: '#6366f1',
        pointBorderColor: '#fff',
        pointRadius: 5,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      animation: { duration: 1200, easing: 'easeInOutQuart' },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, color: labelColor, font: { size: 10 } },
          grid:  { color: gridColor },
          pointLabels: { color: labelColor, font: { size: 12, weight: '600' } },
          angleLines: { color: gridColor },
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

/**
 * Create the Risk Factors bar chart.
 */
function createFactorBarChart(canvasId, factors) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;

  const labels = Object.keys(factors);
  const data   = Object.values(factors);
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const labelColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor  = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Risk Contribution (%)',
        data,
        backgroundColor: [
          'rgba(99,102,241,0.75)',
          'rgba(245,158,11,0.75)',
          'rgba(6,182,212,0.75)',
          'rgba(239,68,68,0.75)',
          'rgba(34,197,94,0.75)',
          'rgba(167,139,250,0.75)',
        ],
        borderColor: [
          '#6366f1','#f59e0b','#06b6d4','#ef4444','#22c55e','#a78bfa'
        ],
        borderWidth: 2,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      animation: { duration: 1200 },
      indexAxis: 'y',
      scales: {
        x: {
          min: 0, max: 100,
          ticks: { color: labelColor, font: { size: 11 } },
          grid:  { color: gridColor },
        },
        y: {
          ticks: { color: labelColor, font: { size: 11, weight: '600' } },
          grid:  { display: false },
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

/**
 * Create the Risk Gauge doughnut chart.
 */
function createGaugeChart(canvasId, score, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;

  const remaining = 100 - score;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const bgColor = isDark ? '#2d3748' : '#e2e8f0';

  new Chart(canvas, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score, remaining],
        backgroundColor: [color, bgColor],
        borderColor: ['transparent', 'transparent'],
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      cutout: '72%',
      animation: { animateRotate: true, duration: 1400, easing: 'easeInOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      }
    }
  });
}

/* Result page initialisation */
function initResultPage() {
  const resultData = window.__RESULT_DATA__;
  if (!resultData) return;

  // Save to history
  addToHistory(resultData);

  // Animate risk circle
  animateRiskCircle(resultData.risk_score, resultData.color || '#6366f1');

  // Animate factor bars
  animateFactorBars();

  // Charts — load Chart.js from CDN first, then draw
  loadChartJS(() => {
    createRadarChart('radarChart', resultData.inputs);
    createFactorBarChart('factorBarChart', resultData.factors);
    createGaugeChart('gaugeChart', resultData.risk_score, resultData.color || '#6366f1');
  });
}

/* ══════════════════════════════════════════════════════════════
   DASHBOARD PAGE
   ══════════════════════════════════════════════════════════════ */

/* State for data explorer */
const tableState = { page: 1, perPage: 10, sort: 'overthinking_risk_score', dir: 'desc', search: '' };

function initDashboard() {
  if (!document.body.classList.contains('dashboard')) return;

  loadChartJS(() => {
    renderDashboardCharts();
    loadTrendChart();
  });

  loadModelPerformance();
  initDataExplorer();
  initRegenerate();
}

function renderDashboardCharts() {
  const resultData = window.__LAST_RESULT__;
  if (!resultData) return;

  createRadarChart('dashRadarChart', resultData.inputs);
  createFactorBarChart('dashFactorChart', resultData.factors);
}

function loadModelPerformance() {
  const container = $('#perfGrid');
  if (!container) return;

  fetch('/api/model-performance')
    .then(r => r.json())
    .then(data => {
      if (data.error) { container.innerHTML = `<p class="text-muted">${data.error}</p>`; return; }

      const metrics = [
        { label: 'MAE',    value: (data.mae  || 0).toFixed(3), tip: 'Mean Absolute Error' },
        { label: 'RMSE',   value: (data.rmse || 0).toFixed(3), tip: 'Root Mean Squared Error' },
        { label: 'R² Score', value: (data.r2 || 0).toFixed(4), tip: 'Coefficient of Determination' },
        { label: 'Trees',  value: data.n_estimators || 200 },
        { label: 'Samples', value: (data.n_samples || 0).toLocaleString() },
        { label: 'Features', value: data.n_features || 8 },
      ];

      container.innerHTML = metrics.map(m => `
        <div class="perf-card" title="${m.tip || ''}">
          <div class="perf-value">${m.value}</div>
          <div class="perf-label">${m.label}</div>
        </div>
      `).join('');

      // Fill dynamic metric rows in model info card (index page About section)
      const maeEl = $('#metricMAE strong');
      const r2El  = $('#metricR2 strong');
      if (maeEl) maeEl.textContent = (data.mae || 0).toFixed(3);
      if (r2El)  r2El.textContent  = (data.r2  || 0).toFixed(4);
    })
    .catch(() => {});
}

/* ── Trend Chart (Feature 2) ─────────────────────────────── */
function loadTrendChart() {
  const canvas  = document.getElementById('trendChart');
  const empty   = document.getElementById('trendChartEmpty');
  const wrapper = document.getElementById('trendChartWrapper');
  if (!canvas) return;

  fetch('/api/trend-data')
    .then(r => r.json())
    .then(data => {
      const history = data.history || [];
      if (history.length < 2) {
        if (empty)   { empty.style.display = ''; }
        if (wrapper) { wrapper.style.display = 'none'; }
        return;
      }

      const labels = history.map(e => {
        const d = new Date(e.ts);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
               ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
      });
      const scores = history.map(e => e.risk_score);
      const colors = history.map(e => e.color || '#6366f1');

      new Chart(canvas, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Risk Score',
            data: scores,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99,102,241,0.1)',
            pointBackgroundColor: colors,
            pointBorderColor: colors,
            pointRadius: 5,
            fill: true,
            tension: 0.35,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => ` Risk Score: ${ctx.parsed.y}`,
              },
            },
          },
          scales: {
            x: {
              ticks: { maxTicksLimit: 8, maxRotation: 30, font: { size: 11 } },
              grid: { display: false },
            },
            y: {
              min: 0, max: 100,
              ticks: { stepSize: 20 },
              grid: { color: 'rgba(0,0,0,0.05)' },
            },
          },
        },
      });
    })
    .catch(() => {
      if (empty)   { empty.style.display = ''; }
      if (wrapper) { wrapper.style.display = 'none'; }
    });
}

/* Data Explorer */
function initDataExplorer() {
  const searchEl = $('#tableSearch');
  if (searchEl) {
    searchEl.addEventListener('input', debounce(() => {
      tableState.search = searchEl.value;
      tableState.page = 1;
      fetchTableData();
    }, 350));
  }
  fetchTableData();
}

function fetchTableData() {
  const { page, perPage, sort, dir, search } = tableState;
  const url = `/api/sample-data?page=${page}&per_page=${perPage}&sort=${sort}&dir=${dir}&search=${encodeURIComponent(search)}`;

  const tableWrapper = $('#dataTableWrapper');
  if (tableWrapper) tableWrapper.style.opacity = '0.5';

  fetch(url)
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      renderTable(data);
      if (tableWrapper) tableWrapper.style.opacity = '1';
    })
    .catch(() => { if (tableWrapper) tableWrapper.style.opacity = '1'; });
}

function renderTable(data) {
  const wrapper = $('#dataTableWrapper');
  if (!wrapper) return;

  const cols = data.columns || [];
  const rows = data.rows    || [];

  const thead = cols.map(c => {
    const active = tableState.sort === c;
    const dirIcon = active ? (tableState.dir === 'asc' ? '↑' : '↓') : '';
    return `<th data-col="${c}" title="Sort by ${c}">${c.replace(/_/g,' ')} ${dirIcon}</th>`;
  }).join('');

  const tbody = rows.map(row =>
    `<tr>${row.map((v, i) => {
      let display = v;
      if (cols[i] === 'overthinking_risk_score') {
        const cls = v < 35 ? 'success' : v < 65 ? 'warning' : 'danger';
        display = `<strong style="color:var(--${cls})">${v}</strong>`;
      }
      return `<td>${display}</td>`;
    }).join('')}</tr>`
  ).join('') || '<tr><td colspan="9" style="text-align:center;padding:24px;color:var(--text-muted)">No data found.</td></tr>';

  // Pagination
  const totalPages = Math.ceil(data.total / data.per_page) || 1;
  const start = (data.page - 1) * data.per_page + 1;
  const end   = Math.min(data.page * data.per_page, data.total);

  wrapper.innerHTML = `
    <div class="data-table-wrapper">
      <table class="data-table" id="dataTable">
        <thead><tr>${thead}</tr></thead>
        <tbody>${tbody}</tbody>
      </table>
    </div>
    <div class="table-pagination">
      <span>Showing ${start}–${end} of ${data.total} rows</span>
      <div class="pagination-buttons">
        <button class="page-btn" id="prevPage" ${data.page <= 1 ? 'disabled' : ''}>← Prev</button>
        <button class="page-btn active">${data.page}</button>
        <button class="page-btn" id="nextPage" ${data.page >= totalPages ? 'disabled' : ''}>Next →</button>
      </div>
    </div>
  `;

  // Sort header clicks
  wrapper.querySelectorAll('.data-table th').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (tableState.sort === col) {
        tableState.dir = tableState.dir === 'asc' ? 'desc' : 'asc';
      } else {
        tableState.sort = col;
        tableState.dir  = 'desc';
      }
      tableState.page = 1;
      fetchTableData();
    });
  });

  // Pagination
  const prev = $('#prevPage', wrapper);
  const next = $('#nextPage', wrapper);
  if (prev) prev.addEventListener('click', () => { tableState.page--; fetchTableData(); });
  if (next) next.addEventListener('click', () => { tableState.page++; fetchTableData(); });
}

function initRegenerate() {
  const btn = $('#regenerateBtn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Regenerating...';
    fetch('/api/regenerate-data', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sync"></i> Generate New Synthetic Data';
        if (data.error) { alert('Error: ' + data.error); return; }
        alert(data.message || 'Data regenerated successfully!');
        fetchTableData();
      })
      .catch(() => {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sync"></i> Generate New Synthetic Data';
        alert('Failed to regenerate data. Please try again.');
      });
  });
}

/* ── Chart.js lazy loader ───────────────────────────────────── */
function loadChartJS(callback) {
  if (typeof Chart !== 'undefined') { callback(); return; }
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
  script.onload = callback;
  document.head.appendChild(script);
}

/* ── Debounce ───────────────────────────────────────────────── */
function debounce(fn, delay) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), delay); };
}

/* ── Boot ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavbar();
  initSliders();
  initForm();
  initDemoReset();
  initHistory();
  loadModelMetrics();
  initSmoothScroll();
  initResultPage();
  initDashboard();
});
