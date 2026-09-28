/**
 * static/js/charts.js
 * ────────────────────
 * Chart.js rendering functions for the Statistics and Evaluation pages.
 *
 * Dependencies: Chart.js (loaded via CDN in base.html)
 *
 * Colour palette — matches CSS design tokens in main.css
 */

const PALETTE = {
  accent:  '#087f8c',
  danger:  '#b42318',
  ok:      '#146c43',
  warn:    '#92400e',
  blue:    '#1d4ed8',
  purple:  '#5b21b6',
  muted:   '#5d6b78',
  // Extended set for multi-series charts
  series: [
    '#087f8c', '#1d4ed8', '#146c43', '#5b21b6',
    '#b42318', '#92400e', '#0891b2', '#7c3aed',
  ],
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/**
 * Safely get a canvas context, destroying any prior Chart instance on it.
 * @param {string} canvasId
 * @returns {CanvasRenderingContext2D | null}
 */
function getCtx(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  // Destroy existing Chart.js instance if present
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
  return canvas.getContext('2d');
}

/**
 * Render a "no data" placeholder on a canvas.
 * @param {string} canvasId
 * @param {string} message
 */
function renderNoData(canvasId, message = 'No data available yet') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#5d6b78';
  ctx.font = '13px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

// ---------------------------------------------------------------------------
// Statistics Page — 10 charts
// ---------------------------------------------------------------------------

/**
 * Top Error Types by Frequency — horizontal bar
 */
function renderErrorTypeBar(data) {
  const ctx = getCtx('chart-error-types');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-error-types'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Occurrences', data: data.values,
        backgroundColor: PALETTE.accent + 'cc', borderRadius: 4 }],
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

/**
 * Fix Success Rate by Error Type — grouped bar
 */
function renderFixSuccessRate(data) {
  const ctx = getCtx('chart-fix-success');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-fix-success'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        { label: 'Success', data: data.success,
          backgroundColor: PALETTE.ok + 'cc', borderRadius: 4 },
        { label: 'Total',   data: data.total,
          backgroundColor: PALETTE.muted + '55', borderRadius: 4 },
      ],
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

/**
 * RCA Confidence Distribution — histogram (bar with uniform bin width)
 */
function renderConfidenceHistogram(data) {
  const ctx = getCtx('chart-confidence-dist');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-confidence-dist'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Count', data: data.values,
        backgroundColor: PALETTE.blue + 'bb', borderRadius: 3 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Confidence bucket' } },
        y: { beginAtZero: true, ticks: { precision: 0 },
             title: { display: true, text: 'RCA count' } },
      },
    },
  });
}

/**
 * Severity Breakdown — doughnut
 */
function renderSeverityDonut(data) {
  const ctx = getCtx('chart-severity');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-severity'); return; }
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: data.labels,
      datasets: [{ data: data.values,
        backgroundColor: [PALETTE.danger, PALETTE.warn, PALETTE.ok, PALETTE.muted] }],
    },
    options: {
      plugins: { legend: { position: 'bottom' } },
      cutout: '55%',
    },
  });
}

/**
 * Fix Source Distribution — stacked bar
 */
function renderFixSourceBar(data) {
  const ctx = getCtx('chart-fix-source');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-fix-source'); return; }
  const sources = Object.keys(data.series);
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: sources.map((src, i) => ({
        label: src,
        data: data.series[src],
        backgroundColor: PALETTE.series[i % PALETTE.series.length] + 'cc',
        borderRadius: 3,
      })),
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
    },
  });
}

/**
 * Error Frequency by Repository — bar
 */
function renderRepoFrequencyBar(data) {
  const ctx = getCtx('chart-repo-freq');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-repo-freq'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Failures', data: data.values,
        backgroundColor: PALETTE.purple + 'cc', borderRadius: 4 }],
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

/**
 * RCA Source Comparison (model vs KB vs heuristic) — grouped bar
 */
function renderRcaSourceBar(data) {
  const ctx = getCtx('chart-rca-source');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-rca-source'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Count', data: data.values,
        backgroundColor: PALETTE.series.slice(0, data.values.length) }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

/**
 * Error Type × Severity matrix — colour-coded table rendered as a custom chart.
 * Renders as a stacked bar (error_type on X, stacks = severities) as a proxy
 * for a heat-map (Chart.js has no native matrix type).
 */
function renderErrorSeverityMatrix(data) {
  const ctx = getCtx('chart-error-severity-matrix');
  if (!ctx || !data?.error_types?.length) { renderNoData('chart-error-severity-matrix'); return; }
  const severities = Object.keys(data.matrix);
  const colours    = { high: PALETTE.danger, medium: PALETTE.warn, low: PALETTE.ok };
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.error_types,
      datasets: severities.map(sev => ({
        label: sev.charAt(0).toUpperCase() + sev.slice(1),
        data: data.matrix[sev],
        backgroundColor: (colours[sev] || PALETTE.muted) + 'cc',
        borderRadius: 3,
      })),
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
    },
  });
}

/**
 * Fix Score Distribution — bar (success/total ratio per error_type)
 */
function renderFixScoreBar(data) {
  const ctx = getCtx('chart-fix-score');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-fix-score'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Success Rate', data: data.values,
        backgroundColor: PALETTE.ok + 'bb', borderRadius: 4 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 1,
        ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
    },
  });
}

/**
 * Knowledge Base Growth Curve — line
 */
function renderKbGrowthLine(data) {
  const ctx = getCtx('chart-kb-growth');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-kb-growth'); return; }
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Cumulative KB entries', data: data.values,
        borderColor: PALETTE.accent, backgroundColor: PALETTE.accent + '22',
        fill: true, tension: 0.3, pointRadius: 3 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

// ---------------------------------------------------------------------------
// Evaluation Page — KPI tiles + 6 charts
// ---------------------------------------------------------------------------

/**
 * Render 4 KPI metric tiles.
 * @param {Object} metrics — {precision, recall, f1, accuracy} each in [0,1]
 */
function renderKpiTiles(metrics) {
  const tiles = [
    { id: 'kpi-precision', label: 'Precision', key: 'precision' },
    { id: 'kpi-recall',    label: 'Recall',    key: 'recall'    },
    { id: 'kpi-f1',        label: 'F1-Score',  key: 'f1'        },
    { id: 'kpi-accuracy',  label: 'Accuracy',  key: 'accuracy'  },
  ];
  for (const t of tiles) {
    const el = document.getElementById(t.id);
    if (!el) continue;
    const val = metrics[t.key];
    const pct = (val != null && val !== 'N/A') ? (val * 100).toFixed(1) + '%' : 'N/A';
    el.querySelector('.kpi-value').textContent = pct;
  }
}

/**
 * Confidence Calibration Curve — line (predicted vs actual)
 */
function renderConfidenceCalibration(data) {
  const ctx = getCtx('chart-calibration');
  if (!ctx || !data?.buckets?.length) { renderNoData('chart-calibration'); return; }
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.buckets,
      datasets: [
        { label: 'Actual success rate', data: data.actual,
          borderColor: PALETTE.accent, pointRadius: 4, tension: 0.2 },
        { label: 'Perfect calibration', data: data.buckets,
          borderColor: PALETTE.muted + '88', borderDash: [6, 4],
          pointRadius: 0, tension: 0 },
      ],
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: {
        x: { title: { display: true, text: 'Predicted confidence' } },
        y: { beginAtZero: true, max: 1,
          title: { display: true, text: 'Actual success rate' },
          ticks: { callback: v => (v * 100).toFixed(0) + '%' } },
      },
    },
  });
}

/**
 * Precision-Recall Curve — line
 */
function renderPrecisionRecallCurve(data) {
  const ctx = getCtx('chart-pr-curve');
  if (!ctx || !data?.recall?.length) { renderNoData('chart-pr-curve'); return; }
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.recall.map(v => (v * 100).toFixed(0) + '%'),
      datasets: [{ label: 'Precision', data: data.precision,
        borderColor: PALETTE.blue, backgroundColor: PALETTE.blue + '22',
        fill: true, tension: 0.3, pointRadius: 3 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Recall' } },
        y: { beginAtZero: true, max: 1,
          title: { display: true, text: 'Precision' },
          ticks: { callback: v => (v * 100).toFixed(0) + '%' } },
      },
    },
  });
}

/**
 * Performance by Error Type — grouped bar (P / R / F1)
 */
function renderPerformanceByErrorType(data) {
  const ctx = getCtx('chart-perf-by-type');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-perf-by-type'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        { label: 'Precision', data: data.precision,
          backgroundColor: PALETTE.blue + 'bb', borderRadius: 3 },
        { label: 'Recall',    data: data.recall,
          backgroundColor: PALETTE.accent + 'bb', borderRadius: 3 },
        { label: 'F1',        data: data.f1,
          backgroundColor: PALETTE.purple + 'bb', borderRadius: 3 },
      ],
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: true, max: 1,
        ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
    },
  });
}

/**
 * KB Hit Rate Trend over time — line
 */
function renderKbHitRateTrend(data) {
  const ctx = getCtx('chart-kb-trend');
  if (!ctx || !data?.dates?.length) { renderNoData('chart-kb-trend'); return; }
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [
        { label: 'KB source %', data: data.kb_pct,
          borderColor: PALETTE.ok, backgroundColor: PALETTE.ok + '22',
          fill: true, tension: 0.3, pointRadius: 3 },
        { label: 'Model source %', data: data.model_pct,
          borderColor: PALETTE.accent, borderDash: [5, 3],
          tension: 0.3, pointRadius: 3, fill: false },
      ],
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: true, max: 100,
        ticks: { callback: v => v + '%' } } },
    },
  });
}

/**
 * Fix Acceptance Rate by source — bar
 */
function renderFixAcceptanceRate(data) {
  const ctx = getCtx('chart-fix-acceptance');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-fix-acceptance'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Acceptance Rate', data: data.values,
        backgroundColor: PALETTE.series.slice(0, data.values.length) }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 1,
        ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
    },
  });
}

// ---------------------------------------------------------------------------
// Stats Page — Archive-powered chart functions (replace sparse fix charts)
// ---------------------------------------------------------------------------

/**
 * Fix Applicability by Error Type — grouped bar (applicable / blocked / manual review)
 * Data shape: {labels, applicable, blocked, manual_review}
 */
function renderFixApplicabilityByType(data) {
  const ctx = getCtx('chart-fix-applicability');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-fix-applicability'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        { label: 'Applicable',     data: data.applicable,    backgroundColor: PALETTE.ok      + 'cc', borderRadius: 3 },
        { label: 'Blocked',        data: data.blocked,       backgroundColor: PALETTE.danger  + 'cc', borderRadius: 3 },
        { label: 'Manual Review',  data: data.manual_review, backgroundColor: PALETTE.warn    + 'cc', borderRadius: 3 },
      ],
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: { x: { stacked: false }, y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

/**
 * RCA Confidence Score Distribution (archive) — bar histogram
 * Data shape: {labels, values}  — 10 bins "0–10%" … "90–100%"
 */
function renderRcaConfidenceHistogram(data) {
  const ctx = getCtx('chart-rca-conf-hist');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-rca-conf-hist'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'RCA runs', data: data.values,
        backgroundColor: PALETTE.blue + 'bb', borderRadius: 3 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Confidence score bucket' } },
        y: { beginAtZero: true, ticks: { precision: 0 }, title: { display: true, text: 'Run count' } },
      },
    },
  });
}

/**
 * Fix Source Distribution (archive) — bar
 * Data shape: {labels, values}  — llm / historical / template / other
 */
function renderFixSourceArchive(data) {
  const ctx = getCtx('chart-fix-source-archive');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-fix-source-archive'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Fix candidates', data: data.values,
        backgroundColor: PALETTE.series.slice(0, data.values.length) }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

/**
 * Fix Acceptance Rate (stats page reuse of evaluation data) — bar
 * Data shape: {labels, values}  — one bar per fix_type, value in [0,1]
 */
function renderFixAcceptanceStatsBar(data) {
  const ctx = getCtx('chart-fix-acceptance-stats');
  if (!ctx || !data?.labels?.length) { renderNoData('chart-fix-acceptance-stats', 'Grows with live feedback'); return; }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{ label: 'Acceptance Rate', data: data.values,
        backgroundColor: PALETTE.series.slice(0, data.values.length) }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 1,
        ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
    },
  });
}


// ---------------------------------------------------------------------------
// Evaluation Page — Archive-powered chart functions
// ---------------------------------------------------------------------------

/**
 * RCA + Fix Success by Error Type — grouped bar
 * Data shape: array of {error_type, high_conf_rate, composite_rate}
 */
function renderRcaFixByType(items) {
  const ctx = getCtx('chart-rca-fix-by-type');
  if (!ctx || !items?.length) { renderNoData('chart-rca-fix-by-type'); return; }
  const labels         = items.map(d => d.error_type);
  const highConfRates  = items.map(d => d.high_conf_rate);
  const compositeRates = items.map(d => d.composite_rate);
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'High-Confidence Rate',  data: highConfRates,  backgroundColor: PALETTE.blue   + 'bb', borderRadius: 3 },
        { label: 'Composite Success Rate',data: compositeRates, backgroundColor: PALETTE.ok     + 'bb', borderRadius: 3 },
      ],
    },
    options: {
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: true, max: 1,
        ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
    },
  });
}

/**
 * Confidence vs Fix Applicability Rate — scatter plot (26 data points)
 * Data shape: {points: [{x: confidence_score, y: applicability_rate, label: error_type}]}
 */
function renderConfVsApplicabilityScatter(data) {
  const ctx = getCtx('chart-conf-vs-applicability');
  if (!ctx || !data?.points?.length) { renderNoData('chart-conf-vs-applicability'); return; }
  const points = data.points.map(p => ({ x: p.x, y: p.y }));
  const labels = data.points.map(p => p.label);
  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'RCA run',
        data: points,
        backgroundColor: PALETTE.purple + 'aa',
        pointRadius: 6,
        pointHoverRadius: 9,
      }],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const i = ctx.dataIndex;
              return `${labels[i]}  conf: ${(ctx.parsed.x * 100).toFixed(1)}%  app: ${(ctx.parsed.y * 100).toFixed(1)}%`;
            },
          },
        },
      },
      scales: {
        x: { min: 0, max: 1,
          title: { display: true, text: 'RCA Confidence Score' },
          ticks: { callback: v => (v * 100).toFixed(0) + '%' } },
        y: { min: 0, max: 1,
          title: { display: true, text: 'Fix Applicability Rate' },
          ticks: { callback: v => (v * 100).toFixed(0) + '%' } },
      },
    },
  });
}

/**
 * Confidence Factors bar — average of 6 RCA factors across archive
 * Data shape: {factor_names: [...], avg_values: [...]}
 */
function renderConfidenceFactorsBar(data) {
  const ctx = getCtx('chart-conf-factors');
  if (!ctx || !data?.factor_names?.length) { renderNoData('chart-conf-factors'); return; }
  const labels = data.factor_names.map(f => f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()));
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Avg score', data: data.avg_values,
        backgroundColor: PALETTE.series.slice(0, data.factor_names.length),
        borderRadius: 4 }],
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, max: 1,
        ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
    },
  });
}

