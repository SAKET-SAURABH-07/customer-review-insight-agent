/**
 * app.js - Vanilla JavaScript driver for Customer Review Insight Agent Dashboard.
 * 
 * Features:
 * - Dual-layer data loading: fetch('data/insights.json') with fallback to window.INSIGHTS_DATA.
 * - Zero external charting libraries: rendered natively via HTML5 2D Canvas context.
 * - Reactive filtering for topics & sentiments without page reload.
 */

let globalData = null;
let currentTopicFilter = "all";
let currentSentimentFilter = "all";

document.addEventListener("DOMContentLoaded", () => {
  initDashboard();
});

async function initDashboard() {
  try {
    // 1. Attempt relative fetch of insights.json
    const res = await fetch("data/insights.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    globalData = await res.json();
  } catch (err) {
    console.warn("Relative fetch failed, attempting fallback window.INSIGHTS_DATA:", err);
    if (window.INSIGHTS_DATA) {
      globalData = window.INSIGHTS_DATA;
    } else {
      showErrorState("Could not load insights data. Please check data/insights.json.");
      return;
    }
  }

  renderHeaderAndKPIs(globalData);
  renderAlerts(globalData.alerts || []);
  renderTrendChart(globalData.weekly_trends || []);
  renderTopicChart(globalData.topic_summary || []);
  setupTableControls();
  renderReviewsTable();

  // Re-render charts on window resize for crisp responsive canvas display
  window.addEventListener("resize", () => {
    if (globalData) {
      renderTrendChart(globalData.weekly_trends || []);
      renderTopicChart(globalData.topic_summary || []);
    }
  });
}

function renderHeaderAndKPIs(data) {
  const kpis = data.kpis || {};
  document.getElementById("kpiTotal").textContent = Number(kpis.total_reviews || 0).toLocaleString();
  document.getElementById("kpiPositive").textContent = `${kpis.positive_pct || 0}%`;
  document.getElementById("kpiNegative").textContent = `${kpis.negative_pct || 0}%`;
  
  const alertsCount = Number(kpis.active_alerts || 0);
  const alertsEl = document.getElementById("kpiAlerts");
  alertsEl.textContent = alertsCount;
  if (alertsCount > 0) {
    alertsEl.style.color = "#f87171";
  }

  if (data.trend_title) {
    const titleEl = document.getElementById("trendChartTitle");
    if (titleEl) titleEl.textContent = data.trend_title;
  }
  if (data.trend_sub) {
    const subEl = document.getElementById("trendChartSub");
    if (subEl) subEl.textContent = data.trend_sub;
  }

  document.getElementById("executiveSummaryText").textContent =
    data.executive_summary || "Automated customer review insights loaded successfully.";

  if (data.generated_at) {
    const d = new Date(data.generated_at);
    document.getElementById("lastUpdatedBadge").textContent = `Updated: ${d.toLocaleDateString()}`;
  }
  // Show data coverage period so users know this is historical, not live
  const periodEl = document.getElementById("dataCoveragePeriod");
  if (periodEl && data.data_period) {
    periodEl.textContent = data.data_period;
  }
}

function renderAlerts(alerts) {
  const container = document.getElementById("alertsContainer");
  container.innerHTML = "";

  if (!alerts || alerts.length === 0) {
    const el = document.createElement("div");
    el.className = "alert-nominal";
    el.innerHTML = `<span>✓</span> <strong>All Systems Nominal:</strong> No operational topic spikes detected (all complaint categories within expected variance).`;
    container.appendChild(el);
    return;
  }

  alerts.forEach(alert => {
    const el = document.createElement("div");
    el.className = "alert-banner";
    el.innerHTML = `<span>⚠️</span> <div><strong>${alert.message}</strong> (z-score: ${alert.z_score})</div>`;
    container.appendChild(el);
  });
}

/* -------------------------------------------------------------
   Native HTML5 Canvas Charts (No D3 / Chart.js dependencies)
------------------------------------------------------------- */

function renderTrendChart(trends) {
  const canvas = document.getElementById("trendChart");
  if (!canvas || !trends.length) return;
  const ctx = canvas.getContext("2d");

  // Handle HiDPI displays
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = (rect.width || 560) * dpr;
  canvas.height = 260 * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width || 560;
  const h = 260;
  ctx.clearRect(0, 0, w, h);

  const padLeft = 45;
  const padRight = 20;
  const padTop = 30;
  const padBottom = 40;
  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  // Draw background grid lines (0% to 100%)
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "right";

  for (let pct = 0; pct <= 100; pct += 25) {
    const y = padTop + chartH - (pct / 100) * chartH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
    ctx.fillText(`${pct}%`, padLeft - 8, y + 4);
  }

  const n = trends.length;
  const stepX = n > 1 ? chartW / (n - 1) : chartW / 2;

  // Helper to draw a trend line
  function drawLine(key, strokeColor, fillColor) {
    ctx.strokeStyle = strokeColor;
    ctx.fillStyle = strokeColor;
    ctx.lineWidth = 2.5;
    ctx.beginPath();

    const points = [];
    trends.forEach((item, i) => {
      const val = item[key] || 0;
      const x = padLeft + i * stepX;
      const y = padTop + chartH - (val / 100) * chartH;
      points.push({ x, y, val });
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw data points with small glow
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = strokeColor;
      ctx.fill();
    });
  }

  // Draw Positive Trend Line (Green)
  drawLine("pos_pct", "#34d399");
  // Draw Negative Trend Line (Red)
  drawLine("neg_pct", "#f87171");

  // Draw X axis labels (weeks or batches)
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "center";
  trends.forEach((item, i) => {
    const x = padLeft + i * stepX;
    let lbl = item.period || "";
    if (lbl.includes("/")) {
      lbl = lbl.split("/")[0].slice(5);
    } else {
      lbl = lbl.replace(/^\d{4}-/, "");
    }
    ctx.fillText(lbl, x, h - 14);
  });
}

function renderTopicChart(topics) {
  const canvas = document.getElementById("topicChart");
  if (!canvas || !topics.length) return;
  const ctx = canvas.getContext("2d");

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = (rect.width || 560) * dpr;
  canvas.height = 260 * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width || 560;
  const h = 260;
  ctx.clearRect(0, 0, w, h);

  // Exclude 'other' from pure complaint view if desired, or show top topics
  const displayTopics = topics.filter(t => t.topic !== "other").slice(0, 6);
  const maxNeg = Math.max(...displayTopics.map(t => t.negative), 10);

  const padLeft = 125;
  const padRight = 50;
  const padTop = 20;
  const padBottom = 20;
  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;
  const barHeight = 22;
  const rowHeight = chartH / displayTopics.length;

  displayTopics.forEach((t, i) => {
    const y = padTop + i * rowHeight + (rowHeight - barHeight) / 2;
    const barW = (t.negative / maxNeg) * chartW;

    // Draw topic label
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(t.topic, padLeft - 10, y + 16);

    // Background track
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(padLeft, y, chartW, barHeight);

    // Negative complaints bar
    ctx.fillStyle = "#f87171";
    ctx.beginPath();
    ctx.roundRect(padLeft, y, Math.max(barW, 4), barHeight, 4);
    ctx.fill();

    // Value label
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "left";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${t.negative} complaints`, padLeft + Math.max(barW, 4) + 8, y + 15);
  });
}

/* -------------------------------------------------------------
   Interactive Audited Customer Reviews Table
------------------------------------------------------------- */

function setupTableControls() {
  const topicSelect = document.getElementById("topicFilter");
  topicSelect.addEventListener("change", (e) => {
    currentTopicFilter = e.target.value;
    renderReviewsTable();
  });

  const buttons = document.querySelectorAll(".sentiment-buttons .filter-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentSentimentFilter = btn.getAttribute("data-sentiment");
      renderReviewsTable();
    });
  });
}

function renderReviewsTable() {
  const tbody = document.getElementById("reviewsTableBody");
  if (!tbody || !globalData) return;

  const reviews = globalData.sample_reviews || [];
  const filtered = reviews.filter(r => {
    const matchTopic = (currentTopicFilter === "all") || (r.topic === currentTopicFilter);
    const matchSent = (currentSentimentFilter === "all") || (r.label === currentSentimentFilter);
    return matchTopic && matchSent;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-center">No customer reviews match the selected filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r => {
    const sentimentClass = `badge-${r.label}`;
    return `
      <tr>
        <td><span class="badge badge-topic">${escapeHtml(r.topic)}</span></td>
        <td><span class="badge ${sentimentClass}">${escapeHtml(r.label)}</span></td>
        <td>${escapeHtml(truncate(r.text, 100))}</td>
        <td style="color:#64748b; font-size:12px;">${escapeHtml(r.date || "N/A")}</td>
      </tr>
    `;
  }).join("");
}

function truncate(str, max) {
  if (!str) return "";
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showErrorState(msg) {
  const tbody = document.getElementById("reviewsTableBody");
  if (tbody) tbody.innerHTML = `<tr><td colspan="4" style="color:#f87171;text-align:center;">${msg}</td></tr>`;
}
