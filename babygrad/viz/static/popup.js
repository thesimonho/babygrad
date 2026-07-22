/*
 * Node popup: click a graph node to see what it holds right now.
 *
 * Shows the node's role, shape, and scope, plus a current value/grad summary
 * (from /node_stats.json). For nodes the recorder tracks over time (params and
 * layer outputs, flagged by series_tag), it also draws a sparkline of the
 * series' mean across steps, lazily importing Observable Plot the first time.
 *
 * Exposed as window.wireNodePopup so dashboard.js can bind it to the cy instance
 * without this file needing to know how the graph is built.
 */

"use strict";

const PLOT_MODULE = "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

let nodeStats = {};
let plotLib = null;
let activeCy = null;
let activeNode = null;

const popup = document.getElementById("node-popup");

window.wireNodePopup = function wireNodePopup(cy) {
  fetchNodeStats();
  cy.on("tap", "node[!isScope]", (event) => showPopup(cy, event.target));
  cy.on("tap", (event) => {
    if (event.target === cy) hidePopup();
  });
  // redraw an open popup as epochs stream in, so its sparkline grows live
  if (window.onLiveEpoch) {
    window.onLiveEpoch(() => {
      if (popup.style.display === "block" && activeNode) {
        showPopup(activeCy, activeNode);
      }
    });
  }
};

/** Load the per-node value snapshot once; the sparkline reads the live history. */
async function fetchNodeStats() {
  try {
    nodeStats = await fetch("/node_stats.json").then((response) => response.json());
  } catch {
    // popups still show structural fields even if the value snapshot fails
  }
}

/** Fill and position the popup for the tapped node. */
function showPopup(cy, node) {
  activeCy = cy;
  activeNode = node;
  const data = node.data();
  popup.innerHTML = "";
  popup.append(header(data), meta(data));

  const stats = nodeStats[data.id];
  if (stats) popup.append(statLine("value", stats.value), statLine("grad", stats.grad));

  const hist = window.liveHistory ?? {};
  if (data.seriesTag && hist[data.seriesTag]) {
    popup.append(sparkline(data.seriesTag));
  }

  anchorPopup(cy);
  popup.style.display = "block";
}

function hidePopup() {
  popup.style.display = "none";
  activeNode = null;
}

/** Role name + kind chip. */
function header(data) {
  const row = document.createElement("div");
  row.className = "np-header";
  const role = document.createElement("strong");
  role.textContent = data.label.split("\n")[0];
  const kind = document.createElement("span");
  kind.className = "np-kind";
  kind.textContent = data.kind;
  row.append(role, kind);
  return row;
}

/** Shape and scope lines. */
function meta(data) {
  const wrap = document.createElement("div");
  wrap.className = "np-meta";
  if (data.rawShape) wrap.append(line("shape", data.rawShape.join("×")));
  if (data.scopePath) wrap.append(scopeLine(data.scopePath));
  return wrap;
}

/**
 * The scope line, with a break opportunity after each "/" so the path wraps at
 * its logical segments rather than mid-name. Built from text nodes and <wbr>
 * elements (not an HTML string) so a segment name is never treated as markup.
 */
function scopeLine(scopePath) {
  const row = document.createElement("div");
  row.className = "np-line";
  const label = document.createElement("span");
  label.className = "np-label";
  label.textContent = "scope";
  const value = document.createElement("span");
  const segments = scopePath.split("/");
  segments.forEach((segment, index) => {
    const isLast = index === segments.length - 1;
    value.append(isLast ? segment : `${segment}/`);
    if (!isLast) value.append(document.createElement("wbr"));
  });
  row.append(label, value);
  return row;
}

/** A "value"/"grad" summary line, or a muted note when the node holds none. */
function statLine(label, summary) {
  if (!summary) return line(label, "—");
  const text = `mean ${fmt(summary.mean)} · min ${fmt(summary.min)} · max ${fmt(summary.max)}`;
  return line(label, text);
}

function line(label, value) {
  const row = document.createElement("div");
  row.className = "np-line";
  row.innerHTML = `<span class="np-label">${label}</span><span>${value}</span>`;
  return row;
}

/**
 * A sparkline of the series' per-step mean. Renders a placeholder immediately and
 * fills it once Observable Plot has loaded (imported lazily, then cached).
 */
function sparkline(tag) {
  const slot = document.createElement("div");
  slot.className = "np-spark";
  const rows = perStepMean(tag, window.liveHistory[tag]);
  loadPlot().then((Plot) => {
    slot.append(
      Plot.plot({
        width: 240,
        height: 56,
        margin: 4,
        axis: null,
        x: { label: null },
        y: { label: null },
        marks: [Plot.line(rows, { x: "step", y: "mean", stroke: "#457b9d" })],
      }),
    );
  });
  return slot;
}

// Per-tag cache of a series' per-step means. showPopup re-renders on every
// streamed epoch, so cache the means and append only the new steps instead of
// remapping the whole (growing) series each time. Cleared on the page reload a
// new run triggers, so it never spans runs.
const meanCache = new Map();

/** One mean per step, sorted by step, reusing cached means for steps already
 * seen and computing only the ones added since the last call. */
function perStepMean(tag, series) {
  const steps = Object.keys(series)
    .map(Number)
    .sort((a, b) => a - b);
  const cached = meanCache.get(tag);
  if (cached && cached.stepCount === steps.length) return cached.rows;

  const from = cached && cached.stepCount < steps.length ? cached.stepCount : 0;
  const kept = from > 0 ? cached.rows : [];
  const rows = kept.concat(
    steps.slice(from).map((step) => ({ step, mean: meanOf(series[String(step)]) })),
  );
  meanCache.set(tag, { stepCount: steps.length, rows });
  return rows;
}

/** A distribution series value collapsed to its mean; a scalar passes through. */
function meanOf(values) {
  return Array.isArray(values)
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : values;
}

/** Import Observable Plot once, caching the module across popups. */
async function loadPlot() {
  if (!plotLib) plotLib = await import(PLOT_MODULE);
  return plotLib;
}

/** Pin the popup to the top-right corner of the graph panel, out of the way of
 * the nodes — steadier than chasing the clicked node around the canvas. */
function anchorPopup(cy) {
  const rect = cy.container().getBoundingClientRect();
  popup.style.left = "auto";
  popup.style.right = `${window.innerWidth - rect.right + 16}px`;
  popup.style.top = `${rect.top + 16}px`;
}

/** Trim a number to a readable width. */
function fmt(value) {
  return Number.parseFloat(value.toPrecision(3)).toString();
}
