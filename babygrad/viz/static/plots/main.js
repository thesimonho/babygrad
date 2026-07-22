/*
 * Plots tab entry point. Lazy: nothing loads until the tab is first opened, then
 * it imports Observable Plot, mounts the sidebar, and draws from the live history,
 * re-drawing (throttled) as epochs stream in.
 *
 * The sidebar controls which registry items are visible; the main area renders
 * only the visible ones, grouped into foldable sections (loss & metrics, then one
 * per scope). Folding a section from either surface keeps both in sync because
 * they read the same shared `state` object.
 */

import { loadPlot } from "./plot-lib.js";
import { buildItems, groupOrder, isVisible } from "./registry.js";
import { createSidebar } from "./controls.js";
import { groupSection, chartCard, checkboxRow, sliderRow, optionsBar, note } from "./dom.js";
import { toggleInSet } from "./util.js";
import { computeDiagnostic } from "./diagnostics.js";
import { scalarChart } from "./charts/scalar.js";
import { ridgeChart } from "./charts/ridge.js";
import { linesChart } from "./charts/lines.js";
import { attachCrosshair, resetCrosshairs, applyCrosshairs } from "./charts/crosshair.js";

const RERENDER_MS = 400;

// Visibility + chart-option state, mutated in place by the sidebar and options
// bars, read by the renderer: `hidden` = items the user unchecked (unknown/new
// tags default to visible), `collapsed` = folded groups, `query` = the
// lowercased search filter, `logScale` = the loss chart's y-axis mode,
// `lossSmoothing`/`diagnosticsSmoothing` = each section's EMA smoothing weight.
const state = {
  hidden: new Set(),
  collapsed: new Set(),
  query: "",
  logScale: false,
  lossSmoothing: 0,
  diagnosticsSmoothing: 0,
};

// The smoothing sliders share this range: 0 (raw) up to a heavy 0.95, in coarse
// steps since the debiased EMA changes little between neighbouring fine values.
const SMOOTHING = { min: 0, max: 0.95, step: 0.05 };

let Plot = null;
let sidebar = null;
let mainEl = null;
let hasInit = false;
let renderPending = false;
let lastRenderAt = 0;
let isSliderDragging = false;

const plotsTab = document.querySelector('.tab[data-panel="plots"]');
if (plotsTab) plotsTab.addEventListener("click", init);

// Render on load if the Plots tab is already active (a ?tab=plots deep-link);
// init is idempotent, so a later tab click is harmless.
window.addEventListener("load", () => {
  const panel = document.querySelector('.panel[data-panel="plots"]');
  if (panel && panel.classList.contains("active")) init();
});

/** Load Observable Plot on first open, mount the sidebar, draw, and subscribe. */
async function init() {
  if (hasInit) return;
  hasInit = true;
  const sidebarEl = document.getElementById("plots-sidebar");
  mainEl = document.getElementById("plots-main");
  try {
    Plot = await loadPlot();
  } catch (error) {
    hasInit = false; // let a later click retry
    mainEl.textContent = `failed to load plots: ${error}`;
    return;
  }
  sidebar = createSidebar(state, renderMain);
  sidebarEl.append(sidebar.el);
  renderAll();
  if (window.onLiveEpoch) window.onLiveEpoch(() => scheduleRender());
}

/** Coalesce the burst of live epochs into at most one redraw per RERENDER_MS.
 * A streaming redraw during a slider drag is skipped (see beginSliderDrag). */
function scheduleRender() {
  if (isSliderDragging) return;
  const now = performance.now();
  if (now - lastRenderAt >= RERENDER_MS) {
    lastRenderAt = now;
    renderAll();
    return;
  }
  if (renderPending) return;
  renderPending = true;
  setTimeout(() => {
    renderPending = false;
    if (isSliderDragging) return;
    lastRenderAt = performance.now();
    renderAll();
  }, RERENDER_MS);
}

/**
 * A slider drag spans many epochs, and each streaming redraw rebuilds the whole
 * options bar — which would tear the slider out from under the cursor. So
 * streaming redraws are paused for the drag's duration; the live history keeps
 * accumulating, and the redraw on the slider's own commit (or the next epoch)
 * catches the charts up. Bracketed by the pointer press/release; a keyboard step
 * commits through the slider's change event without a drag.
 */
function beginSliderDrag() {
  isSliderDragging = true;
}

function endSliderDrag() {
  isSliderDragging = false;
}

/** Refresh the sidebar (new tags may have appeared) and the chart area, sharing
 * one item list so the live history is only scanned once per render. */
function renderAll() {
  const items = buildItems(window.liveHistory ?? {});
  sidebar.refresh(items);
  renderMain(items);
}

/** Render the visible charts into #plots-main, grouped into foldable sections.
 * Called with a prebuilt item list from renderAll, or on its own from a sidebar
 * change — in which case the items are unchanged, so rebuild them here. */
function renderMain(items = buildItems(window.liveHistory ?? {})) {
  const history = window.liveHistory ?? {};
  // Wiping the content resets the scroll container to the top; capture and
  // restore the offset so a redraw (a slider change, a streaming tick) doesn't
  // yank the viewport up from wherever the user had scrolled.
  const scrollTop = mainEl.scrollTop;
  mainEl.innerHTML = "";
  resetCrosshairs(); // charts re-register as they rebuild below
  if (items.length === 0) {
    mainEl.append(note("waiting for data…"));
    return;
  }
  const width = measureWidth();
  for (const [group, groupItems] of groupOrder(items)) {
    const collapsed = state.collapsed.has(group);
    // A collapsed group always shows its (foldable) header; an expanded one is
    // skipped only when it has nothing visible (all unchecked or filtered out).
    if (!collapsed && !groupItems.some((item) => isVisible(item, state))) continue;
    const body = () => groupBody(groupItems, history, width);
    mainEl.append(groupSection(group, body, { collapsed, onToggle: () => toggleCollapsed(group) }));
  }
  applyCrosshairs(); // charts are mounted now; redraw the shared line at the hovered epoch
  mainEl.scrollTop = scrollTop;
}

/**
 * The charts for one expanded group, dispatched on the group's item kind (a group
 * is kind-homogeneous). Scalar items share one combined loss & metrics chart;
 * ridge items each get their own chart in a grid. A later kind (diagnostics) adds
 * a branch here rather than another group-name special case.
 */
function groupBody(groupItems, history, width) {
  const kind = groupItems[0].kind;
  const visible = groupItems.filter((item) => isVisible(item, state));
  if (kind === "scalar") return scalarSection(pick(history, visible.map((item) => item.id)), width);
  if (kind === "diagnostic") return diagnosticsSection(visible, history, width);
  return ridgeGrid(visible, history);
}

/** The diagnostics grid preceded by its options bar (the smoothing slider). */
function diagnosticsSection(visible, history, width) {
  const wrap = document.createElement("div");
  const options = optionsBar(
    smoothingRow(state.diagnosticsSmoothing, (weight) => {
      state.diagnosticsSmoothing = weight;
      renderMain();
    }),
  );
  wrap.append(options, diagnosticGrid(visible, history, width));
  return wrap;
}

// Diagnostics lay out two-up when the area is wide enough for two readable
// charts, else one full-width column. The column count is decided here (not in a
// CSS media query) so each chart is authored at its cell width — Plot bakes the
// width into the svg, so a chart drawn at the wrong width would render blurry.
const COLUMN_GAP = 20;
const MIN_DIAGNOSTIC_WIDTH = 360;

/** A 2×2 grid of diagnostic line charts, collapsing to one column when narrow. */
function diagnosticGrid(visible, history, width) {
  const columns = width >= MIN_DIAGNOSTIC_WIDTH * 2 + COLUMN_GAP ? 2 : 1;
  const cellWidth = columns === 1 ? width : Math.floor((width - COLUMN_GAP) / 2);
  const grid = document.createElement("div");
  grid.className = "diagnostics-grid";
  grid.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
  for (const item of visible) {
    const { series, options } = computeDiagnostic(item.id, history);
    const chartOptions = { ...options, smoothing: state.diagnosticsSmoothing };
    const { node, tipContent } = linesChart(series, Plot, cellWidth, chartOptions);
    grid.append(chartCard(item.label, attachCrosshair(node, tipContent)));
  }
  return grid;
}

/** A responsive grid of ridge charts, one per visible parameter series. */
function ridgeGrid(visible, history) {
  const grid = document.createElement("div");
  grid.className = "plot-grid";
  for (const item of visible) {
    grid.append(chartCard(item.label, ridgeChart(history[item.id], item.id, Plot)));
  }
  return grid;
}

/** The loss & metrics chart preceded by its options bar (the log-scale toggle). */
function scalarSection(subset, width) {
  const wrap = document.createElement("div");
  const options = { logScale: state.logScale, smoothing: state.lossSmoothing };
  const { node, tipContent } = scalarChart(subset, Plot, width, options);
  const chart = attachCrosshair(node, tipContent);
  wrap.append(lossOptions(), chart);
  return wrap;
}

/** The inline options bar for the loss chart: a live log-scale toggle and a
 * smoothing slider. Rebuilt each render from `state`, so both reflect the current
 * mode. */
function lossOptions() {
  return optionsBar(
    checkboxRow("log scale", {
      checked: state.logScale,
      className: "plots-option",
      onChange: (checked) => {
        state.logScale = checked;
        renderMain();
      },
    }),
    smoothingRow(state.lossSmoothing, (weight) => {
      state.lossSmoothing = weight;
      renderMain();
    }),
  );
}

/** The smoothing slider row: 0 (raw) to a heavy 0.95, showing "off" at zero.
 * `commit` stores the new weight and redraws. The drag hooks pause streaming
 * redraws while the slider is held (see beginSliderDrag). */
function smoothingRow(weight, commit) {
  return sliderRow("smoothing", {
    value: weight,
    min: SMOOTHING.min,
    max: SMOOTHING.max,
    step: SMOOTHING.step,
    onDragStart: beginSliderDrag,
    onDragEnd: endSliderDrag,
    onChange: commit,
    format: (value) => (value === 0 ? "off" : value.toFixed(2)),
    className: "plots-option",
  });
}

/** Fold or unfold a group, re-rendering both surfaces so their carets agree. */
function toggleCollapsed(group) {
  toggleInSet(state.collapsed, group);
  renderAll();
}

/** The history restricted to the given tags, preserving each tag's series. */
function pick(history, tags) {
  return Object.fromEntries(tags.filter((tag) => tag in history).map((tag) => [tag, history[tag]]));
}

/** The main area's content width, so the scalar chart spans it (re-measured each
 * draw, so a live redraw re-fits after a window resize). */
function measureWidth() {
  const style = getComputedStyle(mainEl);
  const horizontalPadding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  return Math.max(320, mainEl.clientWidth - horizontalPadding);
}
