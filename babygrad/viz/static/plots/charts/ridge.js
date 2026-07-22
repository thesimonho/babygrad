/*
 * The ridgeline chart for one parameter series: one peak-normalised histogram
 * row per step, earliest at the bottom, so a weight/grad distribution's drift
 * over training reads top-to-bottom. Mirrors the matplotlib PlotVisualizer.
 *
 * Binning re-reads every value of a (potentially transformer-sized) parameter,
 * so a per-tag cache bins only the steps added since the last draw and appends
 * them, reusing the shared bin edges while the new values still fall inside them.
 */

import { formatTick } from "../format.js";
import { note } from "../dom.js";

// Ridge binning, mirroring PlotVisualizer's constants closely enough to read the
// same way. Grad series are heavy-tailed, so their bins span a clipped quantile
// range (matching the clip_quantiles the notebook passes plot_ridge for grads).
const RIDGE_BINS = 40;
const RIDGE_ROW_HEIGHT = 2;
const GRAD_CLIP = [0.02, 0.98];

/**
 * A ridgeline of one parameter series: one peak-normalised histogram row per
 * step, earliest at the bottom, so the distribution's drift over training reads
 * top-to-bottom.
 */
export function ridgeChart(series, tag, Plot) {
  const steps = Object.keys(series)
    .map(Number)
    .sort((a, b) => a - b);
  if (steps.length === 0) return note("no data");
  const rowsData = steps.map((step) => series[String(step)]);
  const clip = tag.endsWith("/grad") ? GRAD_CLIP : null;

  const { rows } = ridgeRows(tag, rowsData, steps, clip);

  // Draw latest steps first so the earliest (bottom) rows land in front, the way
  // the overlapping matplotlib ridge stacks. Sort a copy so the cache stays in
  // step order for the next incremental append.
  const drawRows = rows.slice().sort((a, b) => b.step - a.step);

  const stride = Math.max(1, Math.floor(steps.length / 8));
  const tickOffsets = steps.map((_, i) => i).filter((i) => i % stride === 0);

  return Plot.plot({
    width: 380,
    height: Math.min(300, 60 + steps.length * 3.4),
    marginLeft: 46,
    marginBottom: 34,
    x: { label: "value", ticks: 4, tickFormat: formatTick },
    y: {
      label: "epoch",
      domain: [0, steps.length - 1 + RIDGE_ROW_HEIGHT],
      ticks: tickOffsets,
      tickFormat: (offset) => String(steps[offset]),
    },
    marks: [
      Plot.areaY(drawRows, {
        x: "x",
        y1: "base",
        y2: "top",
        z: "step",
        curve: "basis",
        fill: "#457b9d",
        fillOpacity: 0.55,
        stroke: "#457b9d",
        strokeWidth: 0.7,
      }),
      // A vertical crosshair at the hovered value, with a tip naming the row's
      // epoch and the value under the cursor, so a distribution's drift is legible.
      Plot.ruleX(drawRows, Plot.pointerX({ x: "x", stroke: "#9aa0ac", strokeWidth: 1 })),
      Plot.tip(
        drawRows,
        Plot.pointer({
          x: "x",
          y: "base",
          channels: {
            epoch: { value: "step", label: "epoch" },
            value: { value: "x", label: "value" },
          },
          format: { x: false, y: false, epoch: true, value: formatTick },
        }),
      ),
    ],
  });
}

// Per-tag cache of a ridge's binned rows. Binning re-reads every value of a
// (transformer-sized) parameter, so as epochs stream in we bin only the steps
// added since the last draw and append them — reusing the cached bin edges as
// long as the new values still fall inside them. If the range moved, the shared
// edges are stale, so we rebin the whole series from scratch. Cleared on the
// page reload that a new run triggers, so it never spans runs.
const ridgeCache = new Map();

/** The binned rows for a series, appending only new steps when the cached edges
 * still fit; otherwise a full rebin. Returns the cache entry {edges,rows,...}. */
function ridgeRows(tag, rowsData, steps, clip) {
  const cached = ridgeCache.get(tag);
  if (cached && cached.stepCount === steps.length) return cached;

  const hasGrown = cached && cached.stepCount < steps.length;
  if (hasGrown && valuesWithin(rowsData.slice(cached.stepCount), cached.edges)) {
    const appended = binnedRows(rowsData, cached.edges, cached.centres, steps, cached.stepCount);
    const entry = {
      edges: cached.edges,
      centres: cached.centres,
      rows: cached.rows.concat(appended),
      stepCount: steps.length,
    };
    ridgeCache.set(tag, entry);
    return entry;
  }

  const edges = sharedBinEdges(rowsData, RIDGE_BINS, clip);
  const centres = edges.slice(0, -1).map((edge, i) => (edge + edges[i + 1]) / 2);
  const entry = {
    edges,
    centres,
    rows: binnedRows(rowsData, edges, centres, steps, 0),
    stepCount: steps.length,
  };
  ridgeCache.set(tag, entry);
  return entry;
}

/** Peak-normalised histogram rows for the steps at indices [from, end), one area
 * band per step, keyed to the shared bin edges/centres. */
function binnedRows(rowsData, edges, centres, steps, from) {
  const rows = [];
  for (let rowIndex = from; rowIndex < rowsData.length; rowIndex++) {
    const counts = binCounts(rowsData[rowIndex], edges);
    const peak = Math.max(...counts) || 1;
    counts.forEach((count, i) => {
      rows.push({
        step: steps[rowIndex],
        x: centres[i],
        base: rowIndex,
        top: rowIndex + (count / peak) * RIDGE_ROW_HEIGHT,
      });
    });
  }
  return rows;
}

/** True when every value of the given rows falls inside the bin range, so
 * appending to the existing bins stays faithful (else the range has shifted). */
function valuesWithin(rowsData, edges) {
  const lo = edges[0];
  const hi = edges[edges.length - 1];
  for (const values of rowsData) {
    for (const value of values) {
      if (value < lo || value > hi) return false;
    }
  }
  return true;
}

/** Bin edges spanning every row so all ridge rows share one x-axis; a clip range
 * spans a quantile band instead of min/max for heavy-tailed (grad) series. */
function sharedBinEdges(rows, bins, clip) {
  let lo;
  let hi;
  if (!clip) {
    lo = Math.min(...rows.map((row) => Math.min(...row)));
    hi = Math.max(...rows.map((row) => Math.max(...row)));
  } else {
    const pooled = rows.flat().sort((a, b) => a - b);
    lo = quantile(pooled, clip[0]);
    hi = quantile(pooled, clip[1]);
  }
  if (hi === lo) {
    lo -= 0.5;
    hi += 0.5;
  }
  const width = (hi - lo) / bins;
  return Array.from({ length: bins + 1 }, (_, i) => lo + i * width);
}

/** Value at a quantile of an already-sorted list. */
function quantile(sorted, q) {
  return sorted[Math.min(Math.floor(q * sorted.length), sorted.length - 1)];
}

/** Histogram counts of values against bin edges, clamping outliers to the ends. */
function binCounts(values, edges) {
  const bins = edges.length - 1;
  const lo = edges[0];
  const width = (edges[bins] - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const value of values) {
    const index = Math.max(0, Math.min(Math.floor((value - lo) / width), bins - 1));
    counts[index] += 1;
  }
  return counts;
}
