/*
 * A generic multi-line-over-epochs chart: one line per named series, coloured by
 * label, with a shared crosshair tip that lists every line's value at the hovered
 * epoch. Used by the diagnostics (gradient flow, update:weight ratio, dead-unit %,
 * global grad-norm) — they differ only in their data and a couple of options.
 *
 * options: { yLabel, logY } — an axis label and whether the y-axis is logarithmic.
 */

import { formatTick } from "../format.js";
import { note } from "../dom.js";
import { smoothSeries } from "../smooth.js";

/** A line chart from `[{ label, points: [{ epoch, value }] }]`. */
export function linesChart(seriesList, Plot, width, options = {}) {
  const rows = flattenRows(seriesList, (series) => series.points);
  if (rows.length === 0) return note("no data yet");

  // Smoothing feeds only the drawn lines; the hover tip reads the true values.
  const smoothedRows = flattenRows(seriesList, (series) => smoothSeries(series.points, options.smoothing));

  const y = { label: options.yLabel ?? "value", grid: true, tickFormat: formatTick };
  if (options.logY) y.type = "log";

  return Plot.plot({
    width,
    height: 220,
    marginLeft: 60,
    x: { label: "epoch" },
    y,
    color: { legend: true },
    marks: [
      Plot.line(smoothedRows, { x: "epoch", y: "value", stroke: "label" }),
      valueTip(rows, Plot),
    ],
  });
}

/** Flatten `[{ label, points }]` into `[{ label, epoch, value }]` rows, taking
 * each series' points through `pointsOf` (identity, or a smoothing pass). */
function flattenRows(seriesList, pointsOf) {
  const rows = [];
  for (const series of seriesList) {
    for (const point of pointsOf(series)) {
      rows.push({ label: series.label, epoch: point.epoch, value: point.value });
    }
  }
  return rows;
}

/** A tip listing every line's value at the hovered epoch, so one hover reads the
 * whole column at once. The vertical reference line is a shared DOM overlay (see
 * crosshair.js) so it syncs across every chart, not just this one. */
function valueTip(rows, Plot) {
  const byEpoch = new Map();
  for (const row of rows) {
    const column = byEpoch.get(row.epoch) ?? [];
    column.push(row);
    byEpoch.set(row.epoch, column);
  }
  const tipRows = [...byEpoch.entries()].map(([epoch, column]) => ({
    epoch,
    title: [`epoch ${epoch}`, ...column.map((r) => `${r.label}  ${formatTick(r.value)}`)].join("\n"),
  }));
  return Plot.tip(
    tipRows,
    Plot.pointerX({
      x: "epoch",
      frameAnchor: "top",
      title: "title",
      format: { x: false },
      pointerSize: 0,
    }),
  );
}
