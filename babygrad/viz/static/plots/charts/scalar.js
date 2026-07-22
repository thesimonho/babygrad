/*
 * The loss & metrics chart: a line per scalar series over epochs, with a dual
 * y-axis (loss series left, metrics right) since Observable Plot has one y scale
 * per plot. Metric series are rescaled into the loss range and a right axis maps
 * their ticks back to true values.
 *
 * On top of the lines it draws three training aids, each guarded because the
 * history is filtered by the sidebar (any series may be hidden):
 *   - overfit shading — a band between train loss and val loss (the gap),
 *   - best-epoch markers — a ring at min val-loss and at max metric,
 *   - a log-scale option — toggled from the options bar above the chart.
 *
 * Log scale is compatible with the metric rescaling for free: rescaling by a
 * constant is a vertical shift in log space, so a metric point and its right-axis
 * tick still coincide. The only casualty is the value-0 right tick (log 0 = −∞),
 * which is dropped under log.
 */

import { formatTick } from "../format.js";
import { note } from "../dom.js";
import { smoothByTag } from "../smooth.js";
import { epochTip } from "./crosshair.js";

const LOSS_COLOR = "#457b9d";
const OVERFIT_COLOR = "#e63946";

/**
 * A line chart of the scalar series over epochs. `options.logScale` switches the
 * y-axis to log. With no metrics it degrades to a single-axis chart; with any
 * series hidden it simply omits that line and any aid that depended on it. Returns
 * `{ node, tipContent }`: the Plot node, and a per-epoch readout for the shared
 * crosshair tip (null for the empty-data placeholder).
 */
export function scalarChart(history, Plot, width, options = {}) {
  const loss = [];
  const metric = [];
  for (const [tag, series] of Object.entries(history)) {
    const steps = Object.keys(series);
    if (steps.length === 0 || typeof series[steps[0]] !== "number") continue;
    const bucket = tag.toLowerCase().includes("loss") ? loss : metric;
    for (const step of steps) {
      bucket.push({ tag, epoch: Number(step), value: series[step] });
    }
  }
  if (loss.length === 0 && metric.length === 0) return { node: note("no scalar series yet"), tipContent: null };

  // Loss is the primary (left) axis; if there is no loss series, metrics take it.
  const primary = loss.length > 0 ? loss : metric;
  const secondary = loss.length > 0 ? metric : [];
  const primaryLabel = loss.length > 0 ? "loss" : "value";

  const primaryMax = Math.max(1e-9, ...primary.map((d) => d.value));
  const secondaryMax = Math.max(1e-9, ...secondary.map((d) => d.value));
  const scale = primaryMax / secondaryMax; // maps a metric value into the loss range
  const metricY = (d) => (secondary.length === 0 ? d.value : d.value * scale);

  const isVal = (point) => point.tag.toLowerCase().includes("val");

  // The best-val marker sits on the true (unsmoothed) minimum val loss.
  const valLoss = loss.filter(isVal);

  // Smoothing feeds everything drawn — the lines and the overfit band between
  // them — so the shaded gap keeps hugging the smoothed lines. The band's train
  // and val edges are filtered back out of the smoothed primary so it isn't
  // smoothed twice. The hover tip and best-epoch markers stay on the true values.
  const smoothedPrimary = smoothByTag(primary, options.smoothing);
  const smoothedMetric = smoothByTag(metric, options.smoothing);
  const smoothedTrain = smoothedPrimary.filter((point) => !isVal(point));
  const smoothedVal = smoothedPrimary.filter(isVal);

  const marks = [
    ...overfitMarks(smoothedTrain, smoothedVal, Plot),
    Plot.line(smoothedPrimary, { x: "epoch", y: "value", stroke: "tag" }),
    ...(secondary.length > 0
      ? dualAxisMarks(smoothedMetric, scale, secondaryMax, primaryLabel, options, Plot)
      : []),
    ...bestEpochMarks(valLoss, metric, metricY, Plot),
  ];

  const node = Plot.plot({
    width,
    height: 300,
    marginLeft: 60,
    marginRight: secondary.length > 0 ? 66 : 20,
    x: { label: "epoch" },
    y: yScale(secondary.length > 0, primaryLabel, options),
    color: { legend: true },
    marks,
  });
  // The tip lists every series' honest value (not its rescaled plot height) at the
  // hovered epoch, so one hover reads loss + metrics at once. Synced across charts
  // via the shared crosshair (crosshair.js).
  return { node, tipContent: epochTip([...primary, ...secondary], (d) => `${d.tag}  ${formatTick(d.value)}`) };
}

/** The y-scale config: grid always, an explicit label/format only on a single
 * axis (the dual-axis marks carry their own), and log when the option is set. */
function yScale(isDual, primaryLabel, options) {
  const base = isDual ? { grid: true } : { label: primaryLabel, grid: true, tickFormat: formatTick };
  return options.logScale ? { ...base, type: "log" } : base;
}

/** The dashed metric line plus the two explicit y-axes. Declaring the right axis
 * suppresses Plot's implicit left axis, so the left (loss) axis is re-added. The
 * right ticks map metric values into the loss range; the value-0 tick is dropped
 * under log (log 0 is undefined). */
function dualAxisMarks(metric, scale, secondaryMax, primaryLabel, options, Plot) {
  let rightTicks = Array.from({ length: 5 }, (_, i) => (i * secondaryMax) / 4);
  if (options.logScale) rightTicks = rightTicks.filter((value) => value > 0);
  return [
    Plot.line(metric, {
      x: "epoch",
      y: (d) => d.value * scale,
      stroke: "tag",
      strokeDasharray: "4,3",
    }),
    Plot.axisY({ anchor: "left", label: primaryLabel, tickFormat: formatTick }),
    Plot.axisY({
      anchor: "right",
      label: [...new Set(metric.map((d) => d.tag))].join(", "),
      ticks: rightTicks.map((value) => value * scale),
      tickFormat: (value) => formatTick(value / scale),
    }),
  ];
}

/** A shaded band between train loss and val loss — the generalisation gap. Needs
 * both series present; returns nothing if either is hidden or absent. */
function overfitMarks(train, val, Plot) {
  if (train.length === 0 || val.length === 0) return [];
  const trainByEpoch = new Map(train.map((point) => [point.epoch, point.value]));
  const band = val
    .filter((point) => trainByEpoch.has(point.epoch))
    .map((point) => ({ epoch: point.epoch, train: trainByEpoch.get(point.epoch), val: point.value }));
  if (band.length === 0) return [];
  return [
    Plot.areaY(band, {
      x: "epoch",
      y1: "train",
      y2: "val",
      fill: OVERFIT_COLOR,
      fillOpacity: 0.1,
    }),
  ];
}

/** Ring markers at the best epoch: minimum val loss (primary axis) and maximum
 * metric (positioned with the metric's own y-transform so it sits on its line).
 * Each is guarded on its series being present. */
function bestEpochMarks(val, metric, metricY, Plot) {
  const marks = [];
  if (val.length > 0) {
    const best = val.reduce((a, b) => (b.value < a.value ? b : a));
    marks.push(...ring([best], (d) => d.value, OVERFIT_COLOR, "best val", Plot));
  }
  if (metric.length > 0) {
    const best = metric.reduce((a, b) => (b.value > a.value ? b : a));
    marks.push(...ring([best], metricY, LOSS_COLOR, "best", Plot));
  }
  return marks;
}

/** A hollow ring plus a small label above it, at one point. */
function ring(data, y, color, label, Plot) {
  return [
    Plot.dot(data, { x: "epoch", y, r: 4, stroke: color, fill: "white", strokeWidth: 1.5 }),
    Plot.text(data, { x: "epoch", y, text: () => label, dy: -11, fontSize: 9, fill: color }),
  ];
}

