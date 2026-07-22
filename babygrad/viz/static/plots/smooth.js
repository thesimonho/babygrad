/*
 * Exponential-moving-average smoothing for the line charts, the same debiased EMA
 * TensorBoard uses. A `weight` in [0, 1) trades noise for lag: 0 leaves the series
 * untouched, higher values follow the trend more slowly. The debias term
 * (1 − weightⁱ⁺¹) cancels the cold-start pull toward the initial value, so the
 * first few points aren't dragged down to near zero.
 *
 * Smoothing is a display transform only — it feeds the drawn line (and the
 * overfit band between two smoothed lines), never the hover tooltips or the
 * best-epoch markers, which stay on the true values.
 */

import { groupBy } from "./util.js";

/**
 * A debiased EMA over a series of `{ epoch, value }` points, returning new points
 * with the smoothed value. `weight` 0 (or falsy) returns the series unchanged.
 */
export function smoothSeries(points, weight) {
  if (!weight) return points;
  let ema = 0;
  return points.map((point, index) => {
    ema = weight * ema + (1 - weight) * point.value;
    const debiased = ema / (1 - weight ** (index + 1));
    return { ...point, value: debiased };
  });
}

/**
 * Smooth each tag's series independently inside a flat array of `{ tag, epoch,
 * value }` rows (the shape the scalar chart carries), so lines don't bleed into
 * one another. Order within a tag is by epoch; `weight` 0 returns the rows as-is.
 */
export function smoothByTag(rows, weight) {
  if (!weight) return rows;
  const smoothed = [];
  for (const series of groupBy(rows, (row) => row.tag).values()) {
    series.sort((a, b) => a.epoch - b.epoch);
    smoothed.push(...smoothSeries(series, weight));
  }
  return smoothed;
}
