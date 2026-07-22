/*
 * Training diagnostics computed from the live history — no new recorder tags, all
 * derived from tensors already streamed:
 *
 *   - Gradient flow      — L2 norm of each layer's weight gradient over epochs;
 *                          spots vanishing / exploding gradients across depth.
 *   - Update:weight ratio — ‖Wₜ − Wₜ₋₁‖ / ‖Wₜ‖ per weight tensor (Karpathy's
 *                          rule of thumb ≈ 1e-3); undefined at the first step.
 *   - Dead-unit %        — share of a post-ReLU activation that is ≈ 0.
 *   - Global grad-norm   — L2 norm over every gradient at once; a quick "is
 *                          training stable" line.
 *
 * Each diagnostic declares an `applicable(history)` guard so it only appears when
 * its inputs are present, and a `compute(history)` returning
 * `{ series: [{ label, points }], options }` for the lines chart. Per-step values
 * are cached incrementally (see cache.js) since they never change once recorded.
 */

import { scopeGroup } from "./grouping.js";
import { perStep } from "./cache.js";

// A unit counts as dead when its post-activation magnitude is below this.
const DEAD_EPS = 1e-6;

/** The diagnostics, in display order. */
export const DIAGNOSTICS = [
  { id: "diag/grad-flow", label: "Gradient flow", applicable: hasWeightGrads, compute: gradFlow },
  { id: "diag/update-ratio", label: "Update:weight ratio", applicable: hasUpdatableWeights, compute: updateRatio },
  { id: "diag/dead-units", label: "Dead units %", applicable: hasReluResults, compute: deadUnits },
  { id: "diag/grad-norm", label: "Global grad-norm", applicable: hasGrads, compute: globalGradNorm },
];

/** The diagnostics whose inputs are present in the current history. */
export function applicableDiagnostics(history) {
  return DIAGNOSTICS.filter((diagnostic) => diagnostic.applicable(history));
}

/** Compute one diagnostic's chart data by id. */
export function computeDiagnostic(id, history) {
  const diagnostic = DIAGNOSTICS.find((entry) => entry.id === id);
  return diagnostic ? diagnostic.compute(history) : { series: [], options: {} };
}

// --- which tags feed which diagnostic ---------------------------------------

const isWeightGrad = (tag) => tag.endsWith("/weights/grad");
const isWeights = (tag) => tag.endsWith("/weights");
const isGrad = (tag) => tag.endsWith("/grad");
const isReluResult = (tag) => tag.endsWith("/result") && scopeGroup(tag).startsWith("ReLU");

function tagsMatching(history, predicate) {
  return Object.keys(history).filter(predicate).sort();
}

function hasWeightGrads(history) {
  return Object.keys(history).some(isWeightGrad);
}
function hasUpdatableWeights(history) {
  return Object.keys(history).some((tag) => isWeights(tag) && sortedSteps(history[tag]).length >= 2);
}
function hasReluResults(history) {
  return Object.keys(history).some(isReluResult);
}
function hasGrads(history) {
  return Object.keys(history).some(isGrad);
}

// --- the four computations ---------------------------------------------------

/** One line per weight-gradient tag: its L2 norm at each epoch. */
function gradFlow(history) {
  const series = tagsMatching(history, isWeightGrad).map((tag) => ({
    label: scopeGroup(tag),
    points: perStep(tag, sortedSteps(history[tag]), (step) => ({
      epoch: step,
      value: l2(history[tag][String(step)]),
    })),
  }));
  return { series, options: { yLabel: "‖grad‖" } };
}

/** One line per weight tensor: ‖Wₜ − Wₜ₋₁‖ / ‖Wₜ‖, dropping the undefined first step. */
function updateRatio(history) {
  const series = tagsMatching(history, isWeights)
    .map((tag) => ({ tag, steps: sortedSteps(history[tag]) }))
    .filter((entry) => entry.steps.length >= 2)
    .map(({ tag, steps }) => ({ label: scopeGroup(tag), points: ratioPoints(tag, history[tag], steps) }));
  return { series, options: { yLabel: "‖ΔW‖ / ‖W‖", logY: true } };
}

function ratioPoints(tag, series, steps) {
  const rows = perStep(`${tag}/update-ratio`, steps, (step, index) => {
    if (index === 0) return { epoch: step, value: null };
    const current = series[String(step)];
    const previous = series[String(steps[index - 1])];
    return { epoch: step, value: l2diff(current, previous) / (l2(current) || 1e-12) };
  });
  return rows.filter((row) => row.value !== null);
}

/** One line per post-ReLU activation: the percentage of its units that are ≈ 0. */
function deadUnits(history) {
  const series = tagsMatching(history, isReluResult).map((tag) => ({
    label: scopeGroup(tag),
    points: perStep(`${tag}/dead`, sortedSteps(history[tag]), (step) => ({
      epoch: step,
      value: deadFraction(history[tag][String(step)]),
    })),
  }));
  return { series, options: { yLabel: "% dead" } };
}

/** A single line: the L2 norm over every gradient tensor at each epoch. */
function globalGradNorm(history) {
  const gradTags = tagsMatching(history, isGrad);
  if (gradTags.length === 0) return { series: [], options: {} };
  const steps = sortedSteps(history[gradTags[0]]);
  const points = perStep("diag/grad-norm", steps, (step) => {
    let sumOfSquares = 0;
    for (const tag of gradTags) {
      const values = history[tag][String(step)];
      if (values) for (const value of values) sumOfSquares += value * value;
    }
    return { epoch: step, value: Math.sqrt(sumOfSquares) };
  });
  return { series: [{ label: "all grads", points }], options: { yLabel: "‖all grads‖" } };
}

// --- math helpers ------------------------------------------------------------

function sortedSteps(series) {
  return Object.keys(series)
    .map(Number)
    .sort((a, b) => a - b);
}

function l2(values) {
  let sumOfSquares = 0;
  for (const value of values) sumOfSquares += value * value;
  return Math.sqrt(sumOfSquares);
}

function l2diff(current, previous) {
  let sumOfSquares = 0;
  for (let i = 0; i < current.length; i++) {
    const delta = current[i] - previous[i];
    sumOfSquares += delta * delta;
  }
  return Math.sqrt(sumOfSquares);
}

function deadFraction(values) {
  let dead = 0;
  for (const value of values) if (Math.abs(value) < DEAD_EPS) dead += 1;
  return (100 * dead) / values.length;
}
