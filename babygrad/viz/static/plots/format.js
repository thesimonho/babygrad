/*
 * Shared number formatting for plot axes and tips.
 */

/**
 * Format an axis tick: exponential for very small or very large magnitudes (so
 * tiny gradients don't render as a wall of zeros), plain otherwise.
 */
export function formatTick(value) {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude < 1e-3 || magnitude >= 1e5) return value.toExponential(1);
  return String(Number(value.toPrecision(3)));
}
