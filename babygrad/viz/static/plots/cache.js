/*
 * A shared incremental cache for per-step series: compute each step's value once
 * and append only the steps added since the last call. Mirrors the append-only
 * shape of the ridge chart's bin cache (kept separate — that one also invalidates
 * when a value range shifts), in a small reusable form for the diagnostics:
 * recomputing an L2 norm over a (transformer-sized) tensor for every epoch on
 * every 400ms redraw would be wasteful, and a step's value never changes once
 * recorded.
 *
 * Keyed by an arbitrary string; cleared on the page reload a new run triggers, so
 * a cache never spans runs.
 */

const caches = new Map();

/**
 * The per-step rows for `key`, reusing cached rows for steps already seen and
 * computing only the ones added since the last call. `computeStep(step, index)`
 * returns one row for a step (index is its absolute position in `steps`, for
 * computations that need the previous step). `steps` must be sorted ascending.
 */
export function perStep(key, steps, computeStep) {
  const cached = caches.get(key);
  if (cached && cached.count === steps.length) return cached.rows;

  const from = cached && cached.count < steps.length ? cached.count : 0;
  const kept = from > 0 ? cached.rows : [];
  const fresh = steps.slice(from).map((step, offset) => computeStep(step, from + offset));
  const rows = kept.concat(fresh);
  caches.set(key, { count: steps.length, rows });
  return rows;
}
