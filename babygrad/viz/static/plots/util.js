/*
 * Tiny shared helpers with no DOM or chart knowledge.
 */

/** Add a key to a set if absent, remove it if present. */
export function toggleInSet(set, key) {
  if (set.has(key)) set.delete(key);
  else set.add(key);
}

/** Group items into a Map keyed by `keyFn(item)`, preserving insertion order
 * within each group. */
export function groupBy(items, keyFn) {
  const groups = new Map();
  for (const item of items) {
    const key = keyFn(item);
    const list = groups.get(key) ?? [];
    list.push(item);
    groups.set(key, list);
  }
  return groups;
}
