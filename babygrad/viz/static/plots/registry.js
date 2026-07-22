/*
 * The registry turns the live history into a flat list of plottable items and
 * the visibility predicates the sidebar and main area share. An item is the unit
 * the sidebar toggles and the main area renders; the visibility rules (hidden,
 * folded group, search query) live here so both surfaces agree on what shows.
 */

import { scopeGroup, leafLabel, SCALAR_GROUP, DIAGNOSTICS_GROUP } from "./grouping.js";
import { applicableDiagnostics } from "./diagnostics.js";
import { groupBy } from "./util.js";

/**
 * Flat list of plottable items from the live history: scalar series (loss,
 * val_loss, metrics), array parameter series (weights, bias, γ/β, and each one's
 * gradient), and the computed diagnostics whose inputs are present. Layer
 * activations (the `/result` series) are excluded from the ridge set — they are
 * not parameters — but they do feed the dead-unit diagnostic.
 */
export function buildItems(history) {
  const items = [];
  for (const [tag, series] of Object.entries(history)) {
    const first = Object.values(series)[0];
    if (typeof first === "number") {
      items.push({ id: tag, label: tag, group: SCALAR_GROUP, kind: "scalar" });
    } else if (Array.isArray(first) && !tag.includes("/result")) {
      items.push({ id: tag, label: leafLabel(tag), group: scopeGroup(tag), kind: "ridge" });
    }
  }
  for (const diagnostic of applicableDiagnostics(history)) {
    items.push({ id: diagnostic.id, label: diagnostic.label, group: DIAGNOSTICS_GROUP, kind: "diagnostic" });
  }
  return items;
}

/**
 * Items grouped and ordered for display: the scalar group first, then scopes
 * alphabetically; items within a group sorted by label. Returns
 * `[[groupName, items], …]` so the sidebar tree and the main sections iterate
 * the same order.
 */
export function groupOrder(items) {
  const groups = groupBy(items, (item) => item.group);
  for (const list of groups.values()) {
    list.sort((a, b) => a.label.localeCompare(b.label));
  }
  return [...groups.keys()].sort(byGroupName).map((name) => [name, groups.get(name)]);
}

/** Scalar group first, then Diagnostics, then scope groups alphabetically. */
function byGroupName(a, b) {
  const rank = (group) => (group === SCALAR_GROUP ? 0 : group === DIAGNOSTICS_GROUP ? 1 : 2);
  return rank(a) - rank(b) || a.localeCompare(b);
}

/** Whether an item's label/group/id contains the (already-lowercased) query. */
export function matchesQuery(item, query) {
  if (!query) return true;
  return `${item.label} ${item.group} ${item.id}`.toLowerCase().includes(query);
}

/** Whether an item should render: not hidden, its group not folded, query matches. */
export function isVisible(item, state) {
  return (
    !state.hidden.has(item.id) &&
    !state.collapsed.has(item.group) &&
    matchesQuery(item, state.query)
  );
}
