/*
 * The sidebar: a persistent search box plus a grouped, foldable checkbox tree of
 * every plottable item. The search input is created once and kept (so typing
 * never loses focus to a re-render); only the tree below it is rebuilt.
 *
 * Interaction mutates the shared `state` object in place, then re-renders the
 * tree and calls `onChange` so the main chart area redraws. State lives with the
 * caller (main.js); this module only reads and writes it.
 */

import { groupOrder, matchesQuery } from "./registry.js";
import { foldHeader, checkboxRow } from "./dom.js";
import { toggleInSet } from "./util.js";

/**
 * Build the sidebar. Returns `{ el, refresh(items) }` — append `el` once, then
 * call `refresh` whenever the item list changes (e.g. new tags stream in).
 */
export function createSidebar(state, onChange) {
  const el = document.createElement("div");
  const search = document.createElement("input");
  search.type = "search";
  search.className = "plots-search";
  search.placeholder = "Filter series…";
  const tree = document.createElement("div");
  tree.className = "plots-tree";
  el.append(search, tree);

  let currentItems = [];
  let lastSignature = "";

  search.addEventListener("input", () => {
    state.query = search.value.trim().toLowerCase();
    renderTree();
    onChange();
  });

  /** Rebuild the group tree from the current items and state. */
  function renderTree() {
    tree.innerHTML = "";
    for (const [group, items] of groupOrder(currentItems)) {
      const rows = items.filter((item) => matchesQuery(item, state.query));
      if (rows.length > 0) tree.append(groupNode(group, rows));
    }
  }

  /** One group: a foldable header plus its checkbox rows (hidden when folded). */
  function groupNode(group, items) {
    const wrap = document.createElement("div");
    wrap.className = "plots-group";
    const collapsed = state.collapsed.has(group);
    const onToggle = () => {
      toggleInSet(state.collapsed, group);
      renderTree();
      onChange();
    };
    wrap.append(foldHeader(group, { collapsed, onToggle, className: "plots-group-head" }));
    if (!collapsed) items.forEach((item) => wrap.append(itemRow(item)));
    return wrap;
  }

  /** A single checkbox row toggling one item's visibility. */
  function itemRow(item) {
    return checkboxRow(item.label, {
      checked: !state.hidden.has(item.id),
      className: "plots-item",
      onChange: (checked) => {
        if (checked) state.hidden.delete(item.id);
        else state.hidden.add(item.id);
        onChange();
      },
    });
  }

  return {
    el,
    /**
     * Adopt a fresh item list (from a streamed epoch). The tree is only rebuilt
     * when the set of item ids actually changes — a new tag appeared — so a
     * stable list keeps the tree (and the user's scroll position) untouched
     * through the every-epoch redraws. Fold/search still rebuild it directly.
     */
    refresh(items) {
      currentItems = items;
      const signature = items.map((item) => item.id).join("\n");
      if (signature === lastSignature) return;
      lastSignature = signature;
      renderTree();
    },
  };
}
