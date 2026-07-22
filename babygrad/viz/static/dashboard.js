/*
 * babygrad dashboard frontend.
 *
 * Fetches the neutral graph JSON (/graph.json) and the shared palette
 * (/theme.json), then renders with Cytoscape: scopes become compound parent
 * nodes, so nesting and collapse are expressed by containment rather than
 * coordinates. ELK's `elk.layered` lays the compound DAG out client-side with a
 * directional flow while respecting the scope nesting; cytoscape-expand-collapse
 * folds a scope on demand.
 *
 * All colours come from /theme.json — the same palette the graphviz SVG views
 * use — so the live dashboard and the notebook graphs stay visually identical.
 */

"use strict";

cytoscape.use(window.cytoscapeElk);
cytoscape.use(window.cytoscapeExpandCollapse);

/** Graphviz shape names (from the shared theme) mapped onto Cytoscape shapes. */
const SHAPE = { box: "round-rectangle", ellipse: "ellipse" };

/*
 * ELK layered layout. `hierarchyHandling: INCLUDE_CHILDREN` is what makes the
 * layering span into the compound scope boxes rather than treating each scope as
 * one opaque block, so the whole graph reads as one directional flow. Direction
 * DOWN reads top-to-bottom: inputs at the top, the output at the bottom; flip to
 * RIGHT for a left-to-right read.
 */
const ELK_LAYOUT = {
  name: "elk",
  // No `fit`/`padding` here: framing is owned by frameGraph (run on layoutstop),
  // which clamps a tall graph to a readable zoom rather than shrinking it to fit.
  elk: {
    algorithm: "layered",
    "elk.direction": "DOWN",
    "elk.hierarchyHandling": "INCLUDE_CHILDREN",
    // Orthogonal edge routing reserves lanes between layers so edges don't cut
    // across nodes.
    "elk.edgeRouting": "ORTHOGONAL",
    "elk.layered.spacing.nodeNodeBetweenLayers": 45,
    "elk.spacing.nodeNode": 40,
    "elk.spacing.edgeNode": 20,
    "elk.padding": "[top=28,left=18,bottom=18,right=18]",
  },
};

// A deep net lays out as a tall column, so fitting the whole graph shrinks it to
// an unreadable sliver. Instead frame it at a readable zoom, anchored so the top
// of the graph sits at the top of the viewport — the user scrolls down to follow
// the flow. Only fall back to fit-everything when the graph is short enough that
// fitting already lands above this zoom.
const READABLE_ZOOM = 1.4;

wireTabs();
loadGraph();

/** Toggle the visible panel when a tab is clicked, reflect it in ?tab=, and open
 * the tab named in the URL on load (so ?tab=plots deep-links the Plots tab). */
function wireTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      const wanted = tab.dataset.panel;
      for (const other of tabs) other.classList.toggle("active", other === tab);
      for (const panel of panels)
        panel.classList.toggle("active", panel.dataset.panel === wanted);
      history.replaceState(null, "", `?tab=${wanted}`);
    });
  }

  const wanted = new URLSearchParams(location.search).get("tab");
  const target = [...tabs].find((tab) => tab.dataset.panel === wanted);
  // Defer past this parse so a tab whose lazy render is wired by a later-loaded
  // script (plots.js binds its own click listener) actually renders on deep-link.
  if (target) setTimeout(() => target.click(), 0);
}

/** Fetch the graph payload and shared theme together, then render. */
async function loadGraph() {
  try {
    const [graph, theme] = await Promise.all([
      fetch("/graph.json").then((r) => r.json()),
      fetch("/theme.json").then((r) => r.json()),
    ]);
    renderGraph(graph, theme);
  } catch (error) {
    document.getElementById("hint").textContent =
      `failed to load graph: ${error}`;
  }
}

/** Build the Cytoscape instance from the payload + theme and lay it out. */
function renderGraph(payload, theme) {
  // Before any run connects, the server serves an empty graph — show a waiting
  // note instead of building an empty canvas. A run's `reset` reloads the page.
  if (!payload.nodes || payload.nodes.length === 0) {
    document.getElementById("hint").textContent =
      "waiting for a training run… start one with `--dashboard`";
    return;
  }

  const container = document.getElementById("graph");
  container.style.background = theme.background;

  const cy = cytoscape({
    container,
    elements: toElements(payload, theme),
    style: graphStyle(theme),
    wheelSensitivity: 2, // brisk scroll-zoom
    minZoom: 0.1,
    maxZoom: 8, // let fit zoom in enough to fill the viewport for small graphs
  });

  const api = cy.expandCollapse({
    // Per-scope collapse/expand happens in place: layoutBy null means the
    // extension does NOT re-run the global ELK layout on each toggle. A re-layout
    // would recompute every position from scratch (discarding manual node drags)
    // and, because ELK_LAYOUT fits, re-frame the viewport (discarding the pan) —
    // so a folded/unfolded scope would snap back to the original framing first.
    // "Collapse/Expand all" relayouts explicitly; see wireExpandCollapseAll.
    layoutBy: null,
    animate: true,
    animationDuration: 250,
    undoable: false,
    // Instant fold, no animation. This extension only animates expand (fisheye
    // eases neighbours aside) and never collapse, so fisheye:true gives a
    // one-sided animation. The old symmetric motion was the layoutBy reflow, which
    // we dropped to keep collapse/expand in place — so instant is the consistent
    // choice. (animate/animationDuration above are inert while fisheye is off.)
    fisheye: false,
  });

  // Round the op nodes before laying out, so ELK spaces them at their circular
  // size rather than their oval one.
  circularizeOps(cy);

  // Fold the scopes that arrived collapsed before the single layout pass, so ELK
  // arranges the already-folded graph. Order matters now that collapse no longer
  // triggers its own layout: collapse first, then lay the folded graph out once.
  collapseFlagged(cy, api, payload.scopes);
  relayout(cy);
  wireCollapseOnTap(cy, api);
  wireExpandCollapseAll(cy, api);
  if (window.wireNodePopup) window.wireNodePopup(cy);
}

/**
 * Make op nodes circles instead of label-fitted ovals. The shared node style
 * sizes width and height to the label independently; a single-line label has a
 * fixed height but a width that grows with its text, so short ops (``@``) come out
 * round while long ops (``ReLU``) come out as wide ovals. Setting both dimensions
 * to the larger of the two makes every op a circle that still fits its label.
 * Cytoscape styles can't express max(width, height), so it's done here in JS.
 */
function circularizeOps(cy) {
  cy.nodes('[kind = "OP"]').forEach((node) => {
    const diameter = Math.max(node.width(), node.height());
    node.style({ width: diameter, height: diameter });
  });
}

/**
 * Run the ELK layout and frame the result. Used on load and by "Collapse/Expand
 * all" — deliberate whole-graph actions where re-arranging and re-framing is
 * wanted. Per-scope collapse/expand deliberately skips this (see the layoutBy
 * comment) so a toggled scope stays where the user dragged and panned it.
 */
function relayout(cy) {
  const layout = cy.layout(ELK_LAYOUT);
  layout.one("layoutstop", () => frameGraph(cy));
  layout.run();
}

/** Wire the toolbar button to fold or unfold every scope at once. "Collapse all"
 * folds to the top-level layer boxes (the outermost scope stays open) rather than
 * to a single root box, which is the useful architecture overview. */
function wireExpandCollapseAll(cy, api) {
  const button = document.getElementById("collapse-all");
  if (!button) return;
  let allCollapsed = false;
  button.onclick = () => {
    allCollapsed = !allCollapsed;
    if (allCollapsed) {
      api.collapseAll();
      const topScopes = cy
        .nodes()
        .filter((node) => node.data("isScope") && node.parent().empty());
      if (topScopes.nonempty()) api.expand(topScopes);
    } else {
      api.expandAll();
    }
    button.textContent = allCollapsed ? "Expand all" : "Collapse all";
    // A whole-graph overview action: re-arrange and re-frame (per-scope toggles
    // stay in place; only this deliberate reset relayouts).
    relayout(cy);
  };
}

/**
 * Frame the graph on load: fit it, but if that zooms out below a readable level
 * (a tall graph), clamp to READABLE_ZOOM and pan so the graph's top-centre sits
 * near the top of the viewport instead of shrinking the whole column to fit.
 */
function frameGraph(cy) {
  cy.fit(cy.elements(), 30);
  if (cy.zoom() >= READABLE_ZOOM) return;

  cy.zoom(READABLE_ZOOM);
  const box = cy.elements().boundingBox();
  const centreX = (box.x1 + box.x2) / 2;
  cy.pan({
    x: cy.container().clientWidth / 2 - centreX * READABLE_ZOOM,
    y: 30 - box.y1 * READABLE_ZOOM,
  });
}

/**
 * Map the neutral payload to Cytoscape elements, resolving each node's colour and
 * shape from the theme so the stylesheet can stay data-driven. A scope becomes a
 * parent node; its `outer` is its Cytoscape parent, so the compound tree mirrors
 * the module tree. Null parents sit at the top level.
 */
function toElements(payload, theme) {
  const scopeNodes = payload.scopes.map((scope) => {
    const cluster = theme.clusters[crc32(scope.id) % theme.clusters.length];
    return {
      data: {
        id: scope.id,
        label: scope.label,
        parent: scope.outer ?? undefined,
        isScope: true,
        collapsed: scope.collapsed,
        fill: cluster.fill,
        border: cluster.border,
      },
    };
  });

  const graphNodes = payload.nodes.map((node) => {
    const style = theme.nodes[node.kind] ?? theme.nodes._default;
    return {
      data: {
        id: node.id,
        label: nodeLabel(node),
        parent: node.scope ?? undefined,
        fill: style.fill,
        font: style.font,
        shape: SHAPE[style.shape] ?? "round-rectangle",
        // carried for the click popup, not the layout
        kind: node.kind,
        rawShape: node.shape,
        scopePath: node.scope,
        seriesTag: node.series_tag,
      },
    };
  });

  // Resolve each node's kind once so the per-edge "feeds an op?" test is a map
  // lookup rather than a linear scan of every node per edge.
  const kindByNodeId = new Map(payload.nodes.map((node) => [node.id, node.kind]));
  const edges = payload.edges.map((edge, index) => ({
    data: {
      id: `e${index}`,
      source: edge.source,
      target: edge.target,
      label: edge.label ?? "",
      intoOp: kindByNodeId.get(edge.target) === "OP",
    },
  }));

  return [...scopeNodes, ...graphNodes, ...edges];
}

/** A node's caption: role/label plus its tensor shape when it has one. */
function nodeLabel(node) {
  if (!node.shape) return node.label;
  return `${node.label}\n${node.shape.join("×")}`;
}

/** The Cytoscape stylesheet, driven entirely by per-element theme data. */
function graphStyle(theme) {
  return [
    {
      selector: "node[isScope]",
      style: {
        "background-color": "data(fill)",
        "border-color": "data(border)",
        "border-width": 1.4,
        shape: "round-rectangle",
        label: "data(label)",
        color: "#495057",
        "font-size": 12,
        "font-weight": "bold",
        "text-valign": "top",
        "text-halign": "center",
        "text-margin-y": 4,
        padding: 16,
      },
    },
    {
      selector: "node[!isScope]",
      style: {
        "background-color": "data(fill)",
        shape: "data(shape)",
        label: "data(label)",
        color: "data(font)",
        "font-size": 10,
        "text-valign": "center",
        "text-halign": "center",
        "text-wrap": "wrap",
        width: "label",
        height: "label",
        padding: 9,
      },
    },
    {
      selector: "node.cy-expand-collapse-collapsed-node",
      style: {
        "background-color": "data(fill)",
        "border-color": "data(border)",
        "border-width": 2,
        "text-valign": "center",
        "text-margin-y": 0,
        padding: 12,
      },
    },
    {
      selector: "edge",
      style: {
        width: 1.2,
        "line-color": theme.edge.color,
        "target-arrow-color": theme.edge.color,
        "target-arrow-shape": "triangle",
        "arrow-scale": 0.8,
        "curve-style": "taxi",
        "taxi-direction": "auto",
        "taxi-turn": "40%",
        "taxi-turn-min-distance": 4,
        label: "data(label)",
        "font-size": 9,
        color: theme.edge.fontColor,
        "text-background-color": theme.background,
        "text-background-opacity": 1,
        "text-background-padding": 2,
      },
    },
    {
      selector: "edge[intoOp]",
      style: { "line-style": "dashed" },
    },
  ];
}

/** Fold every scope that arrived flagged collapsed, so the graph opens folded. */
function collapseFlagged(cy, api, scopes) {
  const flagged = scopes
    .filter((scope) => scope.collapsed)
    .map((scope) => scope.id);
  if (flagged.length === 0) return;
  const targets = cy.collection(flagged.map((id) => cy.getElementById(id)));
  api.collapse(targets);
}

/** Click a scope box to fold or unfold it in place. */
function wireCollapseOnTap(cy, api) {
  cy.on("tap", "node[isScope]", (event) => {
    const node = event.target;
    if (api.isCollapsible(node)) api.collapse(node);
    else if (api.isExpandable(node)) api.expand(node);
  });
}

/**
 * CRC-32 (IEEE polynomial), byte-identical to Python's ``zlib.crc32`` for ASCII
 * input. The shared theme picks a scope's cluster colour by this hash of its id,
 * so a scope keeps the same colour here as in the graphviz SVG views.
 */
function crc32(text) {
  let crc = 0xffffffff;
  for (let i = 0; i < text.length; i++) {
    crc ^= text.charCodeAt(i) & 0xff;
    for (let bit = 0; bit < 8; bit++) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}
