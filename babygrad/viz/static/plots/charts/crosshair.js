/*
 * A crosshair shared across every epoch-on-X chart: hovering any one chart draws a
 * vertical reference line — and a value tip — at the same epoch on *all* of them,
 * so the loss and the diagnostics read as one column. Observable Plot tracks the
 * pointer per-plot and won't sync across separate plots (its Plot.tip can only
 * fire on the chart the mouse is physically over), so both the line and the tip
 * are plain DOM overlays (pointer-events: none) positioned from each chart's own
 * x-scale. The shared state is the *semantic epoch*; every chart maps it back to
 * its own pixels with `scale("x").apply` and renders its own tip from a per-chart
 * `tipContent(epoch)` builder. The tip is stacked above the line (appended after
 * it) so the reference line no longer paints over the readout.
 *
 * The charts are torn down and rebuilt on every live redraw, so the registry is
 * cleared each render (resetCrosshairs) and, once the rebuilt charts are in the
 * DOM, the current epoch is re-applied to them (applyCrosshairs) — the line and
 * tip survive a redraw mid-hover instead of blinking out. Re-applying after the
 * charts are mounted matters: their pixel geometry is only measurable once laid
 * out.
 */

const LINE_COLOR = "#9aa0ac";
// The tip sits this many rendered pixels below the top of the plot frame, and is
// nudged this far in from an edge so it never clips out of a narrow chart cell.
const TIP_GAP = 4;
const TIP_MARGIN = 4;

// The epoch under the cursor (semantic, shared), and the charts to redraw when it
// moves. `charts` is rebuilt every render — see resetCrosshairs.
let hoveredEpoch = null;
let charts = [];

/** Clear the chart registry at the start of a render, before the charts rebuild. */
export function resetCrosshairs() {
  charts = [];
}

/** Re-apply the current epoch to every chart, once they are mounted and their
 * pixel geometry is measurable. Called at the end of a render so a redraw during
 * a hover redraws the line at the right place rather than dropping it. */
export function applyCrosshairs() {
  for (const place of charts) place();
}

/**
 * Wrap an epoch-on-X chart node in a synced-crosshair overlay and register it, so
 * hovering it (or any other registered chart) draws a vertical line and a value
 * tip at the shared epoch. `tipContent(epoch)` returns this chart's readout for an
 * epoch (a `\n`-joined string) or null where it has no data there; omit it for a
 * line without a tip. A `note()` placeholder (no `.scale`) is returned unwrapped.
 * Returns the node to append in place of the original.
 */
export function attachCrosshair(node, tipContent) {
  if (typeof node.scale !== "function") return node;
  const svg = node.tagName === "svg" ? node : node.querySelector(":scope > svg");
  if (!svg) return node;

  const wrapper = document.createElement("div");
  wrapper.style.position = "relative";
  const line = crosshairLine();
  const tip = crosshairTip();
  wrapper.append(node, line, tip); // tip last so it paints above the line

  charts.push(() => placeOverlays(node, svg, wrapper, line, tip, tipContent));

  wrapper.addEventListener("mousemove", (event) => setHoveredEpoch(epochAt(node, svg, event)));
  wrapper.addEventListener("mouseleave", () => setHoveredEpoch(null));
  return wrapper;
}

/** The absolutely-positioned overlay line, hidden until an epoch is hovered. */
function crosshairLine() {
  const line = document.createElement("div");
  Object.assign(line.style, {
    position: "absolute",
    width: "1px",
    background: LINE_COLOR,
    pointerEvents: "none",
    display: "none",
    zIndex: "1",
  });
  return line;
}

/** The absolutely-positioned value tip, hidden until an epoch is hovered. Styled
 * from the dashboard's theme variables so it reads like the rest of the UI, and
 * `pre` so the `\n`-joined per-series lines lay out one per row. */
function crosshairTip() {
  const tip = document.createElement("div");
  Object.assign(tip.style, {
    position: "absolute",
    pointerEvents: "none",
    display: "none",
    zIndex: "2", // above the line
    whiteSpace: "pre",
    font: "11px/1.45 system-ui, sans-serif",
    color: "var(--ink)",
    background: "var(--panel)",
    border: "1px solid var(--line)",
    borderRadius: "4px",
    padding: "4px 7px",
    boxShadow: "0 2px 8px rgba(20, 24, 40, 0.12)",
  });
  return tip;
}

/** Set the hovered epoch and redraw every registered chart's line. */
function setHoveredEpoch(epoch) {
  hoveredEpoch = epoch;
  applyCrosshairs();
}

/** The nearest epoch to the cursor's x, snapped to an integer within the domain.
 * Plot's scales work in the svg's internal coordinates, so the rendered cursor
 * offset is divided back by the render scale before inverting. */
function epochAt(node, svg, event) {
  const xScale = node.scale("x");
  const { rect, scale } = svgGeometry(svg);
  const internalX = (event.clientX - rect.left) / scale;
  const [min, max] = xScale.domain;
  return Math.max(min, Math.min(max, Math.round(xScale.invert(internalX))));
}

/** Position (or hide) one chart's vertical line and its value tip at the shared
 * epoch. The epoch may fall outside a chart's domain (e.g. update-ratio drops
 * epoch 0), so hide there. Plot's scale outputs are internal coordinates; they and
 * the svg's offset within the wrapper are converted to rendered pixels via the
 * render scale. */
function placeOverlays(node, svg, wrapper, line, tip, tipContent) {
  const xScale = node.scale("x");
  const [min, max] = xScale.domain;
  if (hoveredEpoch === null || hoveredEpoch < min || hoveredEpoch > max) {
    line.style.display = "none";
    tip.style.display = "none";
    return;
  }
  const { rect, scale } = svgGeometry(svg);
  const wrapRect = wrapper.getBoundingClientRect();
  const [y0, y1] = node.scale("y").range;
  const x = rect.left - wrapRect.left + xScale.apply(hoveredEpoch) * scale;
  const top = rect.top - wrapRect.top + Math.min(y0, y1) * scale;

  line.style.left = `${x}px`;
  line.style.top = `${top}px`;
  line.style.height = `${Math.abs(y0 - y1) * scale}px`;
  line.style.display = "block";

  positionTip(tip, tipContent, x, top, wrapper.clientWidth);
}

/** Fill and place (or hide) the value tip, centred on the line and nudged in from
 * either edge so it never clips out of a narrow chart cell. */
function positionTip(tip, tipContent, x, top, wrapperWidth) {
  const content = tipContent ? tipContent(hoveredEpoch) : null;
  if (!content) {
    tip.style.display = "none";
    return;
  }
  tip.textContent = content;
  tip.style.display = "block";
  const width = tip.offsetWidth; // measurable now it is displayed
  const centred = x - width / 2;
  const maxLeft = wrapperWidth - width - TIP_MARGIN;
  tip.style.left = `${Math.max(TIP_MARGIN, Math.min(centred, maxLeft))}px`;
  tip.style.top = `${top + TIP_GAP}px`;
}

/**
 * Build a `tipContent(epoch)` for a chart from its true (unsmoothed) rows: group
 * the rows by epoch and, per epoch, join a header with one `lineFor(row)` line per
 * series. Returns a function mapping an epoch to that block, or null where the
 * chart has no row at that epoch. Shared by the line charts and the loss chart so
 * both read the whole column at the hovered epoch.
 */
export function epochTip(rows, lineFor) {
  const byEpoch = new Map();
  for (const row of rows) {
    const column = byEpoch.get(row.epoch) ?? [];
    column.push(row);
    byEpoch.set(row.epoch, column);
  }
  const titles = new Map();
  for (const [epoch, column] of byEpoch) {
    titles.set(epoch, [`epoch ${epoch}`, ...column.map(lineFor)].join("\n"));
  }
  return (epoch) => titles.get(epoch) ?? null;
}

/** An svg's client rect and the uniform factor mapping its internal user
 * coordinates to rendered pixels (Plot shrinks the svg to fit via max-width). */
function svgGeometry(svg) {
  const rect = svg.getBoundingClientRect();
  return { rect, scale: rect.width / svg.width.baseVal.value };
}
