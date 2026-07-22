/*
 * A crosshair shared across every epoch-on-X chart: hovering any one chart draws a
 * vertical reference line at the same epoch on all of them, so the loss and the
 * diagnostics read as one column. Observable Plot tracks the pointer per-plot and
 * won't sync across separate plots, so the synced line is a plain DOM overlay
 * (pointer-events: none, so it never blocks a chart's own Plot.tip) positioned
 * from each chart's own x-scale. The shared state is the *semantic epoch*; every
 * chart maps it back to its own pixels with `scale("x").apply`.
 *
 * The charts are torn down and rebuilt on every live redraw, so the registry is
 * cleared each render (resetCrosshairs) and, once the rebuilt charts are in the
 * DOM, the current epoch is re-applied to them (applyCrosshairs) — the line
 * survives a redraw mid-hover instead of blinking out. Re-applying after the
 * charts are mounted matters: their pixel geometry is only measurable once laid
 * out.
 */

const LINE_COLOR = "#9aa0ac";

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
 * hovering it (or any other registered chart) draws a vertical line at the shared
 * epoch. A `note()` placeholder (no `.scale`) is returned unwrapped. Returns the
 * node to append in place of the original.
 */
export function attachCrosshair(node) {
  if (typeof node.scale !== "function") return node;
  const svg = node.tagName === "svg" ? node : node.querySelector(":scope > svg");
  if (!svg) return node;

  const wrapper = document.createElement("div");
  wrapper.style.position = "relative";
  const line = crosshairLine();
  wrapper.append(node, line);

  charts.push(() => positionLine(node, svg, wrapper, line));

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
  });
  return line;
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

/** Position (or hide) one chart's vertical line at the shared epoch. The epoch may
 * fall outside a chart's domain (e.g. update-ratio drops epoch 0), so hide there.
 * Plot's scale outputs are internal coordinates; they and the svg's offset within
 * the wrapper are converted to rendered pixels via the render scale. */
function positionLine(node, svg, wrapper, line) {
  const xScale = node.scale("x");
  const [min, max] = xScale.domain;
  if (hoveredEpoch === null || hoveredEpoch < min || hoveredEpoch > max) {
    line.style.display = "none";
    return;
  }
  const { rect, scale } = svgGeometry(svg);
  const wrapRect = wrapper.getBoundingClientRect();
  const [y0, y1] = node.scale("y").range;
  line.style.left = `${rect.left - wrapRect.left + xScale.apply(hoveredEpoch) * scale}px`;
  line.style.top = `${rect.top - wrapRect.top + Math.min(y0, y1) * scale}px`;
  line.style.height = `${Math.abs(y0 - y1) * scale}px`;
  line.style.display = "block";
}

/** An svg's client rect and the uniform factor mapping its internal user
 * coordinates to rendered pixels (Plot shrinks the svg to fit via max-width). */
function svgGeometry(svg) {
  const rect = svg.getBoundingClientRect();
  return { rect, scale: rect.width / svg.width.baseVal.value };
}
