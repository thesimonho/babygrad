/*
 * Lazy loader for Observable Plot, shared across every chart module so the
 * library is imported (from jsDelivr) at most once per page.
 */

const PLOT_URL = "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

let plotLib = null;

/** Import Observable Plot once, caching the module across callers. */
export async function loadPlot() {
  if (!plotLib) plotLib = await import(PLOT_URL);
  return plotLib;
}
