/*
 * Live epoch stream. Seeds a shared history from /history.json, then subscribes
 * to /events (Server-Sent Events) and merges each epoch delta as it arrives.
 *
 * The merged history lives on window.liveHistory so the Plots tab and the node
 * popup read one growing copy; window.onLiveEpoch(cb) registers a listener that
 * fires after every merge, so those views can re-render. Re-sent steps (after an
 * EventSource reconnect) overwrite the same keys, so merging is idempotent.
 */

"use strict";

window.liveHistory = {};
const epochListeners = [];

window.onLiveEpoch = function onLiveEpoch(callback) {
  epochListeners.push(callback);
};

seedHistory();
subscribe();

/** Load whatever history already exists (a finished or in-progress run). */
async function seedHistory() {
  try {
    window.liveHistory = await fetch("/history.json").then((r) => r.json());
    notify(-1);
  } catch {
    // no history yet; the stream will fill it in
  }
}

/** Open the SSE connection and merge each epoch as it lands. */
function subscribe() {
  const source = new EventSource("/events");
  source.addEventListener("epoch", (event) => {
    const delta = JSON.parse(event.data);
    mergeDelta(delta);
    notify(delta.step);
  });
  // A new training run replaced the board — reload to pick up its graph and
  // history from scratch (SSE only carries epoch deltas, not the new graph).
  source.addEventListener("reset", () => location.reload());
}

/** Fold one epoch's scalars and series into the shared history at its step. */
function mergeDelta(delta) {
  for (const [tag, value] of Object.entries(delta.scalars ?? {})) {
    (window.liveHistory[tag] ??= {})[delta.step] = value;
  }
  for (const [tag, value] of Object.entries(delta.series ?? {})) {
    (window.liveHistory[tag] ??= {})[delta.step] = value;
  }
}

/** Tell every listener a merge happened (step -1 marks the initial seed). */
function notify(step) {
  for (const callback of epochListeners) callback(window.liveHistory, step);
}
