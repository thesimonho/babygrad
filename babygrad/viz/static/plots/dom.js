/*
 * Small DOM builders shared across the plots UI: a foldable titled section, a
 * labelled chart card, and a muted placeholder note. Kept free of chart or state
 * knowledge so any module can use them.
 */

/**
 * A foldable header button: a caret (▸ folded / ▾ open) before its text, wired to
 * `onToggle`. Shared by the sidebar group heads and the main-area section heads so
 * the fold affordance looks and behaves the same in both; `className` styles each.
 */
export function foldHeader(text, { collapsed, onToggle = null, className }) {
  const head = document.createElement("button");
  head.type = "button";
  head.className = className;
  head.textContent = `${collapsed ? "▸" : "▾"} ${text}`;
  if (onToggle) head.addEventListener("click", onToggle);
  return head;
}

/**
 * A titled, foldable section. `buildBody` is a thunk called only when the section
 * is expanded, so a collapsed group never builds (or bins) its charts.
 */
export function groupSection(title, buildBody, { collapsed = false, onToggle = null } = {}) {
  const wrapper = document.createElement("section");
  wrapper.className = "plots-section";
  wrapper.append(foldHeader(title, { collapsed, onToggle, className: "plots-section-head" }));
  if (!collapsed) wrapper.append(buildBody());
  return wrapper;
}

/**
 * A checkbox with a text label. `onChange` receives the new checked state. Shared
 * by the sidebar item rows and the loss chart's options bar; `className` styles
 * each.
 */
export function checkboxRow(text, { checked, onChange, className }) {
  const row = document.createElement("label");
  row.className = className;
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = checked;
  box.addEventListener("change", () => onChange(box.checked));
  const span = document.createElement("span");
  span.textContent = text;
  row.append(box, span);
  return row;
}

/** A horizontal options bar (the `.plots-options` strip) wrapping the given rows. */
export function optionsBar(...rows) {
  const bar = document.createElement("div");
  bar.className = "plots-options";
  bar.append(...rows);
  return bar;
}

/**
 * A labelled range slider with a live value readout. `onChange` fires with the
 * final number when a drag ends or a keyboard step commits, while the readout
 * updates continuously during a drag. Optional `onDragStart`/`onDragEnd` bracket
 * a pointer drag (press to release) so a caller can pause work that would rebuild
 * the slider mid-drag; keyboard changes commit through `onChange` alone and need
 * no bracket. `format` renders the readout text.
 */
export function sliderRow(text, { value, min, max, step, onChange, onDragStart, onDragEnd, format, className }) {
  const row = document.createElement("label");
  row.className = className;
  const caption = document.createElement("span");
  caption.textContent = text;
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = min;
  slider.max = max;
  slider.step = step;
  slider.value = value;
  const readout = document.createElement("span");
  readout.className = "plots-slider-value";
  readout.textContent = format(value);
  slider.addEventListener("input", () => {
    readout.textContent = format(Number(slider.value));
  });
  slider.addEventListener("change", () => onChange(Number(slider.value)));
  if (onDragStart) slider.addEventListener("pointerdown", onDragStart);
  if (onDragEnd) {
    slider.addEventListener("pointerup", onDragEnd);
    slider.addEventListener("pointercancel", onDragEnd);
  }
  row.append(caption, slider, readout);
  return row;
}

/** A chart under a small label, for one item inside a group grid. */
export function chartCard(label, node) {
  const card = document.createElement("div");
  card.className = "plot-card";
  const title = document.createElement("h3");
  title.className = "plot-card-title";
  title.textContent = label;
  card.append(title, node);
  return card;
}

/** A muted placeholder when a series group is absent or still empty. */
export function note(text) {
  const paragraph = document.createElement("p");
  paragraph.className = "plots-note";
  paragraph.textContent = text;
  return paragraph;
}
