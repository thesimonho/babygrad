/*
 * How a series tag maps to a sidebar/section group and a short display label.
 * A parameter tag is scoped like `Linear_0/weights` or `Linear_0/weights/grad`;
 * the first path segment names the layer it belongs to, so that becomes the
 * group and the remainder is the item's label within that group.
 */

/** The group a tag belongs to: its first scope segment (`Linear_0/weights` → `Linear_0`). */
export function scopeGroup(tag) {
  const slash = tag.indexOf("/");
  return slash === -1 ? tag : tag.slice(0, slash);
}

/** The label a ridge item shows: the tag minus its group prefix (`Linear_0/weights` → `weights`). */
export function leafLabel(tag) {
  const slash = tag.indexOf("/");
  return slash === -1 ? tag : tag.slice(slash + 1);
}

/** The single group that holds every scalar series (loss, val_loss, metrics). */
export const SCALAR_GROUP = "Loss & metrics";

/** The group that holds the computed training diagnostics. */
export const DIAGNOSTICS_GROUP = "Diagnostics";
