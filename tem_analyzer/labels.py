"""A growing archive of hand-measured true diameters, kept per image.

A label is a fact about an image, not about a measurement: "the particle at
this position is truly this many nm across". It is bound to the image and a
position, never to a detection number - numbers change when the field is
re-analysed, positions do not - so the archive stays valid however much the
measuring algorithm later changes, and the program can always be re-scored
against it.

This module is only the store: load, save, and the few pure operations on it
(add-or-replace a label near a point, find the nearest, list an image's
labels). What the labels are used for lives in the GUI; here they are just
data, deliberately, because that is what the user asked to keep.
"""
import json
import os

from . import config

SCHEMA = 1

#: A label replaces an existing one within this many pixels, so re-clicking a
#: particle edits its value instead of stacking a second label on it.
REPLACE_RADIUS = 15.0


def default_path():
    """Where the archive lives, alongside the settings (see config.data_dir)."""
    return os.path.join(config.data_dir(), "labels.json")


def image_key(path, shape):
    """A stable per-image key: file name plus dimensions.

    The name is what the user recognises; the dimensions disambiguate two
    different images that happen to share a name, and catch a mismatch if a
    label file is applied to the wrong image.
    """
    name = os.path.basename(path) if path else "untitled"
    h, w = (shape[0], shape[1]) if shape is not None else (0, 0)
    return f"{name}|{int(h)}x{int(w)}"


def load(path):
    """Read the archive, or {} on any failure. Never raises."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    store = data.get("images", {})
    return store if isinstance(store, dict) else {}


def save(path, store):
    """Write the archive as JSON, creating the directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": SCHEMA, "images": store}, f,
                  indent=2, ensure_ascii=False)
    return path


def labels_for(store, key):
    """The list of labels recorded for one image (a live reference)."""
    return store.setdefault(key, [])


def add_or_replace(store, key, cx, cy, true_nm, prog_nm=None, when=None):
    """Record a label at (cx, cy), replacing any already within REPLACE_RADIUS.

    Returns (labels, replaced): the image's label list and whether an existing
    label was edited rather than a new one added.
    """
    labels = labels_for(store, key)
    entry = {"cx": int(round(cx)), "cy": int(round(cy)),
             "true_nm": float(true_nm),
             "prog_nm": (float(prog_nm) if prog_nm is not None else None),
             "time": when}
    for i, lab in enumerate(labels):
        if (lab["cx"] - cx) ** 2 + (lab["cy"] - cy) ** 2 <= REPLACE_RADIUS ** 2:
            labels[i] = entry
            return labels, True
    labels.append(entry)
    return labels, False


def remove_near(store, key, cx, cy, radius=REPLACE_RADIUS):
    """Delete the label nearest to (cx, cy) within `radius`. Returns True if one went."""
    labels = labels_for(store, key)
    best, best_d = None, radius ** 2
    for i, lab in enumerate(labels):
        d = (lab["cx"] - cx) ** 2 + (lab["cy"] - cy) ** 2
        if d <= best_d:
            best, best_d = i, d
    if best is None:
        return False
    labels.pop(best)
    return True


def nearest(labels, cx, cy):
    """The label closest to (cx, cy), or None if the list is empty."""
    best, best_d = None, None
    for lab in labels:
        d = (lab["cx"] - cx) ** 2 + (lab["cy"] - cy) ** 2
        if best_d is None or d < best_d:
            best, best_d = lab, d
    return best
