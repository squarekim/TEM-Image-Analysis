"""Persist analysis settings - above all the hand calibration - between runs.

The one value worth keeping across launches is the wall place from a hand
calibration: it is a fraction, independent of image or magnification, and
re-measuring it every session is exactly the friction the calibration was meant
to remove. The analysis parameters are saved with it so a whole configuration
can be carried to another machine or shared between people working on one
specimen.

What is deliberately NOT saved is the scale (nm per pixel): it is read from each
image's own scale bar and applying one image's scale to another would silently
corrupt every measurement. Settings are stored as plain JSON so they can be
read, edited, and version-controlled by hand.
"""
import json
import os

#: Bump only if an older file could be *misread* by newer code; unknown keys
#: are ignored on load, so adding a setting does not need a bump.
SCHEMA = 1

#: The settings the program knows how to save and restore. Scale is absent by
#: design (see the module docstring).
KEYS = (
    "wall_place", "min_area", "max_area", "circularity",
    "hollow", "watershed", "core", "shell", "sphere_edge",
    "edge_auto", "edge_level",
)


def data_dir():
    """Where persisted data (settings, labels) lives.

    Inside a git checkout it is the repo's ``data/`` folder, so a calibration
    or a label archive is tracked and travels with the project - commit and
    push and it is in git, on every clone. Installed elsewhere (a zip, a pip
    install with no repo around it) it falls back to a per-user folder in the
    home directory, so the program still works standalone.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(root, ".git")):
        return os.path.join(root, "data")
    return os.path.join(os.path.expanduser("~"), ".tem_analyzer")


def default_path():
    """Where the auto-persisted settings live, created on first save."""
    return os.path.join(data_dir(), "settings.json")


def save_settings(path, settings):
    """Write the given settings dict to `path` as JSON. Returns the path.

    Only known keys are written, so a caller passing extra state does not leak
    it into the file. The directory is created if missing.
    """
    data = {"schema": SCHEMA}
    for k in KEYS:
        if k in settings:
            data[k] = settings[k]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_settings(path):
    """Read settings from `path`, or return {} if it is missing or unreadable.

    A corrupt or partial file must never stop the program starting, so any
    failure - missing file, bad JSON, wrong types - yields an empty dict and
    the caller keeps its defaults. Only known keys are returned.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data[k] for k in KEYS if k in data}
