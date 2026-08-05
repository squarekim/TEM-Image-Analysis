"""Locate the Qt runtime shipped with PyQt5 before Qt is loaded.

PyQt5 normally finds its own platform plugins, but that lookup fails on some
Windows installs - most often after ``pip install --user``, where PyQt5 and
PyQt5-Qt5 land in the per-user site-packages. Qt then aborts at startup with
"Could not find the Qt platform plugin windows in ''". Pointing Qt at the
plugin directory that ships inside the installed package fixes it.

Import this before anything imports PyQt5.QtWidgets.
"""

import os


def configure():
    try:
        import PyQt5
    except ImportError:
        return

    base = os.path.dirname(os.path.abspath(PyQt5.__file__))
    for qt_dir in ("Qt5", "Qt"):
        plugins = os.path.join(base, qt_dir, "plugins")
        if not os.path.isdir(os.path.join(plugins, "platforms")):
            continue

        if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins

        # Qt's own DLLs sit next to the plugins; on Python 3.8+ they are only
        # resolvable if their directory is registered explicitly.
        qt_bin = os.path.join(base, qt_dir, "bin")
        if os.path.isdir(qt_bin):
            os.environ["PATH"] = qt_bin + os.pathsep + os.environ.get("PATH", "")
            add_dll_directory = getattr(os, "add_dll_directory", None)
            if add_dll_directory is not None:
                try:
                    add_dll_directory(qt_bin)
                except OSError:
                    pass
        return
