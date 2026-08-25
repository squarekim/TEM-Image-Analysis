"""Settings persistence must never be able to stop the program starting.

The calibration is worth keeping between launches, but a settings file is also
the easiest thing to corrupt - hand-edited, half-written by a crash, copied from
an older version. So the one hard rule is that loading can fail in any way and
the caller still gets a usable (empty) result, and the only values that come
back are ones the program knows. This checks the round trip and every failure
mode.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer import config  # noqa: E402


def main():
    failures = []
    d = tempfile.mkdtemp()

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            failures.append(name)

    # Round trip keeps known keys, drops unknown ones.
    p = os.path.join(d, "s.json")
    config.save_settings(p, {"wall_place": 1.07, "min_area": 200,
                             "hollow": True, "not_a_key": "x"})
    back = config.load_settings(p)
    check("known keys survive", back.get("wall_place") == 1.07
          and back.get("min_area") == 200 and back.get("hollow") is True)
    check("unknown key dropped", "not_a_key" not in back)

    # None is a real value (calibration cleared) and must round-trip.
    config.save_settings(p, {"wall_place": None})
    check("None wall_place round-trips",
          config.load_settings(p) == {"wall_place": None})

    # Missing file -> empty, no exception.
    check("missing file -> {}", config.load_settings(os.path.join(d, "no.json")) == {})

    # Corrupt file -> empty, no exception.
    bad = os.path.join(d, "bad.json")
    with open(bad, "w") as f:
        f.write("{ this is not json")
    check("corrupt file -> {}", config.load_settings(bad) == {})

    # A JSON value that is not an object -> empty.
    arr = os.path.join(d, "arr.json")
    with open(arr, "w") as f:
        f.write("[1, 2, 3]")
    check("non-object JSON -> {}", config.load_settings(arr) == {})

    # Saving creates the directory if it does not exist.
    nested = os.path.join(d, "a", "b", "s.json")
    config.save_settings(nested, {"wall_place": 0.9})
    check("save creates directories", os.path.exists(nested))

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
