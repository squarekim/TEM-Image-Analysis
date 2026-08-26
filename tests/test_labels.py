"""The label archive stores facts about images and must not lose them.

Labels are bound to a position, so re-clicking the same particle edits its
value rather than piling a second label on it, and a wrong click can be taken
back. The store is keyed per image so two images keep their own labels, and -
like the settings file - a corrupt file can never stop the program: load
returns {} and the user starts with an empty archive.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer import labels  # noqa: E402


def main():
    failures = []

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            failures.append(name)

    store = {}
    key = labels.image_key("/some/path/img.jpg", (1024, 1024))

    # Add two labels far apart -> two entries.
    labels.add_or_replace(store, key, 100, 100, 95.0, prog_nm=92.0)
    labels.add_or_replace(store, key, 400, 400, 88.0, prog_nm=90.0)
    check("two distinct labels stored", len(labels.labels_for(store, key)) == 2)

    # Re-click near the first -> edit, not a third entry.
    _, replaced = labels.add_or_replace(store, key, 104, 98, 96.5, prog_nm=92.0)
    check("nearby click replaces", replaced
          and len(labels.labels_for(store, key)) == 2)
    check("value updated", labels.nearest(labels.labels_for(store, key), 100, 100)["true_nm"] == 96.5)

    # Remove near the second -> one left.
    check("remove near hits", labels.remove_near(store, key, 398, 402))
    check("one label left", len(labels.labels_for(store, key)) == 1)
    check("remove far misses", not labels.remove_near(store, key, 900, 900, radius=10))

    # A second image keeps its own labels.
    key2 = labels.image_key("/some/path/other.jpg", (512, 512))
    labels.add_or_replace(store, key2, 50, 50, 40.0)
    check("images separated", len(labels.labels_for(store, key)) == 1
          and len(labels.labels_for(store, key2)) == 1)

    # Round trip through a file.
    d = tempfile.mkdtemp()
    path = os.path.join(d, "labels.json")
    labels.save(path, store)
    back = labels.load(path)
    check("round trip preserves both images",
          set(back) == {key, key2}
          and back[key][0]["true_nm"] == 96.5)

    # Failure modes never raise.
    check("missing file -> {}", labels.load(os.path.join(d, "no.json")) == {})
    with open(os.path.join(d, "bad.json"), "w") as f:
        f.write("{ broken")
    check("corrupt file -> {}", labels.load(os.path.join(d, "bad.json")) == {})

    # Same image, different size -> different key (catches a wrong-image mix-up).
    check("size disambiguates key",
          labels.image_key("a.jpg", (100, 100)) != labels.image_key("a.jpg", (200, 200)))

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
