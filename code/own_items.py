"""The item legend for our own recordings: which pixel value in a *_id.png is which object.

The annotator writes one id image per labelled frame, where the pixel value is the index of
that item within its object set, in the order the items appear in data_collection/items.csv.
This module turns that back into the item's attributes, which is what makes a per-object
analysis possible: tone (bright / dark) crossed with the background gives contrast, which is
the one condition ISOD does not contain and the reason the recordings were made.
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS_CSV = f"{ROOT}/data_collection/items.csv"


def item_sets(path=ITEMS_CSV):
    """object set -> [item dicts], index 1..n matching the pixel values in *_id.png."""
    sets = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            s = sets[r["object_set"]]
            s.append(dict(idx=len(s) + 1, item_id=r["item_id"], label=r["label"],
                          tone=r["tone"], shape=r["shape"], specular=r["specular"],
                          height_mm=r["height_mm"], object_set=r["object_set"]))
    return dict(sets)


def contrast(tone, background):
    """How visible the item is in RGB: pale things on the white poster nearly vanish.

    Decided from the recording log (tone was written down per item before any prediction
    was seen, and the background is part of the run name), never from the images.
    """
    if background == "poster":
        return "low" if tone == "bright" else "high"
    return "high" if tone == "bright" else "medium"


if __name__ == "__main__":
    for s, items in sorted(item_sets().items()):
        print(f"{s} ({len(items)} items)")
        for it in items:
            print(f"  {it['idx']:2d} {it['label']:28s} {it['tone']:7s} {it['shape']}")
