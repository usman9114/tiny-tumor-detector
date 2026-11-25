"""Download the Ultralytics brain-tumor detection set and build a fixed
train/val/test split in YOLO format.

The original release only ships train (994) and val (121). We pool all 1115
images and re-split 80/10/10 with a fixed seed so every model is scored on the
same held-out test set. Files named like `00074_249.jpg` carry a patient id
before the underscore; those are kept together in one split to avoid leaking
near-identical neighbouring slices across splits. Other file names carry no
patient id and are treated as independent images (documented limitation).

Usage: python data/prepare.py [--out datasets/brain-tumor] [--seed 0]
"""
import argparse
import random
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
ROOT = Path(__file__).resolve().parent.parent


def group_key(name: str) -> str:
    m = re.match(r"^(\d{5})_\d+\.jpg$", name)
    return m.group(1) if m else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "datasets" / "brain-tumor"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    a = ap.parse_args()
    out = Path(a.out)
    raw = ROOT / "data" / "brain-tumor-raw"

    if not raw.exists():
        raw.mkdir(parents=True)
        zpath = raw / "brain-tumor.zip"
        print("downloading", URL)
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
        zpath.unlink()

    images = sorted(p for p in raw.rglob("*.jpg"))
    groups = defaultdict(list)
    for p in images:
        groups[group_key(p.name)].append(p)
    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)

    total = len(images)
    targets = [f * total for f in a.fracs]
    split_of = {}
    counts = [0, 0, 0]
    for k in keys:
        # fill the split that is furthest below its target
        i = max(range(3), key=lambda j: targets[j] - counts[j])
        split_of[k] = ("train", "val", "test")[i]
        counts[i] += len(groups[k])

    if out.exists():
        shutil.rmtree(out)
    stats = {s: [0, 0] for s in ("train", "val", "test")}  # images, boxes
    for k, files in groups.items():
        s = split_of[k]
        for img in files:
            lbl = raw / "labels" / img.parent.name / (img.stem + ".txt")
            (out / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
            shutil.copy(img, out / "images" / s / img.name)
            nb = 0
            if lbl.exists():
                lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
                nb = len(lines)
                (out / "labels" / s / lbl.name).write_text("\n".join(lines) + "\n")
            stats[s][0] += 1
            stats[s][1] += nb

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: tumor\n"
    )
    print(f"seed={a.seed}  images={total}  groups={len(keys)}  classes: 0=tumor")
    for s, (ni, nb) in stats.items():
        print(f"{s:5s} images={ni:4d}  tumor boxes={nb:4d}")
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
