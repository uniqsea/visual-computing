#!/usr/bin/env python3
"""
Collect all PNG sketches from class subfolders into a single folder.

Output files are renamed to: <class-name>-<index>.png
Example: apple-1.png, apple-2.png, ...

Usage:
  python collect_images.py
The script runs in-place from the dataset directory.
"""

import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    dest = root / "all_27"
    subfolders = [
        p for p in root.iterdir()
        if p.is_dir()
        and p.name != dest.name
        and not p.name.startswith(".")
    ]

    dest.mkdir(exist_ok=True)
    # Clean destination
    for item in dest.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        else:
            shutil.rmtree(item)

    total = 0
    for folder in sorted(subfolders):
        images = sorted(folder.glob("*.png"))
        if not images:
            continue
        for idx, src in enumerate(images, start=1):
            new_name = f"{folder.name}-{idx}{src.suffix}"
            target = dest / new_name
            shutil.copy2(src, target)
            total += 1

    print(f"Collected {total} files into {dest}")


if __name__ == "__main__":
    main()

