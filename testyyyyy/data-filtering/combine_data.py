"""
combine_data.py — combine good + bad data from data_filtering_2 and data_filtering_3
into a single deduplicated JSON, sorted by example_idx.

Usage:
    python data-filtering/combine_data.py [--out output.json]
"""

import argparse
import json
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "local-volume")

SOURCE_FILES = [
    "data_filtering_2/Goedel_good_Numina_again.json",
    "data_filtering_2/Goedel_bad_Numina_again.json",
    "data_filtering_3/Goedel_good_Numina_again.json",
    "data_filtering_3/Goedel_bad_Numina_again.json",
]


def load_entries(path):
    data = json.load(open(path))
    return data if isinstance(data, list) else data["results"]


def combine(source_files):
    seen = set()
    combined = []
    for rel_path in source_files:
        path = os.path.join(BASE, rel_path)
        entries = load_entries(path)
        before = len(combined)
        for entry in entries:
            idx = entry["example_idx"]
            if idx not in seen:
                seen.add(idx)
                combined.append(entry)
        added = len(combined) - before
        skipped = len(entries) - added
        print(f"  {rel_path}: {len(entries)} entries, {added} new" + (f", {skipped} duplicates skipped" if skipped else ""))
    combined.sort(key=lambda e: e["example_idx"])
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(BASE, "Numina_combined_300.json"))
    args = parser.parse_args()

    combined = combine(SOURCE_FILES)
    with open(args.out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved {len(combined)} entries → {args.out}")


if __name__ == "__main__":
    main()
