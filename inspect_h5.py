#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List

try:
    import h5py
except Exception as e:
    print("Error: h5py is not installed. Please install it first.")
    sys.exit(1)


def summarize_h5(path: str, limit: int = 0) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "file": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "groups": [],
        "datasets": [],
        "counts": {"groups": 0, "datasets": 0},
        "total_param_bytes": 0,
    }

    def add_group(name: str):
        summary["groups"].append({"path": name})
        summary["counts"]["groups"] += 1

    def add_dataset(name: str, ds: "h5py.Dataset"):
        item = {
            "path": name,
            "shape": tuple(ds.shape) if ds.shape is not None else None,
            "dtype": str(ds.dtype),
        }
        try:
            item_bytes = int(ds.size) * int(ds.dtype.itemsize)
            item["n_elements"] = int(ds.size)
            item["bytes"] = item_bytes
            summary["total_param_bytes"] += item_bytes
        except Exception:
            pass
        summary["datasets"].append(item)
        summary["counts"]["datasets"] += 1

    with h5py.File(path, "r") as f:
        # Walk all objects
        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                add_group(name)
            elif isinstance(obj, h5py.Dataset):
                add_dataset(name, obj)
        f.visititems(visitor)

    # Optional limiting for printing
    if limit and limit > 0:
        summary["groups"] = summary["groups"][:limit]
        summary["datasets"] = summary["datasets"][:limit]

    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 (.h5) file structure and datasets.")
    parser.add_argument("file", help="Path to the .h5 file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of groups/datasets shown")
    parser.add_argument("--json", dest="json_out", help="Optional path to write JSON summary")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    try:
        summary = summarize_h5(args.file, limit=args.limit)
    except OSError as e:
        print(f"Error opening file: {e}")
        sys.exit(1)

    # Pretty print concise summary
    print("=== H5 Summary ===")
    print(f"File: {summary['file']}")
    print(f"Size: {summary['size_bytes']} bytes")
    print(f"Groups: {summary['counts']['groups']} | Datasets: {summary['counts']['datasets']}")
    print(f"Total parameter bytes (datasets): {summary['total_param_bytes']}")
    print("")

    def print_items(title: str, items: List[Dict[str, Any]]):
        print(title)
        for item in items:
            path = item.get("path")
            shape = item.get("shape")
            dtype = item.get("dtype")
            extra = []
            if "n_elements" in item:
                extra.append(f"n={item['n_elements']}")
            if "bytes" in item:
                extra.append(f"bytes={item['bytes']}")
            extra_str = (" | " + " | ".join(extra)) if extra else ""
            print(f" - {path} :: {shape} :: {dtype}{extra_str}")
        print("")

    print_items("Groups (truncated if limited):", summary["groups"])
    print_items("Datasets (truncated if limited):", summary["datasets"])

    if args.json_out:
        try:
            with open(args.json_out, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"JSON summary written to {args.json_out}")
        except Exception as e:
            print(f"Failed to write JSON: {e}")


if __name__ == "__main__":
    main()
