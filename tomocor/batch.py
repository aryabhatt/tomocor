"""Batch ZNCC/phase-correlation COR estimate over every HDF5 file in a directory.

For each file, loads the 0/180 deg pair from an /exchange-layout HDF5 file, computes the
ZNCC auto-COR (with confidence) and the phase-correlation COR, and writes one CSV row
per file. Sort the CSV by confidence to find files worth checking by eye.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

from .cor import auto_cor, denoise, find_center_pc, metric_view, shift_curve
from .io import load_pair

MIN_OVERLAP_FRAC = 0.3
EXTENSIONS = (".h5", ".hdf5", ".hdf", ".nxs")

FIELDS = ["filename", "cor", "confidence", "zncc", "pc_cor", "pc_peak", "error"]


def process_file(path, avg_tol_deg=0.0, median=3, sigma=3.0, min_overlap_frac=MIN_OVERLAP_FRAC):
    """Run the ZNCC and phase-correlation estimates on one file; returns a CSV-row dict."""
    pair = load_pair(path, avg_tol_deg)
    proj0 = pair.proj0
    projbf = np.ascontiguousarray(pair.proj180[:, ::-1])

    met0 = metric_view(denoise(proj0, median, sigma))
    metbf = metric_view(denoise(projbf, median, sigma))

    curve = shift_curve(met0, metbf, min_overlap_frac)
    auto = auto_cor(met0, metbf, curve=curve)
    pc = find_center_pc(met0, metbf)

    return {
        "filename": str(path),
        "cor": f"{auto.cor:.3f}",
        "confidence": f"{auto.confidence:.4f}",
        "zncc": f"{auto.zncc:.4f}",
        "pc_cor": f"{pc.cor:.3f}",
        "pc_peak": f"{pc.peak:.4f}",
        "error": "",
    }


def find_files(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in EXTENSIONS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory to scan recursively for HDF5 files")
    parser.add_argument(
        "-o", "--out", default=None, help="output CSV path (default: <directory name>.csv)"
    )
    parser.add_argument(
        "--tol", type=float, default=0.0, help="average frames within this many degrees (default: off)"
    )
    parser.add_argument("--median", type=int, default=3, help="median filter size (0 to disable)")
    parser.add_argument("--sigma", type=float, default=3.0, help="gaussian denoise sigma")
    args = parser.parse_args(argv)

    out = args.out if args.out is not None else f"{Path(args.directory).resolve().name}.csv"

    files = find_files(args.directory)
    if not files:
        print(f"No HDF5 files found under {args.directory}", file=sys.stderr)
        return 1

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for path in files:
            print(f"{path} ...", end=" ", file=sys.stderr, flush=True)
            try:
                row = process_file(path, args.tol, args.median, args.sigma)
                print(f"COR {row['cor']}, confidence {row['confidence']}", file=sys.stderr)
            except Exception as exc:
                row = {field: "" for field in FIELDS}
                row["filename"] = str(path)
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {row['error']}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            writer.writerow(row)

    print(f"\nWrote {len(files)} rows to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
