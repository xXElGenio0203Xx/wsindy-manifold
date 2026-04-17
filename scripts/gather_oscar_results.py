#!/usr/bin/env python3
"""Scan all OSCAR experiment dirs and extract R² metrics."""
import csv
import os
import sys


def get_r2(csvpath):
    if not os.path.exists(csvpath):
        return None, None
    rows = list(csv.DictReader(open(csvpath)))
    if not rows:
        return None, None
    vals = [float(r["r2_reconstructed"]) for r in rows if r.get("r2_reconstructed")]
    if not vals:
        return None, None
    return round(sum(vals) / len(vals), 4), len(vals)


def check_wsindy(d):
    if os.path.exists(os.path.join(d, "WSINDy", "identification_results.csv")):
        return "Y"
    if os.path.exists(os.path.join(d, "WSINDy", "identification_summary.json")):
        return "Y(j)"
    return "-"


def check_train(d):
    """Check if training data exists (sim completed)."""
    if os.path.isdir(os.path.join(d, "train")):
        return "Y"
    return "-"


def check_pod(d):
    if os.path.exists(os.path.join(d, "rom_common", "pod_basis.npz")):
        return "Y"
    return "-"


def main():
    roots = [
        "/users/emaciaso/wsindy-manifold/oscar_output",
        "/users/emaciaso/scratch/oscar_output",
    ]

    seen = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        loc = "scratch" if "scratch" in root else "HOME"
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if not os.path.isdir(full):
                continue
            mvar_r2, mvar_n = get_r2(os.path.join(full, "MVAR", "test_results.csv"))
            lstm_r2, lstm_n = get_r2(os.path.join(full, "LSTM", "test_results.csv"))
            ws = check_wsindy(full)
            train = check_train(full)
            pod = check_pod(full)
            has_anything = (
                mvar_r2 is not None
                or lstm_r2 is not None
                or ws != "-"
                or train == "Y"
            )
            if has_anything:
                # Prefer HOME over scratch
                if entry not in seen or loc == "HOME":
                    seen[entry] = (mvar_r2, mvar_n, lstm_r2, lstm_n, ws, train, pod, loc)

    # Print table
    hdr = f"{'Experiment':<45} {'MVAR_R2':>8} {'n':>3} {'LSTM_R2':>8} {'n':>3} {'WS':>4} {'Sim':>4} {'POD':>4} {'Loc':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(seen.keys()):
        mr2, mn, lr2, ln, ws, train, pod, loc = seen[name]
        mr = f"{mr2:.4f}" if mr2 is not None else "   -"
        ms = str(mn) if mn else ""
        lr = f"{lr2:.4f}" if lr2 is not None else "   -"
        ls = str(ln) if ln else ""
        print(f"{name:<45} {mr:>8} {ms:>3} {lr:>8} {ls:>3} {ws:>4} {train:>4} {pod:>4} {loc:>7}")


if __name__ == "__main__":
    main()
