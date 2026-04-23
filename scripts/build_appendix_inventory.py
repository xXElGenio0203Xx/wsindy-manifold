#!/usr/bin/env python3
"""Build a row-level appendix inventory from local and remote result files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appendix_results import (
    CATALOGUE_EXPERIMENTS,
    NOISE_ROWS,
    NCONV_ROWS,
    REMOTE_HOST,
    REMOTE_OUTPUT_ROOT,
    TRACKED_RESULT_FILES,
    extract_catalogue_rom_metrics,
    local_file_flags,
)


def remote_file_flags(experiments: list[str], host: str, remote_root: str) -> dict[str, dict[str, bool]]:
    tracked_literal = json.dumps(list(TRACKED_RESULT_FILES))
    experiments_literal = json.dumps(experiments)
    remote_code = """
import json
import os

remote_root = REMOTE_ROOT_LITERAL
experiments = json.loads(EXPERIMENTS_LITERAL)
tracked = json.loads(TRACKED_LITERAL)
payload = {}
for experiment in experiments:
    base = os.path.join(remote_root, experiment)
    payload[experiment] = {
        relative_path: os.path.isfile(os.path.join(base, relative_path))
        for relative_path in tracked
    }
print(json.dumps(payload))
"""
    remote_code = remote_code.replace("REMOTE_ROOT_LITERAL", repr(remote_root))
    remote_code = remote_code.replace("EXPERIMENTS_LITERAL", repr(experiments_literal))
    remote_code = remote_code.replace("TRACKED_LITERAL", repr(tracked_literal))
    completed = subprocess.run(
        ["ssh", host, "python3", "-"],
        input=remote_code,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Remote inventory failed")
    return json.loads(completed.stdout)


def classify(local_ready: bool, remote_ready: bool) -> str:
    if local_ready:
        return "ready locally"
    if remote_ready:
        return "ready remotely, not synced"
    return "not finished"


def rom_ready(flags: dict[str, bool]) -> bool:
    return flags.get("MVAR/test_results.csv", False) and flags.get("LSTM/test_results.csv", False)


def ws_ready(flags: dict[str, bool]) -> bool:
    return any(
        flags.get(path, False)
        for path in ("summary.json", "WSINDy/test_results.csv", "WSINDy/multifield_model.json")
    )


def build_inventory(host: str, remote_root: str) -> dict[str, object]:
    experiments = set(CATALOGUE_EXPERIMENTS)
    for row in NOISE_ROWS:
        experiments.update(row.rom_candidates)
        experiments.add(row.ws_experiment)
    for row in NCONV_ROWS:
        experiments.add(row.experiment)

    ordered_experiments = sorted(experiments)
    local_flags = {experiment: local_file_flags(experiment, category="catalogue" if experiment in CATALOGUE_EXPERIMENTS else None) for experiment in ordered_experiments}
    remote_flags = remote_file_flags(ordered_experiments, host=host, remote_root=remote_root)

    catalogue_rows = []
    for experiment in CATALOGUE_EXPERIMENTS:
        local = local_flags[experiment]
        remote = remote_flags[experiment]
        recovered = extract_catalogue_rom_metrics(experiment)
        local_ready = recovered["mvar_mean"] is not None and recovered["lstm_mean"] is not None
        catalogue_rows.append(
            {
                "experiment": experiment,
                "local": local,
                "remote": remote,
                "recovered_metrics": recovered,
                "status": classify(local_ready, rom_ready(remote)),
            }
        )

    noise_rows = []
    for row in NOISE_ROWS:
        rom_local = {candidate: local_flags[candidate] for candidate in row.rom_candidates}
        rom_remote = {candidate: remote_flags[candidate] for candidate in row.rom_candidates}
        ws_local = local_flags[row.ws_experiment]
        ws_remote = remote_flags[row.ws_experiment]
        row_local_ready = any(rom_ready(flags) for flags in rom_local.values()) or ws_ready(ws_local)
        row_remote_ready = any(rom_ready(flags) for flags in rom_remote.values()) or ws_ready(ws_remote)
        noise_rows.append(
            {
                "label": f"{row.regime} eta={row.eta_display}",
                "rom_candidates": rom_local,
                "rom_remote": rom_remote,
                "ws_local": ws_local,
                "ws_remote": ws_remote,
                "status": classify(row_local_ready, row_remote_ready),
            }
        )

    nconv_rows = []
    for row in NCONV_ROWS:
        local = local_flags[row.experiment]
        remote = remote_flags[row.experiment]
        nconv_rows.append(
            {
                "label": f"N={row.n_value}",
                "experiment": row.experiment,
                "local": local,
                "remote": remote,
                "status": classify(rom_ready(local) or ws_ready(local), rom_ready(remote) or ws_ready(remote)),
            }
        )

    return {
        "host": host,
        "remote_root": remote_root,
        "tracked_files": list(TRACKED_RESULT_FILES),
        "catalogue": catalogue_rows,
        "noise": noise_rows,
        "n_convergence": nconv_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--remote-root", default=REMOTE_OUTPUT_ROOT)
    parser.add_argument("--output", default="thesis/appendices/appendix_inventory.json")
    args = parser.parse_args()

    inventory = build_inventory(host=args.host, remote_root=args.remote_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
