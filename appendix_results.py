from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_OUTPUT_ROOT = REPO_ROOT / "oscar_output"
LOCAL_SYSTEMATICS_ROOT = LOCAL_OUTPUT_ROOT / "systematics"
REMOTE_HOST = "oscar"
REMOTE_OUTPUT_ROOT = "/users/emaciaso/scratch/oscar_output"

TRACKED_RESULT_FILES = (
    "summary.json",
    "MVAR/test_results.csv",
    "LSTM/test_results.csv",
    "WSINDy/test_results.csv",
    "WSINDy/multifield_model.json",
    "WSINDy/identification_summary.json",
)

CATALOGUE_MAIN = {
    "NDYN04_gas_thesis_final",
    "NDYN05_blackhole_thesis_final",
    "NDYN06_supernova_thesis_final",
    "NDYN08_pure_vicsek_thesis_final",
    "NDYN04_gas_VS_thesis_final",
    "NDYN05_blackhole_VS_thesis_final",
    "NDYN06_supernova_VS_thesis_final",
    "NDYN07_crystal_thesis_final",
    "NDYN07_crystal_VS_thesis_final",
}

CATALOGUE_PRODUCTION_OVERRIDE = {
    "NDYN04_gas_thesis_final": (0.996, 0.996),
    "NDYN04_gas_VS_thesis_final": (0.558, 0.110),
    "NDYN05_blackhole_thesis_final": (0.994, 0.992),
    "NDYN05_blackhole_VS_thesis_final": (0.704, -0.214),
    "NDYN06_supernova_thesis_final": (0.683, 0.508),
    "NDYN06_supernova_VS_thesis_final": (0.244, 0.163),
    "NDYN07_crystal_thesis_final": (0.996, 0.585),
    "NDYN07_crystal_VS_thesis_final": (0.292, -0.779),
    "NDYN08_pure_vicsek_thesis_final": (0.654, 0.229),
}

# This list is the intended appendix row set, not an inference from local files.
CATALOGUE_EXPERIMENTS = (
    "DO_CS01_swarm_C01_l05",
    "DO_CS01_swarm_C01_l05_VS",
    "DO_CS02_swarm_C05_l3",
    "DO_CS03_swarm_C09_l3",
    "DO_DM01_dmill_C09_l05",
    "DO_DM01_dmill_C09_l05_VS",
    "DO_DR01_dring_C01_l01",
    "DO_DR01_dring_C01_l01_VS",
    "DO_DR02_dring_C09_l09",
    "DO_DR02_dring_C09_l09_VS",
    "DO_EC01_esccol_C2_l3",
    "DO_EC02_esccol_C3_l05",
    "DO_EC02_esccol_C3_l05_VS",
    "DO_ES01_escsym_C3_l09",
    "DO_ES01_escsym_C3_l09_VS",
    "DO_EU01_escuns_C2_l2",
    "DO_EU01_escuns_C2_l2_VS",
    "DO_EU02_escuns_C3_l3",
    "DO_SM01_mill_C05_l01",
    "DO_SM01_mill_C05_l01_VS",
    "DO_SM02_mill_C3_l01",
    "DO_SM02_mill_C3_l01_VS",
    "DO_SM03_mill_C2_l05",
    "DO_SM03_mill_C2_l05_VS",
    "NDYN01_crawl",
    "NDYN01_crawl_VS",
    "NDYN02_flock",
    "NDYN02_flock_VS",
    "NDYN04_gas_VS_thesis_final",
    "NDYN04_gas_VS_tier1_bic",
    "NDYN04_gas_VS_tier1_w5",
    "NDYN04_gas_preproc_sqrt_none",
    "NDYN04_gas_preproc_sqrt_scale",
    "NDYN04_gas_thesis_final",
    "NDYN04_gas_tier1_bic",
    "NDYN04_gas_tier1_w5",
    "NDYN05_blackhole_VS_thesis_final",
    "NDYN05_blackhole_VS_tier1_bic",
    "NDYN05_blackhole_VS_tier1_w5",
    "NDYN05_blackhole_thesis_final",
    "NDYN05_blackhole_tier1_bic",
    "NDYN05_blackhole_tier1_w5",
    "NDYN06_supernova_VS_thesis_final",
    "NDYN06_supernova_thesis_final",
    "NDYN06_supernova_tier1_bic",
    "NDYN06_supernova_tier1_w5",
    "NDYN07_crystal_VS_thesis_final",
    "NDYN07_crystal_thesis_final",
    "NDYN08_pure_vicsek_thesis_final",
    "NDYN09_longrange",
    "NDYN10_shortrange",
    "NDYN10_shortrange_VS",
    "NDYN13_chaos",
    "NDYN13_chaos_VS",
    "NDYN14_varspeed",
)


@dataclass(frozen=True)
class NoiseRow:
    regime: str
    eta_display: str
    base_experiment: str
    native: bool = False

    @property
    def rom_candidates(self) -> tuple[str, ...]:
        return (f"{self.base_experiment}_slim", self.base_experiment)

    @property
    def ws_experiment(self) -> str:
        return f"{self.base_experiment}_ws"


NOISE_ROWS = (
    NoiseRow("Gas", "0.15", "NDYN04_gas_eta0p15"),
    NoiseRow("Gas", "0.50", "NDYN04_gas_eta0p50"),
    NoiseRow("Gas", "1.00", "NDYN04_gas_eta1p00"),
    NoiseRow("Gas", "1.50", "NDYN04_gas_eta1p50", native=True),
    NoiseRow("Gas", "2.00", "NDYN04_gas_eta2p00"),
    NoiseRow("Blackhole", "0.05", "NDYN05_blackhole_eta0p05"),
    NoiseRow("Blackhole", "0.15", "NDYN05_blackhole_eta0p15", native=True),
    NoiseRow("Blackhole", "0.50", "NDYN05_blackhole_eta0p50"),
    NoiseRow("Blackhole", "1.00", "NDYN05_blackhole_eta1p00"),
    NoiseRow("Blackhole", "1.50", "NDYN05_blackhole_eta1p50"),
    NoiseRow("Supernova", "0.05", "NDYN06_supernova_eta0p05"),
    NoiseRow("Supernova", "0.15", "NDYN06_supernova_eta0p15", native=True),
    NoiseRow("Supernova", "0.50", "NDYN06_supernova_eta0p50"),
    NoiseRow("Supernova", "1.00", "NDYN06_supernova_eta1p00"),
    NoiseRow("Supernova", "1.50", "NDYN06_supernova_eta1p50"),
    NoiseRow("Crystal", "0.05", "NDYN07_crystal_eta0p05"),
    NoiseRow("Crystal", "0.10", "NDYN07_crystal_eta0p10", native=True),
    NoiseRow("Crystal", "0.50", "NDYN07_crystal_eta0p50"),
    NoiseRow("Crystal", "1.00", "NDYN07_crystal_eta1p00"),
    NoiseRow("Crystal", "1.50", "NDYN07_crystal_eta1p50"),
)


@dataclass(frozen=True)
class NConvergenceRow:
    n_value: int
    experiment: str


NCONV_ROWS = (
    NConvergenceRow(50, "NDYN08_pure_vicsek_N0050"),
    NConvergenceRow(100, "NDYN08_pure_vicsek_N0100"),
    NConvergenceRow(200, "NDYN08_pure_vicsek_N0200"),
    NConvergenceRow(300, "NDYN08_pure_vicsek_N0300"),
    NConvergenceRow(500, "NDYN08_pure_vicsek_N0500"),
    NConvergenceRow(1000, "NDYN08_pure_vicsek_N1000"),
)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def display_catalogue_name(experiment: str) -> str:
    return experiment.removesuffix("_thesis_final")


def iter_local_bases(experiment: str, *, category: str | None = None) -> Iterable[Path]:
    if category == "catalogue":
        yield LOCAL_SYSTEMATICS_ROOT / experiment
    yield LOCAL_OUTPUT_ROOT / experiment


def find_local_file(experiment: str, relative_path: str, *, category: str | None = None) -> Path | None:
    for base in iter_local_bases(experiment, category=category):
        path = base / relative_path
        if path.exists():
            return path
    return None


def local_file_flags(experiment: str, *, category: str | None = None) -> dict[str, bool]:
    return {
        relative_path: find_local_file(experiment, relative_path, category=category) is not None
        for relative_path in TRACKED_RESULT_FILES
    }


def parse_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_csv_field(path: Path | None, field: str) -> float | None:
    if path is None or not path.exists():
        return None
    values = [parse_float(row.get(field)) for row in read_csv_rows(path)]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return statistics.mean(values)


def csv_field_for_test_id(path: Path | None, field: str, test_id: str = "0") -> float | None:
    if path is None or not path.exists():
        return None
    rows = read_csv_rows(path)
    for row in rows:
        row_test_id = row.get("test_id")
        if row_test_id == test_id:
            return parse_float(row.get(field))
    if rows:
        return parse_float(rows[0].get(field))
    return None


def load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def extract_rom_metrics(experiment: str, *, category: str | None = None) -> dict[str, float | None]:
    mvar_path = find_local_file(experiment, "MVAR/test_results.csv", category=category)
    lstm_path = find_local_file(experiment, "LSTM/test_results.csv", category=category)
    return {
        "mvar_mean": mean_csv_field(mvar_path, "r2_reconstructed"),
        "lstm_mean": mean_csv_field(lstm_path, "r2_reconstructed"),
        "mvar_gaussian": csv_field_for_test_id(mvar_path, "r2_reconstructed", test_id="0"),
    }


def extract_wsindy_metrics(experiment: str, *, category: str | None = None) -> dict[str, object]:
    payload = None
    summary_path = find_local_file(experiment, "summary.json", category=category)
    summary_payload = load_json(summary_path)
    if summary_payload is not None:
        payload = summary_payload.get("wsindy", {}).get("discovered_pde")

    if not payload:
        model_payload = load_json(find_local_file(experiment, "WSINDy/multifield_model.json", category=category))
        if model_payload is not None and all(field in model_payload for field in ("rho", "px", "py")):
            payload = model_payload

    coefficients: dict[tuple[str, str], float] = {}
    rho_r2 = px_r2 = py_r2 = None
    if payload:
        rho_r2 = parse_float(payload.get("rho", {}).get("r2_weak"))
        px_r2 = parse_float(payload.get("px", {}).get("r2_weak"))
        py_r2 = parse_float(payload.get("py", {}).get("r2_weak"))
        for field in ("rho", "px", "py"):
            field_payload = payload.get(field, {})
            for term, value in field_payload.get("coefficients", {}).items():
                parsed = parse_float(value)
                if parsed is not None:
                    coefficients[(field, term)] = parsed

    results_csv = find_local_file(experiment, "WSINDy/test_results.csv", category=category)
    if results_csv is not None:
        if rho_r2 is None:
            rho_r2 = mean_csv_field(results_csv, "weak_r2_rho")
        if px_r2 is None:
            px_r2 = mean_csv_field(results_csv, "weak_r2_px")
        if py_r2 is None:
            py_r2 = mean_csv_field(results_csv, "weak_r2_py")

    return {
        "rho_r2": rho_r2,
        "px_r2": px_r2,
        "py_r2": py_r2,
        "coefficients": coefficients,
    }


def format_table_value(value: float | None, *, dagger_threshold: float = -2.0, math_negative: bool = False) -> str:
    if value is None:
        return "---"
    if value < dagger_threshold:
        return r"---$^\dagger$"
    rendered = f"{value:.3f}"
    if math_negative and value < 0:
        return f"${rendered}$"
    return rendered


def format_eta(row: NoiseRow) -> str:
    if row.native:
        return row.eta_display + r"$\star$"
    return row.eta_display


def catalogue_rows() -> list[dict[str, object]]:
    rows = []
    for experiment in CATALOGUE_EXPERIMENTS:
        if experiment in CATALOGUE_PRODUCTION_OVERRIDE:
            mvar_mean, lstm_mean = CATALOGUE_PRODUCTION_OVERRIDE[experiment]
        else:
            metrics = extract_rom_metrics(experiment, category="catalogue")
            mvar_mean = metrics["mvar_mean"]
            lstm_mean = metrics["lstm_mean"]
        rows.append(
            {
                "experiment": experiment,
                "display_name": display_catalogue_name(experiment),
                "mvar_mean": mvar_mean,
                "lstm_mean": lstm_mean,
                "is_main": experiment in CATALOGUE_MAIN,
            }
        )
    return rows


def noise_rows() -> list[dict[str, object]]:
    rows = []
    for row in NOISE_ROWS:
        mvar_mean = lstm_mean = None
        rom_source = None
        for candidate in row.rom_candidates:
            metrics = extract_rom_metrics(candidate)
            if mvar_mean is None and metrics["mvar_mean"] is not None:
                mvar_mean = metrics["mvar_mean"]
                rom_source = candidate
            if lstm_mean is None and metrics["lstm_mean"] is not None:
                lstm_mean = metrics["lstm_mean"]
                rom_source = rom_source or candidate
            if mvar_mean is not None and lstm_mean is not None:
                break

        ws_metrics = extract_wsindy_metrics(row.ws_experiment)
        rows.append(
            {
                "regime": row.regime,
                "eta_display": format_eta(row),
                "base_experiment": row.base_experiment,
                "rom_source": rom_source,
                "ws_source": row.ws_experiment,
                "mvar_mean": mvar_mean,
                "lstm_mean": lstm_mean,
                "rho_r2": ws_metrics["rho_r2"],
                "px_r2": ws_metrics["px_r2"],
                "py_r2": ws_metrics["py_r2"],
                "coefficients": ws_metrics["coefficients"],
            }
        )
    return rows


def nconv_rows() -> list[dict[str, object]]:
    rows = []
    for row in NCONV_ROWS:
        rom_metrics = extract_rom_metrics(row.experiment)
        ws_metrics = extract_wsindy_metrics(row.experiment)
        rows.append(
            {
                "n_value": row.n_value,
                "experiment": row.experiment,
                "mvar_mean": rom_metrics["mvar_mean"],
                "lstm_mean": rom_metrics["lstm_mean"],
                "mvar_gaussian": rom_metrics["mvar_gaussian"],
                "rho_r2": ws_metrics["rho_r2"],
                "px_r2": ws_metrics["px_r2"],
                "py_r2": ws_metrics["py_r2"],
                "coefficients": ws_metrics["coefficients"],
            }
        )
    return rows


def unique_sync_targets() -> dict[str, str]:
    targets: dict[str, str] = {}
    for experiment in CATALOGUE_EXPERIMENTS:
        targets[experiment] = "catalogue"
    for row in NOISE_ROWS:
        for experiment in row.rom_candidates:
            targets.setdefault(experiment, "noise")
        targets.setdefault(row.ws_experiment, "noise")
    for row in NCONV_ROWS:
        targets.setdefault(row.experiment, "nconv")
    return targets
