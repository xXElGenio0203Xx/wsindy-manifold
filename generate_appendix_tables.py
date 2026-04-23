"""Generate data-driven appendix table bodies for noise and N-convergence."""

from __future__ import annotations

from pathlib import Path

from appendix_results import (
    format_table_value,
    nconv_rows,
    noise_rows,
)

NOISE_OUTPUT = Path("thesis/appendices/noise_table_generated.tex")
NOISE_WSONLY_OUTPUT = Path("thesis/appendices/noise_wsonly_table_generated.tex")
NCONV_OUTPUT = Path("thesis/appendices/nconv_table_generated.tex")

# Regimes included in the full MVAR+LSTM+WSINDy noise table (slim data available).
_NOISE_FULL_REGIMES = {"Gas", "Blackhole"}
# Regimes included in the WSINDy-only noise table.
_NOISE_WSONLY_REGIMES = {"Supernova", "Crystal"}


def write_noise_table() -> None:
    """Write the gas+blackhole MVAR/LSTM/WSINDy noise table."""
    lines: list[str] = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Regime} & $\boldsymbol{\eta}$ & \textbf{MVAR} $R^2$ & \textbf{LSTM} $R^2$ & \textbf{WSINDy} $R^2_{\mathrm{wf}}$ \\",
        r"\midrule",
    ]
    previous_regime = None
    for row in noise_rows():
        if row["regime"] not in _NOISE_FULL_REGIMES:
            continue
        if previous_regime is not None and row["regime"] != previous_regime:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    str(row["regime"]),
                    str(row["eta_display"]),
                    format_table_value(row["mvar_mean"], math_negative=True),
                    format_table_value(row["lstm_mean"], math_negative=True),
                    format_table_value(row["rho_r2"], math_negative=True),
                ]
            )
            + r" \\"
        )
        previous_regime = row["regime"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    NOISE_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {NOISE_OUTPUT}")


def write_wsonly_table() -> None:
    """Write the supernova+crystal WSINDy-only noise table."""
    lines: list[str] = [
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"\textbf{Regime} & $\boldsymbol{\eta}$ & \textbf{WSINDy} $R^2_{\mathrm{wf},\rho}$ \\",
        r"\midrule",
    ]
    previous_regime = None
    for row in noise_rows():
        if row["regime"] not in _NOISE_WSONLY_REGIMES:
            continue
        if previous_regime is not None and row["regime"] != previous_regime:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    str(row["regime"]),
                    str(row["eta_display"]),
                    format_table_value(row["rho_r2"], math_negative=True),
                ]
            )
            + r" \\"
        )
        previous_regime = row["regime"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    NOISE_WSONLY_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {NOISE_WSONLY_OUTPUT}")


def write_nconv_table() -> None:
    lines = [
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"$N$ & \textbf{MVAR} $R^2$ & \textbf{LSTM} $R^2$ & \textbf{MVAR} $R^2$ (Gaussian IC) & \textbf{WSINDy} $R^2_{\mathrm{wf},\rho}$ & \textbf{WSINDy} $R^2_{\mathrm{wf},p}$ \\",
        r"\midrule",
    ]
    for row in nconv_rows():
        lines.append(
            " & ".join(
                [
                    str(row["n_value"]),
                    format_table_value(row["mvar_mean"], math_negative=True),
                    format_table_value(row["lstm_mean"], math_negative=True),
                    format_table_value(row["mvar_gaussian"], math_negative=True),
                    format_table_value(row["rho_r2"], math_negative=True),
                    format_table_value(row["px_r2"], math_negative=True),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    NCONV_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {NCONV_OUTPUT}")


def main() -> None:
    write_noise_table()
    write_wsonly_table()
    write_nconv_table()


if __name__ == "__main__":
    main()
