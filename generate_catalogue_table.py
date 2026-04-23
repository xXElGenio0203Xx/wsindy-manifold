"""Generate the appendix regime catalogue from the intended row manifest."""

from __future__ import annotations

from pathlib import Path

from appendix_results import (
    catalogue_rows,
    format_table_value,
    tex_escape,
)

OUTPUT_PATH = Path("thesis/appendices/catalogue_table_generated.tex")


def render_column(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\hline",
        r"Regime & MVAR $R^2$ & LSTM $R^2$ \\",
        r"\hline",
    ]
    for row in rows:
        name = tex_escape(str(row["display_name"]))
        mvar = format_table_value(row["mvar_mean"])
        lstm = format_table_value(row["lstm_mean"])
        if row["is_main"]:
            name = r"\textbf{" + name + "}"
            if row["mvar_mean"] is not None and row["mvar_mean"] > -2.0:
                mvar = r"\textbf{" + mvar + "}"
            if row["lstm_mean"] is not None and row["lstm_mean"] > -2.0:
                lstm = r"\textbf{" + lstm + "}"
        lines.append(f"{name} & {mvar} & {lstm} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return lines


def main() -> None:
    rows = catalogue_rows()
    split = (len(rows) + 1) // 2
    left = rows[:split]
    right = rows[split:]

    lines = [
        r"\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\noindent",
        r"\begin{minipage}[t]{0.49\textwidth}",
        *render_column(left),
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}",
        *render_column(right),
        r"\end{minipage}",
    ]

    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
