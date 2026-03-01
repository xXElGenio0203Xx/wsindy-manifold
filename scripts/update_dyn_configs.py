#!/usr/bin/env python3
"""Add LSTM + WSINDy blocks to DYN2-7 configs for the 3-method comparison."""
from pathlib import Path

experiments = {
    "DYN2_hypervelocity": {
        "comment": "Hypervelocity: 15x speed, strong Morse",
        "morse": True,
        "n_train": 30,
    },
    "DYN3_hypernoisy": {
        "comment": "Hypernoisy: eta=1.5, nearly random walkers",
        "morse": True,
        "n_train": 40,
    },
    "DYN4_blackhole": {
        "comment": "Blackhole: Ca=12 extreme attraction, Cr=0.1",
        "morse": True,
        "n_train": 30,
    },
    "DYN5_supernova": {
        "comment": "Supernova: Ca=0.1, Cr=10 explosive repulsion",
        "morse": True,
        "n_train": 30,
    },
    "DYN6_varspeed": {
        "comment": "Variable speed: forces affect magnitude not just heading",
        "morse": True,
        "n_train": 30,
    },
    "DYN7_pure_vicsek": {
        "comment": "Pure Vicsek: no Morse forces",
        "morse": False,
        "n_train": 30,
    },
}


def make_wsindy_block(exp_info):
    morse_str = "true" if exp_info["morse"] else "false"
    n_train = exp_info["n_train"]

    if exp_info["morse"]:
        lib_comment = (
            "  # rho: div_p, lap_rho, div_rho_gradPhi, lap_rho2, lap_rho3  (5)\n"
            "  # px:  px, p_sq_px, dx_rho, lap_px, rho_dx_Phi              (5)\n"
            "  # py:  py, p_sq_py, dy_rho, lap_py, rho_dy_Phi              (5)"
        )
    else:
        lib_comment = (
            "  # No Morse => drops Phi terms\n"
            "  # rho: div_p, lap_rho, lap_rho2, lap_rho3                   (4)\n"
            "  # px:  px, p_sq_px, dx_rho, lap_px                          (4)\n"
            "  # py:  py, p_sq_py, dy_rho, lap_py                          (4)"
        )

    return (
        "\n"
        "# " + "=" * 64 + "\n"
        f"# WSINDy PDE Discovery -- 3-field system\n"
        f"# {exp_info['comment']}\n"
        "# " + "=" * 64 + "\n"
        "wsindy:\n"
        "  enabled: true\n"
        '  mode: "multifield"\n'
        "\n"
        f"  n_train: {n_train}\n"
        "  subsample: 3\n"
        "  seed: 42\n"
        "\n"
        f"{lib_comment}\n"
        "  multifield_library:\n"
        f"    morse: {morse_str}\n"
        "    rich: false\n"
        "\n"
        "  model_selection:\n"
        "    n_ell: 12\n"
        "    p: [2, 2, 2]\n"
        "    stride: [2, 2, 2]\n"
        "\n"
        "  lambdas:\n"
        "    log_min: -5\n"
        "    log_max: 2\n"
        "    n_points: 60\n"
        "\n"
        "  bootstrap:\n"
        "    enabled: true\n"
        "    B: 50\n"
        "    ci_alpha: 0.05\n"
        "\n"
        "  forecast:\n"
        "    clip_negative: true\n"
        "    mass_conserve: true\n"
        '    method: "auto"\n'
    )


LSTM_BLOCK = """      enabled: true
      hidden_dim: 128
      n_layers: 2
      n_epochs: 200
      lr: 1.0e-3
      patience: 20"""


def main():
    root = Path(__file__).resolve().parent.parent / "configs"

    for exp_name, info in experiments.items():
        cfg_path = root / f"{exp_name}.yaml"
        if not cfg_path.exists():
            print(f"  SKIP {exp_name}: no config file")
            continue

        text = cfg_path.read_text()

        if "wsindy:" in text:
            print(f"  SKIP {exp_name}: wsindy block already exists")
            continue

        # Enable LSTM (replace disabled block)
        text = text.replace(
            "    lstm:\n      enabled: false",
            "    lstm:\n" + LSTM_BLOCK,
        )

        # Append WSINDy block
        text = text.rstrip() + "\n" + make_wsindy_block(info)

        cfg_path.write_text(text)
        print(f"  UPDATED {exp_name}: LSTM enabled + WSINDy block added")

    print("\nDone.")


if __name__ == "__main__":
    main()
