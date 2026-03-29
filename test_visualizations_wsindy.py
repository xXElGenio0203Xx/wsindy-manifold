import numpy as np
import pandas as pd

from run_visualizations import export_oscar_insight_data
from visualizations.compute_metrics import compute_test_metrics
from visualizations.time_analysis import generate_time_resolved_analysis


def test_export_oscar_insight_data_copies_wsindy_files(tmp_path):
    data_dir = tmp_path / "oscar_output" / "EXP_WS"
    output_dir = tmp_path / "predictions" / "EXP_WS"

    wsindy_dir = data_dir / "WSINDy"
    wsindy_dir.mkdir(parents=True)
    (data_dir / "test" / "test_000").mkdir(parents=True)

    (data_dir / "config_used.yaml").write_text("sim: {}\n")
    (data_dir / "summary.json").write_text("{}\n")
    (data_dir / "export_manifest.json").write_text("{}\n")
    (wsindy_dir / "test_results.csv").write_text("run_name,r2_reconstructed\n")
    (wsindy_dir / "runtime_profile.json").write_text("{}\n")
    (wsindy_dir / "multifield_model.json").write_text("{}\n")
    (wsindy_dir / "summary.json").write_text("{}\n")

    test_dir = data_dir / "test"
    (test_dir / "metadata.json").write_text("[]\n")
    (test_dir / "index_mapping.csv").write_text("idx,run_name\n0,test_000\n")
    (test_dir / "test_000" / "r2_vs_time_wsindy.csv").write_text("time,r2_reconstructed\n0.0,1.0\n")
    (test_dir / "test_000" / "metrics_summary_wsindy.json").write_text("{}\n")
    (test_dir / "test_000" / "density_metrics_wsindy.csv").write_text("t,mass_true\n0.0,1.0\n")

    export_oscar_insight_data(
        data_dir=data_dir,
        output_dir=output_dir,
        models_data={"wsindy": {"dir": wsindy_dir}},
    )

    oscar_data = output_dir / "oscar_data"
    assert (oscar_data / "WSINDy" / "test_results.csv").exists()
    assert (oscar_data / "WSINDy" / "runtime_profile.json").exists()
    assert (oscar_data / "WSINDy" / "multifield_model.json").exists()
    assert (oscar_data / "WSINDy" / "summary.json").exists()
    assert (oscar_data / "export_manifest.json").exists()
    assert (oscar_data / "test" / "test_000" / "r2_vs_time_wsindy.csv").exists()
    assert (oscar_data / "test" / "test_000" / "metrics_summary_wsindy.json").exists()
    assert (oscar_data / "test" / "test_000" / "density_metrics_wsindy.csv").exists()
    assert (oscar_data / "experiment_card.json").exists()


def test_time_analysis_backfills_missing_wsindy_columns(tmp_path):
    test_dir = tmp_path / "test"
    run_dir = test_dir / "test_000"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "time_analysis"

    pd.DataFrame(
        {
            "time": [0.0, 0.5, 1.0],
            "r2_reconstructed": [1.0, 0.8, 0.6],
        }
    ).to_csv(run_dir / "r2_vs_time_wsindy.csv", index=False)

    degradation_info = generate_time_resolved_analysis(
        test_metadata=[{"run_name": "test_000", "distribution": "uniform"}],
        test_dir=test_dir,
        mvar_dir=tmp_path / "WSINDy",
        data_dir=tmp_path,
        time_analysis_dir=output_dir,
        model_name="wsindy",
    )

    stats = pd.read_csv(output_dir / "r2_statistics_over_time.csv")
    assert set(stats["metric"]) == {"r2_reconstructed", "r2_latent", "r2_pod"}
    assert degradation_info["best_time"] == 0.0
    assert (output_dir / "r2_mean_over_time.png").exists()
    assert (output_dir / "survival_curves.png").exists()


def test_compute_metrics_treats_wsindy_forecast_start_as_relative_after_alignment(tmp_path):
    test_dir = tmp_path / "test"
    run_dir = test_dir / "test_000"
    run_dir.mkdir(parents=True)

    np.savez_compressed(
        run_dir / "density_true.npz",
        rho=np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]], dtype=np.float32),
        times=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        xgrid=np.array([0.0], dtype=np.float32),
        ygrid=np.array([0.0], dtype=np.float32),
    )
    np.savez_compressed(
        run_dir / "density_pred_wsindy.npz",
        rho=np.array([[[3.0]], [[4.0]]], dtype=np.float32),
        times=np.array([2.0, 3.0], dtype=np.float32),
        forecast_start_idx=2,
    )
    np.savez_compressed(
        run_dir / "trajectory.npz",
        traj=np.zeros((2, 1, 2), dtype=np.float32),
        vel=np.zeros((2, 1, 2), dtype=np.float32),
        times=np.array([2.0, 3.0], dtype=np.float32),
    )

    metrics_df, test_predictions, _ = compute_test_metrics(
        test_metadata=[{"run_name": "test_000", "distribution": "uniform"}],
        test_dir=test_dir,
        x_train_mean=np.zeros(1, dtype=np.float32),
        ic_types=["uniform"],
        output_dir=tmp_path / "out",
        model_name="wsindy",
    )

    assert metrics_df.loc[0, "forecast_start_idx"] == 0
    assert metrics_df.loc[0, "T_conditioning"] == 0
    assert test_predictions["test_000"]["forecast_start_idx"] == 0
