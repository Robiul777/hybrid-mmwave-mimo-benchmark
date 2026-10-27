"""Hybrid mmWave MIMO beamforming benchmark under imperfect CSI.

This is a reproducible narrowband research simulator, not a 3GPP-compliant
link-level implementation. It compares digital and hybrid precoding using
small-scale geometric mmWave channels and evaluates rates on the true channel.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_N_TX = 64
DEFAULT_N_USERS = 4
DEFAULT_N_PATHS = 3
DEFAULT_N_RF = 4
SNR_GRID_DB = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
CSI_NMSE_GRID_DB = np.array([-30, -20, -10, 0], dtype=float)
RF_CHAIN_GRID = np.array([4, 6, 8], dtype=int)


def ula_steering_vector(n_ant: int, angle_rad: float) -> np.ndarray:
    """Unit-norm half-wavelength ULA steering vector."""
    n = np.arange(n_ant)
    return np.exp(1j * np.pi * n * np.sin(angle_rad)) / np.sqrt(n_ant)


def make_dft_codebook(n_ant: int) -> np.ndarray:
    """Unitary DFT beam codebook with constant-modulus entries."""
    n = np.arange(n_ant)[:, None]
    m = np.arange(n_ant)[None, :]
    return (np.exp(1j * 2.0 * np.pi * n * m / n_ant) / np.sqrt(n_ant)).astype(np.complex128)


def generate_channel(
    rng: np.random.Generator,
    n_tx: int = DEFAULT_N_TX,
    n_users: int = DEFAULT_N_USERS,
    n_paths: int = DEFAULT_N_PATHS,
) -> np.ndarray:
    """Generate a K x Nt sparse geometric mmWave channel.

    Each user has a random cluster center and several nearby paths. Path gains
    are independent circular complex Gaussian variables with total expected
    power one before the sqrt(Nt) array-gain factor.
    """
    if n_tx < 2 or n_users < 1 or n_paths < 1:
        raise ValueError("n_tx >= 2, n_users >= 1, and n_paths >= 1 are required")

    h = np.zeros((n_users, n_tx), dtype=np.complex128)
    for user in range(n_users):
        center = rng.uniform(-np.pi / 3.0, np.pi / 3.0)
        angles = np.clip(center + rng.normal(0.0, 0.07, n_paths), -np.pi / 2, np.pi / 2)
        gains = (rng.normal(size=n_paths) + 1j * rng.normal(size=n_paths)) / np.sqrt(2.0 * n_paths)
        for gain, angle in zip(gains, angles):
            h[user] += np.sqrt(n_tx) * gain * ula_steering_vector(n_tx, float(angle))
    return h


def add_csi_error(
    h_true: np.ndarray,
    nmse_linear: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex Gaussian CSI error with the requested ensemble NMSE."""
    if nmse_linear < 0:
        raise ValueError("nmse_linear must be nonnegative")
    channel_power = float(np.mean(np.abs(h_true) ** 2))
    error_scale = np.sqrt(nmse_linear * channel_power / 2.0)
    error = error_scale * (
        rng.normal(size=h_true.shape) + 1j * rng.normal(size=h_true.shape)
    )
    return h_true + error


def normalize_precoder(w: np.ndarray) -> np.ndarray:
    """Normalize total transmit power so ||W||_F^2 = 1."""
    norm = float(np.linalg.norm(w, ord="fro"))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Precoder has zero or invalid norm")
    return w / norm


def rzf_precoder(
    h_est: np.ndarray,
    snr_db: float,
    nmse_linear: float = 0.0,
    uncertainty_aware: bool = False,
) -> np.ndarray:
    """Fully digital regularized zero-forcing precoder from estimated CSI."""
    k = h_est.shape[0]
    rho = 10.0 ** (snr_db / 10.0)
    alpha = k / max(rho, 1e-12)
    if uncertainty_aware:
        # Simple uncertainty-aware regularization heuristic, evaluated empirically.
        alpha += k * max(nmse_linear, 0.0)
    gram = h_est @ h_est.conj().T
    w = h_est.conj().T @ np.linalg.solve(gram + alpha * np.eye(k), np.eye(k))
    return normalize_precoder(w)


def omp_analog_precoder(
    h_est: np.ndarray,
    n_rf: int,
    codebook: np.ndarray,
) -> np.ndarray:
    """Greedy OMP selection of constant-modulus analog beams."""
    n_tx = h_est.shape[1]
    if not 1 <= n_rf <= n_tx:
        raise ValueError("n_rf must be between 1 and n_tx")
    target = h_est.conj().T  # Nt x K
    residual = target.copy()
    selected: list[int] = []
    for _ in range(n_rf):
        correlations = codebook.conj().T @ residual
        scores = np.linalg.norm(correlations, axis=1)
        if selected:
            scores[np.asarray(selected)] = -np.inf
        idx = int(np.argmax(scores))
        selected.append(idx)
        f_rf = codebook[:, selected]
        digital_approx = np.linalg.lstsq(f_rf, target, rcond=None)[0]
        residual = target - f_rf @ digital_approx
    return codebook[:, selected]


def dft_analog_precoder(
    h_est: np.ndarray,
    n_rf: int,
    codebook: np.ndarray,
) -> np.ndarray:
    """Select per-user strongest DFT beams, avoiding duplicate beams.

    One beam is allocated to each user's strongest available direction first.
    Any additional RF chains are filled using aggregate channel energy. This
    makes the codebook baseline distinct from OMP's joint residual minimization.
    """
    n_tx = h_est.shape[1]
    n_users = h_est.shape[0]
    if not 1 <= n_rf <= n_tx:
        raise ValueError("n_rf must be between 1 and n_tx")
    beamspace = codebook.conj().T @ h_est.conj().T  # [n_tx, n_users]
    power = np.abs(beamspace) ** 2
    selected: list[int] = []

    # Give each user its strongest unused codebook beam.
    user_order = np.argsort(np.max(power, axis=0))[::-1]
    for user in user_order:
        if len(selected) >= n_rf:
            break
        for beam_idx in np.argsort(power[:, user])[::-1]:
            idx = int(beam_idx)
            if idx not in selected:
                selected.append(idx)
                break

    # Use remaining RF chains for the strongest aggregate unused beams.
    aggregate = np.sum(power, axis=1)
    for beam_idx in np.argsort(aggregate)[::-1]:
        if len(selected) >= n_rf:
            break
        idx = int(beam_idx)
        if idx not in selected:
            selected.append(idx)
    return codebook[:, selected]


def hybrid_precoder(
    h_est: np.ndarray,
    snr_db: float,
    n_rf: int,
    nmse_linear: float,
    codebook: np.ndarray,
    selection: str = "omp",
    uncertainty_aware: bool = False,
) -> np.ndarray:
    """Build an analog-plus-digital hybrid precoder."""
    if n_rf < h_est.shape[0]:
        raise ValueError(
            "For simultaneous one-stream-per-user transmission, n_rf must be "
            "at least the number of users."
        )
    if selection == "omp":
        f_rf = omp_analog_precoder(h_est, n_rf, codebook)
    elif selection == "dft":
        f_rf = dft_analog_precoder(h_est, n_rf, codebook)
    else:
        raise ValueError(f"Unknown beam selection method: {selection}")

    h_eff = h_est @ f_rf
    k = h_est.shape[0]
    rho = 10.0 ** (snr_db / 10.0)
    alpha = k / max(rho, 1e-12)
    if uncertainty_aware:
        alpha += k * max(nmse_linear, 0.0)
    gram = h_eff @ h_eff.conj().T
    w_bb = h_eff.conj().T @ np.linalg.solve(gram + alpha * np.eye(k), np.eye(k))
    return normalize_precoder(f_rf @ w_bb)


def rate_metrics(
    h_true: np.ndarray,
    precoder: np.ndarray,
    snr_db: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate user rates on the true channel, including multiuser interference."""
    rho = 10.0 ** (snr_db / 10.0)
    effective = h_true @ precoder
    power = np.abs(effective) ** 2
    desired = np.diag(power)
    interference = np.maximum(power.sum(axis=1) - desired, 0.0)
    sinr = rho * desired / (rho * interference + 1.0)
    user_rates = np.log2(1.0 + sinr)
    metrics = {
        "sum_se": float(np.sum(user_rates)),
        "mean_user_rate": float(np.mean(user_rates)),
        "p5_user_rate": float(np.percentile(user_rates, 5)),
        "outage_user_fraction": float(np.mean(user_rates < 1.0)),
    }
    return metrics, user_rates


def csi_nmse_db(h_est: np.ndarray, h_true: np.ndarray) -> float:
    num = np.sum(np.abs(h_est - h_true) ** 2)
    den = np.sum(np.abs(h_true) ** 2) + 1e-12
    return float(10.0 * np.log10(num / den + 1e-12))


def evaluate_methods(
    h_true: np.ndarray,
    h_est: np.ndarray,
    snr_db: float,
    nmse_linear: float,
    n_rf: int,
    codebook: np.ndarray,
) -> dict[str, tuple[dict[str, float], np.ndarray]]:
    """Evaluate digital and hybrid baselines on one channel realization."""
    precoders: dict[str, np.ndarray] = {
        "Digital-RZF-Estimated-CSI": rzf_precoder(h_est, snr_db),
        "Digital-RZF-Uncertainty-Aware": rzf_precoder(
            h_est, snr_db, nmse_linear, uncertainty_aware=True
        ),
        "Hybrid-OMP-RZF": hybrid_precoder(
            h_est, snr_db, n_rf, nmse_linear, codebook, selection="omp"
        ),
        "Hybrid-OMP-Robust-RZF": hybrid_precoder(
            h_est, snr_db, n_rf, nmse_linear, codebook,
            selection="omp", uncertainty_aware=True
        ),
        "DFT-Codebook-RZF": hybrid_precoder(
            h_est, snr_db, n_rf, nmse_linear, codebook, selection="dft"
        ),
        "DFT-Codebook-Robust-RZF": hybrid_precoder(
            h_est, snr_db, n_rf, nmse_linear, codebook,
            selection="dft", uncertainty_aware=True
        ),
        "Digital-RZF-Perfect-CSI": rzf_precoder(h_true, snr_db),
    }
    return {name: rate_metrics(h_true, w, snr_db) for name, w in precoders.items()}


def _record_condition(
    raw_rows: list[dict],
    user_rows: list[dict],
    rng: np.random.Generator,
    codebook: np.ndarray,
    n_realizations: int,
    sweep: str,
    x_name: str,
    x_value: float,
    snr_db: float,
    csi_nmse_db_value: float,
    n_rf: int,
) -> None:
    nmse_linear = 10.0 ** (csi_nmse_db_value / 10.0)
    for realization in range(n_realizations):
        h_true = generate_channel(rng)
        h_est = add_csi_error(h_true, nmse_linear, rng)
        actual_nmse = csi_nmse_db(h_est, h_true)
        results = evaluate_methods(h_true, h_est, snr_db, nmse_linear, n_rf, codebook)
        for method, (metrics, user_rates) in results.items():
            row = {
                "sweep": sweep,
                "x_name": x_name,
                "x_value": float(x_value),
                "snr_db": float(snr_db),
                "csi_nmse_db_target": float(csi_nmse_db_value),
                "csi_nmse_db_actual": float(actual_nmse),
                "n_rf": int(n_rf),
                "realization": int(realization),
                "method": method,
                **metrics,
            }
            raw_rows.append(row)
            for user_id, user_rate in enumerate(user_rates):
                user_rows.append({
                    "sweep": sweep,
                    "x_name": x_name,
                    "x_value": float(x_value),
                    "snr_db": float(snr_db),
                    "csi_nmse_db_target": float(csi_nmse_db_value),
                    "n_rf": int(n_rf),
                    "realization": int(realization),
                    "method": method,
                    "user_id": int(user_id),
                    "user_rate": float(user_rate),
                })


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(raw: list[dict], users: list[dict], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(raw)
    summary = (
        df.groupby(["sweep", "x_name", "x_value", "method", "n_rf"], as_index=False)
        .agg(
            sum_se_mean=("sum_se", "mean"),
            sum_se_std=("sum_se", "std"),
            mean_user_rate_mean=("mean_user_rate", "mean"),
            p5_user_rate_mean=("p5_user_rate", "mean"),
            outage_fraction_mean=("outage_user_fraction", "mean"),
            actual_csi_nmse_db_mean=("csi_nmse_db_actual", "mean"),
        )
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    plot_specs = [
        ("snr", "x_value", "Spectral efficiency vs SNR", "SNR (dB)", "spectral_efficiency_vs_snr.png"),
        ("csi", "x_value", "Spectral efficiency vs CSI error", "Target CSI NMSE (dB)", "spectral_efficiency_vs_csi_error.png"),
        ("rf", "x_value", "Spectral efficiency vs RF-chain count", "Number of RF chains", "spectral_efficiency_vs_rf_chains.png"),
    ]
    for sweep, xcol, title, xlabel, filename in plot_specs:
        sub = summary[summary["sweep"] == sweep]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for method in sorted(sub["method"].unique()):
            part = sub[sub["method"] == method].sort_values(xcol)
            ax.errorbar(
                part[xcol], part["sum_se_mean"], yerr=part["sum_se_std"].fillna(0),
                marker="o", capsize=2, linewidth=1.6, label=method
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sum spectral efficiency (bit/s/Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    user_df = pd.DataFrame(users)
    cdf_data = user_df[
        (user_df["sweep"] == "csi") &
        np.isclose(user_df["x_value"], -10.0) &
        (user_df["n_rf"] == DEFAULT_N_RF)
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in sorted(cdf_data["method"].unique()):
        values = np.sort(cdf_data.loc[cdf_data["method"] == method, "user_rate"].to_numpy())
        if values.size:
            y = np.arange(1, values.size + 1) / values.size
            ax.plot(values, y, label=method, linewidth=1.6)
    ax.set_title("Per-user rate CDF (20 dB SNR, target CSI NMSE = −10 dB)")
    ax.set_xlabel("User spectral efficiency (bit/s/Hz)")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / "per_user_rate_cdf.png", dpi=180)
    plt.close(fig)


def run_experiment(n_realizations: int, seed: int, out_dir: Path) -> None:
    if n_realizations < 2:
        raise ValueError("Use at least two realizations to estimate variability")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    codebook = make_dft_codebook(DEFAULT_N_TX)
    raw_rows: list[dict] = []
    user_rows: list[dict] = []

    print(f"Monte Carlo realizations per sweep point: {n_realizations}")
    print(f"Array: {DEFAULT_N_TX} antennas; users: {DEFAULT_N_USERS}; paths/user: {DEFAULT_N_PATHS}")
    print(f"SNR sweep: {SNR_GRID_DB.tolist()} dB; CSI NMSE sweep: {CSI_NMSE_GRID_DB.tolist()} dB")

    # Sweep 1: SNR, with moderate CSI error and four RF chains.
    for i, snr in enumerate(SNR_GRID_DB):
        print(f"SNR sweep {i + 1}/{len(SNR_GRID_DB)}: {snr:g} dB")
        _record_condition(raw_rows, user_rows, rng, codebook, n_realizations,
                          "snr", "snr_db", snr, float(snr), -20.0, DEFAULT_N_RF)

    # Sweep 2: CSI uncertainty at 20 dB SNR and four RF chains.
    for i, csi_db in enumerate(CSI_NMSE_GRID_DB):
        print(f"CSI sweep {i + 1}/{len(CSI_NMSE_GRID_DB)}: target NMSE {csi_db:g} dB")
        _record_condition(raw_rows, user_rows, rng, codebook, n_realizations,
                          "csi", "csi_nmse_db", csi_db, 20.0, float(csi_db), DEFAULT_N_RF)

    # Sweep 3: RF-chain count at 20 dB SNR and −20 dB target CSI NMSE.
    for i, n_rf in enumerate(RF_CHAIN_GRID):
        print(f"RF-chain sweep {i + 1}/{len(RF_CHAIN_GRID)}: {int(n_rf)} chains")
        _record_condition(raw_rows, user_rows, rng, codebook, n_realizations,
                          "rf", "n_rf", float(n_rf), 20.0, -20.0, int(n_rf))

    _write_csv(out_dir / "metrics_raw.csv", raw_rows)
    _write_csv(out_dir / "user_rates.csv", user_rows)
    _make_plots(raw_rows, user_rows, out_dir)
    config = {
        "seed": seed,
        "n_realizations_per_sweep_point": n_realizations,
        "n_tx": DEFAULT_N_TX,
        "n_users": DEFAULT_N_USERS,
        "n_paths_per_user": DEFAULT_N_PATHS,
        "default_n_rf": DEFAULT_N_RF,
        "snr_grid_db": SNR_GRID_DB.tolist(),
        "csi_nmse_grid_db": CSI_NMSE_GRID_DB.tolist(),
        "rf_chain_grid": RF_CHAIN_GRID.tolist(),
        "channel_model": "narrowband sparse geometric ULA, 3 paths per user",
        "precoder_power_normalization": "Frobenius norm squared equals 1",
        "methods": [
            "Digital-RZF-Estimated-CSI",
            "Digital-RZF-Uncertainty-Aware",
            "Hybrid-OMP-RZF",
            "Hybrid-OMP-Robust-RZF",
            "DFT-Codebook-RZF",
            "DFT-Codebook-Robust-RZF",
            "Digital-RZF-Perfect-CSI",
        ],
        "uncertainty_aware_regularization": "alpha = K / SNR_linear + K * target_CSI_NMSE_linear; target NMSE is supplied to the algorithm in simulation",
        "limitations": [
            "Not a 3GPP-compliant or wideband OFDM simulator",
            "No path loss, blockage, phase-shifter quantization, or hardware nonlinearity",
            "Uncertainty-aware RZF uses a simple regularization heuristic, not a proven optimum",
        ],
    }
    (out_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved metrics, summaries, configuration, and plots to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-realizations", type=int, default=100,
                        help="Monte Carlo realizations per sweep point (default: 100)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_experiment(args.n_realizations, args.seed, args.outdir)


if __name__ == "__main__":
    main()
