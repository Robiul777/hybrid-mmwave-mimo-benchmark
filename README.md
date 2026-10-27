# Hybrid mmWave MIMO Beamforming Under Imperfect CSI — Version 3

A reproducible Python benchmark for comparing fully digital and hybrid precoding in a narrowband multiuser mmWave MIMO downlink when the transmitter has imperfect channel-state information (CSI) and a limited number of RF chains.

## Engineering question

How do RF-chain constraints and CSI estimation error affect multiuser spectral efficiency, and can a simple uncertainty-aware RZF regularization rule reduce that loss compared with conventional hybrid precoding?

This repository is an experimental benchmark, not a claim that the uncertainty-aware method is universally superior. Results must be measured across the included sweeps.

## Implemented methods

- **Digital-RZF-Estimated-CSI:** fully digital regularized zero-forcing based on the estimated channel.
- **Digital-RZF-Uncertainty-Aware:** digital RZF with an additional CSI-error-dependent regularization term.
- **Hybrid-OMP-RZF:** greedy OMP selection of constant-modulus DFT analog beams plus digital RZF.
- **Hybrid-OMP-Robust-RZF:** OMP analog beams plus uncertainty-aware digital regularization.
- **DFT-Codebook-RZF:** gives each user its strongest unused DFT beam first, then fills extra RF chains by aggregate channel energy; this provides a distinct codebook-selection baseline from OMP.
- **DFT-Codebook-Robust-RZF:** uses the same DFT analog-beam selection as DFT-Codebook-RZF, but adds the CSI-NMSE-dependent regularization heuristic to the digital baseband RZF stage. It tests whether the heuristic also helps a stronger hybrid baseline.
- **Digital-RZF-Perfect-CSI:** idealized reference using the true channel at the transmitter.

All methods are evaluated using the **true channel** so that imperfect CSI affects the precoder design, not the scoring channel. Total transmit power is normalized consistently across methods.

## Default system model

| Parameter | Value |
|---|---:|
| BS antennas | 64-element half-wavelength ULA |
| Single-antenna users | 4 |
| Geometric paths per user | 3 |
| Default RF chains | 4 |
| SNR sweep | 0 to 30 dB in 5 dB steps |
| Target CSI NMSE sweep | −30, −20, −10, 0 dB |
| RF-chain sweep | 4, 6, 8 |
| Monte Carlo realizations | 100 per sweep point by default |

The channel is generated from a sparse geometric model with random user angle clusters and complex path gains. CSI error is complex Gaussian and scaled to a target ensemble NMSE.

## Metrics and outputs

The benchmark reports:

- Sum spectral efficiency (bit/s/Hz)
- Mean and 5th-percentile user spectral efficiency
- Fraction of users below 1 bit/s/Hz (defined here as an outage threshold)
- Actual CSI NMSE for each realization
- Mean and standard deviation across Monte Carlo realizations

Generated files include `metrics_raw.csv`, `summary.csv`, `user_rates.csv`, `experiment_config.json`, and plots for SNR, CSI error, RF-chain count, and per-user rate CDF.

## Run locally or in Colab

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --n-realizations 100 --seed 2026 --outdir results
```

See [`COLAB_START_HERE.md`](COLAB_START_HERE.md) for notebook instructions and [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for the model, assumptions, and interpretation guide.

## Research-quality checks

1. Confirm all unit tests pass.
2. Run a smoke test before the full Monte Carlo run.
3. Report means and variability, not only the best point.
4. Treat perfect-CSI digital RZF as an idealized reference, not a directly deployable baseline.
5. Do not assume uncertainty-aware regularization wins; inspect the measured results.
6. Re-run with multiple seeds before making strong claims.
7. Compare DFT-Codebook-Robust-RZF directly with DFT-Codebook-RZF and Hybrid-OMP-Robust-RZF; do not assume the added regularization improves every condition.

## Limitations

This is a narrowband, small-scale research simulator. It does not implement 3GPP TR 38.901, wideband OFDM, path loss, blockage, phase-shifter quantization, RF impairments, channel estimation pilots, or standards-compliant 5G NR signaling. It is suitable for algorithm prototyping and controlled comparisons, not as a standards conformance simulator.
