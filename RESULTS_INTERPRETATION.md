# Version 3 full-run results: DFT robust-RZF baseline

## Run configuration

- Seed: `2026`
- Monte Carlo realizations: 100 per sweep point
- Array: 64 transmit antennas, 4 single-antenna users, 3 paths per user
- Sweeps: SNR 0–30 dB; target CSI NMSE −30 to 0 dB; RF-chain count 4, 6, 8
- Unit tests: 11 passed
- Raw metrics: 9,800 rows across seven methods
- Summary metrics: 98 rows
- Per-user rate samples: 39,200 rows

## What changed

`DFT-Codebook-Robust-RZF` was added. It uses exactly the same DFT analog-beam selection as `DFT-Codebook-RZF`; only the digital RZF regularization changes. It uses the same heuristic as the OMP robust method:

`alpha = K / SNR_linear + K * target_CSI_NMSE_linear`

The target CSI NMSE is supplied to the algorithm in this simulation. A real transmitter would need an estimate of the CSI-error level; this benchmark does not estimate it.

## CSI sweep at 20 dB SNR and four RF chains

Mean sum spectral efficiency (bit/s/Hz):

| Target CSI NMSE | DFT-RZF | DFT robust-RZF | Difference | OMP-RZF | OMP robust-RZF | Digital estimated CSI |
|---:|---:|---:|---:|---:|---:|---:|
| -30 dB | 35.381 | 35.384 | +0.003 | 21.855 | 22.164 | 39.441 |
| -20 dB | 32.375 | 32.409 | +0.034 | 18.078 | 20.265 | 36.442 |
| -10 dB | 24.649 | 24.960 | +0.311 | 12.299 | 17.856 | 28.617 |
| 0 dB | 13.406 | 15.086 | +1.680 | 7.240 | 12.447 | 17.651 |

## Paired comparisons

`paired_comparisons.csv` contains paired differences between each robust method and its non-robust counterpart. Each difference uses the same 100 channel realizations for both methods at that sweep point. The reported 95% intervals are unadjusted, t-based Monte Carlo confidence intervals for the mean paired difference; they are not evidence of generalization to other channel models.

- At target CSI NMSE −30 dB, DFT robust-RZF gains only about 0.003 bit/s/Hz over DFT-RZF.
- At −20 dB, the gain is about 0.034 bit/s/Hz.
- At −10 dB, the gain is about 0.311 bit/s/Hz.
- At 0 dB, the gain is about 1.680 bit/s/Hz.

This supports a limited conclusion: uncertainty-aware regularization of the DFT hybrid precoder becomes more useful as CSI quality worsens. The effect is very small at high CSI quality. In the SNR sweep, which fixes target CSI NMSE at −20 dB, the gain remains small (roughly 0.005–0.037 bit/s/Hz) and the paired interval at 30 dB includes zero.

The robust OMP method improves more strongly over ordinary OMP, especially at poor CSI quality. However, it still trails the DFT-codebook methods in this simulator. The new method is a stronger comparison baseline, not evidence that the current proposal is best overall.

## Interpretation limits

- The perfect-CSI digital RZF method is an idealized reference, not a strict capacity bound.
- The channel is narrowband and geometric, with a ULA and no path loss, blockage, phase-shifter quantization, or RF nonlinearity.
- The CSI error is synthetic complex Gaussian noise scaled to a target NMSE.
- Results use one seed. Repeat the benchmark with additional seeds before making strong claims.
- The regularization rule is a heuristic, not a derived optimum. Tune it only on separate validation conditions, then freeze it before final testing.
