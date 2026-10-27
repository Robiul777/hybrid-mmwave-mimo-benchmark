# Research notes

## Hypothesis

When transmitter CSI is noisy, the regularization level used by RZF affects the trade-off between noise enhancement and residual multiuser interference. A simple uncertainty-aware regularization term may improve robustness in some regimes, while hybrid precoding introduces an additional loss due to the limited analog beam subspace.

## Methods

For each channel realization, the benchmark generates a true multiuser channel `H`, adds complex Gaussian error to obtain `H_hat`, designs each precoder from `H_hat` (except the perfect-CSI reference), normalizes total transmit power, and evaluates all methods using `H`.

The sparse geometric channel uses a half-wavelength ULA and three paths per user. OMP greedily selects DFT beams to approximate the estimated channel subspace. The codebook baseline assigns each user its strongest unused DFT beam, then fills any extra RF chains by aggregate channel energy. This avoids duplicating OMP, which with an orthogonal DFT dictionary and aggregate residual-energy scoring can select the same beams as aggregate-energy ranking. Digital weights use RZF on either the full estimated channel or the effective channel after analog beam selection.

## Uncertainty-aware regularization

The experimental heuristic uses `alpha = K / SNR_linear + K * NMSE_linear`. This is a transparent baseline, not a derived optimum. It is applied to both OMP-selected and DFT-selected analog beams. `Hybrid-OMP-Robust-RZF` tests uncertainty-aware regularization after OMP selection; `DFT-Codebook-Robust-RZF` applies the same rule after per-user DFT beam selection. In each pair, the analog selection is held fixed and only the digital regularization changes. If it does not help, report that result rather than tuning against the test set.

## Interpretation

- The perfect-CSI digital RZF curve is an idealized reference under the same channel and power normalization, not a strict information-theoretic capacity bound.
- The RF-chain sweep uses 4, 6, and 8 chains for 4 simultaneous users; conventional hybrid precoding needs at least as many RF chains as independent data streams. More RF chains should generally improve the available analog subspace, but Monte Carlo variability and multiuser geometry can change individual outcomes.
- High CSI error may degrade all estimated-CSI methods; an uncertainty-aware method is not guaranteed to help every channel or SNR.
- The outage threshold is defined by this project as a user rate below 1 bit/s/Hz; it is a benchmark metric, not a standards-defined outage target.
- Compare paired methods on the same realization when doing formal statistical tests. The current summary reports mean and standard deviation; paired confidence intervals or bootstrap intervals are recommended before publication.

## Suggested ablations

1. 32, 64, and 128 transmit antennas.
2. 4, 6, and 8 RF chains for the current four-user configuration.
3. CSI NMSE from −30 to 0 dB.
4. Compare the regularization coefficient on a validation grid, then freeze it before final testing.
5. 1, 3, and 5 paths per user.
6. Multiple random seeds.
7. Optional phase quantization of analog beams as a later extension.

## Scope

This is a narrowband algorithm benchmark, not a 3GPP-compliant 5G NR or wideband OFDM simulator. It should be described accurately in a resume and README.
