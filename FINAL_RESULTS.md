# Independent CSI-Mismatch Test

## Experimental setup
- 64 transmit antennas and 4 users
- 4 RF chains; 3 paths per user
- Narrowband geometric channel simulation
- SNR: 20 dB
- 200 independent test realizations per actual CSI NMSE
- Validation-selected assumed CSI NMSE: 0 dB

## Results

The selected uncertainty-aware Hybrid-OMP-RZF configuration
improved mean sum spectral efficiency in all four test conditions.

| Actual CSI NMSE (dB) | Standard SE | Robust SE | Gain (bit/s/Hz) |
|---:|---:|---:|---:|
| -30 | 21.8738 | 27.5661 | 5.6922 |\n| -20 | 17.4100 | 25.7566 | 8.3466 |\n| -10 | 10.9926 | 20.1060 | 9.1134 |\n| 0 | 6.3138 | 11.5387 | 5.2249 |\n
## Limitations
The uncertainty-aware regularization rule is a heuristic.
The selected setting maximizes average sum spectral efficiency,
not fairness. User outage and lower-tail rates must also be
considered. This is a simulation prototype, not a 3GPP-compliant
or hardware-validated 5G/6G implementation.

See `test_summary.csv` and `test_raw.csv` for measured results.
