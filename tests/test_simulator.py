import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as sim


class HybridMIMOBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.codebook = sim.make_dft_codebook(sim.DEFAULT_N_TX)
        self.h = sim.generate_channel(self.rng)
        self.h_est = sim.add_csi_error(self.h, 0.01, self.rng)

    def test_channel_shape_and_finite(self):
        self.assertEqual(self.h.shape, (sim.DEFAULT_N_USERS, sim.DEFAULT_N_TX))
        self.assertTrue(np.all(np.isfinite(self.h)))

    def test_dft_codebook_is_unitary_and_constant_modulus(self):
        d = self.codebook
        self.assertTrue(np.allclose(d.conj().T @ d, np.eye(sim.DEFAULT_N_TX), atol=1e-10))
        self.assertTrue(np.allclose(np.abs(d), 1 / np.sqrt(sim.DEFAULT_N_TX)))

    def test_csi_error_has_approximately_requested_nmse(self):
        rng = np.random.default_rng(99)
        hs, estimates = [], []
        for _ in range(100):
            h = sim.generate_channel(rng)
            hs.append(h)
            estimates.append(sim.add_csi_error(h, 0.1, rng))
        h = np.concatenate(hs, axis=0)
        hhat = np.concatenate(estimates, axis=0)
        measured = np.sum(np.abs(hhat - h) ** 2) / np.sum(np.abs(h) ** 2)
        self.assertAlmostEqual(10 * np.log10(measured), -10.0, delta=0.6)

    def test_precoders_have_unit_total_power(self):
        rho_db = 15.0
        w_digital = sim.rzf_precoder(self.h_est, rho_db)
        w_omp = sim.hybrid_precoder(self.h_est, rho_db, 4, 0.01, self.codebook, "omp")
        w_dft = sim.hybrid_precoder(self.h_est, rho_db, 4, 0.01, self.codebook, "dft")
        w_dft_robust = sim.hybrid_precoder(
            self.h_est, rho_db, 4, 0.01, self.codebook,
            selection="dft", uncertainty_aware=True
        )
        for w in [w_digital, w_omp, w_dft, w_dft_robust]:
            self.assertAlmostEqual(float(np.linalg.norm(w, "fro") ** 2), 1.0, places=8)

    def test_analog_precoders_are_constant_modulus(self):
        for f in [
            sim.omp_analog_precoder(self.h_est, 4, self.codebook),
            sim.dft_analog_precoder(self.h_est, 4, self.codebook),
        ]:
            self.assertEqual(f.shape, (sim.DEFAULT_N_TX, 4))
            self.assertTrue(np.allclose(np.abs(f), 1 / np.sqrt(sim.DEFAULT_N_TX)))

    def test_rates_are_finite_and_nonnegative(self):
        w = sim.rzf_precoder(self.h_est, 20.0)
        metrics, rates = sim.rate_metrics(self.h, w, 20.0)
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertTrue(np.all(rates >= 0))
        self.assertGreaterEqual(metrics["sum_se"], 0)
        self.assertGreaterEqual(metrics["outage_user_fraction"], 0)
        self.assertLessEqual(metrics["outage_user_fraction"], 1)

    def test_invalid_rf_chain_count_is_rejected(self):
        with self.assertRaises(ValueError):
            sim.omp_analog_precoder(self.h_est, sim.DEFAULT_N_TX + 1, self.codebook)

    def test_hybrid_precoder_requires_enough_rf_chains_for_users(self):
        with self.assertRaises(ValueError):
            sim.hybrid_precoder(self.h_est, 20.0, sim.DEFAULT_N_USERS - 1,
                                0.01, self.codebook, selection="omp")


    def test_dft_robust_rzf_is_included_and_finite(self):
        methods = sim.evaluate_methods(
            self.h, self.h_est, 20.0, 0.1, 4, self.codebook
        )
        self.assertIn("DFT-Codebook-Robust-RZF", methods)
        metrics, rates = methods["DFT-Codebook-Robust-RZF"]
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertTrue(np.isfinite(metrics["sum_se"]))
        self.assertGreaterEqual(metrics["sum_se"], 0.0)

    def test_dft_robust_rzf_changes_regularization_under_uncertain_csi(self):
        w_standard = sim.hybrid_precoder(
            self.h_est, 20.0, 4, 0.1, self.codebook,
            selection="dft", uncertainty_aware=False
        )
        w_robust = sim.hybrid_precoder(
            self.h_est, 20.0, 4, 0.1, self.codebook,
            selection="dft", uncertainty_aware=True
        )
        self.assertFalse(np.allclose(w_standard, w_robust, atol=1e-10))
        self.assertAlmostEqual(float(np.linalg.norm(w_robust, "fro") ** 2), 1.0, places=8)

    def test_dft_baseline_is_distinct_from_omp_on_seeded_channel(self):
        f_omp = sim.omp_analog_precoder(self.h_est, 4, self.codebook)
        f_dft = sim.dft_analog_precoder(self.h_est, 4, self.codebook)
        self.assertFalse(np.allclose(f_omp @ f_omp.conj().T,
                                     f_dft @ f_dft.conj().T, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
