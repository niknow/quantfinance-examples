import itertools
from unittest import TestCase
import numpy as np

from pyqfin.models.black_scholes import Analytic


class TestAnalytic(TestCase):

    def setUp(self) -> None:
        self.sigmas = np.array([0.01, 0.05, 0.2, 0.5, 1.5])
        self.rs = np.array([-0.03, -0.01, 0, 0.01, 0.03])
        self.qs = np.array([0., 0.01, 0, 0.5, 0.10])
        self.taus = np.array([3/12, 9/12, 1., 5., 10.])
        self.ks = np.array([80., 95., 105., 120.3])
        self.s0s = np.array([80., 90., 100., 110., 120])
        return super().setUp()

    def test_regression(self):
        self.bsa = Analytic.fromParams(0.3, r=0.01)
        c = self.bsa.price(100, 1, 95, 'c')
        p = self.bsa.price(100, 1, 95, 'p')
        np.testing.assert_almost_equal(c, 14.780463438292863)
        np.testing.assert_almost_equal(p, 8.835197644463832)
    
    def test_put_call_parity(self):
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs, self.qs, self.taus, self.ks, self.s0s):
            with self.subTest():
                self.bsa = Analytic.fromParams(sigma, r, q)
                c = self.bsa.price(s0, tau, k, 'c')
                p = self.bsa.price(s0, tau, k, 'p')
                df = np.exp(- r * tau)
                np.testing.assert_almost_equal(c - p, np.exp(-q * tau) * s0 - df * k)

    def test_delta(self):
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs, self.qs, self.taus, self.ks, self.s0s):
            bump = 1e-4
            with self.subTest():
                self.bsa = Analytic.fromParams(sigma, r, q)
                c = self.bsa.price(s0, tau, k, 'c')
                p = self.bsa.price(s0, tau, k, 'p')
                c_bump = self.bsa.price(s0 + bump, tau, k, 'c')
                p_bump = self.bsa.price(s0 + bump, tau, k, 'p')
                c_delta = self.bsa.delta(s0, tau, k, 'c')
                p_delta = self.bsa.delta(s0, tau, k, 'p')
                np.testing.assert_almost_equal((c_bump - c) / bump, c_delta, decimal=2)
                np.testing.assert_almost_equal((p_bump - p) / bump, p_delta, decimal=2)

    def test_gamma(self):
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs, self.qs, self.taus, self.ks, self.s0s):
            bump = 1e-4
            with self.subTest():
                self.bsa = Analytic.fromParams(sigma, r, q)
                c = self.bsa.price(s0, tau, k, 'c')
                p = self.bsa.price(s0, tau, k, 'p')
                c_up = self.bsa.price(s0 + bump, tau, k, 'c')
                c_down = self.bsa.price(s0 - bump, tau, k, 'c')
                p_up = self.bsa.price(s0 + bump, tau, k, 'p')
                p_down = self.bsa.price(s0 - bump, tau, k, 'p')
                c_gamma = self.bsa.gamma(s0, tau, k)
                p_gamma = c_gamma
                np.testing.assert_almost_equal((c_up - 2*c + c_down) / bump**2, c_gamma, decimal=2)
                np.testing.assert_almost_equal((p_up - 2*p + p_down) / bump**2, p_gamma, decimal=2)

    def test_vega(self):
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs, self.qs, self.taus, self.ks, self.s0s):
            bump = 1e-6
            with self.subTest():
                self.bsa = Analytic.fromParams(sigma, r, q)
                self.bsa_bumped = Analytic.fromParams(sigma+bump, r, q)
                c = self.bsa.price(s0, tau, k, 'c')
                p = self.bsa.price(s0, tau, k, 'p')
                c_bumped = self.bsa_bumped.price(s0, tau, k, 'c')
                p_bumped = self.bsa_bumped.price(s0, tau, k, 'p')
                c_vega = self.bsa.vega(s0, tau, k)
                p_vega = c_vega
                np.testing.assert_almost_equal((c_bumped - c) / bump, c_vega, decimal=2)
                np.testing.assert_almost_equal((p_bumped - p) / bump, p_vega, decimal=2)

    def test_theta(self):
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs, self.qs, self.taus, self.ks, self.s0s):
            bump = 1e-10
            with self.subTest():
                self.bsa = Analytic.fromParams(sigma, r, q)
                c = self.bsa.price(s0, tau, k, 'c')
                p = self.bsa.price(s0, tau, k, 'p')
                c_bumped = self.bsa.price(s0, tau - bump, k, 'c')
                p_bumped = self.bsa.price(s0, tau - bump, k, 'p')
                c_theta = self.bsa.theta(s0, tau, k, 'c')
                p_theta = self.bsa.theta(s0, tau, k, 'p')
                print(sigma, r, q, tau, k, s0, c)
                np.testing.assert_almost_equal((c_bumped - c) / bump, c_theta, decimal=2)
                np.testing.assert_almost_equal((p_bumped - p) / bump, p_theta, decimal=2)

    def test_rho(self):
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs, self.qs, self.taus, self.ks, self.s0s):
            bump = 1e-4
            with self.subTest():
                self.bsa = Analytic.fromParams(sigma, r, q)
                self.bsa_bumped = Analytic.fromParams(sigma, r+bump, q)
                c = self.bsa.price(s0, tau, k, 'c')
                p = self.bsa.price(s0, tau, k, 'p')
                c_bumped = self.bsa_bumped.price(s0, tau, k, 'c')
                p_bumped = self.bsa_bumped.price(s0, tau, k, 'p')
                c_rho = self.bsa.rho(s0, tau, k, 'c')
                p_rho = self.bsa.rho(s0, tau, k, 'p')
                np.testing.assert_almost_equal((c_bumped - c) / bump, c_rho, decimal=2)
                np.testing.assert_almost_equal((p_bumped - p) / bump, p_rho, decimal=2)

    def test_vs_ref(self):
        import scipy.stats as si
        def black_scholes_call_div(S, K, T, r, q, sigma):
            # S: spot price
            # K: strike price
            # T: time to maturity
            # r: interest rate
            # q: rate of continuous dividend paying asset
            # sigma: volatility of underlying asset

            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (
                        sigma * np.sqrt(T))
            d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (
                        sigma * np.sqrt(T))

            call = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(
                -r * T) * si.norm.cdf(d2, 0.0, 1.0))

            return call
        s0=100
        sigma=0.2
        tau=0.7
        q=0.03
        r=0.02
        k=95
        self.bsa = Analytic.fromParams(sigma, r, q)
        print(self.bsa.price(s0, tau, k))
        print(black_scholes_call_div(s0, k, tau, r, q, sigma))
