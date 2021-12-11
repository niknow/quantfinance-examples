import itertools
from unittest import TestCase
import numpy as np
import sympy as sy
from sympy.stats import Normal, cdf

from pyqfin.models.black_scholes import Analytic


class BlackScholesSympy:

    def __init__(self):
        sigma, r, q, s0, k, T, t = sy.symbols('sigma r q s0 k T t')
        self.sigma = sigma
        self.r = r
        self.q = q
        self.s0 = s0
        self.k = k
        self.T = T
        self.t = t
        self.tau = self.T - self.t

    def call(self):
        Phi = cdf(Normal('x', 0.0, 1.0))
        dp = 1 / (self.sigma * sy.sqrt(self.tau)) * (
                sy.log(self.s0 / self.k) + (
                self.r - self.q + self.sigma ** 2 / 2) * self.tau)
        dm = dp - sy.sqrt(self.tau) * self.sigma
        fwd = sy.exp((self.r - self.q) * self.tau) * self.s0
        df = sy.exp(-self.r * self.tau)
        c = df * (fwd * Phi(dp) - self.k * Phi(dm))
        return c

    def put(self):
        c = self.call()
        return c - sy.exp(-self.q * self.tau) * self.s0 + sy.exp(
            -self.r * self.tau) * self.k


class TestAnalytic(TestCase):

    def setUp(self) -> None:
        self.sigmas = np.array([0.01, 0.05, 0.2, 0.5, 1.5])
        self.rs = np.array([-0.03, -0.01, 0, 0.01, 0.03])
        self.qs = np.array([0., 0.01, 0, 0.5, 0.10])
        self.taus = np.array([3 / 12, 9 / 12, 1., 5., 10.])
        self.ks = np.array([80., 95., 105., 120.3])
        self.s0s = np.array([80., 90., 100., 110., 120])
        return super().setUp()

    def test_regression(self):
        self.bsa = Analytic.fromParams(0.3, 0.01)
        c = self.bsa.price(100, 1, 95, 'c')
        p = self.bsa.price(100, 1, 95, 'p')
        np.testing.assert_almost_equal(c, 14.780463438292863)
        np.testing.assert_almost_equal(p, 8.835197644463832)

    def test_put_call_parity(self):
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs,
                                                         self.qs, self.taus,
                                                         self.ks, self.s0s):
            with self.subTest():
                self.bsa = Analytic.fromParams(sigma, r, q)
                c = self.bsa.price(s0, tau, k, 'c')
                p = self.bsa.price(s0, tau, k, 'p')
                df = np.exp(- r * tau)
                np.testing.assert_almost_equal(c - p,
                                               np.exp(-q * tau) * s0 - df * k)

    def test_vs_sympy(self):
        self.bss = BlackScholesSympy()
        c = self.bss.call()
        p = self.bss.put()
        c_price_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k], c)
        p_price_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k], p)
        T = 11
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs,
                                                         self.qs, self.taus,
                                                         self.ks, self.s0s):
            with self.subTest():
                t = T - tau
                self.bsa = Analytic.fromParams(sigma, r, q)
                c_price = self.bsa.price(s0, tau, k, 'c')
                p_price = self.bsa.price(s0, tau, k, 'p')
                np.testing.assert_almost_equal(
                    c_price_sym(sigma, r, q, s0, T, t, k),
                    c_price)
                np.testing.assert_almost_equal(
                    p_price_sym(sigma, r, q, s0, T, t, k),
                    p_price)

    def test_delta_sympy(self):
        self.bss = BlackScholesSympy()
        c = self.bss.call()
        p = self.bss.put()
        c_delta_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k], sy.diff(c, self.bss.s0))
        p_delta_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k], sy.diff(p, self.bss.s0))
        T = 11
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs,
                                                         self.qs, self.taus,
                                                         self.ks, self.s0s):
            with self.subTest():
                t = T - tau
                self.bsa = Analytic.fromParams(sigma, r, q)
                c_delta = self.bsa.delta(s0, tau, k, 'c')
                p_delta = self.bsa.delta(s0, tau, k, 'p')
                np.testing.assert_almost_equal(
                    c_delta_sym(sigma, r, q, s0, T, t, k),
                    c_delta)
                np.testing.assert_almost_equal(
                    p_delta_sym(sigma, r, q, s0, T, t, k),
                    p_delta)

    def test_gamma_sympy(self):
        self.bss = BlackScholesSympy()
        c = self.bss.call()
        p = self.bss.put()
        c_gamma_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k],
                                  sy.diff(sy.diff(c, self.bss.s0), self.bss.s0))
        p_gamma_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k],
                                  sy.diff(sy.diff(p, self.bss.s0), self.bss.s0))
        T = 11
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs,
                                                         self.qs, self.taus,
                                                         self.ks, self.s0s):
            with self.subTest():
                t = T - tau
                self.bsa = Analytic.fromParams(sigma, r, q)
                c_gamma = self.bsa.gamma(s0, tau, k)
                p_gamma = c_gamma
                np.testing.assert_almost_equal(
                    c_gamma_sym(sigma, r, q, s0, T, t, k),
                    c_gamma)
                np.testing.assert_almost_equal(
                    p_gamma_sym(sigma, r, q, s0, T, t, k),
                    p_gamma)

    def test_vega_sympy(self):
        self.bss = BlackScholesSympy()
        c = self.bss.call()
        p = self.bss.put()
        c_vega_sym = sy.lambdify([self.bss.sigma,
                                  self.bss.r,
                                  self.bss.q,
                                  self.bss.s0,
                                  self.bss.T,
                                  self.bss.t,
                                  self.bss.k], sy.diff(c, self.bss.sigma))
        p_vega_sym = sy.lambdify([self.bss.sigma,
                                  self.bss.r,
                                  self.bss.q,
                                  self.bss.s0,
                                  self.bss.T,
                                  self.bss.t,
                                  self.bss.k], sy.diff(p, self.bss.sigma))
        T = 11
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs,
                                                         self.qs, self.taus,
                                                         self.ks, self.s0s):
            with self.subTest():
                t = T - tau
                self.bsa = Analytic.fromParams(sigma, r, q)
                c_vega = self.bsa.vega(s0, tau, k)
                p_vega = c_vega
                np.testing.assert_almost_equal(
                    c_vega_sym(sigma, r, q, s0, T, t, k),
                    c_vega)
                np.testing.assert_almost_equal(
                    p_vega_sym(sigma, r, q, s0, T, t, k),
                    p_vega)

    def test_theta_sympy(self):
        self.bss = BlackScholesSympy()
        c = self.bss.call()
        p = self.bss.put()
        c_theta_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k], sy.diff(c, self.bss.t))
        p_theta_sym = sy.lambdify([self.bss.sigma,
                                   self.bss.r,
                                   self.bss.q,
                                   self.bss.s0,
                                   self.bss.T,
                                   self.bss.t,
                                   self.bss.k], sy.diff(p, self.bss.t))
        T = 11
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs,
                                                         self.qs, self.taus,
                                                         self.ks, self.s0s):
            with self.subTest():
                t = T - tau
                self.bsa = Analytic.fromParams(sigma, r, q)
                c_theta = self.bsa.theta(s0, tau, k, 'c')
                p_theta = self.bsa.theta(s0, tau, k, 'p')
                np.testing.assert_almost_equal(
                    c_theta_sym(sigma, r, q, s0, T, t, k),
                    c_theta)
                np.testing.assert_almost_equal(
                    p_theta_sym(sigma, r, q, s0, T, t, k),
                    p_theta)

    def test_rho_sympy(self):
        self.bss = BlackScholesSympy()
        c = self.bss.call()
        p = self.bss.put()
        c_rho_sym = sy.lambdify([self.bss.sigma,
                                 self.bss.r,
                                 self.bss.q,
                                 self.bss.s0,
                                 self.bss.T,
                                 self.bss.t,
                                 self.bss.k], sy.diff(c, self.bss.r))
        p_rho_sym = sy.lambdify([self.bss.sigma,
                                 self.bss.r,
                                 self.bss.q,
                                 self.bss.s0,
                                 self.bss.T,
                                 self.bss.t,
                                 self.bss.k], sy.diff(p, self.bss.r))
        T = 11
        for sigma, r, q, tau, k, s0 in itertools.product(self.sigmas, self.rs,
                                                         self.qs, self.taus,
                                                         self.ks, self.s0s):
            with self.subTest():
                t = T - tau
                print(sigma, r, q, tau, k, s0)
                self.bsa = Analytic.fromParams(sigma, r, q)
                c_rho = self.bsa.rho(s0, tau, k, 'c')
                p_rho = self.bsa.rho(s0, tau, k, 'p')
                np.testing.assert_almost_equal(
                    c_rho_sym(sigma, r, q, s0, T, t, k),
                    c_rho)
                np.testing.assert_almost_equal(
                    p_rho_sym(sigma, r, q, s0, T, t, k),
                    p_rho)
