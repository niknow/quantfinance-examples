from unittest import TestCase
import numpy as np
import pyqfin.models.heston as h
from numpy.testing import assert_almost_equal


class TestAnalytic(TestCase):

    def test_ap_vs_alan(self):
        """
        Compare vs reference prices from
        https://financepress.com/2019/02/15/heston-model-reference-prices/
        """
        params = h.Params(v0=0.04, kappa=4, theta=0.25,
                          sigma=1, rho=-0.5, r=0.01, q=0.02)
        a = h.Analytic(params)
        tau = 1
        s0 = 100
        decimals = 10
        assert_almost_equal(a.price_option(tau, sk=80, s0=s0, pc='p'),
                            7.958878113256768285213263077598987193482161301733,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=80, s0=s0, pc='c'),
                            26.774758743998854221382195325726949201687074848341,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=90, s0=s0, pc='p'),
                            12.017966707346304987709573290236471654992071308187,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=90, s0=s0, pc='c'),
                            20.933349000596710388139445766564068085476194042256,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=100, s0=s0, pc='p'),
                            17.055270961270109413522653999411000974895436309183,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=100, s0=s0, pc='c'),
                            16.070154917028834278213466703938231827658768230714,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=110, s0=s0, pc='p'),
                            23.017825898442800538908781834822560777763225722188,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=110, s0=s0, pc='c'),
                            12.132211516709844867860534767549426052805766831181,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=120, s0=s0, pc='p'),
                            29.811026202682471843340682293165857439167301370697,
                            decimal=decimals)
        assert_almost_equal(a.price_option(tau, sk=120, s0=s0, pc='c'),
                            9.024913483457835636553375454092357136489051667150,
                            decimal=decimals)


class TestSimulation(TestCase):

    def test_simulation(self):
        h_sim = h.Simulation(
            params=h.Params(v0=0.2, kappa=0.3, theta=0.04, sigma=0.4, rho=-0.6,
                            r=0.03, q=0.01),
            time_grid=np.linspace(0, 1, 100),
            npaths=100000)
        h_sim.simulate(s0=100)
        decimals = 2
        # moments
        assert_almost_equal(h_sim.analytic.s_log_mean(h_sim.time_grid,
                                                      np.log(h_sim.s0)),
                            np.log(h_sim.s_).mean(axis=0), decimal=decimals)
        assert_almost_equal(h_sim.analytic.s_log_var(h_sim.time_grid),
                            np.log(h_sim.s_).var(axis=0), decimal=decimals)
        assert_almost_equal(h_sim.analytic.v_mean(h_sim.time_grid),
                            h_sim.v_.mean(axis=0), decimal=decimals)
        assert_almost_equal(h_sim.analytic.v_var(h_sim.time_grid),
                            h_sim.v_.var(axis=0), decimal=decimals)
        assert_almost_equal(h_sim.analytic.s_log_v_cov(h_sim.time_grid),
                            np.array([np.cov(np.log(h_sim.s_[:, i]),
                                             h_sim.v_[:, i])[0, 1] for i in
                                      range(h_sim.time_grid.shape[0])]),
                            decimal=decimals)
        # pricing

        strikes = np.linspace(90, 110, 10)
        ti = round(h_sim.time_grid.shape[0] / 2)
        maturities = h_sim.time_grid[ti:]
        strikes_, maturities_ = np.meshgrid(strikes, maturities)
        call_analytic = np.array([[h_sim.analytic.price_option_cui(tau, k, h_sim.s0, 'c')
                                   for k in strikes] for tau in maturities])
        put_analytic = np.array([[h_sim.analytic.price_option_cui(tau, k, h_sim.s0, 'p')
                                  for k in strikes] for tau in maturities])
        call_mc = np.array([[h_sim.price_option(0, t, k, 'c').mean() for k in strikes]
                            for t in range(ti, h_sim.time_grid.shape[0])])
        put_mc = np.array([[h_sim.price_option(0, t, k, 'p').mean() for k in strikes]
                           for t in range(ti, h_sim.time_grid.shape[0])])

        assert_almost_equal(put_analytic, put_mc)
        assert_almost_equal(call_analytic, call_mc)
