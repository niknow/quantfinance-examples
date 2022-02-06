import numpy as np
from scipy.integrate import quad, trapz
from scipy.stats import norm

from pyqfin.models.model import ParametersBase, AnalyticBase, SimulationBase
import pyqfin.models.black_scholes as bs



"""
An implementation of the Heston model using numpy. We use the
follwing formulation of the Heston model:
\begin{align*}
    dS_t &=0 (r-q) S_t dt + \sqrt{v_t} S_t dW \\
    dv_t &= \kappa(\theta - v_t)dt + \sigma \sqrt{v_t}dW,
\end{align*}
where $dW dZ = \rho dt$, $v$ represents a stochastic volatility,
$S_t$ is the stock price with spot s0, $r$ is the $r$, $\kappa$
is the rate of mean reversion, $\theta$ is the long-term mean of
the volatility and $\sigma$ is the volatility of volatility.

These implementations are based off various publications:

[1] Andersen. "Efficient Simulation of the Heston Stochastic
Volatility Model", SSRN 2007, http://ssrn.com/abstract=946405

[2] Cui et al. "Full and fast calibration of the Heston stochastic volatility model",
https://doi.org/10.1016/j.ejor.2017.05.018

[3] Le Floch. "An adaptive Filon quadrature for stochastic volatility model",
https://ssrn.com/abstract=3304016"
"""


class Params(ParametersBase):

    def __init__(self, v0: float, kappa: float, theta: float, sigma: float, rho: float, r: float, q: float = 0) -> None:
        """
        :param v0:    spot variance
        :param kappa: mean reversion of variance
        :param theta: long term variance
        :param sigma: volatility of variance
        :param rho:   correlation of the driving BMs
        :param r:     the assumed risk-free rate
        :param q:     the dividend rate
        """
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.q = q
        super().__init__()


class Analytic(AnalyticBase):
    """
        This class contains analytic functions needed for the Heston model.
    """
    def characteristic_function(self, st: float, tau: float, u: float) -> float:
        """
        Computes the Heston characteristic function
        $$ \Psi_{\log(S_T/S_0)}(u) = E[e^{i u \log(S_T/S_0)}] $$
        for valuing an option at $t$ with maturity $T$, hence time
        to maturity $\tau := T - t$, and stock spot value $S_t$,
        based off [2, Eq. (18)].

        :param st:  spot stock at t
        :param tau: time to maturity
        :param u:   function argument
        """
        xi = self.kappa() - self.sigma() * self.rho() * 1j * u  # [2, Eq. (11a)]
        d = np.sqrt(xi ** 2 + self.sigma() ** 2 * (u ** 2 + 1j * u))  # [2, Eq. (11b)]
        a1 = (u ** 2 + 1j * u) * np.sinh(d * tau / 2)  # Eq. (15b)
        a2 = (d / self.v0()) * np.cosh(d * tau / 2) + xi / self.v0() * np.sinh(d * tau / 2)  # [2, Eq. (15c)]
        a = a1 / a2  # [2, Eq. (15a)]
        fwd = st * np.exp((self.r() - self.q()) * tau)
        # [2, Eq. (17.b)]:
        dd = np.log(d / self.v0()) + (self.kappa() - d) * tau / 2
        dd -= np.log((d + xi) / (2 * self.v0()) + (d - xi) / (2 * self.v0()) * np.exp(- d * tau))
        # [2, Eq. (18):]
        res = 1j * u * np.log(fwd / st)
        res -= self.kappa() * self.theta() * self.rho() * tau * 1j * u / self.sigma()
        res -= a
        res += 2 * self.kappa() * self.theta() / self.sigma() ** 2 * dd
        return np.exp(res)

    def v_mean(self, t: float, s: float = 0, vs: float = None) -> float:
        """
        Computes the conditional mean of the variance process $V_t$ given
        $V_s$, $s \leq t$, c.f. [1, Corrolary 1].

        :param t: future time
        :param s: current time
        :param vs: variance at vs
        :return: $\mathbb{E}[V_t | V_s]$
        """
        if vs is None:
            vs = self.v0()
        return self.theta() + np.exp(-self.kappa() * (t-s)) * (vs - self.theta())

    def v_var(self, t: float, s: float = 0, vs: float = None) -> float:
        """
        Computes the conditional variance of the variance process $V_t$ given
        $V_s$, $s \leq t$. c.f. [1, Corrolary 1].

        :param t: future time
        :param s: current time
        :param vs: variance at vs
        :return: $\mathbb{V}[V_t | V_s]$
        """
        if vs is None:
            vs = self.v0()
        e = 1 - np.exp(-self.kappa() * (t-s))
        res = vs * self.sigma()**2 * np.exp(-self.kappa() * (t-s)) / self.kappa() * e
        res += self.theta() * self.sigma()**2 / (2 * self.kappa()) * e**2
        return res

    def _omega1(self, t: float) -> float:
        """
        Auxiliary function, see [1, Appendix A].

        :param t: time parameter
        :return: $\Omega_1(t)$
        """
        res = (1 + self.kappa() * t) * self.sigma()**2
        res -= 2 * self.rho() * self.kappa() * self.sigma() * (2 + self.kappa() * t)
        res += 2 * self.kappa() ** 2
        res *= 4 * np.exp(-self.kappa() * t)
        res += np.exp(-2 * self.kappa() * t) * self.sigma()**2
        res += (2 * self.kappa() * t - 5) * self.sigma()**2
        res -= 8 * self.rho() * self.kappa() * self.sigma() * (self.kappa() * t - 2)
        res += 8 * self.kappa()**2 * (self.kappa() * t - 1)
        return res

    def _omega2(self, t: float) -> float:
        """
        Auxiliary function, see [1, Appendix A].

        :param t: time parameter
        :return: $\Omega_2(t)$
        """
        res = - self.kappa() * t * self.sigma()**2
        res += 2 * self.rho() * self.sigma() * self.kappa() * (1 + self.kappa() * t)
        res -= 2 * self.kappa() ** 2
        res *= 2 * np.exp(-self.kappa() * t)
        res -= np.exp(-2 * self.kappa() * t) * self.sigma() **2
        res += self.sigma()**2 + 4 * self.kappa()**2
        res -= 4 * self.kappa() * self.rho() * self.sigma()
        return res

    def _omega3(self, t: float) -> float:
        """
        Auxiliary function, see [1, Appendix A].

        :param t: time parameter
        :return: $\Omega_3(t)$
        """
        res = t - 2 * self.rho() / self.sigma() * (1 + self.kappa() * t)
        res *= 2 * self.kappa() * np.exp(-self.kappa() * t)
        res += np.exp(-2 * self.kappa() * t)
        res += (4 * self.kappa() * self.rho() - self.sigma()) / self.sigma()
        return res

    def _omega4(self, t: float) -> float:
        """
        Auxiliary function, see [1, Appendix A].

        :param t: time parameter
        :return: $\Omega_4(t)$
        """
        res = 1 - self.kappa() * t
        res += 2 * self.rho() * self.kappa() ** 2 * t / self.sigma()
        res *= np.exp(-self.kappa() * t)
        res -= np.exp(- 2 * self.kappa() * t)
        return res

    def s_log_mean(self, t: float, x0: float) -> float:
        """
        Computes the mean of the log of the stock, see [1, Appendix A].

        :param t: time parameter
        :return: $\mathbb{E}[\ln(S_t)]$
        """
        res = (self.theta() - self.v0()) / (2 * self.kappa())
        res *= (1 - np.exp(-self.kappa() * t))
        res += x0 - self.theta() * t / 2
        return (self.r() - self.q()) * t + res

    def s_log_var(self, t: float) -> float:
        """
        Computes the variance of the log of the stock, see [1, Appendix A].

        :param t: time parameter
        :return: $\mathbb{V}[\ln(S_t)]$
        """
        res = self.theta() * self._omega1(t) / 8
        res += self.v0() * self._omega2(t) / 4
        res /= self.kappa()**3
        return res

    def s_log_v_cov(self, t: float) -> float:
        """
        Computes the covariance of the log of the stock and the
        variance process, see [1, Appendix A].

        :param t: time parameter
        :return: $\mathbb{Cov}[\ln(S_t), V_t]$
        """
        res = self.theta() * self.sigma() ** 2 * self._omega3(t) / 4
        res += self.v0() * self.sigma() ** 2 * self._omega4(t) / 2
        res /= self.kappa() ** 2
        return res


    def _price_option_int_ubound(self, tau: float, eps: float) -> float:
        """
        Calculates the truncation limit for the integration in the Heston
        pricer.

        :param tau:     time to maturity
        :param eps:     relative tolerance
        :return:
        """
        ltiny = np.log(eps)
        cinf = (self.v0() + self.kappa() * self.theta() * tau) / self.sigma() * np.sqrt(1 - self.rho() ** 2)  #[3, (24)]

        # for Newton solver
        def func1(u):
            # [3, (27)]
            return -cinf * u - np.log(u) - ltiny, -cinf - 1 / u

        def func2(u):
            # [3, (28)]
            return - 0.5 * self.v0() * tau * u * u - np.log(u) - ltiny, -  self.v0() * tau * u - 1 / u

        # Newton iteration
        zero1 = -np.log(eps) / cinf  # starting value
        zero2 = np.sqrt(-2 * np.log(eps) / self.v0() / tau)
        nsteps = 5

        for j in range(nsteps):
            nom, denom = func1(zero1)
            zero1 = zero1 - nom / denom
            nom, denom = func2(zero2)
            zero2 = zero2 - nom / denom
        return max(zero1, zero2) + 1.

    def _price_option_integrand(self, s0: float, tau: float, sk: float):
        """
        Implements the integrand for the Heston call option pricer
        $$ \operatorname{Re}\Big( e^{-iux}\frac{\phi_B(u-\tfrac{i}{2}) - \phi(u-\tfrac{i}{2})}{u^2+\tfrac{1}{4}} \Big), $$
        where $\phi_B(z) := \Psi_{\ln(F_T/F_0)}(z) = e^{-\tfrac{1}{2}\sigma_B^2 T(z^2+iz)}$
        is the characteristic function of the normalized log-forward in the Black-Scholes model,
        $\phi:=\Psi_{\ln(F_T/F_0)}$ is the characteristic function of the normalized log-forward in the
        Heston model and $x := \ln(K/F)$. This is the formulation from [3, Eq. (18)].
        :param s0:   spot stock
        :param tau:  time to maturity
        :param sk:   strike
        :return:
        """

        fwd = s0 * np.exp((self.r() - self.q()) * tau)

        def integrand(u):
            u_shifted = u - 0.5 * 1j
            psi = self.characteristic_function(s0, tau, u_shifted)
            psi *= np.exp(-1j * (self.r()-self.q()) * tau * u_shifted)
            res = np.exp(-0.5 * tau * self.v0() * (u ** 2 + 0.25)) - psi
            res *= np.exp(-1j * u * np.log(sk / fwd)) / (u ** 2 + 0.25)
            return np.real(res)

        return integrand

    def price_option(self, tau: float, sk: float, s0: float, pc: chr = 'c', eps: float =1e-10, num_grid_points: int =1000) -> float:
        """
        Computes the price of a European option under the Heston model. The
        pricing formula implemented is based off Andersen/Piterbarg as
        discussed in [3, (18)].

        :param s0:  spot stock
        :param tau: time to maturity
        :param sk:  strike
        :param pc:  'c' if call, 'p' if put option
        """
        int_ubound = self._price_option_int_ubound(tau, eps)
        int_domain = np.linspace(0, int_ubound, num_grid_points)
        int_fun = self._price_option_integrand(s0, tau, sk)
        int_value = trapz(int_fun(int_domain), x=int_domain)
        price_bs = bs.Analytic(bs.Params(np.sqrt(self.v0()), self.r(), self.q())).price(s0, tau, sk, pc)
        fwd = s0 * np.exp((self.r() - self.q()) * tau)
        return price_bs + np.sqrt(fwd * sk) / np.pi * np.exp(-self.r() * tau) * int_value


class Simulation(SimulationBase):

    def __init__(self, params, time_grid, npaths) -> None:
        super().__init__(params)
        self.analytic = Analytic(params)
        self.time_grid = time_grid
        self.ntimes = self.time_grid.shape[0]
        self.npaths = npaths
        self.s0 = None
        self.v_ = np.zeros((self.npaths, self.ntimes))
        self.s_ = np.zeros((self.npaths, self.ntimes))

    def simulate(self, s0, seed=1, z=None, method='qe'):
        """

        :param s0:     spot value of the stock
        :param seed:   seed of random number generator
        :param z:      np.array of shape (npaths, ntimes)
        :param method: 'qe' for QE-Scheme and 'te' for 'truncated euler'
        :return:
        """
        np.random.seed(seed)
        self.s0 = s0
        dt = self.time_grid[1:] - self.time_grid[:-1]
        if method == 'te':
            if z is None:
                rho = self.params.rho
                z = np.random.multivariate_normal(
                    np.array([0, 0]),
                    np.array([[1, rho], [rho, 1]]),
                    (self.npaths, self.ntimes - 1))
                self._simulate_variance_te(z[:, :, 1], dt)
                self._simulate_stock_te(z[:, :, 0], dt)
        elif method == 'qe':
            uv = np.random.uniform(0, 1, (self.npaths, self.ntimes - 1))
            self._simulate_variance_qe(uv)
            zx = np.random.normal(0, 1, (self.npaths, self.ntimes - 1))
            self._simulate_stock_qe(zx, dt)

    def _simulate_stock_te(self, zx, dt):
        vp = np.maximum(self.v_, 0)
        a = -0.5 * vp
        b = np.sqrt(vp)
        self.s_[:, 0] = np.log(self.s0)
        for i in range(self.ntimes - 1):
            self.s_[:, i + 1] = self.s_[:, i] + a[:, i] * dt[i] + b[:, i] * np.sqrt(dt[i]) * zx[:, i]
        self.s_ = np.exp(self.s_) * np.exp((self.params.r-self.params.q) * self.time_grid)

    def _simulate_variance_te(self, zv, dt):
        self.v_[:, 0] = self.params.v0
        for i in range(self.ntimes - 1):
            v_trunc = np.maximum(self.v_[:, i], 0)
            v_drift = self.params.kappa * (self.params.theta - v_trunc) * dt[i]
            v_diff_c = np.sqrt(v_trunc) * self.params.sigma * np.sqrt(dt[i])
            self.v_[:, i + 1] = self.v_[:, i] + v_drift + v_diff_c * zv[:, i]

    def _simulate_variance_qe(self, uv):
        """
        This implements the QE scheme based on [1, Sect. 3.2.4].
        :param zu:
        :param dt:
        :return:
        """
        self.v_[:, 0] = self.params.v0

        psi_c = 1.5
        for i in range(self.ntimes - 1):
            m = self.analytic.v_mean(self.time_grid[i+1], self.time_grid[i], self.v_[:, i])
            s2 = self.analytic.v_var(self.time_grid[i+1], self.time_grid[i], self.v_[:, i])
            psi = s2 / m ** 2

            idx = psi <= psi_c

            # psi <= psi_c
            psi1 = psi[idx]
            b2 = 2 / psi1 - 1 + np.sqrt(2 / psi1) * np.sqrt(2 / psi1 - 1)  # [1, Eq. (27)]
            a = m[idx] / (1 + b2)    # [1, Eq. (28)]
            b = np.sqrt(b2)
            zv = norm.ppf(uv[idx, i])
            self.v_[idx, i+1] = a * (b + zv) ** 2  # [1, Eq. (23)]

            # psi > psi_c
            psi2 = psi[~idx]
            p = (psi2 - 1) / (psi2 + 1)   # [1, Eq. (29)]
            beta = 2 / (m[~idx] * (psi2 + 1))  # [1, Eq. (30)]
            idy = p < uv[~idx, i]
            self.v_[~idx, i+1][idy] = np.log((1-p[idy]) / (1-uv[~idx, i][idy])) / beta[idy]  # [1, Eq. (25)]

    def _simulate_stock_qe(self, z, dt):
        """
        Implements the simulation of the log stock to combine with the QE
        scheme for the variance, see [1, Eq. (33)].

        :param z:   Gaussians of shape (npaths, ntimes - 1)
        :param dt:  time deltas of shape (ntimes - 1)
        :return:
        """
        g1 = 0.5
        g2 = 0.5
        k0 = - self.params.rho * self.params.kappa * self.params.theta / self.params.sigma * dt
        h = self.params.kappa * self.params.rho / self.params.sigma - 0.5
        k1 = g1 * dt * h - self.params.rho / self.params.sigma
        k2 = g2 * dt * h + self.params.rho / self.params.sigma
        k3 = g1 * dt * (1 - self.params.rho ** 2)
        k4 = g2 * dt * (1 - self.params.rho ** 2)
        self.s_[:, 0] = np.log(self.s0)
        for i in range(self.ntimes - 1):
            drift = k0[i] + k1[i] * self.v_[:, i] + k2[i] * self.v_[:, i+1]
            diff = np.sqrt(k3[i] * self.v_[:, i] + k4[i] * self.v_[:, i + 1])
            self.s_[:, i+1] = self.s_[:, i] + drift + diff * z[:, i]
        self.s_ = np.exp(self.s_) * np.exp((self.params.r - self.params.q) * self.time_grid)

    def v_means(self):
        return self.analytic.v_mean(self.time_grid)

    def v_vars(self):
        return self.analytic.v_var(self.time_grid)

    def price_option(self, t: float, t_mat: float, sk: float, pc: chr = 'c'):
        """
        Computes the price distribution of a European option.

        param t:     time index as of which we want to price
        param t_mat: time index of option maturity
        param sk:    the strike of the option
        param pc:    put call flag: 'c' for call, 'p' for put

        returns: price of call option maturing after tau
        """
        df = np.exp(- self.params.r * (self.time_grid[t_mat] - self.time_grid[t]))
        if pc == 'c':
            return df * np.maximum(self.s_[:, t_mat] - sk, 0)
        elif pc == 'p':
            return df * np.maximum(sk - self.s_[:, t_mat], 0)
        else:
            raise ValueError(
                "black_scholes:Simulation:price: flag %s is invalid." % pc)
