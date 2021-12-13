import numpy as np
import scipy.stats as stats
from pyqfin.models.model import ParametersBase, AnalyticBase, SimulationBase


class Parameters(ParametersBase):

    def __init__(self, sigma: float, r: float, q: float = 0) -> None:
        """
        :param sigma: volatility
        :param r:     risk-free rate
        :param q:     dividend rate
        """
        self.sigma = sigma
        self.r = r
        self.q = q
        super().__init__()


class Analytic(AnalyticBase):

    @classmethod
    def fromParamValues(cls, sigma, r, q=0):
        return cls(Parameters(sigma, r, q))

    def _dp(self, s0: float, tau: float, k: float) -> float:
        """
        Computes the auxilliary quantity

        $$d_+ := \frac{1}{\sqrt{\sigma \tau}}(\ln(\tfrac{s0}{k} + (r-q - \tfrac{1}{2} \sigma^2) \tau))$$

        from the Black-Scholes formula.
        
        param s0:     the current spot price of the stock
        param tau:    the remaining time to maturity of the option
        param k:      the strike of the option
        
        returns: d1 as per Black-Scholes formula (scalar)
        """
        return 1 / (self.sigma() * np.sqrt(tau)) * (np.log(s0 / k)
                + (self.r() - self.q() + self.sigma() ** 2 / 2) * tau) \


    def price(self, s0: float, tau: float, k: float, pc: chr = 'c') -> float:
        """
        Computes the price of a call option in the Black/Scholes model.
        
        param s0:  the current spot price of the stock
        param tau: time to maturity of the option    
        param k:   the strike of the option
        param pc:  put call flag: 'c' for call, 'p' for put
        
        returns: price of call option maturing after tau
        """

        Phi = stats.norm(loc=0, scale=1).cdf
        dp = self._dp(s0, tau, k)
        dm = dp - self.sigma() * np.sqrt(tau)
        fwd = np.exp((self.r() - self.q()) * tau) * s0
        df = np.exp(-self.r() * tau)
        if pc == 'c':
            return df * (fwd * Phi(dp) - k * Phi(dm))
        elif pc == 'p':
            return df * (k * Phi(-dm) - fwd * Phi(-dp))
        else:
            raise ValueError(
                "black_scholes:Analytic:price_option: flag %s is invalid." % pc)

    def delta(self, s0: float, tau: float, k: float, pc: chr = 'c') -> float:
        Phi = stats.norm(loc=0., scale=1.).cdf
        dp = self._dp(s0, tau, k)
        delta_call = np.exp(-self.q() * tau) * Phi(dp)
        if pc == 'c':
            return delta_call
        elif pc == 'p':
            return delta_call - np.exp(-self.q() * tau)
        else:
            raise ValueError(
                "black_scholes:Analytic:delta: flag %s is invalid." % pc)

    def gamma(self, s0: float, tau: float, k: float) -> float:
        phi = stats.norm(loc=0, scale=1).pdf
        dp = self._dp(s0, tau, k)
        return np.exp(-self.q() * tau) * phi(dp) / (
                    s0 * self.sigma() * np.sqrt(tau))

    def vega(self, s0: float, tau: float, k: float) -> float:
        phi = stats.norm(loc=0, scale=1).pdf
        dp = self._dp(s0, tau, k)
        return np.exp(-self.q() * tau) * s0 * phi(dp) * np.sqrt(tau)

    def theta(self, s0: float, tau: float, k: float, pc: chr = 'c') -> float:
        Phi = stats.norm(loc=0, scale=1).cdf
        phi = stats.norm(loc=0, scale=1).pdf
        dp = self._dp(s0, tau, k)
        dm = dp - self.sigma() * np.sqrt(tau)
        theta_call = self.q() * np.exp(-self.q() * tau) * s0 * Phi(dp) \
                   - self.r() * np.exp(-self.r() * tau) * k * Phi(dm) \
                   - np.exp(-self.q() * tau) * self.sigma() * s0 * phi(dp) / (
                               2 * np.sqrt(tau))
        if pc == 'c':
            return theta_call
        elif pc == 'p':
            return theta_call - self.q() * np.exp(-self.q() * tau) * s0 + self.r() * np.exp(-self.r() * tau) * k
        else:
            raise ValueError(
                "black_scholes:Analytic:theta: flag %s is invalid." % pc)

    def rho(self, s0: float, tau: float, k: float, pc: chr = 'c') -> float:
        Phi = stats.norm(loc=0, scale=1).cdf
        dp = self._dp(s0, tau, k)
        dm = dp - self.sigma() * np.sqrt(tau)
        if pc == 'c':
            return tau * np.exp(-self.r() * tau) * k * Phi(dm)
        elif pc == 'p':
            return - tau * np.exp(-self.r() * tau) * k * Phi(-dm)
        else:
            raise ValueError(
                "black_scholes:Analytic:rho: flag %s is invalid." % pc)


class Simulation(SimulationBase):

    def __init__(self, params, time_grid, npaths) -> None:
        self.params = params
        self.analytic = Analytic(params)
        self.time_grid = time_grid
        self.ntimes = self.time_grid.shape[0]
        self.npaths = npaths
        self.s_ = None

    def simulate(self, s0, seed=1, z=None):
        np.random.seed(seed)
        if z is None:
            z = np.random.standard_normal((self.npaths, self.ntimes - 1))
        self._simulate_with(s0, z)
        return self

    def _simulate_with(self, s0, z):
        delta = self.time_grid[1:] - self.time_grid[:-1]
        paths = self.params.s0 * np.cumprod(np.exp((self.params.r - self.params.q - self.params.sigma ** 2 / 2) * delta + self.params.sigma * np.sqrt(delta) * z), axis=1)
        self.s_ = np.transpose(np.c_[np.ones(self.npaths) * s0, paths])
