import numpy as np
import scipy.stats as stats
from pyqfin.models.model import ParametersBase, AnalyticBase, SimulationBase


class Params(ParametersBase):

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
        return cls(Params(sigma, r, q))

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


    def stock_mean(self, s0: float, t: float) -> float:
        return s0 * np.exp((self.r() - self.q()) * t)

    def stock_var(self, s0: float, t: float) -> float:
        return s0**2 * (np.exp(self.sigma()**2 * t) - 1) * np.exp(2*(self.r() - self.q() - 0.5 * self.sigma()**2)*t + self.sigma()**2 * t)

    def stock_autocov(self, s0: float, t1: float, t2: float) -> float:
        if t1 <= t2:
            return np.exp(self.sigma() ** 2 * (t1 + t2) / 2) * (np.exp(self.sigma()**2 * t1) - 1) * s0**2 * np.exp((self.r() - self.q() - 0.5 * self.sigma()**2) * (t1 + t2))
        else:
            return self.stock_autocov(s0, t2, t1)

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


def implied_volatility(price, r, s0, tm, sk, call, pc: chr = 'c') -> float:
    """
    Computes the implied volatility of a European option.

    :param r: risk-free rate
    :param s0: value of underlying stock price at t=0
    :param tm: time to maturity of the option
    :param sk: strike of the option
    param pc:  put call flag: 'c' for call, 'p' for put
    """
    from py_vollib.black_scholes.implied_volatility import \
        implied_volatility as pyvimp
    return pyvimp(price, s0, sk, tm, r, pc)


class Simulation(SimulationBase):

    def __init__(self, params, time_grid, npaths) -> None:
        super().__init__(params)
        self.analytic = Analytic(params)
        self.time_grid = time_grid
        self.ntimes = self.time_grid.shape[0]
        self.npaths = npaths
        self.s0 = None
        self.s_ = None

    def simulate(self, s0, seed=1, z=None):
        np.random.seed(seed)
        self.s0 = s0
        if z is None:
            z = np.random.standard_normal((self.npaths, self.ntimes - 1))
        self._simulate_with(z)
        return self

    def _simulate_with(self, z):
        delta = self.time_grid[1:] - self.time_grid[:-1]
        paths = self.s0 * np.cumprod(np.exp((self.params.r - self.params.q - self.params.sigma ** 2 / 2) * delta + self.params.sigma * np.sqrt(delta) * z), axis=1)
        self.s_ = np.c_[np.ones(self.npaths) * self.s0, paths]

    def price(self, t: int, t_mat: int, k: float, pc: chr = 'c') -> float:
        """
        Computes the price distribution of a European option.

        param t:     time index as of which we want to price
        param t_mat: time index of option maturity
        param k:     the strike of the option
        param pc:    put call flag: 'c' for call, 'p' for put

        returns: price of call option maturing after tau
        """
        df = np.exp(- self.params.r * (self.time_grid[t_mat] - self.time_grid[t]))
        if pc == 'c':
            return df * np.maximum(self.s_[:, t_mat] - k, 0)
        elif pc == 'p':
            return df * np.maximum(k - self.s_[:, t_mat], 0)
        else:
            raise ValueError(
                "black_scholes:Simulation:price: flag %s is invalid." % pc)

    def stock_means(self):
        return np.array([self.analytic.stock_mean(self.s0, t) for t in self.time_grid])

    def stock_vars(self):
        return np.array([self.analytic.stock_var(self.s0, t) for t in self.time_grid])

    def stock_autocovs(self):
        return np.array([[self.analytic.stock_autocov(self.s0, t0, t1)
                          for t0 in self.time_grid]
                         for t1 in self.time_grid])
