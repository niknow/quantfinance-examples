import numpy as np
from scipy.stats import norm
from pyqfin.models.model import ParametersBase, AnalyticBase, SimulationBase


class Params(ParametersBase):

    def __init__(self, kappa, sigma, f0) -> None:
        self.kappa = kappa
        self.sigma = sigma
        self.f0 = f0


class Analytic(AnalyticBase):

    def kappa(self):
        return self.params.kappa

    def sigma(self):
        return self.params.sigma

    def f0(self):
        return self.params.f0

    def c(self, t):
        """
        Implements the auxilliary function
        $$ c(t) = f(0,t) + \frac{e^{-\kappa t} |\sigma|^2}{\kappa^2} \Big(  e^{\kappa t} - 1  -  \sinh(\kappa t) \Big) $$
        """
        return self.f0() + np.exp(-self.kappa() * t) / (self.kappa() ** 2) * \
            (np.exp(self.kappa() * t) - 1 - np.sinh(self.kappa() * t))

    def g(self, s, t):
        """
        Implements the auxilliary function
        $$ G(s,t) &= \frac{1}{\kappa}(1 - e^{-\kappa(t-s)}) $$
        """
        return 1 / self.kappa() * (1 - np.exp(-self.kappa() * (t - s)))

    def y(self, t):
        """
        Implements the auxilliary function
        $$ y(t) = \frac{\sigma^2}{2\kappa}( 1 - e^{-2 \kappa t}) $$
        """
        return self.sigma()**2 / (2 * self.kappa()) * (1 - np.exp(-2 * self.kappa() * t))

    def r_mean_cond(self, rs, s, t):
        """
        Implements the analytic conditional mean of the short rate
        $$ \mathbb{E}[r(t) \mid \mathcal{F}_s] = r(s)e^{-\kappa(t-s)} + c(t) - e^{-\kappa(t-s)}c(s) $$
        """
        return rs * np.exp(-self.kappa() * (t-s)) + self.c(t) - np.exp(-self.kappa() * (t-s)) * self.c(s)

    def r_var_cond(self, s, t):
        """
        Implements the analytic conditional variance of the short rate
        $$ \mathbb{V}[r(t) \mid \mathcal{F}_s] = e^{-2 \kappa t} \frac{\sigma^2}{2\kappa} (e^{2 \kappa t} - e^{2 \kappa s}) $$
        """
        return np.exp(- 2 * self.kappa() * t) / (2 * self.kappa()) * self.sigma() ** 2 * \
               (np.exp(2 * self.kappa() * t) - np.exp(2 * self.kappa() * s))

    def x_mean_cond(self, xs, s, t):
        """
        Implements the conditional mean of the state variable
        $$ \mathbb{E}[x(t) \mid \mathcal{F}_s] = e^{-\kappa (t-s)} x(s) + e^{- \kappa t} \frac{|\sigma|^2}{\kappa^2}\Big(\cosh(\kappa t) - \cosh(\kappa s) \Big) $$
        """
        return np.exp(-self.kappa() * (t-s)) * xs + \
               np.exp(-self.kappa() * t) * self.sigma()**2 / self.kappa()**2 * \
               (np.cosh(self.kappa() * t) - np.cosh(self.kappa() * s))

    def x_mean(self, t):
        """
        Implements the mean of the state variable
        $$ \mathbb{E}[x(t)] = \frac{\sigma^2}{2 \kappa^2}\Big(1 - e^{- \kappa t} \Big)^2 $$
        """
        return self.sigma() ** 2 * (1 - np.exp(- self.kappa() * t)) ** 2 / ( 2 *self.kappa()**2 )
    
    def r_mean(self, t):
        return self.x_mean(t) + self.f0()

    def x_var_cond(self, s, t):
        """
        Computes the conditional variance of the state variable
        $$ \mathbb{V}[x(t)]} = \frac{\sigma^2}{2 \kappa} (1 - e^{-2 \kappa (t-s)})) $$
        """
        return self.sigma()**2 / (2 * self.kappa()) * (1 - np.exp(-2 * self.kappa() * (t-s)))

    def x_var(self, t):
        """
        Computes the variance of the state variable
        $$ \mathbb{V}[x(t)]} = \frac{\sigma^2}{2 \kappa} (1 - e^{-2 \kappa t)})) $$
        """
        return self.sigma()**2 * (1 - np.exp(- 2 * self.kappa() * t)) / (2*self.kappa())
    
    def r_var(self, t):
        return self.x_var(t)

    def x_cov(self, s, t):
        """
        Computes the autocovariance of the state variable
        $$ \operatorname{Cov}[x(s),x(t)] = \frac{\sigma^2}{\kappa} e^{-\kappa t} \sinh(\kappa s) $$
        """
        return self.sigma()**2 / self.kappa() * np.exp(-self.kappa() * max(s, t)) * np.sinh(min(s, t) * self.kappa())

    def I_aux(self, s, t):
        """
        Computes the auxilliary quantity
        $$ m_I(s,t) = \frac{\sigma^2}{4 \kappa^3} \Big( 2 \kappa (t-s)  - (e^{-2 \kappa t} - e^{-2 \kappa s})  + 4 \cosh(\kappa s) (e^{- \kappa t} - e^{- \kappa s}) \Big) $$
        """
        return self.sigma()**2 / (4 * self.kappa()**3) * ( 2 * self.kappa() * (t-s) - (np.exp(-2 * self.kappa() * t) - np.exp(-2 * self.kappa() * s)) + 4 * np.cosh(self.kappa() * s) * (np.exp(- self.kappa() * t) - np.exp(- self.kappa() * s)) )

    def I_mean(self, t):
        """
        Computes the mean
        $$ \operatorname{E}[I(t)] = \frac{\sigma^2}{4 \kappa^3} \Big( 2 \kappa t  - (e^{-2 \kappa t} - 1)  + 4 (e^{- \kappa t} - 1) \Big) $$ 
        """
        return self.I_aux(0, t)

    def I_mean_cond(self, Is, xs, s, t):
        """
        Computes the conditional mean
        $$ \mathbb{E}[I(t) \mid I(s), x(s)] = I(s) + x(s) G(s,t) + m_I(s,t) $$
        """
        return Is + xs * self.g(s,t) + self.I_aux(s, t)

    def I_var_cond(self, s, t):
        """
        Computes the conditional variance
        $$ \mathbb{V}[I(t) \mid I(s), x(s)] = 2 m_I(s,t) - y(s) G(s,t)^2 $$
        """
        return 2 * self.I_aux(s, t) - self.y(s) * self.g(s, t)**2

    def I_var(self, t):
        """
        Computes the variance
        $$ \mathbb{V}[I(t)] = 2 \mathbb{E}[I(t)] $$
        """
        return 2 * self.I_mean(t)
    
    def I_cov(self, s, t):
        """
        Computes the covariance
        $$ \operatorname{Cov}[I(s), I(t)] = \frac{\sigma^2}{\kappa^3} \Big( e^{-\kappa t} \sinh(\kappa s)  - 1 +e^{-\kappa s} - e^{-\kappa t}(e^{\kappa s}-1) + \kappa s  \Big)
        """
        if s <= t:
            return self.sigma()**2 / self.kappa()**3 * ( np.exp(-self.kappa() *  t) * np.sinh(self.kappa() * s)  - 1 + np.exp(-self.kappa() * s) - np.exp(-self.kappa() * t) * (np.exp(self.kappa() * s)-1) + self.kappa() * s )
        else:
            return self.I_cov(t, s)

    def Ix_cov(self, s, t):
        """
        Computes the conditional covariance 
        $$ \operatorname{Cov}[x(t), I(t) \mid I(s), x(s)] = \frac{\sigma^2}{2 \kappa^2} \Big(1 - e^{- \kappa (t-s)}\Big)^2 $$
        """
        return self.sigma()**2 / (2 * self.kappa()**2) * (1 - np.exp(- self.kappa() * (t-s)))**2
    
    def Ix_cross_cov(self, s, t):
        """
        Computes the cross covariance 
        $$ \operatorname{Cov}[x(s), I(t)] = e^{-s \kappa} \frac{\sigma^2}{\kappa^2}(1 -e^{\kappa (s\wedge t)}  - \kappa e^{-\kappa t}(s \wedge t))
        """
        return np.exp(-self.kappa() * s) * self.sigma()**2 / self.kappa()**2 * (1 - np.exp(-self.kappa() * min(s,t))  - self.kappa() * np.exp(-self.kappa() * t) * min(s,t))
    
    def bond_reconstitution(self, xs, s, t):
        """
        Computes the bond reconstitution for $s \leq t$
        $$ P(s,t) = \frac{P(0,t)}{P(0,s)} \exp(-x(s) G(s,t) - \frac{1}{2} y(s) G(s,t)^2)$$
        """
        return np.exp(-self.params.f0 * (t-s)) * np.exp(-xs * self.g(s,t) - 0.5 * self.y(s) * self.g(s,t)**2)

    def _v(self, t, t_e, t_m):
        return self.sigma()**2 * self.g(t_e,t_m)**2 * (1 - np.exp(-2 * self.kappa() * (t_e-t))) / (2 * self.kappa())

    def _capfloorlet(self, xt, t, t_e, t_m, k):
        v = self._v(t, t_e, t_m)
        tau = t_m - t_e
        P_t_te = self.bond_reconstitution(xt, t, t_e)
        P_t_tm = self.bond_reconstitution(xt, t, t_m)
        d = 1/np.sqrt(v) * np.log( (1 + tau * k) *  P_t_tm / P_t_te)
        dp = d + np.sqrt(v) / 2
        dm = d - np.sqrt(v) / 2
        return (P_t_te, P_t_tm, dp, dm)

    def caplet(self, xt, t, t_e, t_m, k):
        P_t_te, P_t_tm, dp, dm = self._capfloorlet(xt, t, t_e, t_m, k)
        return P_t_te * norm.cdf(-dm) - (1 + k * (t_m - t_e)) * P_t_tm * norm.cdf(-dp)

    def floorlet(self, xt, t, t_e, t_m, k):
        P_t_te, P_t_tm, dp, dm = self._capfloorlet(xt, t, t_e, t_m, k)
        return P_t_tm * (1 + k * (t_m - t_e)) * norm.cdf(dp) - P_t_te * norm.cdf(dm)
    
    def bond_option(self, t_e, t_m, k, call=True):
        v = self._v(0, t_e, t_m)
        P_t_t_e = self.bond_reconstitution(0, 0, t_e)
        P_t_t_m = self.bond_reconstitution(0, 0, t_m)
        d = 1 / np.sqrt(v) * np.log(P_t_t_m / (P_t_t_e * k))
        dp = d + np.sqrt(v) / 2
        dm = d - np.sqrt(v) / 2
        if call:
            return P_t_t_m * norm.cdf(dp) - P_t_t_e * k * norm.cdf(dm)
        else: # put
            return P_t_t_e * k * norm.cdf(-dm) - P_t_t_m * norm.cdf(-dp)


class Simulation(SimulationBase):

    def __init__(self, params, time_grid, npaths) -> None:
        self.params = params
        self.analytic = Analytic(params)
        self.time_grid = time_grid
        self.ntimes = self.time_grid.shape[0]
        self.npaths = npaths
        self.method = None
        self.x_ = None
        self.w_ = None
        self.r_ = None
        self.I_ = None
        self.I_disc_ = np.zeros((self.npaths, self.ntimes))
        self.df_ = None
        self.df_disc = None

    def simulate(self, seed=1, zx=None, zI=None,  method='inc'):
        self.method = method
        np.random.seed(seed)
        if self.method == 'inc':
            if zx is None:
                zx = np.random.standard_normal((self.npaths, self.ntimes - 1))
            if zI is None:
                zI = np.random.standard_normal((self.npaths, self.ntimes - 1))
            self._simulate_with_inc(zx, zI)
        elif self.method == 'fcov':
            self._simulate_fcov()
        elif self.method == 'ql':
            return self._simulate_with_ql()
        else:
            raise ValueError("Method %s is not valid." % self.method)
        self.df_ = np.exp(-self.I_) * np.exp(-self.params.f0 * self.time_grid)
        return self

    def _simulate_with_inc(self, zx, zI):
        self.x_ = np.zeros((self.npaths, self.ntimes))
        self.I_ = np.zeros((self.npaths, self.ntimes))
        for i in range(1, self.ntimes):
            x_mean = self.x_mean_cond(i)
            x_std = np.sqrt(self.x_var_cond(i-1, i))
            I_mean_cond = self.I_mean_cond(i-1, i)
            I_std_cond = np.sqrt(self.I_var_cond(i-1, i))
            Ix_cov = self.Ix_cov(i-1, i)
            rho = Ix_cov / x_std / I_std_cond
            zxI = rho * zx[:, i-1] + np.sqrt(1-rho**2) * zI[:,i-1]
            self.x_[:, i] = x_mean + x_std * zx[:, i-1]
            self.I_[:, i] = I_mean_cond + I_std_cond * zxI
        self.r_ = self.x_ + self.params.f0

    def _simulate_fcov(self):
        x_means = self.x_means()
        I_means = self.I_means()
        means = np.hstack((x_means[1:],I_means[1:]))
        x_cov = self.x_covs()
        I_cov = self.I_covs()
        Ix_cross = self.Ix_cross()
        cov = np.vstack((np.hstack((x_cov[1:,1:], Ix_cross[1:,1:])),
                         np.hstack((Ix_cross[1:,1:].T, I_cov[1:,1:]))))
        sim = np.random.multivariate_normal(mean=means, cov=cov, size=self.npaths)
        self.x_ = np.insert(sim[:,:self.ntimes-1], 0, 0, axis=1)
        self.I_ = np.insert(sim[:,self.ntimes-1:], 0, 0, axis=1)
        self.r_ = self.x_ + self.params.f0

    def _simulate_with_ql(self):
        try:
            import QuantLib as ql
        except ImportError:
            raise ImportError("QuantLib not installed. Skipping reconciliation.")

        def generate_paths(num_paths, timestep):
            arr = np.zeros((num_paths, timestep+1))
            for i in range(num_paths):
                sample_path = seq.next()
                path = sample_path.value()
                time = [path.time(j) for j in range(len(path))]
                value = [path[j] for j in range(len(path))]
                arr[i, :] = np.array(value)
            return np.array(time), arr

        timestep = self.time_grid.shape[0] - 1
        length = self.time_grid[-1]
        forward_rate = self.params.f0
        day_count = ql.Thirty360()
        todays_date = ql.Date(15, 1, 2015)
        ql.Settings.instance().evaluationDate = todays_date
        spot_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
        spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)
        hw_process = ql.HullWhiteProcess(spot_curve_handle, self.params.kappa, self.params.sigma)
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
        seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)
        time, paths = generate_paths(self.npaths, timestep)
        self.r_ = paths
        self.x_ = self.r_ - self.params.f0

    def discretize(self):
        delta = self.time_grid[1:] - self.time_grid[:-1]
        for i in range(1, self.ntimes):
            self.I_disc_[:, i] = self.I_disc_[:, i-1] + self.x_[:, i-1] * delta[i-1]
        self.df_disc = np.exp(-self.I_disc_) * np.exp(-self.params.f0 * self.time_grid)
    
    def p(self, j, i):
        return self.analytic.bond_reconstitution(self.x_[:,j], self.time_grid[j], self.time_grid[i])

    def price_mc_caplet(self, i, i_e, i_m, k):
        tau = self.time_grid[i_m] - self.time_grid[i_e]
        return self.df_[:, i_m] / self.df_[:, i] * np.maximum(1. / self.p(i_e, i_m) - (1 + k * tau), 0)
    
    def price_caplet(self, i, i_e, i_m, k):
         return self.analytic.caplet(self.x_[:, i], self.time_grid[i], self.time_grid[i_e], self.time_grid[i_m], k)
    
    def x_means(self):
        return self.analytic.x_mean(self.time_grid)
    
    def r_means(self):
        return self.analytic.x_mean(self.time_grid) + self.params.f0
    
    def x_vars(self):
        return self.analytic.x_var(self.time_grid)
    
    def r_vars(self):
        return self.analytic.r_var(self.time_grid)

    def x_covs(self):
        return np.array([[self.analytic.x_cov(s, t) for t in self.time_grid] for s in self.time_grid])

    def I_means(self):
        return self.analytic.I_mean(self.time_grid)

    def I_vars(self):
        return self.analytic.I_var(self.time_grid)
    
    def Ix_covs(self):
        return self.analytic.Ix_cov(0, self.time_grid)

    def I_covs(self):
        return np.array([[self.analytic.I_cov(s,t) for t in self.time_grid] for s in self.time_grid])
    
    def Ix_cross(self):
        return np.array([[self.analytic.Ix_cross_cov(s,t) for t in self.time_grid] for s in self.time_grid])

    def x_mean_cond(self, i):
        return self.analytic.x_mean_cond(self.x_[:, i-1], self.time_grid[i-1], self.time_grid[i])

    def x_var_cond(self, j, i):
        return self.analytic.x_var_cond(self.time_grid[j], self.time_grid[i])

    def I_mean_cond(self, j, i):
        return self.analytic.I_mean_cond(self.I_[:, j], self.x_[:, j], self.time_grid[j], self.time_grid[i])

    def I_var_cond(self, j, i):
        return self.analytic.I_var_cond(self.time_grid[j], self.time_grid[i])

    def Ix_cov(self, j, i):
        return self.analytic.Ix_cov(self.time_grid[j], self.time_grid[i])
