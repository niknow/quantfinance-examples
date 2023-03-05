import numpy as np
import scipy.stats as ss


def independent_samples(sigma, n, nsims, seed=1):
    np.random.seed(seed)
    return ss.norm.rvs(loc=0, scale=sigma, size=(n, nsims))


def _create_windows(y, m, r):
    n = y.shape[0]
    return np.array([y[i:i + m, :] for i in range(0, n - m + 1, r)])


def overlap_samples(y, m, r=1):
    return np.sum(_create_windows(y, m, r), axis=1)


def threshold(sigma, q, m):
    return ss.norm.ppf(q, loc=0, scale=sigma) * np.sqrt(m)


def empirical_quantile(x, q):
    return ss.scoreatpercentile(x, q * 100, interpolation_method='lower')


def exceedance_count(sigma, n, nsims, seed, m, r, h):
    y = independent_samples(sigma, n, nsims, seed)
    x = overlap_samples(y, m, r)
    return np.sum(x > h, axis=0)


def observed_frequencies(x, h):
    o = np.zeros((h.shape[0] - 1, x.shape[1]))
    nsims = x.shape[1]
    for w in range(nsims):
        o[:, w] = np.histogram(x[:, w], bins=h)[0]
    return o


def chi_squared_samples(sigma, n, m, nsims, gamma, h, r=1, seed=1):
    y = independent_samples(sigma, n, nsims, seed)
    x = overlap_samples(y, m, r)
    e = (gamma[1:] - gamma[:-1]) * x.shape[0]
    o = observed_frequencies(x, h)
    return np.sum((o - e[:, np.newaxis]) ** 2 / e[:, np.newaxis], axis=0)
