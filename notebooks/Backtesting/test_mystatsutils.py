import numpy as np
import mystatsutils
from unittest import TestCase


class TestMyStatsUtils(TestCase):

    def test_create_windows(self):
        self.n = 100
        self.nsims = 7
        self.m = 10
        y = mystatsutils.independent_samples(sigma=0.1, n=self.n, nsims=self.nsims, seed=1)
        self.assertEqual(y.shape, (self.n, self.nsims))
        for r in range(1, 11):
            windows = mystatsutils._create_windows(y, m=self.m, r=r)
            nr = int(np.ceil((self.n - self.m + 1) / r))
            self.assertEqual(windows.shape, (nr, self.m, self.nsims))
            for i in range(nr):
                for j in range(self.m):
                    np.testing.assert_array_almost_equal(
                        y[i * r + j, :],
                        windows[i, j, :]
                    )

    def test_overlap_samples(self):
        self.n = 100
        self.nsims = 7
        self.m = 10
        y = mystatsutils.independent_samples(sigma=0.1, n=self.n, nsims=self.nsims, seed=1)
        self.assertEqual(y.shape, (self.n, self.nsims))
        for r in range(1, 11):
            o = mystatsutils.overlap_samples(y, self.m, r)
            nr = int(np.ceil((self.n - self.m + 1) / r))
            self.assertEqual(o.shape, (nr, self.nsims))
            for i in range(nr):
                np.testing.assert_array_almost_equal(o[i, :], np.sum(y[i * r:i * r + self.m, :], axis=0))

    def test_observer_frequencies(self):
        self.x = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
                           [2, 3, 4, 4, 4, 4, 5, 5, 5, 5]]).T
        self.h = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        self.o = mystatsutils.observed_frequencies(self.x, self.h)
        self.assertEqual(self.o.shape, (self.h.shape[0] - 1, self.x.shape[1]))
        np.testing.assert_array_almost_equal(
            self.o[:, 0],
            np.array([1, 2, 3, 4, 0])
        )
        np.testing.assert_array_almost_equal(
            self.o[:, 1],
            np.array([0, 1, 1, 4, 4])
        )
