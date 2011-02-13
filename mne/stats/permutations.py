"""T-test with permutations
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

from math import sqrt
import numpy as np


def permutation_t_test(X, n_permutations=10000, tail=0):
    """One sample/paired sample permutation test based on a t-statistic.

    This function can perform the test on one variable or
    simultaneously on multiple variables. When applying the test to multiple
    variables, the "tmax" method is used for adjusting the p-values of each
    variable for multiple comparisons. Like Bonferroni correction, this method
    adjusts p-values in a way that controls the family-wise error rate.
    However, the permutation method will be more
    powerful than Bonferroni correction when different variables in the test
    are correlated.

    Parameters
    ----------
    X : array of shape [n_samples x n_tests]
        Data of size number of samples (aka number of observations) times
        number of tests (aka number of variables)

    n_permutations : int
        Number of permutations

    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the alternative hypothesis is that the
        mean of the data is greater than 0 (upper tailed test).  If tail is 0,
        the alternative hypothesis is that the mean of the data is different
        than 0 (two tailed test).  If tail is -1, the alternative hypothesis
        is that the mean of the data is less than 0 (lower tailed test).
    """
    n_samples, n_tests = X.shape

    X2 = np.mean(X**2, axis=0) # precompute moments
    mu0 = np.mean(X, axis=0)
    dof_scaling = sqrt(n_samples / (n_samples - 1.0))
    std0 = np.sqrt(X2 - mu0**2) * dof_scaling # get std with variance splitting
    T0 = np.mean(X, axis=0) / (std0 / sqrt(n_samples))
    perms = np.sign(np.random.randn(n_permutations, n_samples))
    mus = np.dot(perms, X) / float(n_samples)
    stds = np.sqrt(X2[None,:] - mus**2) * dof_scaling # std with splitting
    max_abs = np.max(np.abs(mus) / (stds / sqrt(n_samples)), axis=1) # t-max
    H0 = np.sort(max_abs)

    scaling = float(n_permutations + 1)

    if tail == 0:
        p_values = 1.0 - np.searchsorted(H0, np.abs(T0)) / scaling
    elif tail == 1:
        p_values = 1.0 - np.searchsorted(H0, T0) / scaling
    elif tail == -1:
        p_values = 1.0 - np.searchsorted(H0, -T0) / scaling

    return p_values, T0, H0

permutation_t_test.__test__ = False # for nosetests

if __name__ == '__main__':
    # 1 sample t-test
    n_samples, n_tests = 30, 5
    X = np.random.randn(n_samples, n_tests)
    X[:,:2] += 1
    p_values, T0, H0 = permutation_t_test(X, n_permutations=999, tail=1)
    is_significant = p_values < 0.05
    print "T stats : ", T0
    print "p_values : ", p_values
    print "Is significant : ", is_significant

    # 2 samples t-test
    n_samples, n_tests = 30, 5
    X1 = np.random.randn(n_samples, n_tests)
    X2 = np.random.randn(n_samples, n_tests)
    X1[:,:2] += 2
    p_values, T0, H0 = permutation_t_test(X1 - X2, n_permutations=999)
    print "T stats : ", T0
    print "p_values : ", p_values
    print "Is significant : ", is_significant

    # import pylab as pl
    # pl.close('all')
    # pl.hist(H0)
    # y_min, y_max = pl.ylim()
    # pl.vlines(T0, y_min, y_max, color='g', linewidth=2, linestyle='--')
    # pl.show()
