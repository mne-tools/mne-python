"""T-test with permutations
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Fernando Perez (bin_perm_rep function)
#
# License: Simplified BSD

from math import sqrt
import numpy as np


def bin_perm_rep(ndim, a=0, b=1):
    """bin_perm_rep(ndim) -> ndim permutations with repetitions of (a,b).

    Returns an array with all the possible permutations with repetitions of
    (0,1) in ndim dimensions.  The array is shaped as (2**ndim,ndim), and is
    ordered with the last index changing fastest.  For examble, for ndim=3:

    Examples:

    >>> bin_perm_rep(3)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1]])
    """

    # Create the leftmost column as 0,0,...,1,1,...
    nperms = 2**ndim
    perms = np.empty((nperms, ndim), type(a))
    perms.fill(a)
    half_point = nperms / 2
    perms[half_point:, 0] = b
    # Fill the rest of the table by sampling the pervious column every 2 items
    for j in range(1, ndim):
        half_col = perms[::2, j-1]
        perms[:half_point, j] = half_col
        perms[half_point:, j] = half_col

    return perms


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

    n_permutations : int or 'all'
        Number of permutations. If n_permutations is 'all' all possible
        permutations are tested (2**n_samples). It's the exact test, that
        can be untractable when the number of samples is big (e.g. > 20).
        If n_permutations >= 2**n_samples then the exact test is performed

    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the alternative hypothesis is that the
        mean of the data is greater than 0 (upper tailed test).  If tail is 0,
        the alternative hypothesis is that the mean of the data is different
        than 0 (two tailed test).  If tail is -1, the alternative hypothesis
        is that the mean of the data is less than 0 (lower tailed test).

    Returns
    -------
    p_values : array of shape [n_tests]
        P-values for all the tests (aka variables)

    T0 : array of shape [n_tests]
        T-statistic for all variables

    H0 : array of shape [n_permutations]
        T-statistic obtained by permutations and t-max trick for multiple
        comparison.

    Notes
    -----
    A reference (among many) in field of neuroimaging:
    Nichols, T. E. & Holmes, A. P. (2002). Nonparametric permutation tests
    for functional neuroimaging: a primer with examples.
    Human Brain Mapping, 15, 1-25.
    Overview of standard nonparametric randomization and permutation
    testing applied to neuroimaging data (e.g. fMRI)
    DOI: http://dx.doi.org/10.1002/hbm.1058
    """
    n_samples, n_tests = X.shape

    do_exact = False
    if n_permutations is 'all' or (n_permutations >= 2**n_samples - 1):
        do_exact = True
        n_permutations = 2**n_samples - 1

    X2 = np.mean(X**2, axis=0) # precompute moments
    mu0 = np.mean(X, axis=0)
    dof_scaling = sqrt(n_samples / (n_samples - 1.0))
    std0 = np.sqrt(X2 - mu0**2) * dof_scaling # get std with variance splitting
    T0 = np.mean(X, axis=0) / (std0 / sqrt(n_samples))

    if do_exact:
        perms = bin_perm_rep(n_samples, a=1, b=-1)[1:,:]
    else:
        perms = np.sign(0.5 - np.random.rand(n_permutations, n_samples))

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
    n_permutations = 50000
    # n_permutations = 'exact'
    X = np.random.randn(n_samples, n_tests)
    X[:,:2] += 0.6
    p_values, T0, H0 = permutation_t_test(X, n_permutations, tail=1)
    is_significant = p_values < 0.05
    print 80*"-"
    print "-------- 1-sample t-test :"
    print "T stats : ", T0
    print "p_values : ", p_values
    print "Is significant : ", is_significant

    print 80*"-"
    print "-------- Comparison analytic vs permutation :"
    p_values, T0, H0 = permutation_t_test(X, n_permutations, tail=1)
    print "--- permutation_t_test :"
    print "T stats : ", T0
    print "p_values : ", p_values
    print "Is significant : ", is_significant

    from scipy import stats
    T0, p_values = stats.ttest_1samp(X[:,0], 0)
    print "--- scipy.stats.ttest_1samp :"
    print "T stats : ", T0
    print "p_values : ", p_values

    # 2 samples t-test
    X1 = np.random.randn(n_samples, n_tests)
    X2 = np.random.randn(n_samples, n_tests)
    X1[:,:2] += 2
    p_values, T0, H0 = permutation_t_test(X1 - X2, n_permutations)
    print 80*"-"
    print "-------- 2-samples t-test :"
    print "T stats : ", T0
    print "p_values : ", p_values
    print "Is significant : ", is_significant

    # import pylab as pl
    # pl.close('all')
    # pl.hist(H0)
    # y_min, y_max = pl.ylim()
    # pl.vlines(T0, y_min, y_max, color='g', linewidth=2, linestyle='--')
    # pl.show()
