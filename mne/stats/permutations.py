"""T-test with permutations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

from math import sqrt
import numpy as np

from ..utils import check_random_state, verbose, logger
from ..parallel import parallel_func


def _max_stat(X, X2, perms, dof_scaling):
    """Aux function for permutation_t_test (for parallel comp)."""
    n_samples = len(X)
    mus = np.dot(perms, X) / float(n_samples)
    stds = np.sqrt(X2[None, :] - mus * mus) * dof_scaling  # std with splitting
    max_abs = np.max(np.abs(mus) / (stds / sqrt(n_samples)), axis=1)  # t-max
    return max_abs


@verbose
def permutation_t_test(X, n_permutations=10000, tail=0, n_jobs=None,
                       seed=None, verbose=None):
    """One sample/paired sample permutation test based on a t-statistic.

    This function can perform the test on one variable or
    simultaneously on multiple variables. When applying the test to multiple
    variables, the "tmax" method is used for adjusting the p-values of each
    variable for multiple comparisons. Like Bonferroni correction, this method
    adjusts p-values in a way that controls the family-wise error rate.
    However, the permutation method will be more
    powerful than Bonferroni correction when different variables in the test
    are correlated (see :footcite:`NicholsHolmes2002`).

    Parameters
    ----------
    X : array, shape (n_samples, n_tests)
        Samples (observations) by number of tests (variables).
    n_permutations : int | 'all'
        Number of permutations. If n_permutations is 'all' all possible
        permutations are tested. It's the exact test, that
        can be untractable when the number of samples is big (e.g. > 20).
        If n_permutations >= 2**n_samples then the exact test is performed.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the alternative hypothesis is that the
        mean of the data is greater than 0 (upper tailed test).  If tail is 0,
        the alternative hypothesis is that the mean of the data is different
        than 0 (two tailed test).  If tail is -1, the alternative hypothesis
        is that the mean of the data is less than 0 (lower tailed test).
    %(n_jobs)s
    %(seed)s
    %(verbose)s

    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observed for all variables.
    p_values : array of shape [n_tests]
        P-values for all the tests (a.k.a. variables).
    H0 : array of shape [n_permutations]
        T-statistic obtained by permutations and t-max trick for multiple
        comparison.

    Notes
    -----
    If ``n_permutations >= 2 ** (n_samples - (tail == 0))``,
    ``n_permutations`` and ``seed`` will be ignored since an exact test
    (full permutation test) will be performed.

    References
    ----------
    .. footbibliography::
    """
    from .cluster_level import _get_1samp_orders
    n_samples, n_tests = X.shape
    X2 = np.mean(X ** 2, axis=0)  # precompute moments
    mu0 = np.mean(X, axis=0)
    dof_scaling = sqrt(n_samples / (n_samples - 1.0))
    std0 = np.sqrt(X2 - mu0 ** 2) * dof_scaling  # get std with var splitting
    T_obs = np.mean(X, axis=0) / (std0 / sqrt(n_samples))
    rng = check_random_state(seed)
    orders, _, extra = _get_1samp_orders(n_samples, n_permutations, tail, rng)
    perms = 2 * np.array(orders) - 1  # from 0, 1 -> 1, -1
    logger.info('Permuting %d times%s...' % (len(orders), extra))
    parallel, my_max_stat, n_jobs = parallel_func(_max_stat, n_jobs)
    max_abs = np.concatenate(parallel(my_max_stat(X, X2, p, dof_scaling)
                                      for p in np.array_split(perms, n_jobs)))
    max_abs = np.concatenate((max_abs, [np.abs(T_obs).max()]))
    H0 = np.sort(max_abs)
    if tail == 0:
        p_values = (H0 >= np.abs(T_obs[:, np.newaxis])).mean(-1)
    elif tail == 1:
        p_values = (H0 >= T_obs[:, np.newaxis]).mean(-1)
    elif tail == -1:
        p_values = (-H0 <= T_obs[:, np.newaxis]).mean(-1)
    return T_obs, p_values, H0


def bootstrap_confidence_interval(arr, ci=.95, n_bootstraps=2000,
                                  stat_fun='mean', random_state=None):
    """Get confidence intervals from non-parametric bootstrap.

    Parameters
    ----------
    arr : ndarray, shape (n_samples, ...)
        The input data on which to calculate the confidence interval.
    ci : float
        Level of the confidence interval between 0 and 1.
    n_bootstraps : int
        Number of bootstraps.
    stat_fun : str | callable
        Can be "mean", "median", or a callable operating along ``axis=0``.
    random_state : int | float | array_like | None
        The seed at which to initialize the bootstrap.

    Returns
    -------
    cis : ndarray, shape (2, ...)
        Containing the lower boundary of the CI at ``cis[0, ...]`` and the
        upper boundary of the CI at ``cis[1, ...]``.
    """
    if stat_fun == "mean":
        def stat_fun(x):
            return x.mean(axis=0)
    elif stat_fun == 'median':
        def stat_fun(x):
            return np.median(x, axis=0)
    elif not callable(stat_fun):
        raise ValueError("stat_fun must be 'mean', 'median' or callable.")
    n_trials = arr.shape[0]
    indices = np.arange(n_trials, dtype=int)  # BCA would be cool to have too
    rng = check_random_state(random_state)
    boot_indices = rng.choice(indices, replace=True,
                              size=(n_bootstraps, len(indices)))
    stat = np.array([stat_fun(arr[inds]) for inds in boot_indices])
    ci = (((1 - ci) / 2) * 100, ((1 - ((1 - ci) / 2))) * 100)
    ci_low, ci_up = np.percentile(stat, ci, axis=0)
    return np.array([ci_low, ci_up])


def _ci(arr, ci=.95, method="bootstrap", n_bootstraps=2000, random_state=None):
    """Calculate confidence interval. Aux function for plot_compare_evokeds."""
    if method == "bootstrap":
        return bootstrap_confidence_interval(arr, ci=ci,
                                             n_bootstraps=n_bootstraps,
                                             random_state=random_state)
    else:
        from . import _parametric_ci
        return _parametric_ci(arr, ci=ci)
