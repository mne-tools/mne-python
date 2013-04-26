import numpy as np
from scipy import stats
from scipy.signal import detrend
from ..utils import split_list
from ..parallel import parallel_func

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: Simplified BSD


# The following function is a rewriting of scipy.stats.f_oneway
# Contrary to the scipy.stats.f_oneway implementation it does not
# copy the data while keeping the inputs unchanged.
def _f_oneway(*args):
    """
    Performs a 1-way ANOVA.

    The one-way ANOVA tests the null hypothesis that 2 or more groups have
    the same population mean. The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements should be given as arguments.

    Returns
    -------
    F-value : float
        The computed F-value of the test
    p-value : float
        The associated p-value from the F-distribution

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.

    1. The samples are independent
    2. Each sample is from a normally distributed population
    3. The population standard deviations of the groups are all equal.  This
       property is known as homocedasticity.

    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (`stats.kruskal`_) although with
    some loss of power

    The algorithm is from Heiman[2], pp.394-7.

    See scipy.stats.f_oneway that should give the same results while
    being less efficient

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 14.
           http://faculty.vassar.edu/lowry/ch14pt1.html

    .. [2] Heiman, G.W.  Research Methods in Statistics. 2002.

    """
    n_classes = len(args)
    n_samples_per_class = np.array([len(a) for a in args])
    n_samples = np.sum(n_samples_per_class)
    ss_alldata = reduce(lambda x, y: x + y,
                        [np.sum(a ** 2, axis=0) for a in args])
    sums_args = [np.sum(a, axis=0) for a in args]
    square_of_sums_alldata = reduce(lambda x, y: x + y, sums_args) ** 2
    square_of_sums_args = [s ** 2 for s in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0
    for k, _ in enumerate(args):
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    prob = stats.fprob(dfbn, dfwn, f)
    return f, prob


def f_oneway(*args):
    """Call scipy.stats.f_oneway, but return only f-value"""
    return _f_oneway(*args)[0]


# The following functions based on MATLAB code by Rik Henson
# and Python code from the pyvttble toolbox by Roger Lew.
def r_anova_twoway(data, factor_levels, factor_labels=None, alpha=0.05,
        correction=False, n_jobs=1, return_pvals=True):
    """ 2 way repeated measures ANOVA for fully balanced designs

    data : ndarray
        3D array where the first two dimensions are compliant
        with a subjects X conditions scheme:

        first factor repeats slowest:

                    A1B1 A1B2 A2B1 B2B2
        subject 1   1.34 2.53 0.97 1.74
        subject ... .... .... .... ....
        subject k   2.45 7.90 3.09 4.76

        The last dimensions is thought to carry the observations
        for mass univariate analysis
    factor_levels : list-like
        The number of levels per factor.
    alpha : float
        The significance threshold.
    correction : bool.
        The correction method to be employed. If True, sphericity
        correction using the Greenhouse-Geisser method will be applied.
    return_pvals : bool
        If True, return p values corresponding to f values
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).

    Returns
    -------
    f_vals : ndarray
        An array of f values with length corresponding to the number
        of effects estimated. The shape depends on the number of effects
        estimated.
    p_vals : ndarray
        If not requested via return_pvals, defaults to an empty array.
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    n_obs = data.shape[2]
    n_replications = data.shape[0]
    parallel, parallel_anova, _ = parallel_func(_r_anova, n_jobs)

    sc, sy, = [], []
    for n_levels in factor_levels:
        sc.append([np.ones([n_levels, 1]),
            detrend(np.eye(n_levels), type='constant')])
        sy.append([np.ones([n_levels, 1]) / n_levels, np.eye(n_levels)])

    df1 = np.prod(np.array(factor_levels) - 1)
    df2 = df1 * (n_replications - 1)
    results = parallel(parallel_anova(d, factor_levels,
                n_replications, sc, sy, alpha, correction, df1, df2)
                for d in split_list(np.rollaxis(data, 2), n_jobs))

    fvals, eps = zip(*results)
    fvals = np.concatenate(fvals)
    df1, df2 = np.zeros(n_obs) + df1, np.zeros(n_obs) + df2

    if correction:
        eps = np.concatenate(eps)
        df1, df2 = [eps * d for d in df1, df2]

    if return_pvals:
        if not correction:
            pvals = np.c_[[stats.f(df1, df2).sf(fv) for fv in fvals.T]]
        else:
            pvals = np.c_[[stats.f(df1, df2).sf(fv) for fv, df1, df2 in
                          zip(fvals.T, df1.T, df2.T)]]
    else:
        pvals = np.empty(0)

    return fvals, pvals


def _r_anova(data, factor_levels, n_replications, sc, sy,
            alpha, correction, df1, df2):
    """ Aux Function """

    n_factors, n_effects = 2, 3  # hard coded for now
    iter_contrasts = [(1, 0, 0), (0, 1, 1), (1, 1, 1)]
    fvals, epsilon = [], []
    for obs in data if data.ndim == 3 else data[None:, ]:
        this_fvals, this_epsilon = [], []
        for (c1, c2, c3) in iter_contrasts:
            c_ = np.kron(sc[0][c1], sc[c3][c2])  # compute design matrix
            y_ = np.dot(obs, c_)
            b_ = np.mean(y_, axis=0)
            ss = np.sum(y_ * b_.T)
            mse = (np.sum(np.diag(np.dot(y_.T, y_))) - ss) / df2
            mss = ss / df1
            f_value = mss / mse
            this_fvals.append(f_value)
            if correction:
                v_ = np.cov(y_, rowvar=False)  # sample covariance
                this_epsilon.append(np.trace(v_) ** 2 \
                    / (df1 * np.trace(np.dot(v_.T, v_))))

        fvals.append(this_fvals)
        if correction:
            epsilon.append(this_epsilon)

    return fvals, epsilon
