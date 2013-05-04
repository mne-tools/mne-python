import numpy as np
from scipy import stats
from scipy.signal import detrend
from ..utils import split_list
from ..parallel import parallel_func

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

DEFAULTS = {
    'parse': {
        'A': [0],
        'B': [1],
        'A+B': [0, 1],
        'A:B': [2],
        'A*B': [0, 1, 2]
        },
    'iter_contrasts': np.array([(1, 0, 1), (0, 1, 1), (1, 1, 1)])
 }


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
# and Python code from the pvttble toolbox by Roger Lew.
def r_anova_twoway(data, factor_levels, effects='A*B', alpha=0.05,
                   correction=False, n_jobs=1,
                   return_pvals=True):
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
        for mass univariate analysis.
    factor_levels : list-like
        The number of levels per factor.
    effects : str
        A string denoting the effect to be returned. The following
        mapping is currently supported:
            'A': main effect of A
            'B': main effect of B
            'A:B': interaction effect
            'A+B': both main effects
            'A*B': all three effects
    alpha : float
        The significance threshold.
    correction : bool
        The correction method to be employed if one factor has more than two
        levels. If True, sphericity correction using the Greenhouse-Geisser
        method will be applied.
    return_pvals : bool
        If True, return p values corresponding to f values.
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

    if n_jobs > 1 and data.shape[2] < 2:
        raise ValueError('You cannot use parallel jobs with less'
                         'than two observations per subject.')

    if effects.upper() not in DEFAULTS['parse']:
        raise ValueError('The value passed for `effects` is not supported.'
            ' Please consider the documentation.')

    effect_picks = DEFAULTS['parse'][effects]
    n_obs = data.shape[2]
    n_replications = data.shape[0]

    sc, sy, = [], []  # setup contrasts
    for n_levels in factor_levels:
        sc.append([np.ones([n_levels, 1]),
            detrend(np.eye(n_levels), type='constant')])
        sy.append([np.ones([n_levels, 1]) / n_levels, np.eye(n_levels)])

    df1 = np.prod(np.array(factor_levels) - 1)
    df2 = df1 * (n_replications - 1)

    if n_jobs > 1:
        parallel, parallel_anova, _ = parallel_func(_r_anova, n_jobs)
        results = parallel(parallel_anova(d, factor_levels, effect_picks,
                    n_replications, sc, sy, alpha, correction, df1, df2)
                    for d in split_list(np.rollaxis(data, 2), n_jobs))
        fvals, eps = zip(*results)
        fvals = np.concatenate(fvals, -1)
        eps = np.concatenate(eps, -1) if correction else eps
    else:
        fvals, eps = _r_anova(np.rollaxis(data, 2), factor_levels,
            effect_picks, n_replications, sc, sy, alpha, correction,
            df1, df2)

    fvals = np.array(fvals) if len(fvals) > 1 else np.array(fvals)[None, :]
    df1, df2 = np.zeros(n_obs) + df1, np.zeros(n_obs) + df2

    if correction:
        eps = np.array(eps) if len(eps) > 1 else np.array(eps)[None, :].T
        df1, df2 = [d[None, :] * eps for d in df1, df2]

    if return_pvals:
        if not correction:
            pvals = np.c_[[stats.f(df1, df2).sf(fv) for fv in fvals]]
        else:
            pvals = np.c_[[stats.f(df1_, df2_).sf(fv) for fv, df1_, df2_ in
                          zip(fvals, df1, df2)]]
    else:
        pvals = np.empty(0)

    return fvals, pvals


def _r_anova(data, factor_levels, effect_picks, n_replications, sc, sy,
             alpha, correction, df1, df2):
    """ Aux Function """
    fvals, epsilon = [], []
    for (c1, c2, c3) in DEFAULTS['iter_contrasts'][effect_picks]:
        c_ = np.kron(sc[0][c1], sc[c3][c2])  # compute design matrix
        y = np.dot(data, c_)
        b = np.mean(y, axis=1)[:, np.newaxis, :]
        ss = np.sum(np.sum(y * b, axis=2), axis=1)
        mse = (np.sum(np.sum(y * y, axis=2), axis=1) - ss) / (df2 / df1)
        fvals.append(ss / mse)
        if correction:
            # sample covariances, leave off "/ (y.shape[1] - 1)" norm because
            # it falls out. the below line is faster than the equivalent:
            # v = np.array([np.dot(y_.T, y_) for y_ in y])
            v = np.array(map(np.dot, y.swapaxes(2, 1), y))
            v = (np.array(map(np.trace, v)) ** 2 /
                  (df1 * np.sum(np.sum(v * v, axis=2), axis=1)))
            epsilon.append(v)

    return fvals, epsilon
