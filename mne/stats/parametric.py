# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
from functools import reduce
from string import ascii_uppercase
from ..utils import _check_option

# The following function is a rewriting of scipy.stats.f_oneway
# Contrary to the scipy.stats.f_oneway implementation it does not
# copy the data while keeping the inputs unchanged.


def ttest_1samp_no_p(X, sigma=0, method='relative'):
    """Perform one-sample t-test.

    This is a modified version of :func:`scipy.stats.ttest_1samp` that avoids
    a (relatively) time-consuming p-value calculation, and can adjust
    for implausibly small variance values :footcite:`RidgwayEtAl2012`.

    Parameters
    ----------
    X : array
        Array to return t-values for.
    sigma : float
        The variance estimate will be given by ``var + sigma * max(var)`` or
        ``var + sigma``, depending on "method". By default this is 0 (no
        adjustment). See Notes for details.
    method : str
        If 'relative', the minimum variance estimate will be sigma * max(var),
        if 'absolute' the minimum variance estimate will be sigma.

    Returns
    -------
    t : array
        T-values, potentially adjusted using the hat method.

    Notes
    -----
    To use the "hat" adjustment method :footcite:`RidgwayEtAl2012`, a value
    of ``sigma=1e-3`` may be a reasonable choice.

    You can use the conversion from ``scipy.stats.distributions.t.ppf``::

        thresh = -scipy.stats.distributions.t.ppf(p_thresh, n_samples - 1) / 2.

    to convert a desired p-value threshold to 2-tailed t-value threshold.
    For one-tailed tests, ``thresh`` in the above should be multiplied by 2
    (and for ``tail=-1``, multiplied by ``-1``).

    References
    ----------
    .. footbibliography::
    """
    _check_option('method', method, ['absolute', 'relative'])
    var = np.var(X, axis=0, ddof=1)
    if sigma > 0:
        limit = sigma * np.max(var) if method == 'relative' else sigma
        var += limit
    return np.mean(X, axis=0) / np.sqrt(var / X.shape[0])


def ttest_ind_no_p(a, b, equal_var=True, sigma=0.):
    """Independent samples t-test without p calculation.

    This is a modified version of :func:`scipy.stats.ttest_ind`. It operates
    along the first axis. The ``sigma`` parameter provides an optional "hat"
    adjustment (see :func:`ttest_1samp_no_p` and :footcite:`RidgwayEtAl2012`).

    Parameters
    ----------
    a : array-like
        The first array.
    b : array-like
        The second array.
    equal_var : bool
        Assume equal variance. See :func:`scipy.stats.ttest_ind`.
    sigma : float
        The regularization. See :func:`ttest_1samp_no_p`.

    Returns
    -------
    t : array
        T values.

    References
    ----------
    .. footbibliography::
    """
    v1 = np.var(a, axis=0, ddof=1)
    v2 = np.var(b, axis=0, ddof=1)
    n1 = a.shape[0]
    n2 = b.shape[0]
    if equal_var:
        df = n1 + n2 - 2.0
        var = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        var = var * (1.0 / n1 + 1.0 / n2)
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        var = vn1 + vn2
    if sigma > 0:
        var += sigma * np.max(var)
    denom = np.sqrt(var)
    d = np.mean(a, 0) - np.mean(b, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(d, denom)
    return t


def f_oneway(*args):
    """Perform a 1-way ANOVA.

    The one-way ANOVA tests the null hypothesis that 2 or more groups have
    the same population mean. The test is applied to samples from two or
    more groups, possibly with differing sizes :footcite:`Lowry2014`.

    This is a modified version of :func:`scipy.stats.f_oneway` that avoids
    computing the associated p-value.

    Parameters
    ----------
    *args : array_like
        The sample measurements should be given as arguments.

    Returns
    -------
    F-value : float
        The computed F-value of the test.

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.

    1. The samples are independent
    2. Each sample is from a normally distributed population
    3. The population standard deviations of the groups are all equal.  This
       property is known as homocedasticity.

    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (:func:`scipy.stats.kruskal`)
    although with some loss of power

    The algorithm is from Heiman :footcite:`Heiman2002`, pp.394-7.

    References
    ----------
    .. footbibliography::
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
    return f


def _map_effects(n_factors, effects):
    """Map effects to indices."""
    if n_factors > len(ascii_uppercase):
        raise ValueError('Maximum number of factors supported is 26')

    factor_names = list(ascii_uppercase[:n_factors])

    if isinstance(effects, str):
        if '*' in effects and ':' in effects:
            raise ValueError('Not "*" and ":" permitted in effects')
        elif '+' in effects and ':' in effects:
            raise ValueError('Not "+" and ":" permitted in effects')
        elif effects == 'all':
            effects = None
        elif len(effects) == 1 or ':' in effects:
            effects = [effects]
        elif '+' in effects:
            # all main effects
            effects = effects.split('+')
        elif '*' in effects:
            pass  # handle later
        else:
            raise ValueError('"{}" is not a valid option for "effects"'
                             .format(effects))
    if isinstance(effects, list):
        bad_names = [e for e in effects if e not in factor_names]
        if len(bad_names) > 1:
            raise ValueError('Effect names: {} are not valid. They should '
                             'the first `n_factors` ({}) characters from the'
                             'alphabet'.format(bad_names, n_factors))

    indices = list(np.arange(2 ** n_factors - 1))
    names = list()
    for this_effect in indices:
        contrast_idx = _get_contrast_indices(this_effect + 1, n_factors)
        this_code = (n_factors - 1) - np.where(contrast_idx == 1)[0]
        this_name = [factor_names[e] for e in this_code]
        this_name.sort()
        names.append(':'.join(this_name))

    if effects is None or isinstance(effects, str):
        effects_ = names
    else:
        effects_ = effects

    selection = [names.index(sel) for sel in effects_]
    names = [names[sel] for sel in selection]

    if isinstance(effects, str):
        if '*' in effects:
            # hierarchical order of effects
            # the * based effect can be used as stop index
            sel_ind = names.index(effects.replace('*', ':')) + 1
            names = names[:sel_ind]
            selection = selection[:sel_ind]

    return selection, names


def _get_contrast_indices(effect_idx, n_factors):  # noqa: D401
    """Henson's factor coding, see num2binvec."""
    binrepr = np.binary_repr(effect_idx, n_factors)
    return np.array([int(i) for i in binrepr], dtype=int)


def _iter_contrasts(n_subjects, factor_levels, effect_picks):
    """Set up contrasts."""
    from scipy.signal import detrend
    sc = []
    n_factors = len(factor_levels)
    # prepare computation of Kronecker products
    for n_levels in factor_levels:
        # for each factor append
        # 1) column vector of length == number of levels,
        # 2) square matrix with diagonal == number of levels

        # main + interaction effects for contrasts
        sc.append([np.ones([n_levels, 1]),
                   detrend(np.eye(n_levels), type='constant')])

    for this_effect in effect_picks:
        contrast_idx = _get_contrast_indices(this_effect + 1, n_factors)
        c_ = sc[0][contrast_idx[n_factors - 1]]
        for i_contrast in range(1, n_factors):
            this_contrast = contrast_idx[(n_factors - 1) - i_contrast]
            c_ = np.kron(c_, sc[i_contrast][this_contrast])
        df1 = np.linalg.matrix_rank(c_)
        df2 = df1 * (n_subjects - 1)
        yield c_, df1, df2


def f_threshold_mway_rm(n_subjects, factor_levels, effects='A*B',
                        pvalue=0.05):
    """Compute F-value thresholds for a two-way ANOVA.

    Parameters
    ----------
    n_subjects : int
        The number of subjects to be analyzed.
    factor_levels : list-like
        The number of levels per factor.
    effects : str
        A string denoting the effect to be returned. The following
        mapping is currently supported:

            * ``'A'``: main effect of A
            * ``'B'``: main effect of B
            * ``'A:B'``: interaction effect
            * ``'A+B'``: both main effects
            * ``'A*B'``: all three effects

    pvalue : float
        The p-value to be thresholded.

    Returns
    -------
    F_threshold : list | float
        List of F-values for each effect if the number of effects
        requested > 2, else float.

    See Also
    --------
    f_oneway
    f_mway_rm

    Notes
    -----
    .. versionadded:: 0.10
    """
    from scipy.stats import f
    effect_picks, _ = _map_effects(len(factor_levels), effects)

    F_threshold = []
    for _, df1, df2 in _iter_contrasts(n_subjects, factor_levels,
                                       effect_picks):
        F_threshold.append(f(df1, df2).isf(pvalue))

    return F_threshold if len(F_threshold) > 1 else F_threshold[0]


def f_mway_rm(data, factor_levels, effects='all',
              correction=False, return_pvals=True):
    """Compute M-way repeated measures ANOVA for fully balanced designs.

    Parameters
    ----------
    data : ndarray
        3D array where the first two dimensions are compliant
        with a subjects X conditions scheme where the first
        factor repeats slowest::

                        A1B1 A1B2 A2B1 A2B2
            subject 1   1.34 2.53 0.97 1.74
            subject ... .... .... .... ....
            subject k   2.45 7.90 3.09 4.76

        The last dimensions is thought to carry the observations
        for mass univariate analysis.
    factor_levels : list-like
        The number of levels per factor.
    effects : str | list
        A string denoting the effect to be returned. The following
        mapping is currently supported (example with 2 factors):

            * ``'A'``: main effect of A
            * ``'B'``: main effect of B
            * ``'A:B'``: interaction effect
            * ``'A+B'``: both main effects
            * ``'A*B'``: all three effects
            * ``'all'``: all effects (equals 'A*B' in a 2 way design)

        If list, effect names are used: ``['A', 'B', 'A:B']``.
    correction : bool
        The correction method to be employed if one factor has more than two
        levels. If True, sphericity correction using the Greenhouse-Geisser
        method will be applied.
    return_pvals : bool
        If True, return p-values corresponding to F-values.

    Returns
    -------
    F_vals : ndarray
        An array of F-statistics with length corresponding to the number
        of effects estimated. The shape depends on the number of effects
        estimated.
    p_vals : ndarray
        If not requested via return_pvals, defaults to an empty array.

    See Also
    --------
    f_oneway
    f_threshold_mway_rm

    Notes
    -----
    .. versionadded:: 0.10
    """
    from scipy.stats import f

    if data.ndim == 2:  # general purpose support, e.g. behavioural data
        data = data[:, :, np.newaxis]
    elif data.ndim > 3:  # let's allow for some magic here.
        data = data.reshape(
            data.shape[0], data.shape[1], np.prod(data.shape[2:]))

    effect_picks, _ = _map_effects(len(factor_levels), effects)
    n_obs = data.shape[2]
    n_replications = data.shape[0]

    # put last axis in front to 'iterate' over mass univariate instances.
    data = np.rollaxis(data, 2)
    fvalues, pvalues = [], []
    for c_, df1, df2 in _iter_contrasts(n_replications, factor_levels,
                                        effect_picks):
        y = np.dot(data, c_)
        b = np.mean(y, axis=1)[:, np.newaxis, :]
        ss = np.sum(np.sum(y * b, axis=2), axis=1)
        mse = (np.sum(np.sum(y * y, axis=2), axis=1) - ss) / (df2 / df1)
        fvals = ss / mse
        fvalues.append(fvals)
        if correction:
            # sample covariances, leave off "/ (y.shape[1] - 1)" norm because
            # it falls out.
            v = np.array([np.dot(y_.T, y_) for y_ in y])
            v = (np.array([np.trace(vv) for vv in v]) ** 2 /
                 (df1 * np.sum(np.sum(v * v, axis=2), axis=1)))
            eps = v

        df1, df2 = np.zeros(n_obs) + df1, np.zeros(n_obs) + df2
        if correction:
            # numerical imprecision can cause eps=0.99999999999999989
            # even with a single category, so never let our degrees of
            # freedom drop below 1.
            df1, df2 = [np.maximum(d[None, :] * eps, 1.) for d in (df1, df2)]

        if return_pvals:
            pvals = f(df1, df2).sf(fvals)
        else:
            pvals = np.empty(0)
        pvalues.append(pvals)

    # handle single effect returns
    return [np.squeeze(np.asarray(vv)) for vv in (fvalues, pvalues)]


def _parametric_ci(arr, ci=.95):
    """Calculate the `ci`% parametric confidence interval for `arr`."""
    mean = arr.mean(0)
    if len(arr) < 2:  # can't compute standard error
        sigma = np.full_like(mean, np.nan)
        return mean, sigma
    from scipy import stats
    sigma = stats.sem(arr, 0)
    return stats.t.interval(ci, loc=mean, scale=sigma, df=arr.shape[0])
