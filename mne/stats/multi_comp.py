# Authors: Josef Pktd and example from H Raja and rewrite from Vincent Davis
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# Code borrowed from statsmodels
#
# License: BSD (3-clause)

import numpy as np
from scipy import stats, optimize
from ..fixes import partial


def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def fdr_correction(pvals, alpha=0.05, method='indep'):
    """P-value correction with False Discovery Rate (FDR)

    Correction for multiple comparison using FDR.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    Reference:
    Genovese CR, Lazar NA, Nichols T.
    Thresholding of statistical maps in functional neuroimaging using the false
    discovery rate. Neuroimage. 2002 Apr;15(4):870-8.
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected


def bonferroni_correction(pval, alpha=0.05):
    """P-value correction with Bonferroni method

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    """
    pval = np.asarray(pval)
    pval_corrected = pval * float(pval.size)
    reject = pval < alpha
    return reject, pval_corrected


def _local_fdr_h0_err(f, xb, par):
    """Error function for H0 subdensity used in local FDR optimization.
    """
    mu, sigma, alo, ahi = par
    sl = slice(np.argmin(np.abs(xb - alo)), np.argmin(np.abs(xb - ahi)))
    if sl.start == sl.stop:
        raise ValueError('insufficient n_bins, restart with higher value')
    f0 = stats.norm.pdf(xb[sl], loc=mu, scale=sigma)
    f1 = f[sl]
    f0 = f0 / f0.max() * f1.max()
    return np.sum((f0 - f1) ** 2) / np.sum(f1 ** 2) - f1.sum()


def _local_fdr(data, n_bins=100, h0_maxiter=500, decimate=1):
    """Local false discovery rate correction for univariate data.

    Where FDR corrects multiple comparisons based on the CDF, lFDR corrects
    based on the PDF, and provides a per-sample estimate of the false discovery
    rate, assuming a Gaussian null hypothesis.

    Parameters
    ----------
    data : array, any shape
        Set of observations on which to estimate FDR.
    n_bins : int
        Number of points at which to evaluate the density.
    h0_maxtier : int
        Max no of iteratios for H0 fit optimization.
    decimate : int
        Factor by which to decimate data prior to density estimation.

    Returns
    -------
    xb : array, (n_bins*2,)
        Points at which the densities are evaluated.
    f : array, (n_bins*2,)
        Density of x.
    cf : array, (n_bins*2,)
        Estimated "center density" of H0.
    fdr : array, (n_bins*2,)
        Estimated false discovery rate.
    mu : float
        Mean of H0 distribution.
    sig : float
        Std of H0 distribution.
    """

    x = data.reshape(-1)[::decimate]

    # initial estimate of density
    k = stats.gaussian_kde(x)
    xb = np.r_[x.min():x.max():1j * n_bins]
    f = k(xb)
    f /= f.sum()

    # adapt estimation points' density to actual density
    F = np.cumsum(f)
    Fb = np.r_[0.0:1.0:1j * n_bins]
    dxb = np.interp(Fb, F, xb + (xb[1] - xb[0]) / 2.0)
    xb = np.unique(np.r_[xb, dxb])
    xb.sort()
    f = k(xb)
    f /= f.sum()

    # initial search params
    mu0, sig0 = x.mean(), x.std()
    par0 = mu0, sig0, mu0 - sig0, mu0 + sig0

    # optimize error wrt density
    err_func = partial(_local_fdr_h0_err, f, xb)
    mu, sig, _, _ = optimize.fmin(err_func, par0, disp=0, maxiter=h0_maxiter)

    # compute center density
    cf = stats.norm.pdf(xb, loc=mu, scale=sig)
    cf = cf / cf.max() * f.max()

    # lfdr
    fdr = np.clip(cf / f, 0.0, 1.0)

    return xb, f, cf, fdr, mu, sig


def local_fdr_correction(data, n_bins=100, h0_maxiter=500, decimate=1):
    """Correction for multiple comparison based on local FDR.

    Rather than working directly with the p-values, local FDR determines
    the FDR of each sample based on empirical and assumed H0 distributions,
    and thresholds on the sample's FDR value.

    As with fdr_correction, this operates on the entire flattened array.

    Parameters
    ----------
    data : array, any shape
        Set of observations on which to estimate FDR.
    n_bins : int
        Number of points at which to evaluate the density.
    h0_maxtier : int
        Max no of iteratios for H0 fit optimization.
    decimate : int
        Factor by which to decimate data prior to density estimation.

    Returns
    -------
    qvals : array, data.shape
        Values of FDR for each element of data.

    Notes
    -----

    The use of a Gaussian in this implementation is not required, in fact, the
    distribution to be fit does not even have to have an analytic PDF, rather
    it is simply necessary to be able to compute a numerical density for the
    H0.

    Reference:
    Efron B, Tibshirani R.
    Empirical Bayes methods and false discovery rates for microarrays.
    Genetic epidemiology 2002; 23(1): 70-86.
    """

    xb, _, _, fdr, _, _ = _local_fdr(data, n_bins=n_bins,
                                     h0_maxiter=h0_maxiter, decimate=decimate)
    qvals = np.interp(data, xb, fdr, 0.0, 0.0)
    return qvals
