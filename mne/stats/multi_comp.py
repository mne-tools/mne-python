# Authors: Josef Pktd and example from H Raja and rewrite from Vincent Davis
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# Code borrowed from statsmodels
#
# License: BSD (3-clause)

import numpy as np

from scipy import stats, optimize

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


def local_fdr(data, nbins=100, h0_maxiter=500, decimate=1):
    """
    Implements the local false discovery rate correction, due to Efron 2005, 
    for univariate data.

    Where FDR performs a correction based on the CDF, lFDR performs a
    correction based on the PDF, which works by computing density statistics on
    `x`, where a "null" hypothesis distribution H0, here a Gaussian
    distribution, is fit to the center of the data, and this provides an
    estimation of how likely each bin comes from the Gaussian distribution, a
    measure called the local false discovery rate.

    The use of a Gaussian in this implementation is not required, in fact, the
    distribution to be fit does not even have to have an analytic PDF, rather
    it is simply necessary to be able to compute a numerical density for the
    H0.

    This function performs the density and FDR estimation, while
    `local_fdr_correction` provides an interface similar to `fdr_correction`.


    Parameters
    ----------
    nbins : int
        Number of points at which to evaluate the density.
    h0_maxtier : int
        Max no of iteratios for H0 fit optimization
    decimate : int
        Factor by which to decimate data prior to density estimation

    Returns
    -------
    xb : array
        Points at which the densities are evaluated
    f : array
        Density of `x`
    cf : array
        Estimated "center density" of H0
    fdr : array
        Estimated false discovery rate 
    mu : float
        Mean of H0 distribution
    sig : float
        Std of H0 distribution

    Notes
    -----

    Reference:
    Efron B, Local false discovery rates. 2005

    """

    x = data.reshape(-1)[::decimate]

    # initial estimate of density
    k = stats.gaussian_kde(x)
    xb = np.r_[x.min() : x.max() : 1j*nbins]
    f = k(xb)
    f /= f.sum()
    
    # adapt estimation points' density to actual density
    F = np.cumsum(f)
    Fb = np.r_[0.0:1.0:1j*nbins]
    dxb = np.interp(Fb, F, xb + (xb[1] - xb[0])/2.0)
    xb = np.unique(np.r_[xb, dxb])
    xb.sort()                
    f = k(xb)
    f /= f.sum()

    # estimate center sub-density
    def err(par):
        mu, sigma, alo, ahi = par
        sl = slice(np.argmin(np.abs(xb - alo)), np.argmin(np.abs(xb - ahi)))
        if sl.start == sl.stop:
            raise ValueError('insufficient nbins, restart with higher value')
        f0 = stats.norm.pdf(xb[sl], loc=mu, scale=sigma)
        f1 = f[sl]
        f0 = f0/f0.max()*f1.max()
        return np.sum((f0 - f1)**2)/np.sum(f1**2) - f1.sum()
    
    # initialize opt with guess
    mu0, sig0 = x.mean(), x.std()
    par0 = mu0, sig0, mu0-sig0, mu0+sig0
    mu, sig, _, _ = optimize.fmin(err, par0, disp=0, maxiter=h0_maxiter)

    # compute center density
    cf = stats.norm.pdf(xb, loc=mu, scale=sig)
    cf = cf/cf.max()*f.max()

    # lfdr 
    fdr = np.clip(cf/f, 0.0, 1.0)

    return xb, f, cf, fdr, mu, sig


def local_fdr_correction(data, q=0.2, **lfdr_kwds):
    """
    Correction for multiple comparison based on local FDR.

    Rather than working directly with the p-values, local FDR determines
    the FDR of each sample based on empirical and assumed H0 distributions,
    and thresholds on the sample's FDR value.

    Parameters
    ----------

    data : array_like
        set of observations to test
    q : float
        acceptable false discovery rate below which H0 rejected

    **lfdr_kwds : 
        keywords to pass to the local_fdr routine
  
    Returns
    -------

    reject : array, bool
        True if H0 rejected else False
    qvals : array
        Values of FDR for each element of data

    """

    xb, _, _, fdr, _, _ = local_fdr(data, **lfdr_kwds)
    qvals = np.interp(data, xb, fdr, 0.0, 0.0)
    return qvals <= q, qvals
    




