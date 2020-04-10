# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..defaults import _handle_default
from ..fixes import _safe_svd
from ..utils import warn, logger, sqrtm_sym, eigh


# For the reference implementation of eLORETA (force_equal=False),
# 0 < loose <= 1 all produce solutions that are (more or less)
# the same as free orientation (loose=1) and quite different from
# loose=0 (fixed). If we do force_equal=True, we get a visibly smooth
# transition from 0->1. This is probably because this mode behaves more like
# sLORETA and dSPM in that it weights each orientation for a given source
# uniformly (which is not the case for the reference eLORETA implementation).
#
# If we *reapply the orientation prior* after each eLORETA iteration,
# we can preserve the smooth transition without requiring force_equal=True,
# which is probably more representative of what eLORETA should do. But this
# does not produce results that pass the eye test.

def _compute_eloreta(inv, lambda2, options):
    """Compute the eLORETA solution."""
    from .inverse import compute_rank_inverse, _compute_reginv
    options = _handle_default('eloreta_options', options)
    eps, max_iter = options['eps'], options['max_iter']
    force_equal = bool(options['force_equal'])  # None means False

    # Reassemble the gain matrix (should be fast enough)
    if inv['eigen_leads_weighted']:
        # We can probably relax this if we ever need to
        raise RuntimeError('eLORETA cannot be computed with weighted eigen '
                           'leads')
    G = np.dot(inv['eigen_fields']['data'].T * inv['sing'],
               inv['eigen_leads']['data'].T).astype(np.float64)
    n_nzero = compute_rank_inverse(inv)
    G /= np.sqrt(inv['source_cov']['data'])
    # restore orientation prior
    source_std = np.ones(G.shape[1])
    if inv['orient_prior'] is not None:
        source_std *= np.sqrt(inv['orient_prior']['data'])
    G *= source_std
    # We do not multiply by the depth prior, as eLORETA should compensate for
    # depth bias.
    n_src = inv['nsource']
    n_chan, n_orient = G.shape
    n_orient //= n_src
    assert n_orient in (1, 3)
    logger.info('    Computing optimized source covariance (eLORETA)...')
    if n_orient == 3:
        logger.info('        Using %s orientation weights'
                    % ('uniform' if force_equal else 'independent',))
    # src, sens, 3
    G_3 = _get_G_3(G, n_orient)
    if n_orient != 1 and not force_equal:
        # Outer product
        R_prior = (source_std.reshape(n_src, 1, 3) *
                   source_std.reshape(n_src, 3, 1))
    else:
        R_prior = source_std ** 2

    def _normalize_R(G, R):
        """Normalize R so that lambda2 is consistent."""
        if n_orient == 1 or force_equal:
            R_Gt = R[:, np.newaxis] * G.T
        else:
            R_Gt = np.matmul(R, G_3).reshape(n_src * 3, -1)
        G_R_Gt = np.dot(G, R_Gt)
        norm = np.trace(G_R_Gt) / n_nzero
        G_R_Gt /= norm
        R /= norm
        return G_R_Gt

    # The following was adapted under BSD license by permission of Guido Nolte
    if force_equal or n_orient == 1:
        R_shape = (n_src * n_orient,)
        R = np.ones(R_shape)
    else:
        R_shape = (n_src, n_orient, n_orient)
        R = np.empty(R_shape)
        R[:] = np.eye(n_orient)[np.newaxis]
    R *= R_prior
    G_R_Gt = _normalize_R(G, R)
    extra = ' (this make take a while)' if n_orient == 3 else ''
    logger.info('        Fitting up to %d iterations%s...'
                % (max_iter, extra))
    for kk in range(max_iter):
        # 1. Compute inverse of the weights (stabilized) and C
        s, u = eigh(G_R_Gt)
        s = abs(s)
        sidx = np.argsort(s)[::-1][:n_nzero]
        s, u = s[sidx], u[:, sidx]
        with np.errstate(invalid='ignore'):
            s = np.where(s > 0, 1 / (s + lambda2), 0)
        N = np.dot(u * s, u.T)
        del s

        # Update the weights
        R_last = R.copy()
        if n_orient == 1:
            R[:] = 1. / np.sqrt((np.dot(N, G) * G).sum(0))
        else:
            M = np.matmul(np.matmul(G_3, N[np.newaxis]), G_3.swapaxes(-2, -1))
            if force_equal:
                _, s = sqrtm_sym(M, inv=True)
                R[:] = np.repeat(1. / np.mean(s, axis=-1), 3)
            else:
                R[:], _ = sqrtm_sym(M, inv=True)
        R *= R_prior  # reapply our prior, eLORETA undoes it
        G_R_Gt = _normalize_R(G, R)

        # Check for weight convergence
        delta = (np.linalg.norm(R.ravel() - R_last.ravel()) /
                 np.linalg.norm(R_last.ravel()))
        logger.debug('            Iteration %s / %s ...%s (%0.1e)'
                     % (kk + 1, max_iter, extra, delta))
        if delta < eps:
            logger.info('        Converged on iteration %d (%0.2g < %0.2g)'
                        % (kk, delta, eps))
            break
    else:
        warn('eLORETA weight fitting did not converge (>= %s)' % eps)
    logger.info('        Updating inverse with weighted eigen leads')
    G /= source_std  # undo our biasing
    G_3 = _get_G_3(G, n_orient)
    _normalize_R(G, R)
    if n_orient == 1 or force_equal:
        R_sqrt = np.sqrt(R)
    else:
        R_sqrt = sqrtm_sym(R)[0]
    assert R_sqrt.shape == R_shape
    A = _R_sqrt_mult(G, R_sqrt)
    del R, G  # the rest will be done in terms of R_sqrt and A
    eigen_fields, sing, eigen_leads = _safe_svd(A, full_matrices=False)
    inv['sing'][:] = sing
    inv['reginv'][:] = _compute_reginv(inv, lambda2)
    inv['eigen_leads_weighted'] = True
    inv['eigen_leads']['data'][:] = _R_sqrt_mult(eigen_leads, R_sqrt).T
    inv['eigen_fields']['data'][:] = eigen_fields.T
    # XXX in theory we should set inv['source_cov'] properly.
    # For fixed ori (or free ori with force_equal=True), we can as these
    # are diagonal matrices. But for free ori without force_equal, it's a
    # block diagonal 3x3 and we have no efficient way of storing this (and
    # storing a covariance matrix with (20484 * 3) ** 2 elements is not going
    # to work. So let's just set to nan for now.
    # It's not used downstream anyway now that we set
    # eigen_leads_weighted = True.
    inv['source_cov']['data'].fill(np.nan)
    logger.info('[done]')


def _get_G_3(G, n_orient):
    if n_orient == 1:
        return None
    else:
        return np.ascontiguousarray(
            G.reshape(G.shape[0], -1, n_orient).transpose(1, 2, 0))


def _R_sqrt_mult(other, R_sqrt):
    """Do other @ R ** 0.5."""
    if R_sqrt.ndim == 1:
        assert other.shape[1] == R_sqrt.size
        out = R_sqrt * other
    else:
        assert R_sqrt.shape[1:3] == (3, 3)
        assert other.shape[1] == np.prod(R_sqrt.shape[:2])
        assert other.ndim == 2
        n_src = R_sqrt.shape[0]
        n_chan = other.shape[0]
        out = np.matmul(
            R_sqrt, other.reshape(n_chan, n_src, 3).transpose(1, 2, 0)
        ).reshape(n_src * 3, n_chan).T
    return out
