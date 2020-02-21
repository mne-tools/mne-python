# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..defaults import _handle_default
from ..fixes import _safe_svd, pinv
from ..utils import (warn, logger, _svd_lwork, _repeated_svd, sqrtm_sym)


# For the reference implementation of eLORETA (force_equal=False),
# 0 < loose <= 1 all produce solutions that are (more or less)
# the same as free orientation (loose=1) and quite different from
# loose=0 (fixed). If we do force_equal=True, we get a visibly smooth
# transition from 0->1. This is probably because this mode behaves more like
# sLORETA and dSPM in that it weights each orientation for a given source
# uniformly (which is not the case for the reference eLORETA implementation).

def _compute_eloreta(inv, lambda2, options):
    """Compute the eLORETA solution."""
    options = _handle_default('eloreta_options', options)
    eps, max_iter = options['eps'], options['max_iter']
    force_equal = options['force_equal']
    if force_equal is None:  # default -> figure it out
        is_loose = not (inv['orient_prior'] is None or
                        np.allclose(inv['orient_prior']['data'], 1.))
        force_equal = True if is_loose else False
    force_equal = bool(force_equal)

    # Reassemble the gain matrix (should be fast enough)
    if inv['eigen_leads_weighted']:
        # We can probably relax this if we ever need to
        raise RuntimeError('eLORETA cannot be computed with weighted eigen '
                           'leads')
    # This "G" is really "A" in our forumalation:
    # A = G @ R_sqrt
    # But let's proceed with eLORETA as if this weighted gain matrix
    # were the correct one (it makes things like lambda reasonable).
    G = np.dot(inv['eigen_fields']['data'].T * inv['sing'],
               inv['eigen_leads']['data'].T).astype(np.float64)
    n_nzero = int(round((G * G).sum()))
    orig_R = inv['source_cov']['data']
    # G /= np.sqrt(orig_R)
    del orig_R
    n_src = inv['nsource']
    n_chan, n_orient = G.shape
    n_orient //= n_src
    assert n_orient in (1, 3)
    if n_orient == 3:
        logger.info('        Using %s orientation weights'
                    % ('uniform' if force_equal else 'independent',))
        # src, sens, 3
        G_3 = np.ascontiguousarray(G.reshape(-1, n_src, 3).transpose(1, 2, 0))
    else:
        G_3 = None

    # The following was adapted under BSD license by permission of Guido Nolte
    shape = (n_src,)
    shape += () if force_equal or n_orient == 1 else (n_orient, n_orient)
    W = np.empty(shape)
    W[:] = 1. if force_equal or n_orient == 1 else np.eye(n_orient)[np.newaxis]
    extra = ' (this make take a while)' if n_orient == 3 else ''
    logger.info('        Fitting up to %d iterations%s...'
                % (max_iter, extra))
    svd_lwork = _svd_lwork((G.shape[0], G.shape[0]))
    for kk in range(max_iter):
        # 1. Compute inverse of the weights (stabilized) and C
        R = _w_inv(W, n_orient, force_equal)
        if n_orient == 1 or force_equal:
            R_Gt = G.T * np.repeat(R, n_orient)[:, np.newaxis]
        else:
            R_Gt = np.matmul(R, G_3).reshape(n_src * 3, -1)
        G_R_Gt = np.dot(G, R_Gt)
        loading_factor = np.trace(G_R_Gt) / n_nzero
        u, s, v = _repeated_svd(G_R_Gt, lwork=svd_lwork)
        s = s[:n_nzero] / (s[:n_nzero] ** 2 + lambda2 * loading_factor ** 2)
        N = np.dot(v.T[:, :n_nzero] * s, u.T[:n_nzero])

        # Update the weights
        W_last = W.copy()
        if n_orient == 1:
            W[:] = np.sqrt((np.dot(N, G) * G).sum(0))
        else:
            w, s, = sqrtm_sym(
                np.matmul(np.matmul(G_3, N[np.newaxis]), G_3.swapaxes(-2, -1)))
            if force_equal:
                W[:] = np.mean(s, axis=-1)
            else:
                W[:] = w

        # Check for weight convergence
        delta = (np.linalg.norm(W.ravel() - W_last.ravel()) /
                 np.linalg.norm(W_last.ravel()))
        logger.debug('            Iteration %s / %s ...%s (%0.1e)'
                     % (kk + 1, max_iter, extra, delta))
        if delta < eps:
            logger.info('        Converged on iteration %d (%0.2g < %0.2g)'
                        % (kk, delta, eps))
            break
    else:
        warn('eLORETA weight fitting did not converge (>= %s)' % eps)
    logger.info('        Assembling eLORETA kernel and modifying inverse')

    R = _w_inv(W, n_orient, force_equal)
    if n_orient == 1 or force_equal:
        R_Gt = G.T * np.repeat(R, n_orient)[:, np.newaxis]
    else:
        R_Gt = np.matmul(R, G_3).reshape(n_src * 3, -1)
    G_R_Gt = np.dot(G, R_Gt)
    """
    if n_orient == 1 or force_equal:
        G *= np.sqrt(R)
    else:
        G = np.matmul(G_3.swapaxes(1, 2),
                      sqrtm_sym(R)[0]).swapaxes(0, 1).reshape(n_chan, -1)
    inv['source_cov']['data'] = R
    del R
    eigen_fields, sing, eigen_leads = _safe_svd(G, full_matrices=False)
    loading_factor = np.trace(np.dot(G, G.T)) / n_nzero
    del G
    with np.errstate(invalid='ignore'):  # if lambda2==0
        reginv = np.where(sing > 0, sing / (sing ** 2 + lambda2 * loading_factor ** 2), 0)
    inv['eigen_leads']['data'][:] = eigen_leads.T
    inv['sing'][:] = sing
    inv['reginv'][:] = reginv
    inv['eigen_fields']['data'][:] = eigen_fields.T
    "" = np.zeros((n_src * n_orient, n_chan))
    for ii in range(n_src):
        sl = slice(n_orient * ii, n_orient * (ii + 1))
        K[sl] = np.dot(W_inv[ii], np.dot(G.T[sl], M))
    # Avoid the scaling to get to currents
    K /= np.sqrt(inv['source_cov']['data'])[:, np.newaxis]
    # eLORETA seems to break our simple relationships with noisenorm etc.,
    # but we can get around it by making our eventual dots do the right thing
    eigen_leads, reginv, eigen_fields = _safe_svd(K, full_matrices=False)
    inv['eigen_leads']['data'] = eigen_leads
    inv['reginv'] = reginv
    inv['eigen_fields']['data'] = eigen_fields
    """
    inv['source_cov']['data'].fill(1.)
    K = np.zeros((n_src * n_orient, n_chan))
    loading_factor = np.trace(G_R_Gt) / n_nzero
    u, s, v = _repeated_svd(G_R_Gt, lwork=svd_lwork)
    s = s[:n_nzero] / (s[:n_nzero] ** 2 + lambda2 * loading_factor ** 2)
    N = np.dot(v.T[:, :n_nzero] * s, u.T[:n_nzero])
    for ii in range(n_src):
        sl = slice(n_orient * ii, n_orient * (ii + 1))
        K[sl] = np.dot(R[ii], np.dot(G.T[sl], N))
    # eLORETA seems to break our simple relationships with noisenorm etc.,
    # but we can get around it by making our eventual dots do the right thing
    eigen_leads, reginv, eigen_fields = _safe_svd(K, full_matrices=False)
    inv['eigen_leads']['data'] = eigen_leads
    inv['reginv'] = reginv
    inv['eigen_fields']['data'] = eigen_fields
    logger.info('[done]')


def _w_inv(W, n_orient, force_equal):
    """Invert weights to compute the source covariance matrix."""
    if n_orient == 1 or force_equal:
        R = 1. / W
    else:
        # Here we use a single-precision-suitable `rcond` (given our
        # 3x3 matrix size) because the inv could be saved in single
        # precision.
        R = pinv(W, rcond=1e-7, hermitian=True)
    return R
