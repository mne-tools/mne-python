# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..defaults import _handle_default
from ..fixes import _safe_svd
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
    A = np.dot(inv['eigen_fields']['data'].T * inv['sing'],
               inv['eigen_leads']['data'].T).astype(np.float64)
    n_nzero = int(round((A * A).sum()))
    G = A / np.sqrt(inv['source_cov']['data'])
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

    def _normalize_R(G, R):
        """Normalize R so that lambda2 is consistent."""
        if n_orient == 1 or force_equal:
            R_Gt = np.repeat(R, n_orient)[:, np.newaxis] * G.T
        else:
            R_Gt = np.matmul(R, G_3).reshape(n_src * 3, -1)
        G_R_Gt = np.dot(G, R_Gt)
        norm = np.trace(G_R_Gt) / n_nzero
        G_R_Gt /= norm
        R /= norm
        return G_R_Gt

    # The following was adapted under BSD license by permission of Guido Nolte
    shape = (n_src,)
    shape += () if force_equal or n_orient == 1 else (n_orient, n_orient)
    R = np.empty(shape)
    R[:] = 1. if force_equal or n_orient == 1 else np.eye(n_orient)[np.newaxis]
    G_R_Gt = _normalize_R(G, R)
    extra = ' (this make take a while)' if n_orient == 3 else ''
    logger.info('        Fitting up to %d iterations%s...'
                % (max_iter, extra))
    svd_lwork = _svd_lwork((G.shape[0], G.shape[0]))
    for kk in range(max_iter):
        # 1. Compute inverse of the weights (stabilized) and C
        u, s, v = _repeated_svd(G_R_Gt, lwork=svd_lwork)
        s = s[:n_nzero] / (s[:n_nzero] ** 2 + lambda2)
        N = np.dot(v.T[:, :n_nzero] * s, u.T[:n_nzero])

        # Update the weights
        R_last = R.copy()
        if n_orient == 1:
            R[:] = 1. / np.sqrt((np.dot(N, G) * G).sum(0))
        else:
            M = np.matmul(np.matmul(G_3, N[np.newaxis]), G_3.swapaxes(-2, -1))
            R, s = sqrtm_sym(M, inv=True)
            if force_equal:
                R = 1. / np.mean(1. / s, axis=-1)
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
    logger.info('        Assembling eLORETA kernel and modifying inverse')

    if n_orient == 1 or force_equal:
        R_sqrt = np.sqrt(np.repeat(R, n_orient))
        A = G * R_sqrt
    else:
        R_sqrt = sqrtm_sym(R)[0]
        A = np.matmul(R_sqrt, G_3).reshape(n_src * 3, -1).T
    u, s, v = _safe_svd(A, full_matrices=False)
    with np.errstate(invalid='ignore'):  # if lambda2==0
        reginv = np.where(s > 0, s / (s ** 2 + lambda2), 0)
    inv['eigen_leads_weighted'] = True
    eigen_leads, eigen_fields = v.T, u.T
    if n_orient == 1 or force_equal:
        eigen_leads *= R_sqrt[:, np.newaxis]
    else:
        eigen_leads_3 = eigen_leads.reshape(
            n_chan, n_src, n_orient).transpose(1, 2, 0)
        eigen_leads[:] = np.matmul(
            R_sqrt, eigen_leads_3).reshape(n_src * n_orient, n_chan)
    inv['sing'][:] = s
    inv['eigen_leads']['data'][:] = eigen_leads
    inv['reginv'][:] = reginv
    inv['eigen_fields']['data'][:] = eigen_fields
    M = np.dot(eigen_leads, reginv[:, np.newaxis] * eigen_fields)
    # The direct way from their paper
    u, s, v = _repeated_svd(G_R_Gt, lwork=svd_lwork)
    s = s[:n_nzero] / (s[:n_nzero] ** 2 + lambda2)
    N = np.dot(v.T[:, :n_nzero] * s, u.T[:n_nzero])
    if n_orient == 1 or force_equal:
        R_Gt = np.repeat(R, n_orient)[:, np.newaxis] * G.T
    else:
        R_Gt = np.matmul(R, G_3)
        R_Gt.shape = (n_src * n_orient, n_chan)
    M_ = np.dot(R_Gt, N)
    # 1. Fix here:
    # np.testing.assert_allclose(M_, M)  # XXX FIX HERE, del delow
    eigen_leads, reginv, eigen_fields = _safe_svd(M_, full_matrices=False)
    inv['eigen_leads']['data'][:] = eigen_leads
    inv['reginv'][:] = reginv
    inv['eigen_fields']['data'][:] = eigen_fields
    # 2. Fix loose
    # 3. Fix force_fixed=True
    logger.info('[done]')
