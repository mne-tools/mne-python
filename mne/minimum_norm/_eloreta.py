# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from scipy import linalg

from ..defaults import _handle_default
from ..fixes import _safe_svd
from ..utils import warn, logger


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

    # eps=1e-6, max_iter=20, force_equal=False):
    # Reassemble the gain matrix (should be fast enough)
    if inv['eigen_leads_weighted']:
        # We can probably relax this if we ever need to
        raise RuntimeError('eLORETA cannot be computed with weighted eigen '
                           'leads')
    G = np.dot(inv['eigen_fields']['data'].T * inv['sing'],
               inv['eigen_leads']['data'].T)
    # Getting this term right does not seem to be totally necessary...
    # n_nzero = gain.shape[0]
    n_nzero = int(round((G * G).sum()))
    n_src = inv['nsource']
    n_chan, n_orient = G.shape
    n_orient //= n_src
    assert n_orient in (1, 3)
    if n_orient == 3:
        logger.info('        Using %s orientation weights'
                    % ('uniform' if force_equal else 'independent',))

    # The following was adapted under BSD license by permission of Guido Nolte
    shape = (n_src,)
    shape += () if force_equal or n_orient == 1 else (n_orient, n_orient)
    W = np.empty(shape)
    W[:] = 1. if force_equal or n_orient == 1 else np.eye(n_orient)[np.newaxis]
    # Here we keep the weights normalized to roughly n_src * n_orient.
    # Not sure if there is a better way to normalize.
    extra = ' (this make take a while)' if n_orient == 3 else ''
    logger.info('        Fitting up to %d iterations%s...'
                % (max_iter, extra))
    for kk in range(max_iter):
        # Compute inverse of the weights (stabilized) and corresponding M
        M, _ = _compute_eloreta_inv(G, W, n_orient, n_nzero, lambda2,
                                    force_equal)

        # Update the weights
        W_last = W.copy()
        if n_orient == 1:
            W[:] = np.sqrt((np.dot(M, G) * G).sum(0))
            W /= W.sum() / n_src
        else:
            norm = 0.
            for ii in range(n_src):
                sl = slice(n_orient * ii, n_orient * (ii + 1))
                this_w, this_s = _sqrtm_sym(
                    np.dot(np.dot(G[:, sl].T, M), G[:, sl]))
                W[ii] = np.mean(this_s) if force_equal else this_w
                norm += np.mean(this_s)
            W /= norm / n_src

        # Check for weight convergence
        delta = (linalg.norm(W.ravel() - W_last.ravel()) /
                 linalg.norm(W_last.ravel()))
        logger.debug('            Iteration %s / %s ...%s'
                     % (kk + 1, max_iter, extra))
        if delta < eps:
            logger.info('        Converged on iteration %d (%0.2g < %0.2g)'
                        % (kk, delta, eps))
            break
    else:
        warn('eLORETA weight fitting did not converge (>= %s)' % eps)
    logger.info('        Assembling eLORETA kernel and modifying inverse')
    M, W_inv = _compute_eloreta_inv(G, W, n_orient, n_nzero, lambda2,
                                    force_equal)
    K = np.zeros((n_src * n_orient, n_chan))
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
    logger.info('[done]')
    return W


def _compute_eloreta_inv(G, W, n_orient, n_nzero, lambda2, force_equal):
    """Invert weights and compute M."""
    W_inv = np.empty_like(W)
    n_src = W_inv.shape[0]
    if n_orient == 1 or force_equal:
        W_inv[:] = 1. / W
    else:
        for ii in range(n_src):
            # Here we use a single-precision-suitable `rcond` (given our
            # 3x3 matrix size) because the inv could be saved in single
            # precision.
            W_inv[ii] = linalg.pinv2(W[ii], rcond=1e-7)

    # Weight the gain matrix
    W_inv_Gt = np.empty_like(G).T
    for ii in range(n_src):
        sl = slice(n_orient * ii, n_orient * (ii + 1))
        W_inv_Gt[sl, :] = np.dot(W_inv[ii], G[:, sl].T)

    # Compute the inverse, normalizing by the trace
    G_W_inv_Gt = np.dot(G, W_inv_Gt)
    G_W_inv_Gt *= n_nzero / np.trace(G_W_inv_Gt)
    u, s, v = linalg.svd(G_W_inv_Gt)
    s = s / (s ** 2 + lambda2)
    M = np.dot(v.T[:, :n_nzero] * s[:n_nzero], u.T[:n_nzero])
    return M, W_inv


def _sqrtm_sym(C):
    """Compute the square root of a symmetric matrix."""
    # Same as linalg.sqrtm(C) but faster, also yields the eigenvalues
    s, u = linalg.eigh(C)
    mask = s > s.max() * 1e-7
    u = u[:, mask]
    s = np.sqrt(s[mask])
    a = np.dot(s * u, u.T)
    return a, s
