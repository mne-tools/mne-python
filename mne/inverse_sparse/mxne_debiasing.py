# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from math import sqrt

import numpy as np

from ..utils import check_random_state, fill_doc, logger, verbose


@fill_doc
def power_iteration_kron(A, C, max_iter=1000, tol=1e-3, random_state=0):
    """Find the largest singular value for the matrix kron(C.T, A).

    It uses power iterations.

    Parameters
    ----------
    A : array
        An array
    C : array
        An array
    max_iter : int
        Maximum number of iterations
    %(random_state)s

    Returns
    -------
    L : float
        largest singular value

    Notes
    -----
    http://en.wikipedia.org/wiki/Power_iteration
    """
    AS_size = C.shape[0]
    rng = check_random_state(random_state)
    B = rng.randn(AS_size, AS_size)
    B /= np.linalg.norm(B, "fro")
    ATA = np.dot(A.T, A)
    CCT = np.dot(C, C.T)
    L0 = np.inf
    for _ in range(max_iter):
        Y = np.dot(np.dot(ATA, B), CCT)
        L = np.linalg.norm(Y, "fro")

        if abs(L - L0) < tol:
            break

        B = Y / L
        L0 = L
    return L


@verbose
def compute_bias(M, G, X, max_iter=1000, tol=1e-6, n_orient=1, verbose=None):
    """Compute scaling to correct amplitude bias.

    It solves the following optimization problem using FISTA:

    min 1/2 * (|| M - GDX ||fro)^2
    s.t. D >= 1 and D is a diagonal matrix

    Reference for the FISTA algorithm:
    Amir Beck and Marc Teboulle
    A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
    Problems, SIAM J. Imaging Sci., 2(1), 183-202. (20 pages)
    http://epubs.siam.org/doi/abs/10.1137/080716542

    Parameters
    ----------
    M : array
        measurement data.
    G : array
        leadfield matrix.
    X : array
        reconstructed time courses with amplitude bias.
    max_iter : int
        Maximum number of iterations.
    tol : float
        The tolerance on convergence.
    n_orient : int
        The number of orientations (1 for fixed and 3 otherwise).
    %(verbose)s

    Returns
    -------
    D : array
        Debiasing weights.
    """
    n_sources = X.shape[0]

    lipschitz_constant = 1.1 * power_iteration_kron(G, X)

    # initializations
    D = np.ones(n_sources)
    Y = np.ones(n_sources)
    t = 1.0

    for i in range(max_iter):
        D0 = D

        # gradient step
        R = M - np.dot(G * Y, X)
        D = Y + np.sum(np.dot(G.T, R) * X, axis=1) / lipschitz_constant
        # Equivalent but faster than:
        # D = Y + np.diag(np.dot(np.dot(G.T, R), X.T)) / lipschitz_constant

        # prox ie projection on constraint
        if n_orient != 1:  # take care of orientations
            # The scaling has to be the same for all orientations
            D = np.mean(D.reshape(-1, n_orient), axis=1)
            D = np.tile(D, [n_orient, 1]).T.ravel()
        D = np.maximum(D, 1.0)

        t0 = t
        t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t**2))
        Y.fill(0.0)
        dt = (t0 - 1.0) / t
        Y = D + dt * (D - D0)

        Ddiff = np.linalg.norm(D - D0, np.inf)

        if Ddiff < tol:
            logger.info(
                f"Debiasing converged after {i} iterations "
                f"max(|D - D0| = {Ddiff:e} < {tol:e})"
            )
            break
    else:
        Ddiff = np.linalg.norm(D - D0, np.inf)
        logger.info(
            f"Debiasing did not converge after {max_iter} iterations! "
            f"max(|D - D0| = {Ddiff:e} >= {tol:e})"
        )
    return D
