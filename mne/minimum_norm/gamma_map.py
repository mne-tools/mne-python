import numpy as np
import pylab as pl
from scipy import linalg


def gamma_map_inverse(M, G, noise_cov, maxit=500, tol=1e-20,
                      maxit_n_active=200, update_mode=1, gammas=None,
                      display_energy=True):
    """Hierarchical Bayes (Gamma-MAP)

    Parameters
    ----------
    evoked : instance of Evoked or list of instance of Evoked
        Evoked data to invert
    forward : dict
        Forward operator
    noise_cov : instance of Covariance
        Noise covariance to compute whitener
    maxit : int
        Maximum number of iterations
    tol : float
        Tolerance parameter
    maxit_n_active : integer

    gammas : array

    Returns
    -------
    stc : dict
        Source time courses

    References
    ----------
    Wipf et al. Analysis of Empirical Bayesian Methods for Neuroelectromagnetic
    Source Localization. Advances in Neural Information Processing Systems (2007)
    """
    M = M.copy()
    noise_cov = noise_cov.copy()

    if gammas is None:
        gammas = np.ones(G.shape[1], dtype=np.float)

    eps = np.finfo(float).eps

    n_sensors, n_times = M.shape
    Minit = M.copy()

    MMt = np.dot(M, M.T)
    normalize_constant = linalg.norm(MMt, ord='fro')
    M /= np.sqrt(normalize_constant)
    Minit /= np.sqrt(normalize_constant)
    MMt /= normalize_constant
    noise_cov /= normalize_constant

    G_normalize_constant = linalg.norm(G, ord=np.inf)
    G /= G_normalize_constant

    energy = np.nan

    Ginit = G.copy()
    n_points = G.shape[1]
    n_active = n_points
    active_set = np.arange(n_points)
    E = []

    counter_n_active_fixed = 0

    for k in np.arange(maxit):
        counter_n_active_fixed += 1

        gammas[np.isnan(gammas)] = 0.0

        gidx = (np.abs(gammas) > eps)
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            G = G[:,gidx]
            counter_n_active_fixed = 0

        CM = noise_cov + np.dot(G * gammas[np.newaxis, :], G.T)

        # Invert CM keeping symmetry
        U, S, V = linalg.svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        CM = np.dot(U * S, U.T)
        CMinv = np.dot(U / (S + eps), U.T)

        CMinvG = np.dot(CMinv, G)
        A = np.dot(CMinvG.T, M)

        if update_mode == 1:
            # Default update rule for the gammas
            gammas = gammas**2 * np.mean(np.abs(A)**2, axis=1) + gammas * (1 - gammas * np.sum(G * CMinvG).T)
        elif update_mode == 2:
            # MacKay fixed point update (equivalent to Variational-Bayes Sato update in hbi_inverse.m)
            gammas *= np.mean(np.abs(A)**2, axis=1) / np.sum(G * CMinvG).T
        elif update_mode == 3:
            # modified MacKay fixed point update
            gammas *= np.sqrt(np.mean(np.abs(A)**2, axis=1) / np.sum(G * CMinvG).T)

        energy_old = energy
        _, logdet_CM = np.linalg.slogdet(CM)

        # log likelihood of gaussian density
        energy = 0.5*(n_times * (logdet_CM + n_sensors) * np.log(2 * np.pi)) + np.abs(np.trace(np.dot(MMt, CMinv)))
        print energy
        E.append(energy)

        if display_energy:
            if (k > 2) and ((k % 1) == 0):
                pl.plot(k, energy,'ro');
                pl.xlabel('Iteration')
                pl.ylabel('Cost function')
                pl.show()

        err = np.abs(energy - energy_old) / np.abs(energy_old)

        if err < tol:
            break

        if counter_n_active_fixed > maxit_n_active:
            break

        if n_active == 0:
            break

    if k < maxit - 1:
        print('\nConvergence reached !\n')
    else:
        print('\nConvergence NOT reached !\n')

    full_gammas = np.zeros(n_points)
    full_gammas[active_set] = gammas
    gammas = full_gammas

    Ginv = np.dot(gammas[:, np.newaxis] * Ginit.T, CMinv)
    Ginv /= G_normalize_constant
    X = np.dot(Ginv, Minit)
    X = np.dot(X, np.sqrt(normalize_constant))

    return X

if __name__ == '__main__':
    X = np.zeros((10, 4))
    X[1] = 1.0
    G = np.random.randn(5, 10)
    M = np.dot(G, X)
    std_dev = 0.1
    M += std_dev * np.random.randn(*M.shape)
    noise_cov = std_dev * np.eye(5)

    X_hat = gamma_map_inverse(M, G, noise_cov, maxit=500, tol=1e-20,
                              maxit_n_active=200, update_mode=2, gammas=None,
                              display_energy=True)

    print X_hat
