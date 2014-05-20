# Authors: Denis Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)


import numpy as np


def find_outliers(X, threshold=0.0):
    """Find outliers based on Gaussian mixture

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n ica components)
        The outlier indices.
    """
    from sklearn.mixture import GMM
    clf = GMM(n_components=2, n_init=10, random_state=42)
    X = np.abs(X)
    probability = clf.fit(X).score(X)
    bad_ix = np.where(probability <= threshold)[0]
    return bad_ix
