# Authors: Jean-Remi king <jeanremi.king@gmail.com>
#
# License: Simplified BSD

import numpy as np


def set_contrasts(factors, level=0, labels=None):
    """Generates the conditions for n-way binary contrasts

    Parameters
    ----------
    factors : list of lists, shape(n_factors)
              Each factor must be a binary condition: e.g.
              [['left', 'right'], ['visual', 'audio']]
    level : int.
            Level of the analysis.
                0: main effect
                1: main effect + first level interaction
                ...
            Defaults to 0.
    labels : list of strings, shape(n_factors), optional.
            Labels to clarify the contrasts performed: e.g.
            ['side', 'modality']

    Returns
    -------
    contrasts : list of lists, shape (n_contrast,)
                List of elements used to make contrast
    model_labels : list of str, shape (n_contrast,)
                The label of each contrast
    """
    from itertools import combinations, product

    if np.any([len(factor) != 2 for factor in factors]):
        raise ValueError('Supports binary contrasts only.')

    if labels is None:
        labels = ['(%s-%s)' % (factor[0], factor[1]) for factor in factors]
    elif (len(labels) != len(factors) or
          not np.all(isinstance(label, str) for label in labels)):
        raise ValueError('Labels a list of strings, shape (n_factors).')
    labels = np.array(labels, dtype=str)

    sub_labels = [p for p in product(*factors)]
    sub_labels = np.array(['/'.join(label) for label in sub_labels])
    design = np.array([p for p in product(*([[1, -1]] * len(factors)))])
    contrasts = list()
    model_labels = list()
    for level_ in range(level):
        perms = [p for p in combinations(range(len(factors)), level_ + 1)]
        for perm in perms:
            model = np.prod(design[:, perm], axis=1)
            contrast = [sub_labels[model == 1].tolist(),
                        sub_labels[model == -1].tolist()]
            contrasts.append(contrast)
            model_labels.append(' * '.join(labels[np.array(perm)]))
    return contrasts, model_labels


def compute_contrast(epochs, contrasts):
    """Compute n-level contrast
    Parameters
    ----------
        epochs: mne.Epochs
            The epochs.
        contrasts : list of list of str, n_shape(n_contrast, 2, n_events)
            The contrasts
    Returns:
    --------
        X : np.array, shape (n_contrast, n_chan, n_time)
    """
    X = list()
    for contrast in contrasts:
        if len(contrast) != 2:
            raise ValueError('Can only perform binary contrast')
        x = 0.
        for cond in contrast:
            for subcond in cond:
                x += epochs[subcond].average().data
            x *= -1
        X.append(x)
    return np.array(X)
