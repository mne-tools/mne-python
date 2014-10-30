"""Functions to plot decoding results
"""
from __future__ import print_function

# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: Simplified BSD

import numpy as np


def plot_gat_matrix(gat, title=None, vmin=0., vmax=1., tlim=None,
                    ax=None, cmap='RdBu_r', show=True):
    """Plotting function of GeneralizationAcrossTime object

    Predict each classifier. If multiple classifiers are passed, average
    prediction across all classifier to result in a single prediction per
    classifier.

    Parameters
    ----------
    gat : instance of mne.decoding.GeneralizationAcrossTime
        The gat object.
    title : str | None, optional
        Figure title. Defaults to None.
    vmin : float, optional
        Min color value for score. Defaults to None.
    vmax : float, optional
        Max color value for score. Defaults to None.
    tlim : np.ndarray, (train_min, test_max) | None, optional,
        The temporal boundries. defaults to None.
    ax : object | None, optional
        Plot pointer. If None, generate new figure. Defaults to None.
    cmap : str | cmap object
        The color map to be used. Defaults to 'RdBu_r'.
    show : bool, optional, default: True
        If True, the figure will will be shown. Defaults to True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    if not hasattr(gat, 'scores_'):
        raise RuntimeError('Please score your data before trying to plot '
                           'scores')
    import matplotlib.pyplot as plt
    # XXX actually the test seemed wrong and obsolete (D.E.)
    # Check that same amount of testing time per training time
    # assert len(np.unique([len(t) for t in gat.test_times_])) == 1
    # Setup plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Define time limits
    if tlim is None:
        tlim = [gat.test_times_['s'][0][0], gat.test_times_['s'][-1][-1],
                gat.train_times['s'][0], gat.train_times['s'][-1]]
    # Plot scores
    im = ax.imshow(gat.scores_, interpolation='nearest', origin='lower',
                   extent=tlim, vmin=vmin, vmax=vmax,
                   cmap=cmap)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    if not title is None:
        ax.set_title(title)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.colorbar(im, ax=ax)
    if show:
        plt.show()
    return fig if ax is None else ax.get_figure()


def plot_gat_diagonal(gat, title=None, ymin=0., ymax=1., ax=None, show=True,
                      color='b'):
    """Plotting function of GeneralizationAcrossTime object

    Predict each classifier. If multiple classifiers are passed, average
    prediction across all classifier to result in a single prediction per
    classifier.

    Parameters
    ----------
    gat : instance of mne.decoding.GeneralizationAcrossTime
        The gat object.
    title : str | None, optional
        Figure title. Defaults to None.
    ymin : float, optional, defaults to 0.
        Min score value.
    ymax : float, optional, defaults to 1.
        Max score value.
    tlim : np.ndarray, (train_min_max, test_min_max) | None, optional,
        The temporal boundries. Defaults to None.
    ax : object | None, optional
        Plot pointer. If None, generate new figure. Defaults to None.
    show : bool, optional, defaults to True.
        If True, the figure will will be shown. Defaults to True.
    color : str, optional
        Score line color. Defaults to 'steelblue'.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    if not hasattr(gat, 'scores_'):
        raise RuntimeError('Please score your data before trying to plot scores')
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    # detect whether gat is a full matrix or just its diagonal
    if np.all(np.unique([len(t) for t in gat.test_times_['s']]) == 1):
        scores = gat.scores_
    else:
        scores = np.diag(gat.scores_)
    ax.plot(gat.train_times['s'], scores, color=color,
            label="Classif. score")
    ax.axhline(0.5, color='k', linestyle='--', label="Chance level")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Classif. score ({0})'.format(
            'AUC' if 'roc' in repr(gat.scorer_) else r'%'
        ))
    ax.legend(loc='best')
    if show:
        plt.show()
    return fig if ax is None else ax.get_figure()
