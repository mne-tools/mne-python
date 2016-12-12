"""Functions to plot decoding results."""
from __future__ import print_function

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Clement Moutard <clement.moutard@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: Simplified BSD

import numpy as np

from .utils import plt_show
from ..utils import warn


def plot_gat_matrix(gat, title=None, vmin=None, vmax=None, tlim=None,
                    ax=None, cmap='RdBu_r', show=True, colorbar=True,
                    xlabel=True, ylabel=True):
    """Plotting function of GeneralizationAcrossTime object.

    Predict each classifier. If multiple classifiers are passed, average
    prediction across all classifier to result in a single prediction per
    classifier.

    Parameters
    ----------
    gat : instance of mne.decoding.GeneralizationAcrossTime
        The gat object.
    title : str | None
        Figure title. Defaults to None.
    vmin : float | None
        Min color value for scores. If None, sets to min(gat.scores_).
        Defaults to None.
    vmax : float | None
        Max color value for scores. If None, sets to max(gat.scores_).
        Defaults to None.
    tlim : array-like, (4,) | None
        The temporal boundaries. If None, expands to
        [tmin_train, tmax_train, tmin_test, tmax_test]. Defaults to None.
    ax : object | None
        Plot pointer. If None, generate new figure. Defaults to None.
    cmap : str | cmap object
        The color map to be used. Defaults to 'RdBu_r'.
    show : bool
        If True, the figure will be shown. Defaults to True.
    colorbar : bool
        If True, the colorbar of the figure is displayed. Defaults to True.
    xlabel : bool
        If True, the xlabel is displayed. Defaults to True.
    ylabel : bool
        If True, the ylabel is displayed. Defaults to True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    if not hasattr(gat, 'scores_'):
        raise RuntimeError('Please score your data before trying to plot '
                           'scores')
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Define time limits
    if tlim is None:
        tn_times = gat.train_times_['times']
        tt_times = gat.test_times_['times']
        tlim = [tt_times[0][0], tt_times[-1][-1], tn_times[0], tn_times[-1]]

    # Plot scores
    im = ax.imshow(gat.scores_, interpolation='nearest', origin='lower',
                   extent=tlim, vmin=vmin, vmax=vmax, cmap=cmap)
    if xlabel is True:
        ax.set_xlabel('Testing Time (s)')
    if ylabel is True:
        ax.set_ylabel('Training Time (s)')
    if title is not None:
        ax.set_title(title)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    ax.set_xlim(tlim[:2])
    ax.set_ylim(tlim[2:])
    if colorbar is True:
        plt.colorbar(im, ax=ax)
    plt_show(show)
    return fig if ax is None else ax.get_figure()


def plot_gat_times(gat, train_time='diagonal', title=None, xmin=None,
                   xmax=None, ymin=None, ymax=None, ax=None, show=True,
                   color=None, xlabel=True, ylabel=True, legend=True,
                   chance=True, label='Classif. score'):
    """Plot the GeneralizationAcrossTime results.

    Plot the scores of the classifier trained at 'train_time'.

    Parameters
    ----------
    gat : instance of mne.decoding.GeneralizationAcrossTime
        The gat object.
    train_time : 'diagonal' | float | list or array of float
        Plot a 1d array of a portion of gat.scores_.
        If set to 'diagonal', plots the gat.scores_ of classifiers
        trained and tested at identical times
        if set to float | list or array of float, plots scores of the
        classifier(s) trained at (a) specific training time(s).
        Default to 'diagonal'.
    title : str | None
        Figure title. Defaults to None.
    xmin : float | None, optional
        Min time value. Defaults to None.
    xmax : float | None, optional
        Max time value. Defaults to None.
    ymin : float | None, optional
        Min score value. If None, sets to min(scores). Defaults to None.
    ymax : float | None, optional
        Max score value. If None, sets to max(scores). Defaults to None.
    ax : object | None
        Plot pointer. If None, generate new figure. Defaults to None.
    show : bool, optional
        If True, the figure will be shown. Defaults to True.
    color : str
        Score line color. Defaults to 'steelblue'.
    xlabel : bool
        If True, the xlabel is displayed. Defaults to True.
    ylabel : bool
        If True, the ylabel is displayed. Defaults to True.
    legend : bool
        If True, a legend is displayed. Defaults to True.
    chance : bool | float.
        Plot chance level. If True, chance level is estimated from the type
        of scorer. Defaults to None.
    label : str
        Score label used in the legend. Defaults to 'Classif. score'.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    if not hasattr(gat, 'scores_'):
        raise RuntimeError('Please score your data before trying to plot '
                           'scores')
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Find and plot chance level
    if chance is not False:
        if chance is True:
            chance = _get_chance_level(gat.scorer_, gat.y_train_)
        chance = float(chance)
        if np.isfinite(chance):  # don't plot nan chance level
            ax.axhline(chance, color='k', linestyle='--',
                       label="Chance level")
    ax.axvline(0, color='k', label='')

    if isinstance(train_time, (str, float)):
        train_time = [train_time]
        label = [label]
    elif isinstance(train_time, (list, np.ndarray)):
        label = train_time
    else:
        raise ValueError("train_time must be 'diagonal' | float | list or "
                         "array of float.")

    if color is None or isinstance(color, str):
        color = np.tile(color, len(train_time))

    for _train_time, _color, _label in zip(train_time, color, label):
        _plot_gat_time(gat, _train_time, ax, _color, _label)

    if title is not None:
        ax.set_title(title)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    if xlabel is True:
        ax.set_xlabel('Time (s)')
    if ylabel is True:
        ax.set_ylabel('Classif. score ({0})'.format(
                      'AUC' if 'roc' in repr(gat.scorer_) else r'%'))
    if legend is True:
        ax.legend(loc='best')
    plt_show(show)
    return fig if ax is None else ax.get_figure()


def _plot_gat_time(gat, train_time, ax, color, label):
    """Plot a unique score 1d array."""
    # Detect whether gat is a full matrix or just its diagonal
    if np.all(np.unique([len(t) for t in gat.test_times_['times']]) == 1):
        scores = gat.scores_
    elif train_time == 'diagonal':
        # Get scores from identical training and testing times even if GAT
        # is not square.
        scores = np.zeros(len(gat.scores_))
        for train_idx, train_time in enumerate(gat.train_times_['times']):
            for test_times in gat.test_times_['times']:
                # find closest testing time from train_time
                lag = test_times - train_time
                test_idx = np.abs(lag).argmin()
                # check that not more than 1 classifier away
                if np.abs(lag[test_idx]) > gat.train_times_['step']:
                    score = np.nan
                else:
                    score = gat.scores_[train_idx][test_idx]
                scores[train_idx] = score
    elif isinstance(train_time, float):
        train_times = gat.train_times_['times']
        idx = np.abs(train_times - train_time).argmin()
        if train_times[idx] - train_time > gat.train_times_['step']:
            raise ValueError("No classifier trained at %s " % train_time)
        scores = gat.scores_[idx]
    else:
        raise ValueError("train_time must be 'diagonal' or a float.")
    kwargs = dict()
    if color is not None:
        kwargs['color'] = color
    ax.plot(gat.train_times_['times'], scores, label=str(label), **kwargs)


def _get_chance_level(scorer, y_train):
    """Get the chance level."""
    # XXX JRK This should probably be solved within sklearn?
    if scorer.__name__ == 'accuracy_score':
        chance = np.max([np.mean(y_train == c) for c in np.unique(y_train)])
    elif scorer.__name__ == 'roc_auc_score':
        chance = 0.5
    else:
        chance = np.nan
        warn('Cannot find chance level from %s, specify chance level'
             % scorer.__name__)
    return chance
