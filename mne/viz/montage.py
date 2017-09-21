"""Functions to plot EEG sensor montages or digitizer montages."""
from ..utils import check_version, logger
from . import plot_sensors


def plot_montage(montage, scale_factor=20, show_names=True, kind='topomap',
                 show=True):
    """Plot a montage.

    Parameters
    ----------
    montage : instance of Montage or DigMontage
        The montage to visualize.
    scale_factor : float
        Determines the size of the points.
    show_names : bool
        Whether to show the channel names.
    kind : str
        Whether to plot the montage as '3d' or 'topomap' (default).
    show : bool
        Show figure if True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure object.
    """
    from copy import deepcopy
    import numpy as np
    from scipy.spatial.distance import cdist
    from ..channels import Montage, DigMontage
    from .. import create_info

    if isinstance(montage, Montage):
        ch_names = montage.ch_names
        title = montage.kind
    elif isinstance(montage, DigMontage):
        ch_names = montage.point_names
        title = None
    else:
        raise TypeError("montage must be an instance of "
                        "mne.channels.montage.Montage or"
                        "mne.channels.montage.DigMontage")
    if kind not in ['topomap', '3d']:
        raise ValueError("kind must be 'topomap' or '3d'")

    dists = cdist(montage.pos, montage.pos)
    # only consider upper triangular part by setting the rest to np.nan
    dists[np.tril_indices(dists.shape[0])] = np.nan
    dupes = np.argwhere(np.isclose(dists, 0))
    if dupes.any():
        m = deepcopy(montage)
        n_chans = m.pos.shape[0]
        n_dupes = dupes.shape[0]
        idx = np.setdiff1d(m.selection, dupes[:, 1]).tolist()
        logger.info("{} duplicate electrode labels found: ".format(n_dupes))
        logger.info(", ".join([ch_names[d[0]] + "/" + ch_names[d[1]]
                               for d in dupes]))
        logger.info("Plotting {} unique labels.".format(n_chans - n_dupes))
        m.ch_names = [m.ch_names[i] for i in idx]
        ch_names = m.ch_names
        m.pos = m.pos[idx, :]
        m.selection = np.arange(n_chans - n_dupes)
    else:
        m = montage

    info = create_info(ch_names, sfreq=256, ch_types="eeg", montage=m)
    fig = plot_sensors(info, kind=kind, show_names=show_names, show=show,
                       title=title)
    collection = fig.axes[0].collections[0]
    if check_version("matplotlib", "1.4"):
        collection.set_sizes([scale_factor])
    else:
        collection._sizes = [scale_factor]
    return fig
