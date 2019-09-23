"""Functions to plot EEG sensor montages or digitizer montages."""
from copy import deepcopy
import numpy as np
from ..utils import check_version, logger, _check_option
from . import plot_sensors
from .._digitization._utils import _get_fid_coords


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
    fig : instance of matplotlib.figure.Figure
        The figure object.
    """
    from scipy.spatial.distance import cdist
    from ..channels import Montage, DigMontage, make_dig_montage
    from .. import create_info

    if isinstance(montage, Montage):
        ch_names = montage.ch_names
        title = montage.kind
    elif isinstance(montage, DigMontage):
        ch_names = montage.ch_names
        title = None
    else:
        raise TypeError("montage must be an instance of "
                        "mne.channels.montage.Montage or"
                        "mne.channels.montage.DigMontage")
    _check_option('kind', kind, ['topomap', '3d'])

    if len(ch_names) == 0:
        raise RuntimeError('No valid channel positions found.')

    # check for duplicate labels
    if isinstance(montage, Montage):
        pos = montage.pos
    else:
        pos = np.array(list(montage._get_ch_pos().values()))

    dists = cdist(pos, pos)

    # only consider upper triangular part by setting the rest to np.nan
    dists[np.tril_indices(dists.shape[0])] = np.nan
    dupes = np.argwhere(np.isclose(dists, 0))
    if dupes.any():
        montage = deepcopy(montage)
        n_chans = pos.shape[0]
        n_dupes = dupes.shape[0]
        if isinstance(montage, Montage):
            idx = np.setdiff1d(montage.selection, dupes[:, 1]).tolist()
        else:
            idx = np.setdiff1d(np.arange(len(pos)), dupes[:, 1]).tolist()
        logger.info("{} duplicate electrode labels found:".format(n_dupes))
        logger.info(", ".join([ch_names[d[0]] + "/" + ch_names[d[1]]
                               for d in dupes]))
        logger.info("Plotting {} unique labels.".format(n_chans - n_dupes))
        ch_names = [ch_names[i] for i in idx]
        if isinstance(montage, Montage):
            montage.ch_names = ch_names
            montage.pos = pos[idx, :]
            montage.selection = np.arange(n_chans - n_dupes)
        else:
            ch_pos = dict(zip(ch_names, pos[idx, :]))
            # XXX: this might cause trouble if montage was originally in head
            fid, _ = _get_fid_coords(montage.dig)
            montage = make_dig_montage(ch_pos=ch_pos, **fid)

    info = create_info(ch_names, sfreq=256, ch_types="eeg", montage=montage)
    fig = plot_sensors(info, kind=kind, show_names=show_names, show=show,
                       title=title)
    collection = fig.axes[0].collections[0]
    if check_version("matplotlib", "1.4"):
        collection.set_sizes([scale_factor])
    else:
        collection._sizes = [scale_factor]
    return fig
