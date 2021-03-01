"""Functions to plot EEG sensor montages or digitizer montages."""
from copy import deepcopy
import numpy as np
from ..utils import logger, _check_option, _validate_type, verbose
from . import plot_sensors
from ..io._digitization import _get_fid_coords


@verbose
def plot_montage(montage, scale_factor=20, show_names=True, kind='topomap',
                 show=True, sphere=None, verbose=None):
    """Plot a montage.

    Parameters
    ----------
    montage : instance of DigMontage
        The montage to visualize.
    scale_factor : float
        Determines the size of the points.
    show_names : bool
        Whether to show the channel names.
    kind : str
        Whether to plot the montage as '3d' or 'topomap' (default).
    show : bool
        Show figure if True.
    %(topomap_sphere_auto)s
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object.
    """
    from scipy.spatial.distance import cdist
    from ..channels import DigMontage, make_dig_montage
    from ..io import RawArray
    from .. import create_info

    _check_option('kind', kind, ['topomap', '3d'])
    _validate_type(montage, DigMontage, item_name='montage')
    ch_names = montage.ch_names
    title = None

    if len(ch_names) == 0:
        raise RuntimeError('No valid channel positions found.')

    pos = np.array(list(montage._get_ch_pos().values()))

    dists = cdist(pos, pos)

    # only consider upper triangular part by setting the rest to np.nan
    dists[np.tril_indices(dists.shape[0])] = np.nan
    dupes = np.argwhere(np.isclose(dists, 0))
    if dupes.any():
        montage = deepcopy(montage)
        n_chans = pos.shape[0]
        n_dupes = dupes.shape[0]
        idx = np.setdiff1d(np.arange(len(pos)), dupes[:, 1]).tolist()
        logger.info("{} duplicate electrode labels found:".format(n_dupes))
        logger.info(", ".join([ch_names[d[0]] + "/" + ch_names[d[1]]
                               for d in dupes]))
        logger.info("Plotting {} unique labels.".format(n_chans - n_dupes))
        ch_names = [ch_names[i] for i in idx]
        ch_pos = dict(zip(ch_names, pos[idx, :]))
        # XXX: this might cause trouble if montage was originally in head
        fid, _ = _get_fid_coords(montage.dig)
        montage = make_dig_montage(ch_pos=ch_pos, **fid)

    info = create_info(ch_names, sfreq=256, ch_types="eeg")
    raw = RawArray(np.zeros((len(ch_names), 1)), info, copy=None)
    raw.set_montage(montage, on_missing='ignore')
    fig = plot_sensors(info, kind=kind, show_names=show_names, show=show,
                       title=title, sphere=sphere)
    collection = fig.axes[0].collections[0]
    collection.set_sizes([scale_factor])
    return fig
