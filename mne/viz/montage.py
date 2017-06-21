"""Functions to plot EEG sensor montages or digitizer montages."""
from ..utils import check_version
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
    info = create_info(ch_names, sfreq=256, ch_types="eeg", montage=montage)
    fig = plot_sensors(info, kind=kind, show_names=show_names, show=show,
                       title=title)
    collection = fig.axes[0].collections[0]
    if check_version("matplotlib", "1.4"):
        collection.set_sizes([scale_factor])
    else:
        collection._sizes = [scale_factor]
    return fig
