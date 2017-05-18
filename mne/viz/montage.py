"""Functions to plot EEG sensor montages or digitizer montages."""
import mne
from .utils import plot_sensors


def plot_montage(montage, kind='3d', scale_factor=20, show_names=False,
                 show=True):
    """Plot a montage.

    Parameters
    ----------
    montage : instance of Montage
        The montage to visualize.
    kind : str
        Whether to plot the montage as '3d' or 'topomap'.
    scale_factor : float
        Determines the size of the points. Defaults to 20.
    show_names : bool
        Whether to show the channel names. Defaults to False.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure object.
    """
    if isinstance(montage, mne.channels.montage.Montage):
        ch_names = montage.ch_names
    elif isinstance(montage, mne.channels.montage.DigMontage):
        ch_names = montage.point_names
    info = mne.create_info(ch_names, sfreq=256, ch_types="eeg",
                           montage=montage)
    return plot_sensors(info, kind=kind, show_names=show_names, show=show)
