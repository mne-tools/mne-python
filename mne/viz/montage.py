"""Functions to plot EEG sensor montages or digitizer montages."""
import numpy as np

from .utils import plt_show


def plot_montage(montage, scale_factor=20, show_names=False, show=True):
    """Plot a montage.

    Parameters
    ----------
    montage : instance of Montage
        The montage to visualize.
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
    from ..channels.montage import Montage, DigMontage

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(montage, Montage):
        pos = montage.pos
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=scale_factor)
        if show_names:
            ch_names = montage.ch_names
            for ch_name, x, y, z in zip(ch_names, pos[:, 0],
                                        pos[:, 1], pos[:, 2]):
                ax.text(x, y, z, ch_name)
    elif isinstance(montage, DigMontage):
        pos = np.vstack((montage.hsp, montage.elp))
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=scale_factor)
        if show_names:
            if montage.point_names:
                hpi_names = montage.point_names
                for hpi_name, x, y, z in zip(hpi_names, montage.elp[:, 0],
                                             montage.elp[:, 1],
                                             montage.elp[:, 2]):
                    ax.text(x, y, z, hpi_name)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt_show(show)
    return fig
