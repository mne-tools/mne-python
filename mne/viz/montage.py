"""Functions to plot EEG sensor montages or digitizer montages
"""


def plot_montage(montage, scale_factor=1.5, show_names=False, show=True):
    """Plot a montage

    Parameters
    ----------
    montage : instance of Montage
        The montage to visualize.
    scale_factor : float
        Determines the size of the points. Defaults to 1.5.
    show_names : bool
        Whether to show the channel names. Defaults to False.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure object.
    """
    from ..channels.layout import Montage
    from ..io.meas_info import DigMontage

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pos = montage.pos
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if show_names:
        if isinstance(montage, Montage):
            ch_names = montage.ch_names
            for ch_name, x, y, z in zip(ch_names, pos[:, 0],
                                        pos[:, 1], pos[:, 2]):
                ax.text(x, y, z, ch_name)
        elif isinstance(montage, DigMontage):
            if montage.hpi_names:
                hpi_names = montage.hpi_names
                for hpi_name, x, y, z in zip(hpi_names, pos[:, 0],
                                             pos[:, 1], pos[:, 2]):
                    ax.text(x, y, z, hpi_name)
            else:
                raise ValueError('There are no hpi points to show names of.')

    if show:
        plt.show()

    return fig
