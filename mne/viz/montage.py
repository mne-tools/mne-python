import numpy as np

"""Functions to plot EEG sensor montages
"""


def plot_montage(montage, scale_factor=1.5, draw_names=False):
    """Plot EEG sensor montage

    Parameters
    ----------
    montage : instance of Montage
        The montage to visualize
    scale_factor : float
        Determines the size of the points. Defaults to 1.5
    draw_names : bool
        Whether to draw the channel names. Defaults to False

    Returns
    -------
    fig : isntance of mayavi.Scene
        The malab scene object.
    """
    try:
        from mayavi import mlab
    except ImportError:
        from enthought.mayavi import mlab

    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
    pos = montage.pos
    mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2],
                  color=(1.0, 1.0, 1.0), scale_factor=1.5)

    if draw_names:
        for p, n in zip(pos, montage.names):
            mlab.text(p[0], p[1], z=p[2] + scale_factor, text=n, width=0.05)

    mlab.text(0.01, 0.01, montage.kind, width=0.4)
    mlab.view(0, 0)
    return fig
