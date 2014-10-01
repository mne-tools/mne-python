"""Functions to plot EEG sensor montages
"""
try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab


def plot_montage(montage, scale_factor=1.5):
    """Plot EEG sensor montage

    Parameters
    ----------
    montage : instance of Montage
        The montage to visualize
    scale_factor : float
        Determines the size of the points. defaults to 1.5

    Returns
    -------
    fig : isntance of mayavi.Scene
        The malab scene object.
    """
    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
    pos = montage.pos
    mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2],
                  color=(1.0, 1.0, 1.0), scale_factor=1.5)
    mlab.text(0.01, 0.01, montage.kind, width=0.4)
    mlab.view(0, 0)
    return fig
