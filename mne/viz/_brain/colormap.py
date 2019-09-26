# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import numpy as np


def _calculate_lut(lim_cmap, alpha, fmin, fmid, fmax, center=None):
    u"""Transparent color map calculation.

    A colormap may be sequential or divergent. When the colormap is
    divergent indicate this by providing a value for 'center'. The
    meanings of fmin, fmid and fmax are different for sequential and
    divergent colormaps. A sequential colormap is characterised by::

        [fmin, fmid, fmax]

    where fmin and fmax define the edges of the colormap and fmid
    will be the value mapped to the center of the originally chosen colormap.
    A divergent colormap is characterised by::

        [center-fmax, center-fmid, center-fmin, center,
            center+fmin, center+fmid, center+fmax]

    i.e., values between center-fmin and center+fmin will not be shown
    while center-fmid will map to the fmid of the first half of the
    original colormap and center-fmid to the fmid of the second half.

    Parameters
    ----------
    lim_cmap : str | LinearSegmentedColormap
        Color map obtained from MNE._limits_to_control_points.
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    fmin : float
        Min value in colormap.
    fmid : float
        Intermediate value in colormap.
    fmax : float
        Max value in colormap.
    center : float or None
        If not None, center of a divergent colormap, changes the meaning of
        fmin, fmax and fmid.

    Returns
    -------
    cmap : matplotlib.ListedColormap
        Color map with transparency channel.
    """
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    if center is None:
        # 'hot' or another linear color map
        ctrl_pts = (fmin, fmid, fmax)
        scale_pts = ctrl_pts
        rgb_cmap = cm.get_cmap(lim_cmap)
        # take 60% of hot color map, so it will be consistent
        # with mayavi plots
        cmap_size = int(rgb_cmap.N * 0.6)
        cmap = rgb_cmap(np.arange(rgb_cmap.N))[rgb_cmap.N - cmap_size:, :]
        alphas = np.ones(cmap_size)
        step = 2 * (scale_pts[-1] - scale_pts[0]) / rgb_cmap.N
        # coefficients for linear mapping
        # from [ctrl_pts[0], ctrl_pts[1]) interval into [0, 1]
        k = 1 / (ctrl_pts[1] - ctrl_pts[0])
        b = - ctrl_pts[0] * k

        for i in range(0, cmap_size):
            curr_pos = i * step + scale_pts[0]

            if (curr_pos < ctrl_pts[0]):
                alphas[i] = 0
            elif (curr_pos < ctrl_pts[1]):
                alphas[i] = k * curr_pos + b
    else:
        # 'mne' or another divergent color map
        ctrl_pts = (center + fmin, center + fmid, center + fmax)
        scale_pts = (center - fmax, center, center + fmax)
        rgb_cmap = lim_cmap
        cmap = rgb_cmap(np.arange(rgb_cmap.N))
        alphas = np.ones(rgb_cmap.N)
        step = (scale_pts[-1] - scale_pts[0]) / rgb_cmap.N
        # coefficients for linear mapping into [0, 1]
        k_pos = 1 / (ctrl_pts[1] - ctrl_pts[0])
        k_neg = -k_pos
        b = - ctrl_pts[0] * k_pos

        for i in range(0, rgb_cmap.N):
            curr_pos = i * step + scale_pts[0]

            if -ctrl_pts[0] < curr_pos < ctrl_pts[0]:
                alphas[i] = 0
            elif ctrl_pts[0] <= curr_pos < ctrl_pts[1]:
                alphas[i] = k_pos * curr_pos + b
            elif -ctrl_pts[1] < curr_pos <= -ctrl_pts[0]:
                alphas[i] = k_neg * curr_pos + b

    alphas *= alpha
    np.clip(alphas, 0, 1)
    cmap[:, -1] = alphas
    cmap = ListedColormap(cmap)

    return cmap
