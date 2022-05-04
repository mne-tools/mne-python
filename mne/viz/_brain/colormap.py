# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import numpy as np


def create_lut(cmap, n_colors=256, center=None):
    """Return a colormap suitable for setting as a LUT."""
    from .._3d import _get_cmap
    assert not (isinstance(cmap, str) and cmap == 'auto')
    cmap = _get_cmap(cmap)
    lut = np.round(cmap(np.linspace(0, 1, n_colors)) * 255.0).astype(np.int64)
    return lut


def scale_sequential_lut(lut_table, fmin, fmid, fmax):
    """Scale a sequential colormap."""
    assert fmin <= fmid <= fmax  # guaranteed by calculate_lut
    lut_table_new = lut_table.copy()
    n_colors = lut_table.shape[0]
    n_colors2 = n_colors // 2

    if fmax == fmin:
        fmid_idx = 0
    else:
        fmid_idx = np.clip(int(np.round(
            n_colors * ((fmid - fmin) / (fmax - fmin))) - 1), 0, n_colors - 2)

    n_left = fmid_idx + 1
    n_right = n_colors - n_left
    for i in range(4):
        lut_table_new[:fmid_idx + 1, i] = np.interp(
            np.linspace(0, n_colors2 - 1, n_left),
            np.arange(n_colors), lut_table[:, i])
        lut_table_new[fmid_idx + 1:, i] = np.interp(
            np.linspace(n_colors - 1, n_colors2, n_right)[::-1],
            np.arange(n_colors), lut_table[:, i])

    return lut_table_new


def get_fill_colors(cols, n_fill):
    """Get the fill colors for the middle of divergent colormaps."""
    steps = np.linalg.norm(np.diff(cols[:, :3].astype(float), axis=0),
                           axis=1)

    ind = np.flatnonzero(steps[1:-1] > steps[[0, -1]].mean() * 3)
    if ind.size > 0:
        # choose the two colors between which there is the large step
        ind = ind[0] + 1
        fillcols = np.r_[np.tile(cols[ind, :], (n_fill // 2, 1)),
                         np.tile(cols[ind + 1, :],
                                 (n_fill - n_fill // 2, 1))]
    else:
        # choose a color from the middle of the colormap
        fillcols = np.tile(cols[int(cols.shape[0] / 2), :], (n_fill, 1))

    return fillcols


def calculate_lut(lut_table, alpha, fmin, fmid, fmax, center=None,
                  transparent=True):
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
    lim_cmap : Colormap
        Color map obtained from _process_mapdata.
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
    transparent : boolean
        if True: use a linear transparency between fmin and fmid and make
        values below fmin fully transparent (symmetrically for divergent
        colormaps)

    Returns
    -------
    cmap : matplotlib.ListedColormap
        Color map with transparency channel.
    """
    if not fmin <= fmid <= fmax:
        raise ValueError('Must have fmin (%s) <= fmid (%s) <= fmax (%s)'
                         % (fmin, fmid, fmax))
    lut_table = create_lut(lut_table)
    assert lut_table.dtype.kind == 'i'
    divergent = center is not None
    n_colors = lut_table.shape[0]

    # Add transparency if needed
    n_colors2 = n_colors // 2
    if transparent:
        if divergent:
            N4 = np.full(4, n_colors // 4)
            N4[[0, 3, 1, 2][:np.mod(n_colors, 4)]] += 1
            assert N4.sum() == n_colors
            lut_table[:, -1] = np.round(np.hstack([
                np.full(N4[0], 255.),
                np.linspace(0, 255, N4[1])[::-1],
                np.linspace(0, 255, N4[2]),
                np.full(N4[3], 255.)]))
        else:
            lut_table[:n_colors2, -1] = np.round(np.linspace(
                0, 255, n_colors2))
            lut_table[n_colors2:, -1] = 255

    alpha = float(alpha)
    if alpha < 1.0:
        lut_table[:, -1] = np.round(lut_table[:, -1] * alpha)

    if divergent:
        if np.isclose(fmax, fmin, rtol=1e-6, atol=0):
            lut_table = np.r_[
                lut_table[:1],
                get_fill_colors(
                    lut_table[n_colors2 - 3:n_colors2 + 3, :], n_colors - 2),
                lut_table[-1:]]
        else:
            n_fill = int(round(fmin * n_colors2 / (fmax - fmin))) * 2
            lut_table = np.r_[
                scale_sequential_lut(lut_table[:n_colors2, :],
                                     center - fmax, center - fmid,
                                     center - fmin),
                get_fill_colors(
                    lut_table[n_colors2 - 3:n_colors2 + 3, :], n_fill),
                scale_sequential_lut(lut_table[n_colors2:, :][::-1],
                                     center - fmax, center - fmid,
                                     center - fmin)[::-1]]
    else:
        lut_table = scale_sequential_lut(lut_table, fmin, fmid, fmax)

    n_colors = lut_table.shape[0]
    if n_colors != 256:
        lut = np.zeros((256, 4))
        x = np.linspace(1, n_colors, 256)
        for chan in range(4):
            lut[:, chan] = np.interp(x,
                                     np.arange(1, n_colors + 1),
                                     lut_table[:, chan])
        lut_table = lut

    lut_table = lut_table.astype(np.float64) / 255.0
    return lut_table
