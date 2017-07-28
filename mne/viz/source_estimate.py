# -*- coding: utf-8 -*-
# Authors: Jaakko Leppakangas <jaeilepp@gmail.com>
#
# License: Simplified BSD

"""Functions to plot source estimate data."""
from __future__ import print_function

import os.path as op

import numpy as np

from ..source_space import _read_talairach
from ..surface import _compute_nearest, read_surface
from ..utils import has_nibabel, get_subjects_dir
from ..transforms import apply_trans
from .utils import plt_show


def plot_stc_glass_brain(stc, subject, subjects_dir=None, src=None,
                         initial_time=0., display_mode='ortho', colorbar=False,
                         axes=None, title=None, threshold='auto',
                         annotate=True, black_bg=False, cmap=None, alpha=0.7,
                         vmax=None, plot_abs=True, symmetric_cbar="auto",
                         show=True):
    """Plot stc on glass brain.

    Parameters
    ----------
    stc : instance of SourceEstimate | instance of VolSourceEstimate
        Data to plot.
    subject : str
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT.
    subjects_dir : None | str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
        The default is None.
    src : instance of SourceSpace | None
        The source space. Only needed for VolumeSourceEstimate. Has no effect
        when using instance of SourceEstimate.
    initial_time : float
        The time instance to plot in seconds. Defaults to 0.
    display_mode : string
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only, 'ortho' - three cuts are
        performed in orthogonal directions. Possible values are: 'ortho',
        'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r', 'lr', 'lzr', 'lyr',
        'lzry', 'lyrz'. Defaults to 'ortho'.
    colorbar : bool
        If True, display a colorbar on the right of the plots. Defaults to
        False.
    axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height)
        The axes, or the coordinates, in matplotlib figure space,
        of the axes used to display the plot. If None, the complete
        figure is used.
    title : string
        The title displayed on the figure.
    threshold : float | None | 'auto'
        The value used to threshold the image: values below the threshold (in
        absolute value) are plotted as transparent. If None, the image is not
        hresholded. If float, If auto, the threshold is determined magically by
        analysis of the image.
    annotate : bool
        If True, positions and left/right annotation are added to the plot.
    black_bg : bool
        If True, the background of the image is set to be black. If you wish to
        save figures with a black background, you will need to pass
        "facecolor='k', edgecolor='k'" to matplotlib.pyplot.savefig.
    cmap : matplotlib colormap
        The colormap for specified image
    alpha : float
        Alpha transparency for the brain schematics.
    vmax : float
        Upper bound for plotting, passed to matplotlib.pyplot.imshow.
    plot_abs : bool
        If True (default), maximum intensity projection of the absolute value
        will be used (rendering positive and negative values in the same
        manner). If False, the sign of the maximum intensity will be
        represented with different colors. See `nilearn documentation`_
        for examples.
    symmetric_cbar : bool | 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    show : bool
        Whether to show the figure.

    Returns
    -------
    fig : instance of matplotlib.Figure
        The matplotlib figure.

    Notes
    -----
    This function requires nilearn and nibabel.

    .. _nilearn documentation: http://nilearn.github.io/auto_examples/01_plotting/plot_demo_glass_brain_extensive.html
    """  # noqa: E501
    from scipy import sparse
    if has_nibabel(vox2ras_tkr=True):
        import nibabel as nib
    else:
        raise ImportError('This function requires nibabel.')
    try:
        from nilearn import plotting
    except ImportError:
        raise ImportError('This function requires nilearn.')
    from ..source_estimate import VolSourceEstimate, SourceEstimate

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    time_idx = np.argmin(np.abs(stc.times - initial_time))
    xfm = _read_talairach(subject, subjects_dir)
    if isinstance(stc, VolSourceEstimate):
        if src is None:
            raise ValueError('src cannot be None when plotting volume source '
                             'estimate.')
        img = stc.as_volume(src)
        affine = np.dot(xfm, img.affine)
        data = img.get_data()[:, :, :, time_idx]
        img = nib.Nifti1Image(data, affine)
    elif isinstance(stc, SourceEstimate):
        n_vertices = sum(len(v) for v in stc.vertices)
        offset = 0
        surf_to_mri = 0.
        for hi, hemi in enumerate(['lh', 'rh']):
            ribbon = nib.load(op.join(subjects_dir, subject, 'mri',
                                      '%s.ribbon.mgz' % hemi))
            xfm = ribbon.header.get_vox2ras_tkr()
            mri_data = ribbon.get_data()
            ijk = np.array(np.where(mri_data)).T
            xyz = apply_trans(xfm, ijk) / 1000.
            row_ind = np.where(mri_data.ravel())[0]
            data = np.ones(row_ind.size)
            rr = read_surface(op.join(subjects_dir, subject, 'surf',
                                      '%s.white' % hemi))[0]
            rr /= 1000.
            rr = rr[stc.vertices[hi]]
            col_ind = _compute_nearest(rr, xyz) + offset
            surf_to_mri = surf_to_mri + sparse.csr_matrix(
                (data, (row_ind, col_ind)), shape=(mri_data.size, n_vertices))
            offset += len(stc.vertices[hi])

        data = surf_to_mri.dot(stc.data[:, time_idx])
        data.shape = mri_data.shape
        xfm = _read_talairach(subject, subjects_dir)

        affine = np.dot(xfm, ribbon.header.get_vox2ras())
        img = nib.Nifti1Image(data, affine)

    else:
        ValueError('Glass brain plotting is only supported for SourceEstimate '
                   'and VolSourceEstimate. Got %s.' % type(stc))
    fig = plotting.plot_glass_brain(img, display_mode=display_mode,
                                    colorbar=colorbar, axes=axes, title=title,
                                    threshold=threshold, annotate=annotate,
                                    black_bg=black_bg, cmap=cmap, alpha=alpha,
                                    vmax=vmax, plot_abs=plot_abs,
                                    symmetric_cbar=symmetric_cbar)
    plt_show(show)
    return fig
