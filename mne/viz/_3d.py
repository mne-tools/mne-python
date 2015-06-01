"""Functions to make 3D plots with M/EEG data
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

from ..externals.six import string_types, advance_iterator

import os.path as op
import inspect
import warnings
from itertools import cycle
import base64

import numpy as np
from scipy import linalg

from ..io.constants import FIFF
from ..io.pick import pick_types
from ..surface import (get_head_surf, get_meg_helmet_surf, read_surface,
                       transform_surface_to)
from ..transforms import (read_trans, _find_trans, apply_trans,
                          combine_transforms, _get_mri_head_t)
from ..utils import get_subjects_dir, logger, _check_subject
from ..defaults import _handle_default
from .utils import mne_analyze_colormap, _prepare_trellis, COLORS
from ..externals.six import BytesIO


def plot_evoked_field(evoked, surf_maps, time=None, time_label='t = %0.0f ms',
                      n_jobs=1):
    """Plot MEG/EEG fields on head surface and helmet in 3D

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked object.
    surf_maps : list
        The surface mapping information obtained with make_field_map.
    time : float | None
        The time point at which the field map shall be displayed. If None,
        the average peak latency (across sensor types) is used.
    time_label : str
        How to print info about the time instant visualized.
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    fig : instance of mlab.Figure
        The mayavi figure.
    """
    types = [t for t in ['eeg', 'grad', 'mag'] if t in evoked]

    time_idx = None
    if time is None:
        time = np.mean([evoked.get_peak(ch_type=t)[1] for t in types])

    if not evoked.times[0] <= time <= evoked.times[-1]:
        raise ValueError('`time` (%0.3f) must be inside `evoked.times`' % time)
    time_idx = np.argmin(np.abs(evoked.times - time))

    types = [sm['kind'] for sm in surf_maps]

    # Plot them
    from mayavi import mlab
    alphas = [1.0, 0.5]
    colors = [(0.6, 0.6, 0.6), (1.0, 1.0, 1.0)]
    colormap = mne_analyze_colormap(format='mayavi')
    colormap_lines = np.concatenate([np.tile([0., 0., 255., 255.], (127, 1)),
                                     np.tile([0., 0., 0., 255.], (2, 1)),
                                     np.tile([255., 0., 0., 255.], (127, 1))])

    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))

    for ii, this_map in enumerate(surf_maps):
        surf = this_map['surf']
        map_data = this_map['data']
        map_type = this_map['kind']
        map_ch_names = this_map['ch_names']

        if map_type == 'eeg':
            pick = pick_types(evoked.info, meg=False, eeg=True)
        else:
            pick = pick_types(evoked.info, meg=True, eeg=False, ref_meg=False)

        ch_names = [evoked.ch_names[k] for k in pick]

        set_ch_names = set(ch_names)
        set_map_ch_names = set(map_ch_names)
        if set_ch_names != set_map_ch_names:
            message = ['Channels in map and data do not match.']
            diff = set_map_ch_names - set_ch_names
            if len(diff):
                message += ['%s not in data file. ' % list(diff)]
            diff = set_ch_names - set_map_ch_names
            if len(diff):
                message += ['%s not in map file.' % list(diff)]
            raise RuntimeError(' '.join(message))

        data = np.dot(map_data, evoked.data[pick, time_idx])

        x, y, z = surf['rr'].T
        nn = surf['nn']
        # make absolutely sure these are normalized for Mayavi
        nn = nn / np.sum(nn * nn, axis=1)[:, np.newaxis]

        # Make a solid surface
        vlim = np.max(np.abs(data))
        alpha = alphas[ii]
        with warnings.catch_warnings(record=True):  # traits
            mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'])
        mesh.data.point_data.normals = nn
        mesh.data.cell_data.normals = None
        mlab.pipeline.surface(mesh, color=colors[ii], opacity=alpha)

        # Now show our field pattern
        with warnings.catch_warnings(record=True):  # traits
            mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'],
                                                        scalars=data)
        mesh.data.point_data.normals = nn
        mesh.data.cell_data.normals = None
        with warnings.catch_warnings(record=True):  # traits
            fsurf = mlab.pipeline.surface(mesh, vmin=-vlim, vmax=vlim)
        fsurf.module_manager.scalar_lut_manager.lut.table = colormap

        # And the field lines on top
        with warnings.catch_warnings(record=True):  # traits
            mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'],
                                                        scalars=data)
        mesh.data.point_data.normals = nn
        mesh.data.cell_data.normals = None
        with warnings.catch_warnings(record=True):  # traits
            cont = mlab.pipeline.contour_surface(mesh, contours=21,
                                                 line_width=1.0,
                                                 vmin=-vlim, vmax=vlim,
                                                 opacity=alpha)
        cont.module_manager.scalar_lut_manager.lut.table = colormap_lines

    if '%' in time_label:
        time_label %= (1e3 * evoked.times[time_idx])
    mlab.text(0.01, 0.01, time_label, width=0.4)
    mlab.view(10, 60)
    return fig


def _plot_mri_contours(mri_fname, surf_fnames, orientation='coronal',
                       slices=None, show=True, img_output=False):
    """Plot BEM contours on anatomical slices.

    Parameters
    ----------
    mri_fname : str
        The name of the file containing anatomical data.
    surf_fnames : list of str
        The filenames for the BEM surfaces in the format
        ['inner_skull.surf', 'outer_skull.surf', 'outer_skin.surf'].
    orientation : str
        'coronal' or 'transverse' or 'sagittal'
    slices : list of int
        Slice indices.
    show : bool
        Call pyplot.show() at the end.
    img_output : None | tuple
        If tuple (width and height), images will be produced instead of a
        single figure with many axes. This mode is designed to reduce the
        (substantial) overhead associated with making tens to hundreds
        of matplotlib axes, instead opting to re-use a single Axes instance.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure | list
        The figure. Will instead be a list of png images if
        img_output is a tuple.
    """
    import matplotlib.pyplot as plt
    import nibabel as nib

    if orientation not in ['coronal', 'axial', 'sagittal']:
        raise ValueError("Orientation must be 'coronal', 'axial' or "
                         "'sagittal'. Got %s." % orientation)

    # Load the T1 data
    nim = nib.load(mri_fname)
    data = nim.get_data()
    affine = nim.get_affine()

    n_sag, n_axi, n_cor = data.shape
    orientation_name2axis = dict(sagittal=0, axial=1, coronal=2)
    orientation_axis = orientation_name2axis[orientation]

    if slices is None:
        n_slices = data.shape[orientation_axis]
        slices = np.linspace(0, n_slices, 12, endpoint=False).astype(np.int)

    # create of list of surfaces
    surfs = list()

    trans = linalg.inv(affine)
    # XXX : next line is a hack don't ask why
    trans[:3, -1] = [n_sag // 2, n_axi // 2, n_cor // 2]

    for surf_fname in surf_fnames:
        surf = dict()
        surf['rr'], surf['tris'] = read_surface(surf_fname)
        # move back surface to MRI coordinate system
        surf['rr'] = nib.affines.apply_affine(trans, surf['rr'])
        surfs.append(surf)

    if img_output is None:
        fig, axs = _prepare_trellis(len(slices), 4)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 7.0))
        axs = [ax] * len(slices)

        fig_size = fig.get_size_inches()
        w, h = img_output[0], img_output[1]
        w2 = fig_size[0]
        fig.set_size_inches([(w2 / float(w)) * w, (w2 / float(w)) * h])
        plt.close(fig)

    inds = dict(coronal=[0, 1, 2], axial=[2, 0, 1],
                sagittal=[2, 1, 0])[orientation]
    outs = []
    for ax, sl in zip(axs, slices):
        # adjust the orientations for good view
        if orientation == 'coronal':
            dat = data[:, :, sl].transpose()
        elif orientation == 'axial':
            dat = data[:, sl, :]
        elif orientation == 'sagittal':
            dat = data[sl, :, :]

        # First plot the anatomical data
        if img_output is not None:
            ax.clear()
        ax.imshow(dat, cmap=plt.cm.gray)
        ax.axis('off')

        # and then plot the contours on top
        for surf in surfs:
            ax.tricontour(surf['rr'][:, inds[0]], surf['rr'][:, inds[1]],
                          surf['tris'], surf['rr'][:, inds[2]],
                          levels=[sl], colors='yellow', linewidths=2.0)
        if img_output is not None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, img_output[1])
            ax.set_ylim(img_output[0], 0)
            output = BytesIO()
            fig.savefig(output, bbox_inches='tight',
                        pad_inches=0, format='png')
            outs.append(base64.b64encode(output.getvalue()).decode('ascii'))
    if show:
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.,
                            hspace=0.)
        plt.show()

    return fig if img_output is None else outs


def plot_trans(info, trans='auto', subject=None, subjects_dir=None,
               ch_type=None, source=('bem', 'head'), coord_frame='head'):
    """Plot MEG/EEG head surface and helmet in 3D.

    Parameters
    ----------
    info : dict
        The measurement info.
    trans : str | 'auto'
        The full path to the `*-trans.fif` file produced during
        coregistration.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    ch_type : None | 'eeg' | 'meg'
        If None, both the MEG helmet and EEG electrodes will be shown.
        If 'meg', only the MEG helmet will be shown. If 'eeg', only the
        EEG electrodes will be shown.
    source : str
        Type to load. Common choices would be `'bem'` or `'head'`. We first
        try loading `'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'`, and
        then look for `'$SUBJECT*$SOURCE.fif'` in the same directory. Defaults
        to 'bem'. Note. For single layer bems it is recommended to use 'head'.
    coord_frame : str
        Coordinate frame to use.

    Returns
    -------
    fig : instance of mlab.Figure
        The mayavi figure.
    """
    if coord_frame not in ['head', 'meg']:
        raise ValueError('coord_frame must be "head" or "meg"')
    if ch_type not in [None, 'eeg', 'meg']:
        raise ValueError('Argument ch_type must be None | eeg | meg. Got %s.'
                         % ch_type)

    if trans == 'auto':
        # let's try to do this in MRI coordinates so they're easy to plot
        trans = _find_trans(subject, subjects_dir)

    trans = read_trans(trans)

    surfs = [get_head_surf(subject, source=source, subjects_dir=subjects_dir)]
    if ch_type is None or ch_type == 'meg':
        surfs.append(get_meg_helmet_surf(info, trans))
    if coord_frame == 'meg':
        meg_trans = combine_transforms(info['dev_head_t'], trans,
                                       FIFF.FIFFV_COORD_DEVICE,
                                       FIFF.FIFFV_COORD_MRI)
        surfs = [transform_surface_to(surf, 'meg', meg_trans)
                 for surf in surfs]

    # Plot them
    from mayavi import mlab
    alphas = [1.0, 0.5]
    colors = [(0.6, 0.6, 0.6), (0.0, 0.0, 0.6)]

    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))

    for ii, surf in enumerate(surfs):

        x, y, z = surf['rr'].T
        nn = surf['nn']
        # make absolutely sure these are normalized for Mayavi
        nn = nn / np.sum(nn * nn, axis=1)[:, np.newaxis]

        # Make a solid surface
        alpha = alphas[ii]
        with warnings.catch_warnings(record=True):  # traits
            mesh = mlab.pipeline.triangular_mesh_source(x, y, z, surf['tris'])
        mesh.data.point_data.normals = nn
        mesh.data.cell_data.normals = None
        mlab.pipeline.surface(mesh, color=colors[ii], opacity=alpha)

    if ch_type is None or ch_type == 'eeg':
        eeg_locs = [l['eeg_loc'][:, 0] for l in info['chs']
                    if l['eeg_loc'] is not None]

        if len(eeg_locs) > 0:
            eeg_loc = np.array(eeg_locs)

            # Transform EEG electrodes to MRI coordinates
            eeg_loc = apply_trans(trans['trans'], eeg_loc)

            with warnings.catch_warnings(record=True):  # traits
                mlab.points3d(eeg_loc[:, 0], eeg_loc[:, 1], eeg_loc[:, 2],
                              color=(1.0, 0.0, 0.0), scale_factor=0.005)
        else:
            warnings.warn('EEG electrode locations not found. '
                          'Cannot plot EEG electrodes.')

    mlab.view(90, 90)
    return fig


def _limits_to_control_points(clim, stc_data, colormap):
    """Private helper function to convert limits (values or percentiles)
    to control points.

    Note: If using 'mne', generate cmap control points for a directly
    mirrored cmap for simplicity (i.e., no normalization is computed to account
    for a 2-tailed mne cmap).

    Parameters
    ----------
    clim : str | dict
        Desired limits use to set cmap control points.

    Returns
    -------
    ctrl_pts : list (length 3)
        Array of floats corresponding to values to use as cmap control points.
    colormap : str
        The colormap.
    """

    # Based on type of limits specified, get cmap control points
    if colormap == 'auto':
        if clim == 'auto':
            colormap = 'mne' if (stc_data < 0).any() else 'hot'
        else:
            if 'lims' in clim:
                colormap = 'hot'
            else:  # 'pos_lims' in clim
                colormap = 'mne'
    if clim == 'auto':
        # Set upper and lower bound based on percent, and get average between
        ctrl_pts = np.percentile(np.abs(stc_data), [96, 97.5, 99.95])
    elif isinstance(clim, dict):
        # Get appropriate key for clim if it's a dict
        limit_key = ['lims', 'pos_lims'][colormap in ('mne', 'mne_analyze')]
        if colormap != 'auto' and limit_key not in clim.keys():
            raise KeyError('"pos_lims" must be used with "mne" colormap')
        clim['kind'] = clim.get('kind', 'percent')
        if clim['kind'] == 'percent':
            ctrl_pts = np.percentile(np.abs(stc_data),
                                     list(np.abs(clim[limit_key])))
        elif clim['kind'] == 'value':
            ctrl_pts = np.array(clim[limit_key])
            if (np.diff(ctrl_pts) < 0).any():
                raise ValueError('value colormap limits must be strictly '
                                 'nondecreasing')
        else:
            raise ValueError('If clim is a dict, clim[kind] must be '
                             ' "value" or "percent"')
    else:
        raise ValueError('"clim" must be "auto" or dict')
    if len(ctrl_pts) != 3:
        raise ValueError('"lims" or "pos_lims" is length %i. It must be length'
                         ' 3' % len(ctrl_pts))
    ctrl_pts = np.array(ctrl_pts, float)
    if len(set(ctrl_pts)) != 3:
        if len(set(ctrl_pts)) == 1:  # three points match
            if ctrl_pts[0] == 0:  # all are zero
                warnings.warn('All data were zero')
                ctrl_pts = np.arange(3, dtype=float)
            else:
                ctrl_pts *= [0., 0.5, 1]  # all nonzero pts == max
        else:  # two points match
            # if points one and two are identical, add a tiny bit to the
            # control point two; if points two and three are identical,
            # subtract a tiny bit from point two.
            bump = 1e-5 if ctrl_pts[0] == ctrl_pts[1] else -1e-5
            ctrl_pts[1] = ctrl_pts[0] + bump * (ctrl_pts[2] - ctrl_pts[0])

    return ctrl_pts, colormap


def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='auto', time_label='time=%0.2f ms',
                          smoothing_steps=10, transparent=None, alpha=1.0,
                          time_viewer=False, config_opts=None,
                          subjects_dir=None, figure=None, views='lat',
                          colorbar=True, clim='auto'):
    """Plot SourceEstimates with PySurfer

    Note: PySurfer currently needs the SUBJECTS_DIR environment variable,
    which will automatically be set by this function. Plotting multiple
    SourceEstimates with different values for subjects_dir will cause
    PySurfer to use the wrong FreeSurfer surfaces when using methods of
    the returned Brain object. It is therefore recommended to set the
    SUBJECTS_DIR environment variable or always use the same value for
    subjects_dir (within the same Python session).

    Parameters
    ----------
    stc : SourceEstimates
        The source estimates to plot.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None stc.subject will be used. If that
        is None, the environment will be used.
    surface : str
        The type of surface (inflated, white etc.).
    hemi : str, 'lh' | 'rh' | 'split' | 'both'
        The hemisphere to display.
    colormap : str | np.ndarray of float, shape(n_colors, 3 | 4)
        Name of colormap to use or a custom look up table. If array, must
        be (n x 3) or (n x 4) array for with RGB or RGBA values between
        0 and 255. If 'auto', either 'hot' or 'mne' will be chosen
        based on whether 'lims' or 'pos_lims' are specified in `clim`.
    time_label : str
        How to print info about the time instant visualized.
    smoothing_steps : int
        The amount of smoothing
    transparent : bool | None
        If True, use a linear transparency between fmin and fmid.
        None will choose automatically based on colormap type.
    alpha : float
        Alpha value to apply globally to the overlay.
    time_viewer : bool
        Display time viewer GUI.
    config_opts : dict
        Keyword arguments for Brain initialization.
        See pysurfer.viz.Brain.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    figure : instance of mayavi.core.scene.Scene | list | int | None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the Mayavi
        figure by it's id or create a new figure with the given id.
    views : str | list
        View to use. See surfer.Brain().
    colorbar : bool
        If True, display colorbar on scene.
    clim : str | dict
        Colorbar properties specification. If 'auto', set clim automatically
        based on data percentiles. If dict, should contain:

            ``kind`` : str
                Flag to specify type of limits. 'value' or 'percent'.
            ``lims`` : list | np.ndarray | tuple of float, 3 elements
                Note: Only use this if 'colormap' is not 'mne'.
                Left, middle, and right bound for colormap.
            ``pos_lims`` : list | np.ndarray | tuple of float, 3 elements
                Note: Only use this if 'colormap' is 'mne'.
                Left, middle, and right bound for colormap. Positive values
                will be mirrored directly across zero during colormap
                construction to obtain negative control points.


    Returns
    -------
    brain : Brain
        A instance of surfer.viz.Brain from PySurfer.
    """
    from surfer import Brain, TimeViewer
    config_opts = _handle_default('config_opts', config_opts)

    import mayavi
    from mayavi import mlab

    # import here to avoid circular import problem
    from ..source_estimate import SourceEstimate

    if not isinstance(stc, SourceEstimate):
        raise ValueError('stc has to be a surface source estimate')

    if hemi not in ['lh', 'rh', 'split', 'both']:
        raise ValueError('hemi has to be either "lh", "rh", "split", '
                         'or "both"')

    n_split = 2 if hemi == 'split' else 1
    n_views = 1 if isinstance(views, string_types) else len(views)
    if figure is not None:
        # use figure with specified id or create new figure
        if isinstance(figure, int):
            figure = mlab.figure(figure, size=(600, 600))
        # make sure it is of the correct type
        if not isinstance(figure, list):
            figure = [figure]
        if not all(isinstance(f, mayavi.core.scene.Scene) for f in figure):
            raise TypeError('figure must be a mayavi scene or list of scenes')
        # make sure we have the right number of figures
        n_fig = len(figure)
        if not n_fig == n_split * n_views:
            raise RuntimeError('`figure` must be a list with the same '
                               'number of elements as PySurfer plots that '
                               'will be created (%s)' % n_split * n_views)

    # convert control points to locations in colormap
    ctrl_pts, colormap = _limits_to_control_points(clim, stc.data, colormap)

    # Construct cmap manually if 'mne' and get cmap bounds
    # and triage transparent argument
    if colormap in ('mne', 'mne_analyze'):
        colormap = mne_analyze_colormap(ctrl_pts)
        scale_pts = [-1 * ctrl_pts[-1], 0, ctrl_pts[-1]]
        transparent = False if transparent is None else transparent
    else:
        scale_pts = ctrl_pts
        transparent = True if transparent is None else transparent

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    subject = _check_subject(stc.subject, subject, True)
    if hemi in ['both', 'split']:
        hemis = ['lh', 'rh']
    else:
        hemis = [hemi]

    title = subject if len(hemis) > 1 else '%s - %s' % (subject, hemis[0])
    args = inspect.getargspec(Brain.__init__)[0]
    kwargs = dict(title=title, figure=figure, config_opts=config_opts,
                  subjects_dir=subjects_dir)
    if 'views' in args:
        kwargs['views'] = views
    with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(subject, hemi, surface, **kwargs)
    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1
        if hemi_idx == 0:
            data = stc.data[:len(stc.vertices[0])]
        else:
            data = stc.data[len(stc.vertices[0]):]
        vertices = stc.vertices[hemi_idx]
        time = 1e3 * stc.times
        with warnings.catch_warnings(record=True):  # traits warnings
            brain.add_data(data, colormap=colormap, vertices=vertices,
                           smoothing_steps=smoothing_steps, time=time,
                           time_label=time_label, alpha=alpha, hemi=hemi,
                           colorbar=colorbar)

        # scale colormap and set time (index) to display
        brain.scale_data_colormap(fmin=scale_pts[0], fmid=scale_pts[1],
                                  fmax=scale_pts[2], transparent=transparent)

    if time_viewer:
        TimeViewer(brain)
    return brain


def plot_sparse_source_estimates(src, stcs, colors=None, linewidth=2,
                                 fontsize=18, bgcolor=(.05, 0, .1),
                                 opacity=0.2, brain_color=(0.7,) * 3,
                                 show=True, high_resolution=False,
                                 fig_name=None, fig_number=None, labels=None,
                                 modes=('cone', 'sphere'),
                                 scale_factors=(1, 0.6),
                                 verbose=None, **kwargs):
    """Plot source estimates obtained with sparse solver

    Active dipoles are represented in a "Glass" brain.
    If the same source is active in multiple source estimates it is
    displayed with a sphere otherwise with a cone in 3D.

    Parameters
    ----------
    src : dict
        The source space.
    stcs : instance of SourceEstimate or list of instances of SourceEstimate
        The source estimates (up to 3).
    colors : list
        List of colors
    linewidth : int
        Line width in 2D plot.
    fontsize : int
        Font size.
    bgcolor : tuple of length 3
        Background color in 3D.
    opacity : float in [0, 1]
        Opacity of brain mesh.
    brain_color : tuple of length 3
        Brain color.
    show : bool
        Show figures if True.
    high_resolution : bool
        If True, plot on the original (non-downsampled) cortical mesh.
    fig_name :
        Mayavi figure name.
    fig_number :
        Matplotlib figure number.
    labels : ndarray or list of ndarrays
        Labels to show sources in clusters. Sources with the same
        label and the waveforms within each cluster are presented in
        the same color. labels should be a list of ndarrays when
        stcs is a list ie. one label for each stc.
    modes : list
        Should be a list, with each entry being ``'cone'`` or ``'sphere'``
        to specify how the dipoles should be shown.
    scale_factors : list
        List of floating point scale factors for the markers.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    **kwargs : kwargs
        Keyword arguments to pass to mlab.triangular_mesh.
    """
    known_modes = ['cone', 'sphere']
    if not isinstance(modes, (list, tuple)) or \
            not all(mode in known_modes for mode in modes):
        raise ValueError('mode must be a list containing only '
                         '"cone" or "sphere"')
    if not isinstance(stcs, list):
        stcs = [stcs]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    if colors is None:
        colors = COLORS

    linestyles = ['-', '--', ':']

    # Show 3D
    lh_points = src[0]['rr']
    rh_points = src[1]['rr']
    points = np.r_[lh_points, rh_points]

    lh_normals = src[0]['nn']
    rh_normals = src[1]['nn']
    normals = np.r_[lh_normals, rh_normals]

    if high_resolution:
        use_lh_faces = src[0]['tris']
        use_rh_faces = src[1]['tris']
    else:
        use_lh_faces = src[0]['use_tris']
        use_rh_faces = src[1]['use_tris']

    use_faces = np.r_[use_lh_faces, lh_points.shape[0] + use_rh_faces]

    points *= 170

    vertnos = [np.r_[stc.lh_vertno, lh_points.shape[0] + stc.rh_vertno]
               for stc in stcs]
    unique_vertnos = np.unique(np.concatenate(vertnos).ravel())

    from mayavi import mlab
    from matplotlib.colors import ColorConverter
    color_converter = ColorConverter()

    f = mlab.figure(figure=fig_name, bgcolor=bgcolor, size=(600, 600))
    mlab.clf()
    if mlab.options.backend != 'test':
        f.scene.disable_render = True
    with warnings.catch_warnings(record=True):  # traits warnings
        surface = mlab.triangular_mesh(points[:, 0], points[:, 1],
                                       points[:, 2], use_faces,
                                       color=brain_color,
                                       opacity=opacity, **kwargs)

    import matplotlib.pyplot as plt
    # Show time courses
    plt.figure(fig_number)
    plt.clf()

    colors = cycle(colors)

    logger.info("Total number of active sources: %d" % len(unique_vertnos))

    if labels is not None:
        colors = [advance_iterator(colors) for _ in
                  range(np.unique(np.concatenate(labels).ravel()).size)]

    for idx, v in enumerate(unique_vertnos):
        # get indices of stcs it belongs to
        ind = [k for k, vertno in enumerate(vertnos) if v in vertno]
        is_common = len(ind) > 1

        if labels is None:
            c = advance_iterator(colors)
        else:
            # if vertex is in different stcs than take label from first one
            c = colors[labels[ind[0]][vertnos[ind[0]] == v]]

        mode = modes[1] if is_common else modes[0]
        scale_factor = scale_factors[1] if is_common else scale_factors[0]

        if (isinstance(scale_factor, (np.ndarray, list, tuple)) and
                len(unique_vertnos) == len(scale_factor)):
            scale_factor = scale_factor[idx]

        x, y, z = points[v]
        nx, ny, nz = normals[v]
        with warnings.catch_warnings(record=True):  # traits
            mlab.quiver3d(x, y, z, nx, ny, nz, color=color_converter.to_rgb(c),
                          mode=mode, scale_factor=scale_factor)

        for k in ind:
            vertno = vertnos[k]
            mask = (vertno == v)
            assert np.sum(mask) == 1
            linestyle = linestyles[k]
            plt.plot(1e3 * stc.times, 1e9 * stcs[k].data[mask].ravel(), c=c,
                     linewidth=linewidth, linestyle=linestyle)

    plt.xlabel('Time (ms)', fontsize=18)
    plt.ylabel('Source amplitude (nAm)', fontsize=18)

    if fig_name is not None:
        plt.title(fig_name)

    if show:
        plt.show()

    surface.actor.property.backface_culling = True
    surface.actor.property.shading = True

    return surface


def plot_dipole_locations(dipoles, trans, subject, subjects_dir=None,
                          bgcolor=(1, 1, 1), opacity=0.3,
                          brain_color=(0.7, 0.7, 0.7), mesh_color=(1, 1, 0),
                          fig_name=None, fig_size=(600, 600), mode='cone',
                          scale_factor=0.1e-1, colors=None, verbose=None):
    """Plot dipole locations

    Only the location of the first time point of each dipole is shown.

    Parameters
    ----------
    dipoles : list of instances of Dipole | Dipole
        The dipoles to plot.
    trans : dict
        The mri to head trans.
    subject : str
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT.
    subjects_dir : None | str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
        The default is None.
    bgcolor : tuple of length 3
        Background color in 3D.
    opacity : float in [0, 1]
        Opacity of brain mesh.
    brain_color : tuple of length 3
        Brain color.
    mesh_color : tuple of length 3
        Mesh color.
    fig_name : str
        Mayavi figure name.
    fig_size : tuple of length 2
        Mayavi figure size.
    mode : str
        Should be ``'cone'`` or ``'sphere'`` to specify how the
        dipoles should be shown.
    scale_factor : float
        The scaling applied to amplitudes for the plot.
    colors: list of colors | None
        Color to plot with each dipole. If None default colors are used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fig : instance of mlab.Figure
        The mayavi figure.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    from mayavi import mlab
    from matplotlib.colors import ColorConverter
    color_converter = ColorConverter()

    trans = _get_mri_head_t(trans)[0]
    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    fname = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')
    points, faces = read_surface(fname)
    points = apply_trans(trans['trans'], points * 1e-3)

    from .. import Dipole
    if isinstance(dipoles, Dipole):
        dipoles = [dipoles]

    if mode not in ['cone', 'sphere']:
        raise ValueError('mode must be in "cone" or "sphere"')

    if colors is None:
        colors = cycle(COLORS)

    fig = mlab.figure(size=fig_size, bgcolor=bgcolor, fgcolor=(0, 0, 0))
    with warnings.catch_warnings(record=True):  # FutureWarning in traits
        mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
                             faces, color=mesh_color, opacity=opacity)

    for dip, color in zip(dipoles, colors):
        rgb_color = color_converter.to_rgb(color)
        with warnings.catch_warnings(record=True):  # FutureWarning in traits
            mlab.quiver3d(dip.pos[0, 0], dip.pos[0, 1], dip.pos[0, 2],
                          dip.ori[0, 0], dip.ori[0, 1], dip.ori[0, 2],
                          opacity=1., mode=mode, color=rgb_color,
                          scalars=dip.amplitude.max(),
                          scale_factor=scale_factor)
    if fig_name is not None:
        mlab.title(fig_name)
    if fig.scene is not None:  # safe for Travis
        fig.scene.x_plus_view()

    return fig
