"""Functions to make 3D plots with M/EEG data
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

from ..externals.six import string_types, advance_iterator

from distutils.version import LooseVersion

import os
import inspect
import warnings
from itertools import cycle

import numpy as np
from scipy import linalg

from ..io.pick import pick_types
from ..surface import get_head_surf, get_meg_helmet_surf, read_surface
from ..transforms import read_trans, _find_trans, apply_trans
from ..utils import get_subjects_dir, logger, _check_subject
from .utils import mne_analyze_colormap, _prepare_trellis, COLORS


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
                       slices=None, show=True):
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

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
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

    fig, axs = _prepare_trellis(len(slices), 4)

    for ax, sl in zip(axs, slices):

        # adjust the orientations for good view
        if orientation == 'coronal':
            dat = data[:, :, sl].transpose()
        elif orientation == 'axial':
            dat = data[:, sl, :]
        elif orientation == 'sagittal':
            dat = data[sl, :, :]

        # First plot the anatomical data
        ax.imshow(dat, cmap=plt.cm.gray)
        ax.axis('off')

        # and then plot the contours on top
        for surf in surfs:
            if orientation == 'coronal':
                ax.tricontour(surf['rr'][:, 0], surf['rr'][:, 1],
                              surf['tris'], surf['rr'][:, 2],
                              levels=[sl], colors='yellow', linewidths=2.0)
            elif orientation == 'axial':
                ax.tricontour(surf['rr'][:, 2], surf['rr'][:, 0],
                              surf['tris'], surf['rr'][:, 1],
                              levels=[sl], colors='yellow', linewidths=2.0)
            elif orientation == 'sagittal':
                ax.tricontour(surf['rr'][:, 2], surf['rr'][:, 1],
                              surf['tris'], surf['rr'][:, 0],
                              levels=[sl], colors='yellow', linewidths=2.0)

    if show:
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.,
                            hspace=0.)
        plt.show()

    return fig


def plot_trans(info, trans_fname='auto', subject=None, subjects_dir=None,
               ch_type=None, source='bem'):
    """Plot MEG/EEG head surface and helmet in 3D.

    Parameters
    ----------
    info : dict
        The measurement info.
    trans_fname : str | 'auto'
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

    Returns
    -------
    fig : instance of mlab.Figure
        The mayavi figure.
    """

    if ch_type not in [None, 'eeg', 'meg']:
        raise ValueError('Argument ch_type must be None | eeg | meg. Got %s.'
                         % ch_type)

    if trans_fname == 'auto':
        # let's try to do this in MRI coordinates so they're easy to plot
        trans_fname = _find_trans(subject, subjects_dir)

    trans = read_trans(trans_fname)

    surfs = [get_head_surf(subject, source=source, subjects_dir=subjects_dir)]
    if ch_type is None or ch_type == 'meg':
        surfs.append(get_meg_helmet_surf(info, trans))

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


def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='hot', time_label='time=%0.2f ms',
                          smoothing_steps=10, fmin=5., fmid=10., fmax=15.,
                          transparent=True, alpha=1.0, time_viewer=False,
                          config_opts={}, subjects_dir=None, figure=None,
                          views='lat', colorbar=True):
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
        The hemisphere to display. Using 'both' or 'split' requires
        PySurfer version 0.4 or above.
    colormap : str
        The type of colormap to use.
    time_label : str
        How to print info about the time instant visualized.
    smoothing_steps : int
        The amount of smoothing
    fmin : float
        The minimum value to display.
    fmid : float
        The middle value on the colormap.
    fmax : float
        The maximum value for the colormap.
    transparent : bool
        If True, use a linear transparency between fmin and fmid.
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

    Returns
    -------
    brain : Brain
        A instance of surfer.viz.Brain from PySurfer.
    """
    import surfer
    from surfer import Brain, TimeViewer

    if hemi in ['split', 'both'] and LooseVersion(surfer.__version__) < '0.4':
        raise NotImplementedError('hemi type "%s" not supported with your '
                                  'version of pysurfer. Please upgrade to '
                                  'version 0.4 or higher.' % hemi)

    try:
        import mayavi
        from mayavi import mlab
    except ImportError:
        from enthought import mayavi
        from enthought.mayavi import mlab

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
        if not all([isinstance(f, mayavi.core.scene.Scene) for f in figure]):
            raise TypeError('figure must be a mayavi scene or list of scenes')
        # make sure we have the right number of figures
        n_fig = len(figure)
        if not n_fig == n_split * n_views:
            raise RuntimeError('`figure` must be a list with the same '
                               'number of elements as PySurfer plots that '
                               'will be created (%s)' % n_split * n_views)

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir)

    subject = _check_subject(stc.subject, subject, False)
    if subject is None:
        if 'SUBJECT' in os.environ:
            subject = os.environ['SUBJECT']
        else:
            raise ValueError('SUBJECT environment variable not set')

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
    else:
        logger.info('PySurfer does not support "views" argument, please '
                    'consider updating to a newer version (0.4 or later)')
    with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(subject, hemi, surface, **kwargs)
    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1
        if hemi_idx == 0:
            data = stc.data[:len(stc.vertno[0])]
        else:
            data = stc.data[len(stc.vertno[0]):]
        vertices = stc.vertno[hemi_idx]
        time = 1e3 * stc.times
        with warnings.catch_warnings(record=True):  # traits warnings
            brain.add_data(data, colormap=colormap, vertices=vertices,
                           smoothing_steps=smoothing_steps, time=time,
                           time_label=time_label, alpha=alpha, hemi=hemi,
                           colorbar=colorbar)

        # scale colormap and set time (index) to display
        brain.scale_data_colormap(fmin=fmin, fmid=fmid, fmax=fmax,
                                  transparent=transparent)

    if time_viewer:
        TimeViewer(brain)
    return brain


def plot_sparse_source_estimates(src, stcs, colors=None, linewidth=2,
                                 fontsize=18, bgcolor=(.05, 0, .1),
                                 opacity=0.2, brain_color=(0.7,) * 3,
                                 show=True, high_resolution=False,
                                 fig_name=None, fig_number=None, labels=None,
                                 modes=['cone', 'sphere'],
                                 scale_factors=[1, 0.6],
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
    fig_name :
        Mayavi figure name.
    fig_number :
        Matplotlib figure number.
    labels : ndarray or list of ndarrays
        Labels to show sources in clusters. Sources with the same
        label and the waveforms within each cluster are presented in
        the same color. labels should be a list of ndarrays when
        stcs is a list ie. one label for each stc.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    kwargs : kwargs
        Keyword arguments to pass to mlab.triangular_mesh.
    """
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

    try:
        from mayavi import mlab
    except ImportError:
        from enthought.mayavi import mlab

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

        if (isinstance(scale_factor, (np.ndarray, list, tuple))
                and len(unique_vertnos) == len(scale_factor)):
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
