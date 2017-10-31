# -*- coding: utf-8 -*-
"""Functions to make 3D plots with M/EEG data."""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import base64
from distutils.version import LooseVersion
from itertools import cycle
import os.path as op
import warnings
from functools import partial

import numpy as np
from scipy import linalg

from ..defaults import DEFAULTS
from ..externals.six import BytesIO, string_types, advance_iterator
from ..io import _loc_to_coil_trans, Info
from ..io.pick import pick_types
from ..io.constants import FIFF
from ..io.meas_info import read_fiducials
from ..source_space import SourceSpaces, _create_surf_spacing, _check_spacing

from ..surface import (_get_head_surface, get_meg_helmet_surf, read_surface,
                       transform_surface_to, _project_onto_surface,
                       complete_surface_info, mesh_edges,
                       _complete_sphere_surf)
from ..transforms import (read_trans, _find_trans, apply_trans,
                          combine_transforms, _get_trans, _ensure_trans,
                          invert_transform, Transform)
from ..utils import (get_subjects_dir, logger, _check_subject, verbose, warn,
                     _import_mlab, SilenceStdout, has_nibabel, check_version,
                     _ensure_int, deprecated)
from .utils import (mne_analyze_colormap, _prepare_trellis, COLORS, plt_show,
                    tight_layout, figure_nobar)
from ..bem import ConductorModel, _bem_find_surface, _surf_dict, _surf_name


FIDUCIAL_ORDER = (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION,
                  FIFF.FIFFV_POINT_RPA)


def _fiducial_coords(points, coord_frame=None):
    """Generate 3x3 array of fiducial coordinates."""
    if coord_frame is not None:
        points = [p for p in points if p['coord_frame'] == coord_frame]
    points_ = dict((p['ident'], p) for p in points if
                   p['kind'] == FIFF.FIFFV_POINT_CARDINAL)
    if points_:
        return np.array([points_[i]['r'] for i in FIDUCIAL_ORDER])
    else:
        # XXX eventually this should probably live in montage.py
        if coord_frame is None or coord_frame == FIFF.FIFFV_COORD_HEAD:
            # Try converting CTF HPI coils to fiducials
            out = np.empty((3, 3))
            out.fill(np.nan)
            for p in points:
                if p['kind'] == FIFF.FIFFV_POINT_HPI:
                    if np.isclose(p['r'][1:], 0, atol=1e-6).all():
                        out[0 if p['r'][0] < 0 else 2] = p['r']
                    elif np.isclose(p['r'][::2], 0, atol=1e-6).all():
                        out[1] = p['r']
            if np.isfinite(out).all():
                return out
        return np.array([])


def plot_head_positions(pos, mode='traces', cmap='viridis', direction='z',
                        show=True):
    """Plot head positions.

    Parameters
    ----------
    pos : ndarray, shape (n_pos, 10)
        The head position data.
    mode : str
        Can be 'traces' (default) to show position and quaternion traces,
        or 'field' to show the position as a vector field over time.
        The 'field' mode requires matplotlib 1.4+.
    cmap : matplotlib Colormap
        Colormap to use for the trace plot, default is "viridis".
    direction : str
        Can be any combination of "x", "y", or "z" (default: "z") to show
        directional axes in "field" mode.
    show : bool
        Show figure if True. Defaults to True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    from ..chpi import head_pos_to_trans_rot_t
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from mpl_toolkits.mplot3d import axes3d  # noqa: F401, analysis:ignore
    if not isinstance(mode, string_types) or mode not in ('traces', 'field'):
        raise ValueError('mode must be "traces" or "field", got %s' % (mode,))
    trans, rot, t = head_pos_to_trans_rot_t(pos)  # also ensures pos is okay
    # trans, rot, and t are for dev_head_t, but what we really want
    # is head_dev_t (i.e., where the head origin is in device coords)
    use_trans = np.einsum('ijk,ik->ij', rot[:, :3, :3].transpose([0, 2, 1]),
                          -trans) * 1000
    use_rot = rot.transpose([0, 2, 1])
    use_quats = -pos[:, 1:4]  # inverse (like doing rot.T)
    if cmap == 'viridis' and not check_version('matplotlib', '1.5'):
        warn('viridis is unavailable on matplotlib < 1.4, using "YlGnBu_r"')
        cmap = 'YlGnBu_r'
    if mode == 'traces':
        fig, axes = plt.subplots(3, 2, sharex=True)
        labels = ['xyz', ('$q_1$', '$q_2$', '$q_3$')]
        for ii, (quat, coord) in enumerate(zip(use_quats.T, use_trans.T)):
            axes[ii, 0].plot(t, coord, 'k')
            axes[ii, 0].set(ylabel=labels[0][ii], xlim=t[[0, -1]])
            axes[ii, 1].plot(t, quat, 'k')
            axes[ii, 1].set(ylabel=labels[1][ii], xlim=t[[0, -1]])
        for ii, title in enumerate(('Position (mm)', 'Rotation (quat)')):
            axes[0, ii].set(title=title)
            axes[-1, ii].set(xlabel='Time (s)')
    else:  # mode == 'field':
        if not check_version('matplotlib', '1.4'):
            raise RuntimeError('The "field" mode requires matplotlib version '
                               '1.4+')
        fig, ax = plt.subplots(1, subplot_kw=dict(projection='3d'))
        # First plot the trajectory as a colormap:
        # http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        pts = use_trans[:, np.newaxis]
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = Normalize(t[0], t[-2])
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        # now plot the head directions as a quiver
        dir_idx = dict(x=0, y=1, z=2)
        kwargs = _pivot_kwargs()
        for d, length in zip(direction, [1., 0.5, 0.25]):
            use_dir = use_rot[:, :, dir_idx[d]]
            # draws stems, then heads
            array = np.concatenate((t, np.repeat(t, 2)))
            ax.quiver(use_trans[:, 0], use_trans[:, 1], use_trans[:, 2],
                      use_dir[:, 0], use_dir[:, 1], use_dir[:, 2], norm=norm,
                      cmap=cmap, array=array, length=length, **kwargs)
        mins = use_trans.min(0)
        maxs = use_trans.max(0)
        scale = (maxs - mins).max() / 2.
        xlim, ylim, zlim = (maxs + mins)[:, np.newaxis] / 2. + [-scale, scale]
        ax.set(xlabel='x', ylabel='y', zlabel='z',
               xlim=xlim, ylim=ylim, zlim=zlim, aspect='equal')
        ax.view_init(30, 45)
    tight_layout(fig=fig)
    plt_show(show)
    return fig


def _pivot_kwargs():
    """Get kwargs for quiver."""
    kwargs = dict()
    if check_version('matplotlib', '1.5'):
        kwargs['pivot'] = 'tail'
    else:
        import matplotlib
        warn('pivot cannot be set in matplotlib %s (need version 1.5+), '
             'locations are approximate' % (matplotlib.__version__,))
    return kwargs


def plot_evoked_field(evoked, surf_maps, time=None, time_label='t = %0.0f ms',
                      n_jobs=1):
    """Plot MEG/EEG fields on head surface and helmet in 3D.

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
    mlab = _import_mlab()
    alphas = [1.0, 0.5]
    colors = [(0.6, 0.6, 0.6), (1.0, 1.0, 1.0)]
    colormap = mne_analyze_colormap(format='mayavi')
    colormap_lines = np.concatenate([np.tile([0., 0., 255., 255.], (127, 1)),
                                     np.tile([0., 0., 0., 255.], (2, 1)),
                                     np.tile([255., 0., 0., 255.], (127, 1))])

    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
    _toggle_mlab_render(fig, False)

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

        # Make a solid surface
        vlim = np.max(np.abs(data))
        alpha = alphas[ii]
        mesh = _create_mesh_surf(surf, fig)
        with warnings.catch_warnings(record=True):  # traits
            surface = mlab.pipeline.surface(mesh, color=colors[ii],
                                            opacity=alpha, figure=fig)
        surface.actor.property.backface_culling = True

        # Now show our field pattern
        mesh = _create_mesh_surf(surf, fig, scalars=data)
        with warnings.catch_warnings(record=True):  # traits
            fsurf = mlab.pipeline.surface(mesh, vmin=-vlim, vmax=vlim,
                                          figure=fig)
        fsurf.module_manager.scalar_lut_manager.lut.table = colormap
        fsurf.actor.property.backface_culling = True

        # And the field lines on top
        mesh = _create_mesh_surf(surf, fig, scalars=data)
        with warnings.catch_warnings(record=True):  # traits
            cont = mlab.pipeline.contour_surface(
                mesh, contours=21, line_width=1.0, vmin=-vlim, vmax=vlim,
                opacity=alpha, figure=fig)
        cont.module_manager.scalar_lut_manager.lut.table = colormap_lines

    if '%' in time_label:
        time_label %= (1e3 * evoked.times[time_idx])
    with warnings.catch_warnings(record=True):  # traits
        mlab.text(0.01, 0.01, time_label, width=0.4, figure=fig)
        with SilenceStdout():  # setting roll
            mlab.view(10, 60, figure=fig)
    _toggle_mlab_render(fig, True)
    return fig


def _create_mesh_surf(surf, fig=None, scalars=None):
    """Create Mayavi mesh from MNE surf."""
    mlab = _import_mlab()
    nn = surf['nn'].copy()
    # make absolutely sure these are normalized for Mayavi
    norm = np.sum(nn * nn, axis=1)
    mask = norm > 0
    nn[mask] /= norm[mask][:, np.newaxis]
    x, y, z = surf['rr'].T
    with warnings.catch_warnings(record=True):  # traits
        mesh = mlab.pipeline.triangular_mesh_source(
            x, y, z, surf['tris'], scalars=scalars, figure=fig)
    mesh.data.point_data.normals = nn
    mesh.data.cell_data.normals = None
    mesh.update()
    return mesh


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
    try:
        affine = nim.affine
    except AttributeError:  # old nibabel
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
        surf = read_surface(surf_fname, return_dict=True)[-1]
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
    plt_show(show)
    return fig if img_output is None else outs


@deprecated('this function will be removed in version 0.16. '
            'Use plot_alignment instead')
@verbose
def plot_trans(info, trans='auto', subject=None, subjects_dir=None,
               source=('bem', 'head', 'outer_skin'),
               coord_frame='head', meg_sensors=('helmet', 'sensors'),
               eeg_sensors='original', dig=False, ref_meg=False,
               ecog_sensors=True, head=None, brain=None, skull=False,
               src=None, mri_fiducials=False, verbose=None):
    """Plot head, sensor, and source space alignment in 3D.

    Parameters
    ----------
    info : dict
        The measurement info.
    trans : str | 'auto' | dict | None
        The full path to the head<->MRI transform ``*-trans.fif`` file
        produced during coregistration. If trans is None, an identity matrix
        is assumed.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. Can be omitted if ``src`` is provided.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    source : str | list
        Type to load. Common choices would be `'bem'`, `'head'` or
        `'outer_skin'`. If list, the sources are looked up in the given order
        and first found surface is used. We first try loading
        `'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'`, and then look for
        `'$SUBJECT*$SOURCE.fif'` in the same directory. For `'outer_skin'`,
        the subjects bem and bem/flash folders are searched. Defaults to 'bem'.
        Note. For single layer bems it is recommended to use 'head'.
    coord_frame : str
        Coordinate frame to use, 'head', 'meg', or 'mri'.
    meg_sensors : bool | str | list
        Can be "helmet" (equivalent to False) or "sensors" to show the MEG
        helmet or sensors, respectively, or a combination of the two like
        ``['helmet', 'sensors']`` (equivalent to True, default) or ``[]``.
    eeg_sensors : bool | str | list
        Can be "original" (default; equivalent to True) or "projected" to
        show EEG sensors in their digitized locations or projected onto the
        scalp, or a list of these options including ``[]`` (equivalent of
        False).
    dig : bool | 'fiducials'
        If True, plot the digitization points; 'fiducials' to plot fiducial
        points only.
    ref_meg : bool
        If True (default False), include reference MEG sensors.
    ecog_sensors : bool
        If True (default), show ECoG sensors.
    head : bool | None
        If True, show head surface. Can also be None, which will show the
        head surface for MEG and EEG, but hide it if ECoG sensors are
        present.
    brain : bool | str | None
        If True, show the brain surfaces. Can also be a str for
        surface type (e.g., 'pial', same as True), or None (True for ECoG,
        False otherwise).
    skull : bool | str | list of str | list of dict
        Whether to plot skull surface. If string, common choices would be
        'inner_skull', or 'outer_skull'. Can also be a list to plot
        multiple skull surfaces. If a list of dicts, each dict must
        contain the complete surface info (such as you get from
        :func:`mne.make_bem_model`). True is an alias of 'outer_skull'.
        The subjects bem and bem/flash folders are searched for the 'surf'
        files. Defaults to False.
    src : instance of SourceSpaces | None
        If not None, also plot the source space points.

        .. versionadded:: 0.14

    mri_fiducials : bool | str
        Plot MRI fiducials (default False). If ``True``, look for a file with
        the canonical name (``bem/{subject}-fiducials.fif``). If ``str`` it
        should provide the full path to the fiducials file.

        .. versionadded:: 0.14

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig : instance of mlab.Figure
        The mayavi figure.
    """
    from ..forward import _create_meg_coils
    mlab = _import_mlab()
    if meg_sensors is False:  # old behavior
        meg_sensors = 'helmet'
    elif meg_sensors is True:
        meg_sensors = ['helmet', 'sensors']
    if eeg_sensors is False:
        eeg_sensors = []
    elif eeg_sensors is True:
        eeg_sensors = 'original'
    if isinstance(eeg_sensors, string_types):
        eeg_sensors = [eeg_sensors]
    if isinstance(meg_sensors, string_types):
        meg_sensors = [meg_sensors]
    for kind, var in zip(('eeg', 'meg'), (eeg_sensors, meg_sensors)):
        if not isinstance(var, (list, tuple)) or \
                not all(isinstance(x, string_types) for x in var):
            raise TypeError('%s_sensors must be list or tuple of str, got %s'
                            % (type(var),))
    if not all(x in ('helmet', 'sensors') for x in meg_sensors):
        raise ValueError('meg_sensors must only contain "helmet" and "points",'
                         ' got %s' % (meg_sensors,))
    if not all(x in ('original', 'projected') for x in eeg_sensors):
        raise ValueError('eeg_sensors must only contain "original" and '
                         '"projected", got %s' % (eeg_sensors,))

    if not isinstance(info, Info):
        raise TypeError('info must be an instance of Info, got %s'
                        % type(info))
    valid_coords = ['head', 'meg', 'mri']
    if coord_frame not in valid_coords:
        raise ValueError('coord_frame must be one of %s' % (valid_coords,))
    if src is not None:
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be None or SourceSpaces, got %s'
                            % (type(src),))
        src_subject = src[0].get('subject_his_id', None)
        subject = src_subject if subject is None else subject
        if src_subject is not None and subject != src_subject:
            raise ValueError('subject ("%s") did not match the subject name '
                             ' in src ("%s")' % (subject, src_subject))
        src_rr = np.concatenate([s['rr'][s['inuse'].astype(bool)]
                                 for s in src])
        src_nn = np.concatenate([s['nn'][s['inuse'].astype(bool)]
                                 for s in src])
    else:
        src_rr = src_nn = np.empty((0, 3))

    meg_picks = pick_types(info, meg=True, ref_meg=ref_meg)
    eeg_picks = pick_types(info, meg=False, eeg=True, ref_meg=False)
    ecog_picks = pick_types(info, meg=False, ecog=True, ref_meg=False)

    if head is None:
        head = (len(ecog_picks) == 0 and subject is not None)
    if head and subject is None:
        raise ValueError('If head is True, subject must be provided')
    if isinstance(trans, string_types):
        if trans == 'auto':
            # let's try to do this in MRI coordinates so they're easy to plot
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            trans = _find_trans(subject, subjects_dir)
        trans = read_trans(trans, return_all=True)
        exp = None
        for trans in trans:  # we got at least 1
            try:
                trans = _ensure_trans(trans, 'head', 'mri')
            except Exception as exp:
                pass
            else:
                break
        else:
            raise exp
    elif trans is None:
        trans = Transform('head', 'mri')
    elif not isinstance(trans, dict):
        raise TypeError('trans must be str, dict, or None')
    head_mri_t = _ensure_trans(trans, 'head', 'mri')
    dev_head_t = info['dev_head_t']
    del trans

    # Figure out our transformations
    if coord_frame == 'meg':
        head_trans = invert_transform(dev_head_t)
        meg_trans = Transform('meg', 'meg')
        mri_trans = invert_transform(combine_transforms(
            dev_head_t, head_mri_t, 'meg', 'mri'))
    elif coord_frame == 'mri':
        head_trans = head_mri_t
        meg_trans = combine_transforms(dev_head_t, head_mri_t, 'meg', 'mri')
        mri_trans = Transform('mri', 'mri')
    else:  # coord_frame == 'head'
        head_trans = Transform('head', 'head')
        meg_trans = info['dev_head_t']
        mri_trans = invert_transform(head_mri_t)

    # both the head and helmet will be in MRI coordinates after this
    surfs = dict()
    if head:
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        head_surf = _get_head_surface(subject, source=source,
                                      subjects_dir=subjects_dir,
                                      raise_error=False)
        if head_surf is None:
            if isinstance(source, string_types):
                source = [source]
            for this_surf in source:
                if not this_surf.endswith('outer_skin'):
                    continue
                surf_fname = op.join(subjects_dir, subject, 'bem', 'flash',
                                     '%s.surf' % this_surf)
                if not op.exists(surf_fname):
                    surf_fname = op.join(subjects_dir, subject, 'bem',
                                         '%s.surf' % this_surf)
                    if not op.exists(surf_fname):
                        continue
                logger.info('Using %s for head surface.' % this_surf)
                rr, tris = read_surface(surf_fname)
                head_surf = dict(rr=rr / 1000., tris=tris, ntri=len(tris),
                                 np=len(rr), coord_frame=FIFF.FIFFV_COORD_MRI)
                complete_surface_info(head_surf, copy=False, verbose=False)
                break
        if head_surf is None:
            raise IOError('No head surface found for subject %s.' % subject)
        surfs['head'] = head_surf

    if mri_fiducials:
        if mri_fiducials is True:
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            if subject is None:
                raise ValueError("Subject needs to be specified to "
                                 "automatically find the fiducials file.")
            mri_fiducials = op.join(subjects_dir, subject, 'bem',
                                    subject + '-fiducials.fif')
        if isinstance(mri_fiducials, string_types):
            mri_fiducials, cf = read_fiducials(mri_fiducials)
            if cf != FIFF.FIFFV_COORD_MRI:
                raise ValueError("Fiducials are not in MRI space")
        fid_loc = _fiducial_coords(mri_fiducials, FIFF.FIFFV_COORD_MRI)
        fid_loc = apply_trans(mri_trans, fid_loc)
    else:
        fid_loc = []

    if 'helmet' in meg_sensors and len(meg_picks) > 0:
        surfs['helmet'] = get_meg_helmet_surf(info, head_mri_t)
    if brain is None:
        if len(ecog_picks) > 0 and subject is not None:
            brain = 'pial'
        else:
            brain = False
    if brain:
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        brain = 'pial' if brain is True else brain
        for hemi in ['lh', 'rh']:
            fname = op.join(subjects_dir, subject, 'surf',
                            '%s.%s' % (hemi, brain))
            rr, tris = read_surface(fname)
            rr *= 1e-3
            surfs[hemi] = dict(rr=rr, tris=tris, ntri=len(tris), np=len(rr),
                               coord_frame=FIFF.FIFFV_COORD_MRI)
            complete_surface_info(surfs[hemi], copy=False, verbose=False)

    if skull is True:
        skull = 'outer_skull'
    if isinstance(skull, string_types):
        skull = [skull]
    elif not skull:
        skull = []
    if len(skull) > 0 and not isinstance(skull[0], dict):
        skull = sorted(skull)
    skull_alpha = dict()
    skull_colors = dict()
    hemi_val = 0.5
    if src is None or (brain and any(s['type'] == 'surf' for s in src)):
        hemi_val = 1.
    alphas = (4 - np.arange(len(skull) + 1)) * (0.5 / 4.)
    for idx, this_skull in enumerate(skull):
        if isinstance(this_skull, dict):
            from ..bem import _surf_name
            skull_surf = this_skull
            this_skull = _surf_name[skull_surf['id']]
        else:
            skull_fname = op.join(subjects_dir, subject, 'bem', 'flash',
                                  '%s.surf' % this_skull)
            if not op.exists(skull_fname):
                skull_fname = op.join(subjects_dir, subject, 'bem',
                                      '%s.surf' % this_skull)
            if not op.exists(skull_fname):
                raise IOError('No skull surface %s found for subject %s.'
                              % (this_skull, subject))
            logger.info('Using %s for head surface.' % skull_fname)
            rr, tris = read_surface(skull_fname)
            skull_surf = dict(rr=rr / 1000., tris=tris, ntri=len(tris),
                              np=len(rr), coord_frame=FIFF.FIFFV_COORD_MRI)
            complete_surface_info(skull_surf, copy=False, verbose=False)
        skull_alpha[this_skull] = alphas[idx + 1]
        skull_colors[this_skull] = (0.95 - idx * 0.2, 0.85, 0.95 - idx * 0.2)
        surfs[this_skull] = skull_surf

    if src is None and brain is False and len(skull) == 0:
        head_alpha = 1.0
    else:
        head_alpha = alphas[0]

    for key in surfs.keys():
        surfs[key] = transform_surface_to(surfs[key], coord_frame, mri_trans)
    src_rr = apply_trans(mri_trans, src_rr)
    src_nn = apply_trans(mri_trans, src_nn, move=False)

    # determine points
    meg_rrs, meg_tris = list(), list()
    ecog_loc = list()
    hpi_loc = list()
    ext_loc = list()
    car_loc = list()
    eeg_loc = list()
    eegp_loc = list()
    if len(eeg_sensors) > 0:
        eeg_loc = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
        if len(eeg_loc) > 0:
            eeg_loc = apply_trans(head_trans, eeg_loc)
            # XXX do projections here if necessary
            if 'projected' in eeg_sensors:
                eegp_loc, eegp_nn = _project_onto_surface(
                    eeg_loc, surfs['head'], project_rrs=True,
                    return_nn=True)[2:4]
            if 'original' not in eeg_sensors:
                eeg_loc = list()
    del eeg_sensors
    if 'sensors' in meg_sensors:
        coil_transs = [_loc_to_coil_trans(info['chs'][pick]['loc'])
                       for pick in meg_picks]
        coils = _create_meg_coils([info['chs'][pick] for pick in meg_picks],
                                  acc='normal')
        offset = 0
        for coil, coil_trans in zip(coils, coil_transs):
            rrs, tris = _sensor_shape(coil)
            rrs = apply_trans(coil_trans, rrs)
            meg_rrs.append(rrs)
            meg_tris.append(tris + offset)
            offset += len(meg_rrs[-1])
        if len(meg_rrs) == 0:
            warn('MEG electrodes not found. Cannot plot MEG locations.')
        else:
            meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
            meg_tris = np.concatenate(meg_tris, axis=0)
    del meg_sensors
    if dig:
        if dig == 'fiducials':
            hpi_loc = ext_loc = []
        elif dig is not True:
            raise ValueError("dig needs to be True, False or 'fiducials', "
                             "not %s" % repr(dig))
        else:
            hpi_loc = np.array([d['r'] for d in info['dig']
                                if d['kind'] == FIFF.FIFFV_POINT_HPI])
            ext_loc = np.array([d['r'] for d in info['dig']
                               if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
        car_loc = _fiducial_coords(info['dig'])
        # Transform from head coords if necessary
        if coord_frame == 'meg':
            for loc in (hpi_loc, ext_loc, car_loc):
                loc[:] = apply_trans(invert_transform(info['dev_head_t']), loc)
        elif coord_frame == 'mri':
            for loc in (hpi_loc, ext_loc, car_loc):
                loc[:] = apply_trans(head_mri_t, loc)
        if len(car_loc) == len(ext_loc) == len(hpi_loc) == 0:
            warn('Digitization points not found. Cannot plot digitization.')
    del dig
    if len(ecog_picks) > 0 and ecog_sensors:
        ecog_loc = np.array([info['chs'][pick]['loc'][:3]
                             for pick in ecog_picks])

    # initialize figure
    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
    _toggle_mlab_render(fig, False)

    # plot surfaces
    alphas = dict(head=head_alpha, helmet=0.5, lh=hemi_val, rh=hemi_val)
    alphas.update(skull_alpha)
    colors = dict(head=(0.6,) * 3, helmet=(0.0, 0.0, 0.6), lh=(0.5,) * 3,
                  rh=(0.5,) * 3)
    colors.update(skull_colors)
    for key, surf in surfs.items():
        # Make a solid surface
        mesh = _create_mesh_surf(surf, fig)
        with warnings.catch_warnings(record=True):  # traits
            surface = mlab.pipeline.surface(mesh, color=colors[key],
                                            opacity=alphas[key], figure=fig)
        if key != 'helmet':
            surface.actor.property.backface_culling = True

    # plot points
    defaults = DEFAULTS['coreg']
    datas = [eeg_loc,
             hpi_loc,
             ext_loc, ecog_loc]
    colors = [defaults['eeg_color'],
              defaults['hpi_color'],
              defaults['extra_color'], defaults['ecog_color']]
    alphas = [0.8,
              0.5,
              0.25, 0.8]
    scales = [defaults['eeg_scale'],
              defaults['hpi_scale'],
              defaults['extra_scale'], defaults['ecog_scale']]
    for kind, loc in (('dig', car_loc), ('mri', fid_loc)):
        if len(loc) > 0:
            datas.extend(loc[:, np.newaxis])
            colors.extend((defaults['lpa_color'],
                           defaults['nasion_color'],
                           defaults['rpa_color']))
            alphas.extend(3 * (defaults[kind + '_fid_opacity'],))
            scales.extend(3 * (defaults[kind + '_fid_scale'],))

    for data, color, alpha, scale in zip(datas, colors, alphas, scales):
        if len(data) > 0:
            with warnings.catch_warnings(record=True):  # traits
                points = mlab.points3d(data[:, 0], data[:, 1], data[:, 2],
                                       color=color, scale_factor=scale,
                                       opacity=alpha, figure=fig)
                points.actor.property.backface_culling = True
    if len(eegp_loc) > 0:
        with warnings.catch_warnings(record=True):  # traits
            quiv = mlab.quiver3d(
                eegp_loc[:, 0], eegp_loc[:, 1], eegp_loc[:, 2],
                eegp_nn[:, 0], eegp_nn[:, 1], eegp_nn[:, 2],
                color=defaults['eegp_color'], mode='cylinder',
                scale_factor=defaults['eegp_scale'], opacity=0.6, figure=fig)
        quiv.glyph.glyph_source.glyph_source.height = defaults['eegp_height']
        quiv.glyph.glyph_source.glyph_source.center = \
            (0., -defaults['eegp_height'], 0)
        quiv.glyph.glyph_source.glyph_source.resolution = 20
        quiv.actor.property.backface_culling = True
    if len(meg_rrs) > 0:
        color, alpha = (0., 0.25, 0.5), 0.25
        surf = dict(rr=meg_rrs, tris=meg_tris)
        complete_surface_info(surf, copy=False, verbose=False)
        mesh = _create_mesh_surf(surf, fig)
        with warnings.catch_warnings(record=True):  # traits
            surface = mlab.pipeline.surface(mesh, color=color,
                                            opacity=alpha, figure=fig)
        # Don't cull these backfaces
    if len(src_rr) > 0:
        with warnings.catch_warnings(record=True):  # traits
            quiv = mlab.quiver3d(
                src_rr[:, 0], src_rr[:, 1], src_rr[:, 2],
                src_nn[:, 0], src_nn[:, 1], src_nn[:, 2], color=(1., 1., 0.),
                mode='cylinder', scale_factor=3e-3, opacity=0.75, figure=fig)
        quiv.glyph.glyph_source.glyph_source.height = 0.25
        quiv.glyph.glyph_source.glyph_source.center = (0., 0., 0.)
        quiv.glyph.glyph_source.glyph_source.resolution = 20
        quiv.actor.property.backface_culling = True
    with SilenceStdout():
        mlab.view(90, 90, figure=fig)
    _toggle_mlab_render(fig, True)
    return fig


@verbose
def plot_alignment(info, trans=None, subject=None, subjects_dir=None,
                   surfaces=('head',), coord_frame='head',
                   meg=('helmet', 'sensors'), eeg='original',
                   dig=False, ecog=True, src=None, mri_fiducials=False,
                   bem=None, verbose=None):
    """Plot head, sensor, and source space alignment in 3D.

    Parameters
    ----------
    info : dict
        The measurement info.
    trans : str | 'auto' | dict | None
        The full path to the head<->MRI transform ``*-trans.fif`` file
        produced during coregistration. If trans is None, an identity matrix
        is assumed.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. Can be omitted if ``src`` is provided.
    subjects_dir : str | None
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    surfaces : str | list
        Surfaces to plot. Supported values: 'head', 'outer_skin',
        'outer_skull', 'inner_skull', 'brain', 'pial', 'white', 'inflated'.
        Defaults to ('head',).

        .. note:: For single layer BEMs it is recommended to use 'brain'.
    coord_frame : str
        Coordinate frame to use, 'head', 'meg', or 'mri'.
    meg : str | list | bool
        Can be "helmet", "sensors" or "ref" to show the MEG helmet, sensors or
        reference sensors respectively, or a combination like
        ``['helmet', 'sensors']``. True translates to
        ``('helmet', 'sensors', 'ref')``.
    eeg : bool | str | list
        Can be "original" (default; equivalent to True) or "projected" to
        show EEG sensors in their digitized locations or projected onto the
        scalp, or a list of these options including ``[]`` (equivalent of
        False).
    dig : bool | 'fiducials'
        If True, plot the digitization points; 'fiducials' to plot fiducial
        points only.
    ecog : bool
        If True (default), show ECoG sensors.
    src : instance of SourceSpaces | None
        If not None, also plot the source space points.
    mri_fiducials : bool | str
        Plot MRI fiducials (default False). If ``True``, look for a file with
        the canonical name (``bem/{subject}-fiducials.fif``). If ``str`` it
        should provide the full path to the fiducials file.
    bem : list of dict | Instance of ConductorModel | None
        Can be either the BEM surfaces (list of dict), a BEM solution or a
        sphere model. If None, we first try loading
        `'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'`, and then look for
        `'$SUBJECT*$SOURCE.fif'` in the same directory. For `'outer_skin'`,
        the subjects bem and bem/flash folders are searched. Defaults to None.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig : instance of mlab.Figure
        The mayavi figure.

    See Also
    --------
    mne.viz.plot_bem

    Notes
    -----
    This function serves the purpose of checking the validity of the many
    different steps of source reconstruction:

    - Transform matrix (keywords ``trans``, ``meg`` and ``mri_fiducials``),
    - BEM surfaces (keywords ``bem`` and ``surfaces``),
    - sphere conductor model (keywords ``bem`` and ``surfaces``) and
    - source space (keywords ``surfaces`` and ``src``).

    .. versionadded:: 0.15
    """
    from ..forward import _create_meg_coils
    mlab = _import_mlab()

    if eeg is False:
        eeg = list()
    elif eeg is True:
        eeg = 'original'
    if meg is True:
        meg = ('helmet', 'sensors', 'ref')
    elif meg is False:
        meg = list()
    elif isinstance(meg, string_types):
        meg = [meg]
    if isinstance(eeg, string_types):
        eeg = [eeg]

    for kind, var in zip(('eeg', 'meg'), (eeg, meg)):
        if not isinstance(var, (list, tuple)) or \
                not all(isinstance(x, string_types) for x in var):
            raise TypeError('%s must be list or tuple of str, got %s'
                            % (kind, type(var)))
    if not all(x in ('helmet', 'sensors', 'ref') for x in meg):
        raise ValueError('meg must only contain "helmet", "sensors" or "ref", '
                         'got %s' % (meg,))
    if not all(x in ('original', 'projected') for x in eeg):
        raise ValueError('eeg must only contain "original" and '
                         '"projected", got %s' % (eeg,))

    if not isinstance(info, Info):
        raise TypeError('info must be an instance of Info, got %s'
                        % type(info))

    is_sphere = False
    if isinstance(bem, ConductorModel) and bem['is_sphere']:
        if len(bem['layers']) != 4 and len(surfaces) > 1:
            raise ValueError('The sphere conductor model must have three '
                             'layers for plotting skull and head.')
        is_sphere = True

    # Skull:
    skull = list()
    if 'outer_skull' in surfaces:
        if isinstance(bem, ConductorModel) and not bem['is_sphere']:
            skull.append(_bem_find_surface(bem, FIFF.FIFFV_BEM_SURF_ID_SKULL))
        else:
            skull.append('outer_skull')
    if 'inner_skull' in surfaces:
        if isinstance(bem, ConductorModel) and not bem['is_sphere']:
            skull.append(_bem_find_surface(bem, FIFF.FIFFV_BEM_SURF_ID_BRAIN))
        else:
            skull.append('inner_skull')

    surf_dict = _surf_dict.copy()
    surf_dict['outer_skin'] = FIFF.FIFFV_BEM_SURF_ID_HEAD
    if len(skull) > 0 and not isinstance(skull[0], dict):  # list of str
        skull = sorted(skull)
        # list of dict
        if bem is not None and not isinstance(bem, ConductorModel):
            for idx, surf_name in enumerate(skull):
                for this_surf in bem:
                    if this_surf['id'] == surf_dict[surf_name]:
                        skull[idx] = this_surf
                        break
                else:
                    raise ValueError('Could not find the surface for '
                                     '%s.' % surf_name)

    valid_coords = ['head', 'meg', 'mri']
    if coord_frame not in valid_coords:
        raise ValueError('coord_frame must be one of %s' % (valid_coords,))
    if src is not None:
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be None or SourceSpaces, got %s'
                            % (type(src),))
        src_subject = src[0].get('subject_his_id', None)
        subject = src_subject if subject is None else subject
        if src_subject is not None and subject != src_subject:
            raise ValueError('subject ("%s") did not match the subject name '
                             ' in src ("%s")' % (subject, src_subject))
        src_rr = np.concatenate([s['rr'][s['inuse'].astype(bool)]
                                 for s in src])
        src_nn = np.concatenate([s['nn'][s['inuse'].astype(bool)]
                                 for s in src])
    else:
        src_rr = src_nn = np.empty((0, 3))

    ref_meg = 'ref' in meg
    meg_picks = pick_types(info, meg=True, ref_meg=ref_meg)
    eeg_picks = pick_types(info, meg=False, eeg=True, ref_meg=False)
    ecog_picks = pick_types(info, meg=False, ecog=True, ref_meg=False)

    if isinstance(trans, string_types):
        if trans == 'auto':
            # let's try to do this in MRI coordinates so they're easy to plot
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            trans = _find_trans(subject, subjects_dir)
        trans = read_trans(trans, return_all=True)
        exp = None
        for trans in trans:  # we got at least 1
            try:
                trans = _ensure_trans(trans, 'head', 'mri')
            except Exception as exp:
                pass
            else:
                break
        else:
            raise exp
    elif trans is None:
        trans = Transform('head', 'mri')
    elif not isinstance(trans, dict):
        raise TypeError('trans must be str, dict, or None')
    head_mri_t = _ensure_trans(trans, 'head', 'mri')
    dev_head_t = info['dev_head_t']
    del trans

    # Figure out our transformations
    if coord_frame == 'meg':
        head_trans = invert_transform(dev_head_t)
        meg_trans = Transform('meg', 'meg')
        mri_trans = invert_transform(combine_transforms(
            dev_head_t, head_mri_t, 'meg', 'mri'))
    elif coord_frame == 'mri':
        head_trans = head_mri_t
        meg_trans = combine_transforms(dev_head_t, head_mri_t, 'meg', 'mri')
        mri_trans = Transform('mri', 'mri')
    else:  # coord_frame == 'head'
        head_trans = Transform('head', 'head')
        meg_trans = info['dev_head_t']
        mri_trans = invert_transform(head_mri_t)

    # both the head and helmet will be in MRI coordinates after this
    surfs = dict()

    # Head:
    head = any(['outer_skin' in surfaces, 'head' in surfaces])
    if head:
        head_surf = None
        if bem is not None:
            if isinstance(bem, ConductorModel):
                if is_sphere:
                    head_surf = _complete_sphere_surf(bem, 3, 4)
                else:  # BEM solution
                    head_surf = _bem_find_surface(bem,
                                                  FIFF.FIFFV_BEM_SURF_ID_HEAD)
                    complete_surface_info(head_surf, copy=False, verbose=False)
            elif bem is not None:  # list of dict
                for this_surf in bem:
                    if this_surf['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD:
                        head_surf = this_surf
                        break
                else:
                    raise ValueError('Could not find the surface for head.')
        if head_surf is None:
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            surf_fname = op.join(subjects_dir, subject, 'bem', 'flash',
                                 'outer_skin.surf')
            if not op.exists(surf_fname):
                surf_fname = op.join(subjects_dir, subject, 'bem',
                                     'outer_skin.surf')
                if not op.exists(surf_fname):
                    raise IOError('No head surface found for subject '
                                  '%s.' % subject)
            logger.info('Using outer_skin for head surface.')
            rr, tris = read_surface(surf_fname)
            head_surf = dict(rr=rr / 1000., tris=tris, ntri=len(tris),
                             np=len(rr), coord_frame=FIFF.FIFFV_COORD_MRI)
            complete_surface_info(head_surf, copy=False, verbose=False)

        surfs['head'] = head_surf

    if mri_fiducials:
        if mri_fiducials is True:
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            if subject is None:
                raise ValueError("Subject needs to be specified to "
                                 "automatically find the fiducials file.")
            mri_fiducials = op.join(subjects_dir, subject, 'bem',
                                    subject + '-fiducials.fif')
        if isinstance(mri_fiducials, string_types):
            mri_fiducials, cf = read_fiducials(mri_fiducials)
            if cf != FIFF.FIFFV_COORD_MRI:
                raise ValueError("Fiducials are not in MRI space")
        fid_loc = _fiducial_coords(mri_fiducials, FIFF.FIFFV_COORD_MRI)
        fid_loc = apply_trans(mri_trans, fid_loc)
    else:
        fid_loc = []

    if 'helmet' in meg and len(meg_picks) > 0:
        surfs['helmet'] = get_meg_helmet_surf(info, head_mri_t)

    # Brain:
    brain = False
    brain_surfs = np.intersect1d(surfaces, ['brain', 'pial', 'white',
                                            'inflated'])
    if len(brain_surfs) > 1:
        raise ValueError('Only one brain surface can be plotted. '
                         'Got %s.' % brain_surfs)
    elif len(brain_surfs) == 1:
        if brain_surfs[0] == 'brain':
            brain = 'pial'
        else:
            brain = brain_surfs[0]

    if brain:
        if is_sphere:
            if len(bem['layers']) > 0:
                surfs['lh'] = _complete_sphere_surf(bem, 0, 4)  # only plot 1
        else:
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            for hemi in ['lh', 'rh']:
                fname = op.join(subjects_dir, subject, 'surf',
                                '%s.%s' % (hemi, brain))
                rr, tris = read_surface(fname)
                rr *= 1e-3
                surfs[hemi] = dict(rr=rr, ntri=len(tris), np=len(rr),
                                   coord_frame=FIFF.FIFFV_COORD_MRI, tris=tris)
                complete_surface_info(surfs[hemi], copy=False, verbose=False)

    skull_alpha = dict()
    skull_colors = dict()
    hemi_val = 0.5
    if src is None or (brain and any(s['type'] == 'surf' for s in src)):
        hemi_val = 1.
    alphas = (4 - np.arange(len(skull) + 1)) * (0.5 / 4.)
    for idx, this_skull in enumerate(skull):
        if isinstance(this_skull, dict):
            skull_surf = this_skull
            this_skull = _surf_name[skull_surf['id']]
        elif is_sphere:  # this_skull == str
            this_idx = 1 if this_skull == 'inner_skull' else 2
            skull_surf = _complete_sphere_surf(bem, this_idx, 4)
        else:  # str
            skull_fname = op.join(subjects_dir, subject, 'bem', 'flash',
                                  '%s.surf' % this_skull)
            if not op.exists(skull_fname):
                skull_fname = op.join(subjects_dir, subject, 'bem',
                                      '%s.surf' % this_skull)
            if not op.exists(skull_fname):
                raise IOError('No skull surface %s found for subject %s.'
                              % (this_skull, subject))
            logger.info('Using %s for head surface.' % skull_fname)
            rr, tris = read_surface(skull_fname)
            skull_surf = dict(rr=rr / 1000., tris=tris, ntri=len(tris),
                              np=len(rr), coord_frame=FIFF.FIFFV_COORD_MRI)
            complete_surface_info(skull_surf, copy=False, verbose=False)
        skull_alpha[this_skull] = alphas[idx + 1]
        skull_colors[this_skull] = (0.95 - idx * 0.2, 0.85, 0.95 - idx * 0.2)
        surfs[this_skull] = skull_surf

    if src is None and brain is False and len(skull) == 0:
        head_alpha = 1.0
    else:
        head_alpha = alphas[0]

    for key in surfs.keys():
        surfs[key] = transform_surface_to(surfs[key], coord_frame, mri_trans)
    if src is not None:
        if src[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI:
            src_rr = apply_trans(mri_trans, src_rr)
            src_nn = apply_trans(mri_trans, src_nn, move=False)
        elif src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            src_rr = apply_trans(head_trans, src_rr)
            src_nn = apply_trans(head_trans, src_nn, move=False)

    # determine points
    meg_rrs, meg_tris = list(), list()
    ecog_loc = list()
    hpi_loc = list()
    ext_loc = list()
    car_loc = list()
    eeg_loc = list()
    eegp_loc = list()
    if len(eeg) > 0:
        eeg_loc = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
        if len(eeg_loc) > 0:
            eeg_loc = apply_trans(head_trans, eeg_loc)
            # XXX do projections here if necessary
            if 'projected' in eeg:
                eegp_loc, eegp_nn = _project_onto_surface(
                    eeg_loc, surfs['head'], project_rrs=True,
                    return_nn=True)[2:4]
            if 'original' not in eeg:
                eeg_loc = list()
    del eeg
    if 'sensors' in meg:
        coil_transs = [_loc_to_coil_trans(info['chs'][pick]['loc'])
                       for pick in meg_picks]
        coils = _create_meg_coils([info['chs'][pick] for pick in meg_picks],
                                  acc='normal')
        offset = 0
        for coil, coil_trans in zip(coils, coil_transs):
            rrs, tris = _sensor_shape(coil)
            rrs = apply_trans(coil_trans, rrs)
            meg_rrs.append(rrs)
            meg_tris.append(tris + offset)
            offset += len(meg_rrs[-1])
        if len(meg_rrs) == 0:
            warn('MEG electrodes not found. Cannot plot MEG locations.')
        else:
            meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
            meg_tris = np.concatenate(meg_tris, axis=0)
    del meg
    if dig:
        if dig == 'fiducials':
            hpi_loc = ext_loc = []
        elif dig is not True:
            raise ValueError("dig needs to be True, False or 'fiducials', "
                             "not %s" % repr(dig))
        else:
            hpi_loc = np.array([d['r'] for d in info['dig']
                                if d['kind'] == FIFF.FIFFV_POINT_HPI])
            ext_loc = np.array([d['r'] for d in info['dig']
                               if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
        car_loc = _fiducial_coords(info['dig'])
        # Transform from head coords if necessary
        if coord_frame == 'meg':
            for loc in (hpi_loc, ext_loc, car_loc):
                loc[:] = apply_trans(invert_transform(info['dev_head_t']), loc)
        elif coord_frame == 'mri':
            for loc in (hpi_loc, ext_loc, car_loc):
                loc[:] = apply_trans(head_mri_t, loc)
        if len(car_loc) == len(ext_loc) == len(hpi_loc) == 0:
            warn('Digitization points not found. Cannot plot digitization.')
    del dig
    if len(ecog_picks) > 0 and ecog:
        ecog_loc = np.array([info['chs'][pick]['loc'][:3]
                             for pick in ecog_picks])

    # initialize figure
    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
    _toggle_mlab_render(fig, False)

    # plot surfaces
    alphas = dict(head=head_alpha, helmet=0.5, lh=hemi_val, rh=hemi_val)
    alphas.update(skull_alpha)
    colors = dict(head=(0.6,) * 3, helmet=(0.0, 0.0, 0.6), lh=(0.5,) * 3,
                  rh=(0.5,) * 3)
    colors.update(skull_colors)
    for key, surf in surfs.items():
        # Make a solid surface
        mesh = _create_mesh_surf(surf, fig)
        with warnings.catch_warnings(record=True):  # traits
            surface = mlab.pipeline.surface(mesh, color=colors[key],
                                            opacity=alphas[key], figure=fig)
        if key != 'helmet':
            surface.actor.property.backface_culling = True
    if brain and 'lh' not in surfs:  # one layer sphere
        assert bem['coord_frame'] == FIFF.FIFFV_COORD_HEAD
        center = bem['r0'].copy()
        center = apply_trans(head_trans, center)
        mlab.points3d(*center, scale_factor=0.01, color=colors['lh'],
                      opacity=alphas['lh'])

    # plot points
    defaults = DEFAULTS['coreg']
    datas = [eeg_loc,
             hpi_loc,
             ext_loc, ecog_loc]
    colors = [defaults['eeg_color'],
              defaults['hpi_color'],
              defaults['extra_color'], defaults['ecog_color']]
    alphas = [0.8,
              0.5,
              0.25, 0.8]
    scales = [defaults['eeg_scale'],
              defaults['hpi_scale'],
              defaults['extra_scale'], defaults['ecog_scale']]
    for kind, loc in (('dig', car_loc), ('mri', fid_loc)):
        if len(loc) > 0:
            datas.extend(loc[:, np.newaxis])
            colors.extend((defaults['lpa_color'],
                           defaults['nasion_color'],
                           defaults['rpa_color']))
            alphas.extend(3 * (defaults[kind + '_fid_opacity'],))
            scales.extend(3 * (defaults[kind + '_fid_scale'],))

    for data, color, alpha, scale in zip(datas, colors, alphas, scales):
        if len(data) > 0:
            with warnings.catch_warnings(record=True):  # traits
                points = mlab.points3d(data[:, 0], data[:, 1], data[:, 2],
                                       color=color, scale_factor=scale,
                                       opacity=alpha, figure=fig)
                points.actor.property.backface_culling = True
    if len(eegp_loc) > 0:
        with warnings.catch_warnings(record=True):  # traits
            quiv = mlab.quiver3d(
                eegp_loc[:, 0], eegp_loc[:, 1], eegp_loc[:, 2],
                eegp_nn[:, 0], eegp_nn[:, 1], eegp_nn[:, 2],
                color=defaults['eegp_color'], mode='cylinder',
                scale_factor=defaults['eegp_scale'], opacity=0.6, figure=fig)
        quiv.glyph.glyph_source.glyph_source.height = defaults['eegp_height']
        quiv.glyph.glyph_source.glyph_source.center = \
            (0., -defaults['eegp_height'], 0)
        quiv.glyph.glyph_source.glyph_source.resolution = 20
        quiv.actor.property.backface_culling = True
    if len(meg_rrs) > 0:
        color, alpha = (0., 0.25, 0.5), 0.25
        surf = dict(rr=meg_rrs, tris=meg_tris)
        complete_surface_info(surf, copy=False, verbose=False)
        mesh = _create_mesh_surf(surf, fig)
        with warnings.catch_warnings(record=True):  # traits
            surface = mlab.pipeline.surface(mesh, color=color,
                                            opacity=alpha, figure=fig)
        # Don't cull these backfaces
    if len(src_rr) > 0:
        with warnings.catch_warnings(record=True):  # traits
            quiv = mlab.quiver3d(
                src_rr[:, 0], src_rr[:, 1], src_rr[:, 2],
                src_nn[:, 0], src_nn[:, 1], src_nn[:, 2], color=(1., 1., 0.),
                mode='cylinder', scale_factor=3e-3, opacity=0.75, figure=fig)
        quiv.glyph.glyph_source.glyph_source.height = 0.25
        quiv.glyph.glyph_source.glyph_source.center = (0., 0., 0.)
        quiv.glyph.glyph_source.glyph_source.resolution = 20
        quiv.actor.property.backface_culling = True
    with SilenceStdout():
        mlab.view(90, 90, figure=fig)
    _toggle_mlab_render(fig, True)
    return fig


def _make_tris_fan(n_vert):
    """Make tris given a number of vertices of a circle-like obj."""
    tris = np.zeros((n_vert - 2, 3), int)
    tris[:, 2] = np.arange(2, n_vert)
    tris[:, 1] = tris[:, 2] - 1
    return tris


def _sensor_shape(coil):
    """Get the sensor shape vertices."""
    rrs = np.empty([0, 2])
    tris = np.empty([0, 3], int)
    id_ = coil['type'] & 0xFFFF
    if id_ in (2, 3012, 3013, 3011):
        # square figure eight
        # wound by right hand rule such that +x side is "up" (+z)
        long_side = coil['size']  # length of long side (meters)
        offset = 0.0025  # offset of the center portion of planar grad coil
        rrs = np.array([
            [offset, -long_side / 2.],
            [long_side / 2., -long_side / 2.],
            [long_side / 2., long_side / 2.],
            [offset, long_side / 2.],
            [-offset, -long_side / 2.],
            [-long_side / 2., -long_side / 2.],
            [-long_side / 2., long_side / 2.],
            [-offset, long_side / 2.]])
        tris = np.concatenate((_make_tris_fan(4),
                               _make_tris_fan(4) + 4), axis=0)
    elif id_ in (2000, 3022, 3023, 3024):
        # square magnetometer (potentially point-type)
        size = 0.001 if id_ == 2000 else (coil['size'] / 2.)
        rrs = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) * size
        tris = _make_tris_fan(4)
    elif id_ in (4001, 4003, 5002, 7002, 7003,
                 FIFF.FIFFV_COIL_ARTEMIS123_REF_MAG):
        # round magnetometer
        n_pts = 15  # number of points for circle
        circle = np.exp(2j * np.pi * np.arange(n_pts) / float(n_pts))
        circle = np.concatenate(([0.], circle))
        circle *= coil['size'] / 2.  # radius of coil
        rrs = np.array([circle.real, circle.imag]).T
        tris = _make_tris_fan(n_pts + 1)
    elif id_ in (4002, 5001, 5003, 5004, 4004, 4005, 6001, 7001,
                 FIFF.FIFFV_COIL_ARTEMIS123_GRAD,
                 FIFF.FIFFV_COIL_ARTEMIS123_REF_GRAD):
        # round coil 1st order (off-diagonal) gradiometer
        baseline = coil['base'] if id_ in (5004, 4005) else 0.
        n_pts = 16  # number of points for circle
        # This time, go all the way around circle to close it fully
        circle = np.exp(2j * np.pi * np.arange(-1, n_pts) / float(n_pts - 1))
        circle[0] = 0  # center pt for triangulation
        circle *= coil['size'] / 2.
        rrs = np.array([  # first, second coil
            np.concatenate([circle.real + baseline / 2.,
                            circle.real - baseline / 2.]),
            np.concatenate([circle.imag, -circle.imag])]).T
        tris = np.concatenate([_make_tris_fan(n_pts + 1),
                               _make_tris_fan(n_pts + 1) + n_pts + 1])
    # Go from (x,y) -> (x,y,z)
    rrs = np.pad(rrs, ((0, 0), (0, 1)), mode='constant')
    return rrs, tris


def _limits_to_control_points(clim, stc_data, colormap):
    """Convert limits (values or percentiles) to control points.

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
                warn('All data were zero')
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


def _handle_time(time_label, time_unit, times):
    """Handle time label string and units."""
    if time_label == 'auto':
        if time_unit == 's':
            time_label = 'time=%0.3fs'
        elif time_unit == 'ms':
            time_label = 'time=%0.1fms'
    if time_unit == 's':
        times = times
    elif time_unit == 'ms':
        times = 1e3 * times
    else:
        raise ValueError("time_unit needs to be 's' or 'ms', got %r" %
                         (time_unit,))

    return time_label, times


def _key_pressed_slider(event, params):
    """Handle key presses for time_viewer slider."""
    step = 1
    if event.key.startswith('ctrl'):
        step = 5
        event.key = event.key.split('+')[-1]
    if event.key not in ['left', 'right']:
        return
    time_viewer = event.canvas.figure
    value = time_viewer.slider.val
    times = params['stc'].times
    if params['time_unit'] == 'ms':
        times = times * 1000.
    time_idx = np.argmin(np.abs(times - value))
    if event.key == 'left':
        time_idx = np.max((0, time_idx - step))
    elif event.key == 'right':
        time_idx = np.min((len(times) - 1, time_idx + step))
    this_time = times[time_idx]
    time_viewer.slider.set_val(this_time)


def _smooth_plot(this_time, params):
    """Smooth source estimate data and plot with mpl."""
    from ..source_estimate import _morph_buffer
    from mpl_toolkits.mplot3d import art3d
    ax = params['ax']
    stc = params['stc']
    ax.clear()
    times = stc.times
    scaler = 1000. if params['time_unit'] == 'ms' else 1.
    if this_time is None:
        time_idx = 0
    else:
        time_idx = np.argmin(np.abs(times - this_time / scaler))

    if params['hemi_idx'] == 0:
        data = stc.data[:len(stc.vertices[0]), time_idx:time_idx + 1]
    else:
        data = stc.data[len(stc.vertices[0]):, time_idx:time_idx + 1]

    array_plot = _morph_buffer(data, params['vertices'], params['e'],
                               params['smoothing_steps'], params['n_verts'],
                               params['inuse'], params['maps'])

    vmax = np.max(array_plot)
    colors = array_plot / vmax

    transp = 0.8
    faces = params['faces']
    greymap = params['greymap']
    cmap = params['cmap']
    polyc = ax.plot_trisurf(*params['coords'].T, triangles=faces,
                            antialiased=False)
    color_ave = np.mean(colors[faces], axis=1).flatten()
    curv_ave = np.mean(params['curv'][faces], axis=1).flatten()
    facecolors = art3d.PolyCollection.get_facecolors(polyc)

    to_blend = color_ave > params['ctrl_pts'][0] / vmax
    facecolors[to_blend, :3] = ((1 - transp) *
                                greymap(curv_ave[to_blend])[:, :3] +
                                transp * cmap(color_ave[to_blend])[:, :3])
    facecolors[~to_blend, :3] = greymap(curv_ave[~to_blend])[:, :3]
    ax.set_title(params['time_label'] % (times[time_idx] * scaler), color='w')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_zlim(-80, 80)
    ax.figure.canvas.draw()


def _plot_mpl_stc(stc, subject=None, surface='inflated', hemi='lh',
                  colormap='auto', time_label='auto', smoothing_steps=10,
                  subjects_dir=None, views='lat', clim='auto', figure=None,
                  initial_time=None, time_unit='s', background='black',
                  spacing='oct6', time_viewer=False):
    """Plot source estimate using mpl."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.widgets import Slider
    import nibabel as nib
    from scipy import sparse, stats
    from ..source_estimate import _get_subject_sphere_tris
    if hemi not in ['lh', 'rh']:
        raise ValueError("hemi must be 'lh' or 'rh' when using matplotlib. "
                         "Got %s." % hemi)
    kwargs = {'lat': {'elev': 5, 'azim': 0},
              'med': {'elev': 5, 'azim': 180},
              'fos': {'elev': 5, 'azim': 90},
              'cau': {'elev': 5, 'azim': -90},
              'dor': {'elev': 90, 'azim': 0},
              'ven': {'elev': -90, 'azim': 0},
              'fro': {'elev': 5, 'azim': 110},
              'par': {'elev': 5, 'azim': -110}}
    if views not in kwargs:
        raise ValueError("views must be one of ['lat', 'med', 'fos', 'cau', "
                         "'dor' 'ven', 'fro', 'par']. Got %s." % views)
    ctrl_pts, colormap = _limits_to_control_points(clim, stc.data, colormap)
    if colormap == 'auto':
        colormap = mne_analyze_colormap(clim, format='matplotlib')

    time_label, times = _handle_time(time_label, time_unit, stc.times)
    fig = plt.figure(figsize=(6, 6)) if figure is None else figure
    ax = Axes3D(fig)
    hemi_idx = 0 if hemi == 'lh' else 1
    surf = op.join(subjects_dir, subject, 'surf', '%s.%s' % (hemi, surface))
    if spacing == 'all':
        coords, faces = nib.freesurfer.read_geometry(surf)
        inuse = slice(None)
    else:
        stype, sval, ico_surf, src_type_str = _check_spacing(spacing)
        surf = _create_surf_spacing(surf, hemi, subject, stype, ico_surf,
                                    subjects_dir)
        inuse = surf['vertno']
        faces = surf['use_tris']
        coords = surf['rr'][inuse]
        shape = faces.shape
        faces = stats.rankdata(faces, 'dense').reshape(shape) - 1
        faces = np.round(faces).astype(int)  # should really be int-like anyway
    del surf
    vertices = stc.vertices[hemi_idx]
    n_verts = len(vertices)
    tris = _get_subject_sphere_tris(subject, subjects_dir)[hemi_idx]
    e = mesh_edges(tris)
    e.data[e.data == 2] = 1
    n_vertices = e.shape[0]
    maps = sparse.identity(n_vertices).tocsr()
    e = e + sparse.eye(n_vertices, n_vertices)
    cmap = cm.get_cmap(colormap)
    greymap = cm.get_cmap('Greys')

    curv = nib.freesurfer.read_morph_data(
        op.join(subjects_dir, subject, 'surf', '%s.curv' % hemi))[inuse]
    curv = np.clip(np.array(curv > 0, np.int), 0.2, 0.8)
    params = dict(ax=ax, stc=stc, coords=coords, faces=faces,
                  hemi_idx=hemi_idx, vertices=vertices, e=e,
                  smoothing_steps=smoothing_steps, n_verts=n_verts,
                  inuse=inuse, maps=maps, cmap=cmap, curv=curv,
                  ctrl_pts=ctrl_pts, greymap=greymap, time_label=time_label,
                  time_unit=time_unit)
    _smooth_plot(initial_time, params)

    ax.view_init(**kwargs[views])

    try:
        ax.set_facecolor(background)
    except AttributeError:
        ax.set_axis_bgcolor(background)

    if time_viewer:
        time_viewer = figure_nobar(figsize=(4.5, .25))
        fig.time_viewer = time_viewer
        ax_time = plt.axes()
        if initial_time is None:
            initial_time = 0
        slider = Slider(ax=ax_time, label='Time', valmin=times[0],
                        valmax=times[-1], valinit=initial_time,
                        valfmt=time_label)
        time_viewer.slider = slider
        callback_slider = partial(_smooth_plot, params=params)
        slider.on_changed(callback_slider)
        callback_key = partial(_key_pressed_slider, params=params)
        time_viewer.canvas.mpl_connect('key_press_event', callback_key)

        time_viewer.subplots_adjust(left=0.12, bottom=0.05, right=0.75,
                                    top=0.95)
    fig.subplots_adjust(left=0., bottom=0., right=1., top=1.)
    plt.show()
    return fig


def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='auto', time_label='auto',
                          smoothing_steps=10, transparent=None, alpha=1.0,
                          time_viewer=False, subjects_dir=None, figure=None,
                          views='lat', colorbar=True, clim='auto',
                          cortex="classic", size=800, background="black",
                          foreground="white", initial_time=None,
                          time_unit='s', backend='auto', spacing='oct6'):
    """Plot SourceEstimates with PySurfer.

    By default this function uses :mod:`mayavi.mlab` to plot the source
    estimates. If Mayavi is not installed, the plotting is done with
    :mod:`matplotlib.pyplot` (much slower, decimated source space by default).

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
    time_label : str | callable | None
        Format of the time label (a format string, a function that maps
        floating point time values to strings, or None for no label). The
        default is ``time=%0.2f ms``.
    smoothing_steps : int
        The amount of smoothing
    transparent : bool | None
        If True, use a linear transparency between fmin and fmid.
        None will choose automatically based on colormap type. Has no effect
        with mpl backend.
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    time_viewer : bool
        Display time viewer GUI.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    figure : instance of mayavi.core.scene.Scene | instance of matplotlib.figure.Figure | list | int | None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the Mayavi
        figure by it's id or create a new figure with the given id. If an
        instance of matplotlib figure, mpl backend is used for plotting.
    views : str | list
        View to use. See surfer.Brain(). Supported views: ['lat', 'med', 'fos',
        'cau', 'dor' 'ven', 'fro', 'par']. Using multiple views is not
        supported for mpl backend.
    colorbar : bool
        If True, display colorbar on scene. Not available on mpl backend.
    clim : str | dict
        Colorbar properties specification. If 'auto', set clim automatically
        based on data percentiles. If dict, should contain:

            ``kind`` : 'value' | 'percent'
                Flag to specify type of limits.
            ``lims`` : list | np.ndarray | tuple of float, 3 elements
                Note: Only use this if 'colormap' is not 'mne'.
                Left, middle, and right bound for colormap.
            ``pos_lims`` : list | np.ndarray | tuple of float, 3 elements
                Note: Only use this if 'colormap' is 'mne'.
                Left, middle, and right bound for colormap. Positive values
                will be mirrored directly across zero during colormap
                construction to obtain negative control points.

    cortex : str or tuple
        Specifies how binarized curvature values are rendered.
        Either the name of a preset PySurfer cortex colorscheme (one of
        'classic', 'bone', 'low_contrast', or 'high_contrast'), or the name of
        mayavi colormap, or a tuple with values (colormap, min, max, reverse)
        to fully specify the curvature colors. Has no effect with mpl backend.
    size : float or pair of floats
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
        Has no effect with mpl backend.
    background : matplotlib color
        Color of the background of the display window.
    foreground : matplotlib color
        Color of the foreground of the display window. Has no effect with mpl
        backend.
    initial_time : float | None
        The time to display on the plot initially. ``None`` to display the
        first time sample (default).
    time_unit : 's' | 'ms'
        Whether time is represented in seconds ("s", default) or
        milliseconds ("ms").
    backend : 'auto' | 'mayavi' | 'matplotlib'
        Which backend to use. If ``'auto'`` (default), tries to plot with
        mayavi, but resorts to matplotlib if mayavi is not available.

        .. versionadded:: 0.15.0

    spacing : str
        The spacing to use for the source space. Can be ``'ico#'`` for a
        recursively subdivided icosahedron, ``'oct#'`` for a recursively
        subdivided octahedron, or ``'all'`` for all points. In general, you can
        speed up the plotting by selecting a sparser source space. Has no
        effect with mayavi backend. Defaults  to 'oct6'.

        .. versionadded:: 0.15.0

    Returns
    -------
    figure : surfer.viz.Brain | matplotlib.figure.Figure
        An instance of :class:`surfer.Brain` from PySurfer or
        matplotlib figure.
    """  # noqa: E501
    # import here to avoid circular import problem
    from ..source_estimate import SourceEstimate
    if not isinstance(stc, SourceEstimate):
        raise ValueError('stc has to be a surface source estimate')
    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    subject = _check_subject(stc.subject, subject, True)
    if backend not in ['auto', 'matplotlib', 'mayavi']:
        raise ValueError("backend must be 'auto', 'mayavi' or 'matplotlib'. "
                         "Got %s." % backend)
    plot_mpl = backend == 'matplotlib'
    if not plot_mpl:
        try:
            import mayavi
        except ImportError:
            if backend == 'auto':
                warn('Mayavi not found. Resorting to matplotlib 3d.')
                plot_mpl = True
            else:  # 'mayavi'
                raise

    if plot_mpl:
        return _plot_mpl_stc(stc, subject=subject, surface=surface, hemi=hemi,
                             colormap=colormap, time_label=time_label,
                             smoothing_steps=smoothing_steps,
                             subjects_dir=subjects_dir, views=views, clim=clim,
                             figure=figure, initial_time=initial_time,
                             time_unit=time_unit, background=background,
                             spacing=spacing, time_viewer=time_viewer)
    from surfer import Brain, TimeViewer
    initial_time, ad_kwargs, sd_kwargs = _get_ps_kwargs(initial_time)

    if hemi not in ['lh', 'rh', 'split', 'both']:
        raise ValueError('hemi has to be either "lh", "rh", "split", '
                         'or "both"')

    # check `figure` parameter (This will be performed by PySurfer > 0.6)
    if figure is not None:
        if isinstance(figure, int):
            # use figure with specified id
            size_ = size if isinstance(size, (tuple, list)) else (size, size)
            figure = [mayavi.mlab.figure(figure, size=size_)]
        elif not isinstance(figure, (list, tuple)):
            figure = [figure]
        if not all(isinstance(f, mayavi.core.scene.Scene) for f in figure):
            raise TypeError('figure must be a mayavi scene or list of scenes')

    time_label, times = _handle_time(time_label, time_unit, stc.times)
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

    if hemi in ['both', 'split']:
        hemis = ['lh', 'rh']
    else:
        hemis = [hemi]

    title = subject if len(hemis) > 1 else '%s - %s' % (subject, hemis[0])
    with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(subject, hemi=hemi, surf=surface, curv=True,
                      title=title, cortex=cortex, size=size,
                      background=background, foreground=foreground,
                      figure=figure, subjects_dir=subjects_dir,
                      views=views)

    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1
        if hemi_idx == 0:
            data = stc.data[:len(stc.vertices[0])]
        else:
            data = stc.data[len(stc.vertices[0]):]
        vertices = stc.vertices[hemi_idx]
        if len(data) > 0:
            with warnings.catch_warnings(record=True):  # traits warnings
                brain.add_data(data, colormap=colormap, vertices=vertices,
                               smoothing_steps=smoothing_steps, time=times,
                               time_label=time_label, alpha=alpha, hemi=hemi,
                               colorbar=colorbar, min=0, max=1, **ad_kwargs)

        # scale colormap and set time (index) to display
        brain.scale_data_colormap(fmin=scale_pts[0], fmid=scale_pts[1],
                                  fmax=scale_pts[2], transparent=transparent,
                                  **sd_kwargs)

    if initial_time is not None:
        brain.set_time(initial_time)
    if time_viewer:
        TimeViewer(brain)
    return brain


def _get_ps_kwargs(initial_time, require='0.6'):
    """Triage arguments based on PySurfer version."""
    import surfer
    surfer_version = LooseVersion(surfer.__version__)
    if surfer_version < LooseVersion(require):
        raise ImportError("This function requires PySurfer %s (you are "
                          "running version %s). You can update PySurfer "
                          "using:\n\n    $ pip install -U pysurfer" %
                          (require, surfer.__version__))

    ad_kwargs = dict()
    sd_kwargs = dict()
    if initial_time is not None and surfer_version >= LooseVersion('0.7'):
        ad_kwargs['initial_time'] = initial_time
        initial_time = None  # don't set it twice
    if surfer_version >= LooseVersion('0.8'):
        ad_kwargs['verbose'] = False
        sd_kwargs['verbose'] = False
    return initial_time, ad_kwargs, sd_kwargs


def plot_vector_source_estimates(stc, subject=None, hemi='lh', colormap='hot',
                                 time_label='auto', smoothing_steps=10,
                                 transparent=None, brain_alpha=0.4,
                                 overlay_alpha=None, vector_alpha=1.0,
                                 scale_factor=None, time_viewer=False,
                                 subjects_dir=None, figure=None, views='lat',
                                 colorbar=True, clim='auto', cortex='classic',
                                 size=800, background='black',
                                 foreground='white', initial_time=None,
                                 time_unit='s'):
    """Plot VectorSourceEstimates with PySurfer.

    A "glass brain" is drawn and all dipoles defined in the source estimate
    are shown using arrows, depicting the direction and magnitude of the
    current moment at the dipole. Additionally, an overlay is plotted on top of
    the cortex with the magnitude of the current.

    Parameters
    ----------
    stc : VectorSourceEstimate
        The vector source estimate to plot.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None stc.subject will be used. If that
        is None, the environment will be used.
    hemi : str, 'lh' | 'rh' | 'split' | 'both'
        The hemisphere to display.
    colormap : str | np.ndarray of float, shape(n_colors, 3 | 4)
        Name of colormap to use or a custom look up table. If array, must
        be (n x 3) or (n x 4) array for with RGB or RGBA values between
        0 and 255. Defaults to 'hot'.
    time_label : str | callable | None
        Format of the time label (a format string, a function that maps
        floating point time values to strings, or None for no label). The
        default is ``time=%0.2f ms``.
    smoothing_steps : int
        The amount of smoothing
    transparent : bool | None
        If True, use a linear transparency between fmin and fmid.
        None will choose automatically based on colormap type.
    brain_alpha : float
        Alpha value to apply globally to the surface meshes. Defaults to 0.4.
    overlay_alpha : float
        Alpha value to apply globally to the overlay. Defaults to
        ``brain_alpha``.
    vector_alpha : float
        Alpha value to apply globally to the vector glyphs. Defaults to 1.
    scale_factor : float | None
        Scaling factor for the vector glyphs. By default, an attempt is made to
        automatically determine a sane value.
    time_viewer : bool
        Display time viewer GUI.
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

            ``kind`` : 'value' | 'percent'
                Flag to specify type of limits.
            ``lims`` : list | np.ndarray | tuple of float, 3 elements
                Note: Only use this if 'colormap' is not 'mne'.
                Left, middle, and right bound for colormap.
            ``pos_lims`` : list | np.ndarray | tuple of float, 3 elements
                Note: Only use this if 'colormap' is 'mne'.
                Left, middle, and right bound for colormap. Positive values
                will be mirrored directly across zero during colormap
                construction to obtain negative control points.

    cortex : str or tuple
        specifies how binarized curvature values are rendered.
        either the name of a preset PySurfer cortex colorscheme (one of
        'classic', 'bone', 'low_contrast', or 'high_contrast'), or the
        name of mayavi colormap, or a tuple with values (colormap, min,
        max, reverse) to fully specify the curvature colors.
    size : float or pair of floats
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
    background : matplotlib color
        Color of the background of the display window.
    foreground : matplotlib color
        Color of the foreground of the display window.
    initial_time : float | None
        The time to display on the plot initially. ``None`` to display the
        first time sample (default).
    time_unit : 's' | 'ms'
        Whether time is represented in seconds ("s", default) or
        milliseconds ("ms").

    Returns
    -------
    brain : Brain
        A instance of :class:`surfer.Brain` from PySurfer.

    Notes
    -----
    .. versionadded:: 0.15

    If the current magnitude overlay is not desired, set ``overlay_alpha=0``
    and ``smoothing_steps=1``.
    """
    # Import here to avoid circular imports
    from surfer import Brain, TimeViewer
    from ..source_estimate import VectorSourceEstimate

    if not isinstance(stc, VectorSourceEstimate):
        raise ValueError('stc has to be a vector source estimate')

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    subject = _check_subject(stc.subject, subject, True)
    initial_time, ad_kwargs, sd_kwargs = _get_ps_kwargs(initial_time, '0.8')

    if hemi not in ['lh', 'rh', 'split', 'both']:
        raise ValueError('hemi has to be either "lh", "rh", "split", '
                         'or "both"')

    time_label, times = _handle_time(time_label, time_unit, stc.times)

    # convert control points to locations in colormap
    scale_pts, colormap = _limits_to_control_points(clim, stc.data, colormap)
    transparent = True if transparent is None else transparent

    if hemi in ['both', 'split']:
        hemis = ['lh', 'rh']
    else:
        hemis = [hemi]

    if overlay_alpha is None:
        overlay_alpha = brain_alpha
    if overlay_alpha == 0:
        smoothing_steps = 1  # Disable smoothing to save time.

    title = subject if len(hemis) > 1 else '%s - %s' % (subject, hemis[0])
    with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(subject, hemi=hemi, surf='white', curv=True,
                      title=title, cortex=cortex, size=size,
                      background=background, foreground=foreground,
                      figure=figure, subjects_dir=subjects_dir,
                      views=views, alpha=brain_alpha)

    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1
        if hemi_idx == 0:
            data = stc.data[:len(stc.vertices[0])]
        else:
            data = stc.data[len(stc.vertices[0]):]
        vertices = stc.vertices[hemi_idx]
        if len(data) > 0:
            with warnings.catch_warnings(record=True):  # traits warnings
                brain.add_data(data, colormap=colormap, vertices=vertices,
                               smoothing_steps=smoothing_steps, time=times,
                               time_label=time_label, alpha=overlay_alpha,
                               hemi=hemi, colorbar=colorbar,
                               vector_alpha=vector_alpha,
                               scale_factor=scale_factor, **ad_kwargs)

        # scale colormap and set time (index) to display
        brain.scale_data_colormap(fmin=scale_pts[0], fmid=scale_pts[1],
                                  fmax=scale_pts[2], transparent=transparent,
                                  **sd_kwargs)

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
    """Plot source estimates obtained with sparse solver.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    **kwargs : kwargs
        Keyword arguments to pass to mlab.triangular_mesh.

    Returns
    -------
    surface : instance of mlab Surface
        The triangular mesh surface.
    """
    mlab = _import_mlab()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ColorConverter

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

    color_converter = ColorConverter()

    f = mlab.figure(figure=fig_name, bgcolor=bgcolor, size=(600, 600))
    mlab.clf()
    _toggle_mlab_render(f, False)
    with warnings.catch_warnings(record=True):  # traits warnings
        surface = mlab.triangular_mesh(points[:, 0], points[:, 1],
                                       points[:, 2], use_faces,
                                       color=brain_color,
                                       opacity=opacity, **kwargs)
    surface.actor.property.backface_culling = True

    # Show time courses
    fig = plt.figure(fig_number)
    fig.clf()
    ax = fig.add_subplot(111)

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
            ax.plot(1e3 * stcs[k].times, 1e9 * stcs[k].data[mask].ravel(),
                    c=c, linewidth=linewidth, linestyle=linestyle)

    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_ylabel('Source amplitude (nAm)', fontsize=18)

    if fig_name is not None:
        ax.set_title(fig_name)
    plt_show(show)

    surface.actor.property.backface_culling = True
    surface.actor.property.shading = True
    _toggle_mlab_render(f, True)
    return surface


def _toggle_mlab_render(fig, render):
    mlab = _import_mlab()
    if mlab.options.backend != 'test':
        fig.scene.disable_render = not render


def plot_dipole_locations(dipoles, trans, subject, subjects_dir=None,
                          mode='orthoview', coord_frame='mri', idx='gof',
                          show_all=True, ax=None, block=False,
                          show=True, verbose=None):
    """Plot dipole locations.

    If mode is set to 'cone' or 'sphere', only the location of the first
    time point of each dipole is shown else use the show_all parameter.

    The option mode='orthoview' was added in version 0.14.

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
    mode : str
        Currently only ``'orthoview'`` is supported.

        .. versionadded:: 0.14.0
    coord_frame : str
        Coordinate frame to use, 'head' or 'mri'. Defaults to 'mri'.

        .. versionadded:: 0.14.0
    idx : int | 'gof' | 'amplitude'
        Index of the initially plotted dipole. Can also be 'gof' to plot the
        dipole with highest goodness of fit value or 'amplitude' to plot the
        dipole with the highest amplitude. The dipoles can also be browsed
        through using up/down arrow keys or mouse scroll. Defaults to 'gof'.
        Only used if mode equals 'orthoview'.

        .. versionadded:: 0.14.0
    show_all : bool
        Whether to always plot all the dipoles. If True (default), the active
        dipole is plotted as a red dot and it's location determines the shown
        MRI slices. The the non-active dipoles are plotted as small blue dots.
        If False, only the active dipole is plotted.
        Only used if mode equals 'orthoview'.

        .. versionadded:: 0.14.0
    ax : instance of matplotlib Axes3D | None
        Axes to plot into. If None (default), axes will be created.
        Only used if mode equals 'orthoview'.

        .. versionadded:: 0.14.0
    block : bool
        Whether to halt program execution until the figure is closed. Defaults
        to False.
        Only used if mode equals 'orthoview'.

        .. versionadded:: 0.14.0
    show : bool
        Show figure if True. Defaults to True.
        Only used if mode equals 'orthoview'.

        .. versionadded:: 0.14.0
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig : instance of mlab.Figure or matplotlib Figure
        The mayavi figure or matplotlib Figure.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    if mode == 'orthoview':
        fig = _plot_dipole_mri_orthoview(
            dipoles, trans=trans, subject=subject, subjects_dir=subjects_dir,
            coord_frame=coord_frame, idx=idx, show_all=show_all,
            ax=ax, block=block, show=show)
    else:
        raise ValueError('Mode must be "orthoview", got %s.' % (mode,))

    return fig


def snapshot_brain_montage(fig, montage, hide_sensors=True):
    """Take a snapshot of a Mayavi Scene and project channels onto 2d coords.

    Note that this will take the raw values for 3d coordinates of each channel,
    without applying any transforms. If brain images are flipped up/dn upon
    using `imshow`, check your matplotlib backend as this behavior changes.

    Parameters
    ----------
    fig : instance of Mayavi Scene
        The figure on which you've plotted electrodes using
        :func:`mne.viz.plot_alignment`.
    montage : instance of `DigMontage` or `Info` | dict of ch: xyz mappings.
        The digital montage for the electrodes plotted in the scene. If `Info`,
        channel positions will be pulled from the `loc` field of `chs`.
    hide_sensors : bool
        Whether to remove the spheres in the scene before taking a snapshot.

    Returns
    -------
    xy : array, shape (n_channels, 2)
        The 2d location of each channel on the image of the current scene view.
    im : array, shape (m, n, 3)
        The screenshot of the current scene view
    """
    mlab = _import_mlab()
    from ..channels import Montage, DigMontage
    from .. import Info
    if isinstance(montage, (Montage, DigMontage)):
        chs = montage.dig_ch_pos
        ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
    elif isinstance(montage, Info):
        xyz = [ich['loc'][:3] for ich in montage['chs']]
        ch_names = [ich['ch_name'] for ich in montage['chs']]
    elif isinstance(montage, dict):
        if not all(len(ii) == 3 for ii in montage.values()):
            raise ValueError('All electrode positions must be length 3')
        ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in montage.items()])
    else:
        raise ValueError('montage must be an instance of `DigMontage`, `Info`,'
                         ' or `dict`')

    xyz = np.vstack(xyz)
    xy = _3d_to_2d(fig, xyz)
    xy = dict(zip(ch_names, xy))
    pts = fig.children[-1]

    if hide_sensors is True:
        pts.visible = False
    with warnings.catch_warnings(record=True):
        im = mlab.screenshot(fig)
    pts.visible = True
    return xy, im


def _3d_to_2d(fig, xyz):
    """Convert 3d points to a 2d perspective using a Mayavi Scene."""
    from mayavi.core.scene import Scene

    if not isinstance(fig, Scene):
        raise TypeError('fig must be an instance of Scene, '
                        'found type %s' % type(fig))
    xyz = np.column_stack([xyz, np.ones(xyz.shape[0])])

    # Transform points into 'unnormalized' view coordinates
    comb_trans_mat = _get_world_to_view_matrix(fig.scene)
    view_coords = np.dot(comb_trans_mat, xyz.T).T

    # Divide through by the fourth element for normalized view coords
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

    # Transform from normalized view coordinates to display coordinates.
    view_to_disp_mat = _get_view_to_display_matrix(fig.scene)
    xy = np.dot(view_to_disp_mat, norm_view_coords.T).T

    # Pull the first two columns since they're meaningful for 2d plotting
    xy = xy[:, :2]
    return xy


def _get_world_to_view_matrix(scene):
    """Return the 4x4 matrix to transform xyz space to the current view.

    This is a concatenation of the model view and perspective transforms.
    """
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from tvtk.pyface.tvtk_scene import TVTKScene

    if not isinstance(scene, (MayaviScene, TVTKScene)):
        raise TypeError('scene must be an instance of TVTKScene/MayaviScene, '
                        'found type %s' % type(scene))
    cam = scene.camera

    # The VTK method needs the aspect ratio and near and far
    # clipping planes in order to return the proper transform.
    scene_size = tuple(scene.get_size())
    clip_range = cam.clipping_range
    aspect_ratio = float(scene_size[0]) / scene_size[1]

    # Get the vtk matrix object using the aspect ratio we defined
    vtk_comb_trans_mat = cam.get_composite_projection_transform_matrix(
        aspect_ratio, clip_range[0], clip_range[1])
    vtk_comb_trans_mat = vtk_comb_trans_mat.to_array()
    return vtk_comb_trans_mat


def _get_view_to_display_matrix(scene):
    """Return the 4x4 matrix to convert view coordinates to display coordinates.

    It's assumed that the view should take up the entire window and that the
    origin of the window is in the upper left corner.
    """
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from tvtk.pyface.tvtk_scene import TVTKScene

    if not isinstance(scene, (MayaviScene, TVTKScene)):
        raise TypeError('scene must be an instance of TVTKScene/MayaviScene, '
                        'found type %s' % type(scene))

    # normalized view coordinates have the origin in the middle of the space
    # so we need to scale by width and height of the display window and shift
    # by half width and half height. The matrix accomplishes that.
    x, y = tuple(scene.get_size())
    view_to_disp_mat = np.array([[x / 2.0,       0.,   0.,   x / 2.0],
                                 [0.,      -y / 2.0,   0.,   y / 2.0],
                                 [0.,            0.,   1.,        0.],
                                 [0.,            0.,   0.,        1.]])
    return view_to_disp_mat


def _plot_dipole_mri_orthoview(dipole, trans, subject, subjects_dir=None,
                               coord_frame='head', idx='gof', show_all=True,
                               ax=None, block=False, show=True):
    """Plot dipoles on top of MRI slices in 3-D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from .. import Dipole
    if not has_nibabel():
        raise ImportError('This function requires nibabel.')
    import nibabel as nib
    from nibabel.processing import resample_from_to

    if coord_frame not in ['head', 'mri']:
        raise ValueError("coord_frame must be 'head' or 'mri'. "
                         "Got %s." % coord_frame)

    if not isinstance(dipole, Dipole):
        from ..dipole import _concatenate_dipoles
        dipole = _concatenate_dipoles(dipole)
    if idx == 'gof':
        idx = np.argmax(dipole.gof)
    elif idx == 'amplitude':
        idx = np.argmax(np.abs(dipole.amplitude))
    else:
        idx = _ensure_int(idx, 'idx', 'an int or one of ["gof", "amplitude"]')

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    t1 = nib.load(t1_fname)
    vox2ras = t1.header.get_vox2ras_tkr()
    ras2vox = linalg.inv(vox2ras)
    trans = _get_trans(trans, fro='head', to='mri')[0]
    zooms = t1.header.get_zooms()
    if coord_frame == 'head':
        affine_to = trans['trans'].copy()
        affine_to[:3, 3] *= 1000  # to mm
        aff = t1.affine.copy()

        aff[:3, :3] /= zooms
        affine_to = np.dot(affine_to, aff)
        t1 = resample_from_to(t1, ([int(t1.shape[i] * zooms[i]) for i
                                    in range(3)], affine_to))
        dipole_locs = apply_trans(ras2vox, dipole.pos * 1e3) * zooms

        ori = dipole.ori
        scatter_points = dipole.pos * 1e3
    else:
        scatter_points = apply_trans(trans['trans'], dipole.pos) * 1e3
        ori = apply_trans(trans['trans'], dipole.ori, move=False)
        dipole_locs = apply_trans(ras2vox, scatter_points)

    data = t1.get_data()
    dims = len(data)  # Symmetric size assumed.
    dd = dims / 2.
    dd *= t1.header.get_zooms()[0]
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    elif not isinstance(ax, Axes3D):
        raise ValueError('ax must be an instance of Axes3D. '
                         'Got %s.' % type(ax))
    else:
        fig = ax.get_figure()

    gridx, gridy = np.meshgrid(np.linspace(-dd, dd, dims),
                               np.linspace(-dd, dd, dims))

    _plot_dipole(ax, data, dipole_locs, idx, dipole, gridx, gridy, ori,
                 coord_frame, zooms, show_all, scatter_points)
    params = {'ax': ax, 'data': data, 'idx': idx, 'dipole': dipole,
              'dipole_locs': dipole_locs, 'gridx': gridx, 'gridy': gridy,
              'ori': ori, 'coord_frame': coord_frame, 'zooms': zooms,
              'show_all': show_all, 'scatter_points': scatter_points}
    ax.view_init(elev=30, azim=-140)

    callback_func = partial(_dipole_changed, params=params)
    fig.canvas.mpl_connect('scroll_event', callback_func)
    fig.canvas.mpl_connect('key_press_event', callback_func)

    plt_show(show, block=block)
    return fig


def _plot_dipole(ax, data, points, idx, dipole, gridx, gridy, ori, coord_frame,
                 zooms, show_all, scatter_points):
    """Plot dipoles."""
    import matplotlib.pyplot as plt
    point = points[idx]
    xidx, yidx, zidx = np.round(point).astype(int)
    xslice = data[xidx][::-1]
    yslice = data[:, yidx][::-1].T
    zslice = data[:, :, zidx][::-1].T[::-1]
    if coord_frame == 'head':
        zooms = (1., 1., 1.)
    else:
        point = points[idx] * zooms
        xidx, yidx, zidx = np.round(point).astype(int)
    xyz = scatter_points

    ori = ori[idx]
    if show_all:
        colors = np.repeat('y', len(points))
        colors[idx] = 'r'
        size = np.repeat(5, len(points))
        size[idx] = 20
        visibles = range(len(points))
    else:
        colors = 'r'
        size = 20
        visibles = idx

    offset = np.min(gridx)
    ax.scatter(xs=xyz[visibles, 0], ys=xyz[visibles, 1],
               zs=xyz[visibles, 2], zorder=2, s=size, facecolor=colors)
    xx = np.linspace(offset, xyz[idx, 0], xidx)
    yy = np.linspace(offset, xyz[idx, 1], yidx)
    zz = np.linspace(offset, xyz[idx, 2], zidx)
    ax.plot(xx, np.repeat(xyz[idx, 1], len(xx)), zs=xyz[idx, 2], zorder=1,
            linestyle='-', color='r')
    ax.plot(np.repeat(xyz[idx, 0], len(yy)), yy, zs=xyz[idx, 2], zorder=1,
            linestyle='-', color='r')
    ax.plot(np.repeat(xyz[idx, 0], len(zz)),
            np.repeat(xyz[idx, 1], len(zz)), zs=zz, zorder=1,
            linestyle='-', color='r')
    kwargs = _pivot_kwargs()
    ax.quiver(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], ori[0], ori[1],
              ori[2], length=50, color='r', **kwargs)
    dims = np.array([(len(data) / -2.), (len(data) / 2.)])
    ax.set_xlim(-1 * dims * zooms[:2])  # Set axis lims to RAS coordinates.
    ax.set_ylim(-1 * dims * zooms[:2])
    ax.set_zlim(dims * zooms[:2])

    # Plot slices.
    ax.contourf(xslice, gridx, gridy, offset=offset, zdir='x',
                cmap='gray', zorder=0, alpha=.5)
    ax.contourf(gridx, gridy, yslice, offset=offset, zdir='z',
                cmap='gray', zorder=0, alpha=.5)
    ax.contourf(gridx, zslice, gridy, offset=offset,
                zdir='y', cmap='gray', zorder=0, alpha=.5)

    plt.suptitle('Dipole #%s / %s @ %.3fs, GOF: %.1f%%, %.1fnAm\n' % (
        idx + 1, len(dipole.times), dipole.times[idx], dipole.gof[idx],
        dipole.amplitude[idx] * 1e9) +
        '(%0.1f, %0.1f, %0.1f) mm' % tuple(xyz[idx]))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.draw()


def _dipole_changed(event, params):
    """Handle dipole plotter scroll/key event."""
    if event.key is not None:
        if event.key == 'up':
            params['idx'] += 1
        elif event.key == 'down':
            params['idx'] -= 1
        else:  # some other key
            return
    elif event.step > 0:  # scroll event
        params['idx'] += 1
    else:
        params['idx'] -= 1
    params['idx'] = min(max(0, params['idx']), len(params['dipole'].pos) - 1)
    params['ax'].clear()
    _plot_dipole(params['ax'], params['data'], params['dipole_locs'],
                 params['idx'], params['dipole'], params['gridx'],
                 params['gridy'], params['ori'], params['coord_frame'],
                 params['zooms'], params['show_all'], params['scatter_points'])
