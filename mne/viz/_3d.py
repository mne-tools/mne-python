# -*- coding: utf-8 -*-
"""Functions to make 3D plots with M/EEG data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import base64
from distutils.version import LooseVersion
from io import BytesIO
from itertools import cycle
import os.path as op
import warnings
from functools import partial

import numpy as np
from scipy import linalg, sparse

from ..defaults import DEFAULTS
from ..fixes import einsum, _crop_colorbar, _get_img_fdata
from ..io import _loc_to_coil_trans
from ..io.pick import pick_types, _picks_to_idx
from ..io.constants import FIFF
from ..io.meas_info import read_fiducials, create_info
from ..source_space import (_ensure_src, _create_surf_spacing, _check_spacing,
                            _read_mri_info)

from ..surface import (get_meg_helmet_surf, read_surface, _DistanceQuery,
                       transform_surface_to, _project_onto_surface,
                       mesh_edges, _reorder_ccw, _complete_sphere_surf)
from ..transforms import (_find_trans, apply_trans, rot_to_quat,
                          combine_transforms, _get_trans, _ensure_trans,
                          invert_transform, Transform,
                          read_ras_mni_t, _print_coord_trans)
from ..utils import (get_subjects_dir, logger, _check_subject, verbose, warn,
                     has_nibabel, check_version, fill_doc, _pl,
                     _ensure_int, _validate_type, _check_option)
from .utils import (mne_analyze_colormap, _prepare_trellis, _get_color_list,
                    plt_show, tight_layout, figure_nobar, _check_time_unit)
from ..bem import (ConductorModel, _bem_find_surface, _surf_dict, _surf_name,
                   read_bem_surfaces)


verbose_dec = verbose
FIDUCIAL_ORDER = (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION,
                  FIFF.FIFFV_POINT_RPA)


# XXX: to unify with digitization
def _fiducial_coords(points, coord_frame=None):
    """Generate 3x3 array of fiducial coordinates."""
    points = points or []  # None -> list
    if coord_frame is not None:
        points = [p for p in points if p['coord_frame'] == coord_frame]
    points_ = {p['ident']: p for p in points if
               p['kind'] == FIFF.FIFFV_POINT_CARDINAL}
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
                        show=True, destination=None, info=None, color='k',
                        axes=None):
    """Plot head positions.

    Parameters
    ----------
    pos : ndarray, shape (n_pos, 10) | list of ndarray
        The head position data. Can also be a list to treat as a
        concatenation of runs.
    mode : str
        Can be 'traces' (default) to show position and quaternion traces,
        or 'field' to show the position as a vector field over time.
        The 'field' mode requires matplotlib 1.4+.
    cmap : colormap
        Colormap to use for the trace plot, default is "viridis".
    direction : str
        Can be any combination of "x", "y", or "z" (default: "z") to show
        directional axes in "field" mode.
    show : bool
        Show figure if True. Defaults to True.
    destination : str | array-like, shape (3,) | None
        The destination location for the head, assumed to be in head
        coordinates. See :func:`mne.preprocessing.maxwell_filter` for
        details.

        .. versionadded:: 0.16
    info : instance of mne.Info | None
        Measurement information. If provided, will be used to show the
        destination position when ``destination is None``, and for
        showing the MEG sensors.

        .. versionadded:: 0.16
    color : color object
        The color to use for lines in ``mode == 'traces'`` and quiver
        arrows in ``mode == 'field'``.

        .. versionadded:: 0.16
    axes : array-like, shape (3, 2)
        The matplotlib axes to use. Only used for ``mode == 'traces'``.

        .. versionadded:: 0.16

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    from ..chpi import head_pos_to_trans_rot_t
    from ..preprocessing.maxwell import _check_destination
    import matplotlib.pyplot as plt
    _check_option('mode', mode, ['traces', 'field'])
    dest_info = dict(dev_head_t=None) if info is None else info
    destination = _check_destination(destination, dest_info, head_frame=True)
    if destination is not None:
        destination = _ensure_trans(destination, 'head', 'meg')  # probably inv
        destination = destination['trans'][:3].copy()
        destination[:, 3] *= 1000

    if not isinstance(pos, (list, tuple)):
        pos = [pos]
    for ii, p in enumerate(pos):
        p = np.array(p, float)
        if p.ndim != 2 or p.shape[1] != 10:
            raise ValueError('pos (or each entry in pos if a list) must be '
                             'dimension (N, 10), got %s' % (p.shape,))
        if ii > 0:  # concatenation
            p[:, 0] += pos[ii - 1][-1, 0] - p[0, 0]
        pos[ii] = p
    borders = np.cumsum([len(pp) for pp in pos])
    pos = np.concatenate(pos, axis=0)
    trans, rot, t = head_pos_to_trans_rot_t(pos)  # also ensures pos is okay
    # trans, rot, and t are for dev_head_t, but what we really want
    # is head_dev_t (i.e., where the head origin is in device coords)
    use_trans = einsum('ijk,ik->ij', rot[:, :3, :3].transpose([0, 2, 1]),
                       -trans) * 1000
    use_rot = rot.transpose([0, 2, 1])
    use_quats = -pos[:, 1:4]  # inverse (like doing rot.T)
    surf = rrs = lims = None
    if info is not None:
        meg_picks = pick_types(info, meg=True, ref_meg=False, exclude=())
        if len(meg_picks) > 0:
            rrs = 1000 * np.array([info['chs'][pick]['loc'][:3]
                                   for pick in meg_picks], float)
            if mode == 'traces':
                lims = np.array((rrs.min(0), rrs.max(0))).T
            else:  # mode == 'field'
                surf = get_meg_helmet_surf(info)
                transform_surface_to(surf, 'meg', info['dev_head_t'],
                                     copy=False)
                surf['rr'] *= 1000.
    helmet_color = (0.0, 0.0, 0.6)
    if mode == 'traces':
        if axes is None:
            axes = plt.subplots(3, 2, sharex=True)[1]
        else:
            axes = np.array(axes)
        if axes.shape != (3, 2):
            raise ValueError('axes must have shape (3, 2), got %s'
                             % (axes.shape,))
        fig = axes[0, 0].figure

        labels = ['xyz', ('$q_1$', '$q_2$', '$q_3$')]
        for ii, (quat, coord) in enumerate(zip(use_quats.T, use_trans.T)):
            axes[ii, 0].plot(t, coord, color, lw=1., zorder=3)
            axes[ii, 0].set(ylabel=labels[0][ii], xlim=t[[0, -1]])
            axes[ii, 1].plot(t, quat, color, lw=1., zorder=3)
            axes[ii, 1].set(ylabel=labels[1][ii], xlim=t[[0, -1]])
            for b in borders[:-1]:
                for jj in range(2):
                    axes[ii, jj].axvline(t[b], color='r')
        for ii, title in enumerate(('Position (mm)', 'Rotation (quat)')):
            axes[0, ii].set(title=title)
            axes[-1, ii].set(xlabel='Time (s)')
        if rrs is not None:
            pos_bads = np.any([(use_trans[:, ii] <= lims[ii, 0]) |
                               (use_trans[:, ii] >= lims[ii, 1])
                               for ii in range(3)], axis=0)
            for ii in range(3):
                oidx = list(range(ii)) + list(range(ii + 1, 3))
                # knowing it will generally be spherical, we can approximate
                # how far away we are along the axis line by taking the
                # point to the left and right with the smallest distance
                from scipy.spatial.distance import cdist
                dists = cdist(rrs[:, oidx], use_trans[:, oidx])
                left = rrs[:, [ii]] < use_trans[:, ii]
                left_dists_all = dists.copy()
                left_dists_all[~left] = np.inf
                # Don't show negative Z direction
                if ii != 2 and np.isfinite(left_dists_all).any():
                    idx = np.argmin(left_dists_all, axis=0)
                    left_dists = rrs[idx, ii]
                    bads = ~np.isfinite(
                        left_dists_all[idx, np.arange(len(idx))]) | pos_bads
                    left_dists[bads] = np.nan
                    axes[ii, 0].plot(t, left_dists, color=helmet_color,
                                     ls='-', lw=0.5, zorder=2)
                else:
                    axes[ii, 0].axhline(lims[ii][0], color=helmet_color,
                                        ls='-', lw=0.5, zorder=2)
                right_dists_all = dists
                right_dists_all[left] = np.inf
                if np.isfinite(right_dists_all).any():
                    idx = np.argmin(right_dists_all, axis=0)
                    right_dists = rrs[idx, ii]
                    bads = ~np.isfinite(
                        right_dists_all[idx, np.arange(len(idx))]) | pos_bads
                    right_dists[bads] = np.nan
                    axes[ii, 0].plot(t, right_dists, color=helmet_color,
                                     ls='-', lw=0.5, zorder=2)
                else:
                    axes[ii, 0].axhline(lims[ii][1], color=helmet_color,
                                        ls='-', lw=0.5, zorder=2)

        for ii in range(3):
            axes[ii, 1].set(ylim=[-1, 1])

        if destination is not None:
            vals = np.array([destination[:, 3],
                             rot_to_quat(destination[:, :3])]).T.ravel()
            for ax, val in zip(fig.axes, vals):
                ax.axhline(val, color='r', ls=':', zorder=2, lw=1.)

    else:  # mode == 'field':
        from matplotlib.colors import Normalize
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from mpl_toolkits.mplot3d import axes3d  # noqa: F401, analysis:ignore
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
        kwargs = dict(pivot='tail')
        for d, length in zip(direction, [5., 2.5, 1.]):
            use_dir = use_rot[:, :, dir_idx[d]]
            # draws stems, then heads
            array = np.concatenate((t, np.repeat(t, 2)))
            ax.quiver(use_trans[:, 0], use_trans[:, 1], use_trans[:, 2],
                      use_dir[:, 0], use_dir[:, 1], use_dir[:, 2], norm=norm,
                      cmap=cmap, array=array, length=length, **kwargs)
            if destination is not None:
                ax.quiver(destination[0, 3],
                          destination[1, 3],
                          destination[2, 3],
                          destination[dir_idx[d], 0],
                          destination[dir_idx[d], 1],
                          destination[dir_idx[d], 2], color=color,
                          length=length, **kwargs)
        mins = use_trans.min(0)
        maxs = use_trans.max(0)
        if surf is not None:
            ax.plot_trisurf(*surf['rr'].T, triangles=surf['tris'],
                            color=helmet_color, alpha=0.1, shade=False)
            ax.scatter(*rrs.T, s=1, color=helmet_color)
            mins = np.minimum(mins, rrs.min(0))
            maxs = np.maximum(maxs, rrs.max(0))
        scale = (maxs - mins).max() / 2.
        xlim, ylim, zlim = (maxs + mins)[:, np.newaxis] / 2. + [-scale, scale]
        ax.set(xlabel='x', ylabel='y', zlabel='z',
               xlim=xlim, ylim=ylim, zlim=zlim)
        _set_aspect_equal(ax)
        ax.view_init(30, 45)
    tight_layout(fig=fig)
    plt_show(show)
    return fig


def _set_aspect_equal(ax):
    # XXX recent MPL throws an error for 3D axis aspect setting, not much
    # we can do about it at this point
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        pass


@fill_doc
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
    %(n_jobs)s

    Returns
    -------
    fig : instance of mayavi.mlab.Figure
        The mayavi figure.
    """
    # Update the backend
    from .backends.renderer import _Renderer
    types = [t for t in ['eeg', 'grad', 'mag'] if t in evoked]

    time_idx = None
    if time is None:
        time = np.mean([evoked.get_peak(ch_type=t)[1] for t in types])
    del types

    if not evoked.times[0] <= time <= evoked.times[-1]:
        raise ValueError('`time` (%0.3f) must be inside `evoked.times`' % time)
    time_idx = np.argmin(np.abs(evoked.times - time))

    # Plot them
    alphas = [1.0, 0.5]
    colors = [(0.6, 0.6, 0.6), (1.0, 1.0, 1.0)]
    colormap = mne_analyze_colormap(format='mayavi')
    colormap_lines = np.concatenate([np.tile([0., 0., 255., 255.], (127, 1)),
                                     np.tile([0., 0., 0., 255.], (2, 1)),
                                     np.tile([255., 0., 0., 255.], (127, 1))])

    renderer = _Renderer(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))

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
        renderer.surface(surface=surf, color=colors[ii],
                         opacity=alpha)

        # Now show our field pattern
        renderer.surface(surface=surf, vmin=-vlim, vmax=vlim,
                         scalars=data, colormap=colormap)

        # And the field lines on top
        renderer.contour(surface=surf, scalars=data, contours=21,
                         vmin=-vlim, vmax=vlim, opacity=alpha,
                         colormap=colormap_lines)

    if '%' in time_label:
        time_label %= (1e3 * evoked.times[time_idx])
    renderer.text2d(x=0.01, y=0.01, text=time_label)
    renderer.set_camera(azimuth=10, elevation=60)
    renderer.show()
    return renderer.scene()


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
    fig : instance of matplotlib.figure.Figure | list
        The figure. Will instead be a list of png images if
        img_output is a tuple.
    """
    import matplotlib.pyplot as plt
    import nibabel as nib

    _check_option('orientation', orientation, ['coronal', 'axial', 'sagittal'])

    # Load the T1 data
    nim = nib.load(mri_fname)
    data = _get_img_fdata(nim)
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
            with warnings.catch_warnings(record=True):  # no contours
                warnings.simplefilter('ignore')
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


@verbose
def plot_alignment(info=None, trans=None, subject=None, subjects_dir=None,
                   surfaces='head', coord_frame='head',
                   meg=None, eeg='original', fwd=None,
                   dig=False, ecog=True, src=None, mri_fiducials=False,
                   bem=None, seeg=True, fnirs=True, show_axes=False, fig=None,
                   interaction='trackball', verbose=None):
    """Plot head, sensor, and source space alignment in 3D.

    Parameters
    ----------
    info : dict | None
        The measurement info.
        If None (default), no sensor information will be shown.
    %(trans)s
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. Can be omitted if ``src`` is provided.
    %(subjects_dir)s
    surfaces : str | list
        Surfaces to plot. Supported values:

        * scalp: one of 'head', 'outer_skin' (alias for 'head'),
          'head-dense', or 'seghead' (alias for 'head-dense')
        * skull: 'outer_skull', 'inner_skull', 'brain' (alias for
          'inner_skull')
        * brain: one of 'pial', 'white', 'inflated', or 'brain'
          (alias for 'pial').

        Defaults to 'head'.

        .. note:: For single layer BEMs it is recommended to use 'brain'.
    coord_frame : str
        Coordinate frame to use, 'head', 'meg', or 'mri'.
    meg : str | list | bool | None
        Can be "helmet", "sensors" or "ref" to show the MEG helmet, sensors or
        reference sensors respectively, or a combination like
        ``('helmet', 'sensors')`` (same as None, default). True translates to
        ``('helmet', 'sensors', 'ref')``.
    eeg : bool | str | list
        Can be "original" (default; equivalent to True) or "projected" to
        show EEG sensors in their digitized locations or projected onto the
        scalp, or a list of these options including ``[]`` (equivalent of
        False).
    fwd : instance of Forward
        The forward solution. If present, the orientations of the dipoles
        present in the forward solution are displayed.
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
    bem : list of dict | instance of ConductorModel | None
        Can be either the BEM surfaces (list of dict), a BEM solution or a
        sphere model. If None, we first try loading
        `'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'`, and then look for
        `'$SUBJECT*$SOURCE.fif'` in the same directory. For `'outer_skin'`,
        the subjects bem and bem/flash folders are searched. Defaults to None.
    seeg : bool
        If True (default), show sEEG electrodes.
    fnirs : bool
        If True (default), show fNIRS electrodes.

        .. versionadded:: 0.20
    show_axes : bool
        If True (default False), coordinate frame axis indicators will be
        shown:

        * head in pink
        * MRI in gray (if ``trans is not None``)
        * MEG in blue (if MEG sensors are present)

        .. versionadded:: 0.16
    fig : mayavi.mlab.Figure | None
        Mayavi Scene in which to plot the alignment.
        If ``None``, creates a new 600x600 pixel figure with black background.

        .. versionadded:: 0.16
    interaction : str
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.

        .. versionadded:: 0.16
    %(verbose)s

    Returns
    -------
    fig : instance of mayavi.mlab.Figure
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
    from ..forward import _create_meg_coils, Forward
    # Update the backend
    from .backends.renderer import _Renderer

    if eeg is False:
        eeg = list()
    elif eeg is True:
        eeg = 'original'
    if meg is None:
        meg = ('helmet', 'sensors')
        # only consider warning if the value is explicit
        warn_meg = False
    else:
        warn_meg = True

    if meg is True:
        meg = ('helmet', 'sensors', 'ref')
    elif meg is False:
        meg = list()
    elif isinstance(meg, str):
        meg = [meg]
    if isinstance(eeg, str):
        eeg = [eeg]

    _check_option('interaction', interaction, ['trackball', 'terrain'])
    for kind, var in zip(('eeg', 'meg'), (eeg, meg)):
        if not isinstance(var, (list, tuple)) or \
                not all(isinstance(x, str) for x in var):
            raise TypeError('%s must be list or tuple of str, got %s'
                            % (kind, type(var)))
    if not all(x in ('helmet', 'sensors', 'ref') for x in meg):
        raise ValueError('meg must only contain "helmet", "sensors" or "ref", '
                         'got %s' % (meg,))
    if not all(x in ('original', 'projected') for x in eeg):
        raise ValueError('eeg must only contain "original" and '
                         '"projected", got %s' % (eeg,))

    info = create_info(1, 1000., 'misc') if info is None else info
    _validate_type(info, "info")

    if isinstance(surfaces, str):
        surfaces = [surfaces]
    surfaces = list(surfaces)
    for s in surfaces:
        _validate_type(s, "str", "all entries in surfaces")

    is_sphere = False
    if isinstance(bem, ConductorModel) and bem['is_sphere']:
        if len(bem['layers']) != 4 and len(surfaces) > 1:
            raise ValueError('The sphere conductor model must have three '
                             'layers for plotting skull and head.')
        is_sphere = True

    _check_option('coord_frame', coord_frame, ['head', 'meg', 'mri'])
    if src is not None:
        src = _ensure_src(src)
        src_subject = src._subject
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

    if fwd is not None:
        _validate_type(fwd, [Forward])
        fwd_rr = fwd['source_rr']
        if fwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
            fwd_nn = fwd['source_nn'].reshape(-1, 1, 3)
        else:
            fwd_nn = fwd['source_nn'].reshape(-1, 3, 3)

    ref_meg = 'ref' in meg
    meg_picks = pick_types(info, meg=True, ref_meg=ref_meg)
    eeg_picks = pick_types(info, meg=False, eeg=True, ref_meg=False)
    other_bools = dict(ecog=ecog, seeg=seeg, fnirs=fnirs)
    del ecog, seeg, fnirs
    other_keys = sorted(other_bools.keys())
    other_picks = {key: pick_types(info, meg=False, ref_meg=False,
                                   **{key: True}) for key in other_keys}
    if trans == 'auto':
        # let's try to do this in MRI coordinates so they're easy to plot
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        trans = _find_trans(subject, subjects_dir)
    head_mri_t, _ = _get_trans(trans, 'head', 'mri')
    dev_head_t, _ = _get_trans(info['dev_head_t'], 'meg', 'head')
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
        meg_trans = dev_head_t
        mri_trans = invert_transform(head_mri_t)

    # both the head and helmet will be in MRI coordinates after this
    surfs = dict()

    # Head:
    sphere_level = 4
    head = False
    for s in surfaces:
        if s in ('head', 'outer_skin', 'head-dense', 'seghead'):
            if head:
                raise ValueError('Can only supply one head-like surface name')
            surfaces.pop(surfaces.index(s))
            head = True
            head_surf = None
            # Try the BEM if applicable
            if s in ('head', 'outer_skin'):
                if bem is not None:
                    head_missing = (
                        'Could not find the surface for '
                        'head in the provided BEM model, '
                        'looking in the subject directory.')
                    if isinstance(bem, ConductorModel):
                        if is_sphere:
                            head_surf = _complete_sphere_surf(
                                bem, 3, sphere_level, complete=False)
                        else:  # BEM solution
                            try:
                                head_surf = _bem_find_surface(
                                    bem, FIFF.FIFFV_BEM_SURF_ID_HEAD)
                            except RuntimeError:
                                logger.info(head_missing)
                    elif bem is not None:  # list of dict
                        for this_surf in bem:
                            if this_surf['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD:
                                head_surf = this_surf
                                break
                        else:
                            logger.info(head_missing)
            if head_surf is None:
                if subject is None:
                    raise ValueError('To plot the head surface, the BEM/sphere'
                                     ' model must contain a head surface '
                                     'or "subject" must be provided (got '
                                     'None)')
                subject_dir = op.join(
                    get_subjects_dir(subjects_dir, raise_error=True), subject)
                if s in ('head-dense', 'seghead'):
                    try_fnames = [
                        op.join(subject_dir, 'bem', '%s-head-dense.fif'
                                % subject),
                        op.join(subject_dir, 'surf', 'lh.seghead'),
                    ]
                else:
                    try_fnames = [
                        op.join(subject_dir, 'bem', 'outer_skin.surf'),
                        op.join(subject_dir, 'bem', 'flash',
                                'outer_skin.surf'),
                        op.join(subject_dir, 'bem', '%s-head.fif'
                                % subject),
                    ]
                for fname in try_fnames:
                    if op.exists(fname):
                        logger.info('Using %s for head surface.'
                                    % (op.basename(fname),))
                        if op.splitext(fname)[-1] == '.fif':
                            head_surf = read_bem_surfaces(fname)[0]
                        else:
                            head_surf = read_surface(
                                fname, return_dict=True)[2]
                            head_surf['rr'] /= 1000.
                            head_surf.update(coord_frame=FIFF.FIFFV_COORD_MRI)
                        break
                else:
                    raise IOError('No head surface found for subject '
                                  '%s after trying:\n%s'
                                  % (subject, '\n'.join(try_fnames)))
            surfs['head'] = head_surf

    # Skull:
    skull = list()
    for name, id_ in (('outer_skull', FIFF.FIFFV_BEM_SURF_ID_SKULL),
                      ('inner_skull', FIFF.FIFFV_BEM_SURF_ID_BRAIN)):
        if name in surfaces:
            surfaces.pop(surfaces.index(name))
            if bem is None:
                fname = op.join(
                    get_subjects_dir(subjects_dir, raise_error=True),
                    subject, 'bem', name + '.surf')
                if not op.isfile(fname):
                    raise ValueError('bem is None and the the %s file cannot '
                                     'be found:\n%s' % (name, fname))
                surf = read_surface(fname, return_dict=True)[2]
                surf.update(coord_frame=FIFF.FIFFV_COORD_MRI,
                            id=_surf_dict[name])
                surf['rr'] /= 1000.
                skull.append(surf)
            elif isinstance(bem, ConductorModel):
                if is_sphere:
                    if len(bem['layers']) != 4:
                        raise ValueError('The sphere model must have three '
                                         'layers for plotting %s' % (name,))
                    this_idx = 1 if name == 'inner_skull' else 2
                    skull.append(_complete_sphere_surf(
                        bem, this_idx, sphere_level))
                    skull[-1]['id'] = _surf_dict[name]
                else:
                    skull.append(_bem_find_surface(bem, id_))
            else:  # BEM model
                for this_surf in bem:
                    if this_surf['id'] == _surf_dict[name]:
                        skull.append(this_surf)
                        break
                else:
                    raise ValueError('Could not find the surface for %s.'
                                     % name)

    if mri_fiducials:
        if mri_fiducials is True:
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            if subject is None:
                raise ValueError("Subject needs to be specified to "
                                 "automatically find the fiducials file.")
            mri_fiducials = op.join(subjects_dir, subject, 'bem',
                                    subject + '-fiducials.fif')
        if isinstance(mri_fiducials, str):
            mri_fiducials, cf = read_fiducials(mri_fiducials)
            if cf != FIFF.FIFFV_COORD_MRI:
                raise ValueError("Fiducials are not in MRI space")
        fid_loc = _fiducial_coords(mri_fiducials, FIFF.FIFFV_COORD_MRI)
        fid_loc = apply_trans(mri_trans, fid_loc)
    else:
        fid_loc = []

    if 'helmet' in meg and len(meg_picks) > 0:
        surfs['helmet'] = get_meg_helmet_surf(info, head_mri_t)
        assert surfs['helmet']['coord_frame'] == FIFF.FIFFV_COORD_MRI

    # Brain:
    brain = np.intersect1d(surfaces, ['brain', 'pial', 'white', 'inflated'])
    if len(brain) > 1:
        raise ValueError('Only one brain surface can be plotted. '
                         'Got %s.' % brain)
    elif len(brain) == 0:
        brain = False
    else:  # exactly 1
        brain = brain[0]
        surfaces.pop(surfaces.index(brain))
        brain = 'pial' if brain == 'brain' else brain
        if is_sphere:
            if len(bem['layers']) > 0:
                surfs['lh'] = _complete_sphere_surf(
                    bem, 0, sphere_level)  # only plot 1
        else:
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            for hemi in ['lh', 'rh']:
                fname = op.join(subjects_dir, subject, 'surf',
                                '%s.%s' % (hemi, brain))
                surfs[hemi] = read_surface(fname, return_dict=True)[2]
                surfs[hemi]['rr'] /= 1000.
                surfs[hemi].update(coord_frame=FIFF.FIFFV_COORD_MRI)
        brain = True

    # we've looked through all of them, raise if some remain
    if len(surfaces) > 0:
        raise ValueError('Unknown surface type%s: %s'
                         % (_pl(surfaces), surfaces,))

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
            skull_surf = _complete_sphere_surf(bem, this_idx, sphere_level)
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
            skull_surf = read_surface(skull_fname, return_dict=True)[2]
            skull_surf['rr'] /= 1000.
            skull_surf['coord_frame'] = FIFF.FIFFV_COORD_MRI
        skull_alpha[this_skull] = alphas[idx + 1]
        skull_colors[this_skull] = (0.95 - idx * 0.2, 0.85, 0.95 - idx * 0.2)
        surfs[this_skull] = skull_surf

    if src is None and brain is False and len(skull) == 0 and not show_axes:
        head_alpha = 1.0
    else:
        head_alpha = alphas[0]

    for key in surfs.keys():
        # Surfs can sometimes be in head coords (e.g., if coming from sphere)
        surfs[key] = transform_surface_to(surfs[key], coord_frame,
                                          [mri_trans, head_trans], copy=True)

    if src is not None:
        src_rr, src_nn = _update_coord_frame(src[0], src_rr, src_nn,
                                             mri_trans, head_trans)
    if fwd is not None:
        fwd_rr, fwd_nn = _update_coord_frame(fwd, fwd_rr, fwd_nn,
                                             mri_trans, head_trans)

    # determine points
    meg_rrs, meg_tris = list(), list()
    hpi_loc = list()
    ext_loc = list()
    car_loc = list()
    eeg_loc = list()
    eegp_loc = list()
    other_loc = {key: list() for key in other_keys}
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
            if warn_meg:
                warn('MEG sensors not found. Cannot plot MEG locations.')
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
            hpi_loc = np.array([
                d['r'] for d in (info['dig'] or [])
                if (d['kind'] == FIFF.FIFFV_POINT_HPI and
                    d['coord_frame'] == FIFF.FIFFV_COORD_HEAD)])
            ext_loc = np.array([
                d['r'] for d in (info['dig'] or [])
                if (d['kind'] == FIFF.FIFFV_POINT_EXTRA and
                    d['coord_frame'] == FIFF.FIFFV_COORD_HEAD)])
        car_loc = _fiducial_coords(info['dig'], FIFF.FIFFV_COORD_HEAD)
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
    for key, picks in other_picks.items():
        if other_bools[key] and len(picks):
            other_loc[key] = np.array([info['chs'][pick]['loc'][:3]
                                       for pick in picks])
            logger.info('Plotting %d %s location%s'
                        % (len(other_loc[key]), key, _pl(other_loc[key])))

    # initialize figure
    renderer = _Renderer(fig, bgcolor=(0.5, 0.5, 0.5), size=(800, 800))
    if interaction == 'terrain':
        renderer.set_interactive()

    # plot surfaces
    alphas = dict(head=head_alpha, helmet=0.25, lh=hemi_val, rh=hemi_val)
    alphas.update(skull_alpha)
    colors = dict(head=(0.6,) * 3, helmet=(0.0, 0.0, 0.6), lh=(0.5,) * 3,
                  rh=(0.5,) * 3)
    colors.update(skull_colors)
    for key, surf in surfs.items():
        renderer.surface(surface=surf, color=colors[key],
                         opacity=alphas[key],
                         backface_culling=(key != 'helmet'))
    if brain and 'lh' not in surfs:  # one layer sphere
        assert bem['coord_frame'] == FIFF.FIFFV_COORD_HEAD
        center = bem['r0'].copy()
        center = apply_trans(head_trans, center)
        renderer.sphere(center, scale=0.01, color=colors['lh'],
                        opacity=alphas['lh'])
    if show_axes:
        axes = [(head_trans, (0.9, 0.3, 0.3))]  # always show head
        if not np.allclose(head_mri_t['trans'], np.eye(4)):  # Show MRI
            axes.append((mri_trans, (0.6, 0.6, 0.6)))
        if len(meg_picks) > 0:  # Show MEG
            axes.append((meg_trans, (0., 0.6, 0.6)))
        for ax in axes:
            x, y, z = np.tile(ax[0]['trans'][:3, 3], 3).reshape((3, 3)).T
            u, v, w = ax[0]['trans'][:3, :3]
            renderer.sphere(center=np.column_stack((x[0], y[0], z[0])),
                            color=ax[1], scale=3e-3)
            renderer.quiver3d(x=x, y=y, z=z, u=u, v=v, w=w, mode='arrow',
                              scale=2e-2, color=ax[1],
                              scale_mode='scalar', resolution=20,
                              scalars=[0.33, 0.66, 1.0])

    # plot points
    defaults = DEFAULTS['coreg']
    datas = [eeg_loc,
             hpi_loc,
             ext_loc] + list(other_loc[key] for key in other_keys)
    colors = [defaults['eeg_color'],
              defaults['hpi_color'],
              defaults['extra_color']
              ] + [defaults[key + '_color'] for key in other_keys]
    alphas = [0.8,
              0.5,
              0.25] + [0.8] * len(other_keys)
    scales = [defaults['eeg_scale'],
              defaults['hpi_scale'],
              defaults['extra_scale']
              ] + [defaults[key + '_scale'] for key in other_keys]
    assert len(datas) == len(colors) == len(alphas) == len(scales)
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
            renderer.sphere(center=data, color=color, scale=scale,
                            opacity=alpha, backface_culling=True)
    if len(eegp_loc) > 0:
        renderer.quiver3d(
            x=eegp_loc[:, 0], y=eegp_loc[:, 1], z=eegp_loc[:, 2],
            u=eegp_nn[:, 0], v=eegp_nn[:, 1], w=eegp_nn[:, 2],
            color=defaults['eegp_color'], mode='cylinder',
            scale=defaults['eegp_scale'], opacity=0.6,
            glyph_height=defaults['eegp_height'],
            glyph_center=(0., -defaults['eegp_height'], 0),
            glyph_resolution=20,
            backface_culling=True)
    if len(meg_rrs) > 0:
        color, alpha = (0., 0.25, 0.5), 0.25
        surf = dict(rr=meg_rrs, tris=meg_tris)
        renderer.surface(surface=surf, color=color,
                         opacity=alpha, backface_culling=True)
    if len(src_rr) > 0:
        renderer.quiver3d(
            x=src_rr[:, 0], y=src_rr[:, 1], z=src_rr[:, 2],
            u=src_nn[:, 0], v=src_nn[:, 1], w=src_nn[:, 2],
            color=(1., 1., 0.), mode='cylinder', scale=3e-3,
            opacity=0.75, glyph_height=0.25,
            glyph_center=(0., 0., 0.), glyph_resolution=20,
            backface_culling=True)
    if fwd is not None:
        red = (1.0, 0.0, 0.0)
        green = (0.0, 1.0, 0.0)
        blue = (0.0, 0.0, 1.0)
        for ori, color in zip(range(fwd_nn.shape[1]), (red, green, blue)):
            renderer.quiver3d(fwd_rr[:, 0],
                              fwd_rr[:, 1],
                              fwd_rr[:, 2],
                              fwd_nn[:, ori, 0],
                              fwd_nn[:, ori, 1],
                              fwd_nn[:, ori, 2],
                              color=color, mode='arrow', scale=1.5e-3)
    renderer.set_camera(azimuth=90, elevation=90,
                        distance=0.6, focalpoint=(0., 0., 0.))
    renderer.show()
    return renderer.scene()


def _make_tris_fan(n_vert):
    """Make tris given a number of vertices of a circle-like obj."""
    tris = np.zeros((n_vert - 2, 3), int)
    tris[:, 2] = np.arange(2, n_vert)
    tris[:, 1] = tris[:, 2] - 1
    return tris


def _sensor_shape(coil):
    """Get the sensor shape vertices."""
    from scipy.spatial import ConvexHull
    id_ = coil['type'] & 0xFFFF
    pad = True
    # Square figure eight
    if id_ in (FIFF.FIFFV_COIL_NM_122,
               FIFF.FIFFV_COIL_VV_PLANAR_W,
               FIFF.FIFFV_COIL_VV_PLANAR_T1,
               FIFF.FIFFV_COIL_VV_PLANAR_T2,
               ):
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
                               _make_tris_fan(4)[:, ::-1] + 4), axis=0)
    # Square
    elif id_ in (FIFF.FIFFV_COIL_POINT_MAGNETOMETER,
                 FIFF.FIFFV_COIL_VV_MAG_T1,
                 FIFF.FIFFV_COIL_VV_MAG_T2,
                 FIFF.FIFFV_COIL_VV_MAG_T3,
                 FIFF.FIFFV_COIL_KIT_REF_MAG,
                 ):
        # square magnetometer (potentially point-type)
        size = 0.001 if id_ == 2000 else (coil['size'] / 2.)
        rrs = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) * size
        tris = _make_tris_fan(4)
    # Circle
    elif id_ in (FIFF.FIFFV_COIL_MAGNES_MAG,
                 FIFF.FIFFV_COIL_MAGNES_REF_MAG,
                 FIFF.FIFFV_COIL_CTF_REF_MAG,
                 FIFF.FIFFV_COIL_BABY_MAG,
                 FIFF.FIFFV_COIL_BABY_REF_MAG,
                 FIFF.FIFFV_COIL_ARTEMIS123_REF_MAG,
                 ):
        n_pts = 15  # number of points for circle
        circle = np.exp(2j * np.pi * np.arange(n_pts) / float(n_pts))
        circle = np.concatenate(([0.], circle))
        circle *= coil['size'] / 2.  # radius of coil
        rrs = np.array([circle.real, circle.imag]).T
        tris = _make_tris_fan(n_pts + 1)
    # Circle
    elif id_ in (FIFF.FIFFV_COIL_MAGNES_GRAD,
                 FIFF.FIFFV_COIL_CTF_GRAD,
                 FIFF.FIFFV_COIL_CTF_REF_GRAD,
                 FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD,
                 FIFF.FIFFV_COIL_MAGNES_REF_GRAD,
                 FIFF.FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD,
                 FIFF.FIFFV_COIL_KIT_GRAD,
                 FIFF.FIFFV_COIL_BABY_GRAD,
                 FIFF.FIFFV_COIL_ARTEMIS123_GRAD,
                 FIFF.FIFFV_COIL_ARTEMIS123_REF_GRAD,
                 ):
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
    # 3D convex hull (will fail for 2D geometry, can extend later if needed)
    else:
        rrs = coil['rmag_orig'].copy()
        pad = False
        tris = _reorder_ccw(rrs, ConvexHull(rrs).simplices)

    # Go from (x,y) -> (x,y,z)
    if pad:
        rrs = np.pad(rrs, ((0, 0), (0, 1)), mode='constant')
    assert rrs.ndim == 2 and rrs.shape[1] == 3
    return rrs, tris


def _limits_to_control_points(clim, stc_data, colormap, transparent,
                              allow_pos_lims=True, linearize=False):
    """Convert limits (values or percentiles) to control points.

    This function also does the nonlinear scaling of the colormap in the
    case of a diverging colormap, and it forces transparency in the
    alpha channel.
    """
    # Based on type of limits specified, get cmap control points
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if colormap == 'auto':
        if clim == 'auto':
            if allow_pos_lims and (stc_data < 0).any():
                colormap = 'mne'
            else:
                colormap = 'hot'
        else:
            if 'lims' in clim:
                colormap = 'hot'
            else:  # 'pos_lims' in clim
                colormap = 'mne'
    diverging_maps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                      'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr',
                      'seismic']
    diverging_maps += [d + '_r' for d in diverging_maps]
    diverging_maps += ['mne', 'mne_analyze', ]
    if clim == 'auto':
        # this is merely a heuristic!
        if allow_pos_lims and colormap in diverging_maps:
            key = 'pos_lims'
        else:
            key = 'lims'
        clim = {'kind': 'percent', key: [96, 97.5, 99.95]}
    if not isinstance(clim, dict):
        raise ValueError('"clim" must be "auto" or dict, got %s' % (clim,))

    if ('lims' in clim) + ('pos_lims' in clim) != 1:
        raise ValueError('Exactly one of lims and pos_lims must be specified '
                         'in clim, got %s' % (clim,))
    if 'pos_lims' in clim and not allow_pos_lims:
        raise ValueError('Cannot use "pos_lims" for clim, use "lims" '
                         'instead')
    diverging_lims = 'pos_lims' in clim
    ctrl_pts = np.array(clim['pos_lims' if diverging_lims else 'lims'])
    ctrl_pts = np.array(ctrl_pts, float)
    if ctrl_pts.shape != (3,):
        raise ValueError('clim has shape %s, it must be (3,)'
                         % (ctrl_pts.shape,))
    if (np.diff(ctrl_pts) < 0).any():
        raise ValueError('colormap limits must be monotonically '
                         'increasing, got %s' % (ctrl_pts,))
    clim_kind = clim.get('kind', 'percent')
    _check_option("clim['kind']", clim_kind, ['value', 'values', 'percent'])
    if clim_kind == 'percent':
        perc_data = np.abs(stc_data) if diverging_lims else stc_data
        ctrl_pts = np.percentile(perc_data, ctrl_pts)
        logger.info('Using control points %s' % (ctrl_pts,))
    if len(set(ctrl_pts)) == 1:  # three points match
        if ctrl_pts[0] == 0:  # all are zero
            warn('All data were zero')
            ctrl_pts = np.arange(3, dtype=float)
            ticks = [0]
        else:
            ctrl_pts *= [0., 0.5, 1]  # all nonzero pts == max
            ticks = ctrl_pts[-1:]
    elif len(set(ctrl_pts)) == 2:  # two points match
        # if points one and two are identical, add a tiny bit to the
        # control point two; if points two and three are identical,
        # subtract a tiny bit from point two.
        bump = 1e-5 if ctrl_pts[0] == ctrl_pts[1] else -1e-5
        ctrl_pts[1] = ctrl_pts[0] + bump * (ctrl_pts[2] - ctrl_pts[0])
        ticks = ctrl_pts[::2]
    else:
        ticks = ctrl_pts

    if colormap in ('mne', 'mne_analyze'):
        colormap = mne_analyze_colormap([0, 1, 2], format='matplotlib')
    # scale colormap so that the bounds given by scale_pts actually work
    colormap = plt.get_cmap(colormap)
    if diverging_lims:
        # remap -ctrl_norm[2]->ctrl_norm[2] to 0->1
        ctrl_norm = np.concatenate([-ctrl_pts[::-1] / ctrl_pts[2], [0],
                                    ctrl_pts / ctrl_pts[2]]) / 2 + 0.5
        linear_norm = [0, 0.25, 0.5, 0.5, 0.5, 0.75, 1]
        trans_norm = [1, 1, 0, 0, 0, 1, 1]
        scale_pts = [-ctrl_pts[2], ctrl_pts[2]]
        idx = int(ticks[0] == 0)
        ticks = list(-np.array(ticks[idx:])[::-1]) + [0] + list(ticks[idx:])
    else:
        # remap ctrl_norm[0]->ctrl_norm[2] to 0->1
        ctrl_norm = [
            0, (ctrl_pts[1] - ctrl_pts[0]) / (ctrl_pts[2] - ctrl_pts[0]), 1]
        linear_norm = [0, 0.5, 1]
        trans_norm = [0, 1, 1]
        scale_pts = [ctrl_pts[0], ctrl_pts[2]]
    if linearize:  # matplotlib
        # do the piecewise linear transformation
        interp_to = np.linspace(0, 1, 256)
        colormap = np.array(colormap(
            np.interp(interp_to, ctrl_norm, linear_norm)))
        if transparent:
            colormap[:, 3] = np.interp(interp_to, ctrl_norm, trans_norm)
        assert len(scale_pts) == 2
        scale_pts = np.array([scale_pts[0], np.mean(scale_pts), scale_pts[1]])
        colormap = ListedColormap(colormap)
    else:  # mayavi / PySurfer will do the transformation for us
        scale_pts = ctrl_pts
    return colormap, scale_pts, diverging_lims, transparent, ticks


def _handle_time(time_label, time_unit, times):
    """Handle time label string and units."""
    if time_label == 'auto':
        if time_unit == 's':
            time_label = 'time=%0.3fs'
        elif time_unit == 'ms':
            time_label = 'time=%0.1fms'
    _, times = _check_time_unit(time_unit, times)
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
    from ..morph import _morph_buffer
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

    range_ = params['scale_pts'][2] - params['scale_pts'][0]
    colors = (array_plot - params['scale_pts'][0]) / range_

    faces = params['faces']
    greymap = params['greymap']
    cmap = params['cmap']
    polyc = ax.plot_trisurf(*params['coords'].T, triangles=faces,
                            antialiased=False, vmin=0, vmax=1)
    color_ave = np.mean(colors[faces], axis=1).flatten()
    curv_ave = np.mean(params['curv'][faces], axis=1).flatten()
    # matplotlib/matplotlib#11877
    facecolors = polyc._facecolors3d
    colors = cmap(color_ave)
    # alpha blend
    colors[:, :3] *= colors[:, [3]]
    colors[:, :3] += greymap(curv_ave)[:, :3] * (1. - colors[:, [3]])
    colors[:, 3] = 1.
    facecolors[:] = colors
    ax.set_title(params['time_label'] % (times[time_idx] * scaler), color='w')
    _set_aspect_equal(ax)
    ax.axis('off')
    ax.set(xlim=[-80, 80], ylim=(-80, 80), zlim=[-80, 80])
    ax.figure.canvas.draw()


def _plot_mpl_stc(stc, subject=None, surface='inflated', hemi='lh',
                  colormap='auto', time_label='auto', smoothing_steps=10,
                  subjects_dir=None, views='lat', clim='auto', figure=None,
                  initial_time=None, time_unit='s', background='black',
                  spacing='oct6', time_viewer=False, colorbar=True,
                  transparent=True):
    """Plot source estimate using mpl."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.widgets import Slider
    import nibabel as nib
    from scipy import stats
    from ..morph import _get_subject_sphere_tris
    if hemi not in ['lh', 'rh']:
        raise ValueError("hemi must be 'lh' or 'rh' when using matplotlib. "
                         "Got %s." % hemi)
    lh_kwargs = {'lat': {'elev': 0, 'azim': 180},
                 'med': {'elev': 0, 'azim': 0},
                 'ros': {'elev': 0, 'azim': 90},
                 'cau': {'elev': 0, 'azim': -90},
                 'dor': {'elev': 90, 'azim': -90},
                 'ven': {'elev': -90, 'azim': -90},
                 'fro': {'elev': 0, 'azim': 106.739},
                 'par': {'elev': 30, 'azim': -120}}
    rh_kwargs = {'lat': {'elev': 0, 'azim': 0},
                 'med': {'elev': 0, 'azim': 180},
                 'ros': {'elev': 0, 'azim': 90},
                 'cau': {'elev': 0, 'azim': -90},
                 'dor': {'elev': 90, 'azim': -90},
                 'ven': {'elev': -90, 'azim': -90},
                 'fro': {'elev': 16.739, 'azim': 60},
                 'par': {'elev': 30, 'azim': -60}}
    kwargs = dict(lh=lh_kwargs, rh=rh_kwargs)
    _check_option('views', views, sorted(lh_kwargs.keys()))
    colormap, scale_pts, _, _, _ = _limits_to_control_points(
        clim, stc.data, colormap, transparent, linearize=True)
    del transparent

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
    curv = np.clip(np.array(curv > 0, np.int), 0.33, 0.66)
    params = dict(ax=ax, stc=stc, coords=coords, faces=faces,
                  hemi_idx=hemi_idx, vertices=vertices, e=e,
                  smoothing_steps=smoothing_steps, n_verts=n_verts,
                  inuse=inuse, maps=maps, cmap=cmap, curv=curv,
                  scale_pts=scale_pts, greymap=greymap, time_label=time_label,
                  time_unit=time_unit)
    _smooth_plot(initial_time, params)

    ax.view_init(**kwargs[hemi][views])

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

    # add colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(scale_pts[0], scale_pts[2]))
    cax = inset_axes(ax, width="80%", height="5%", loc=8, borderpad=3.)
    plt.setp(plt.getp(cax, 'xticklabels'), color='w')
    sm.set_array(np.linspace(scale_pts[0], scale_pts[2], 256))
    if colorbar:
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb_yticks = plt.getp(cax, 'yticklabels')
        plt.setp(cb_yticks, color='w')
        cax.tick_params(labelsize=16)
        cb.patch.set_facecolor('0.5')
        cax.set(xlim=(scale_pts[0], scale_pts[2]))
    plt.show()
    return fig


@verbose
def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='auto', time_label='auto',
                          smoothing_steps=10, transparent=True, alpha=1.0,
                          time_viewer=False, subjects_dir=None, figure=None,
                          views='lat', colorbar=True, clim='auto',
                          cortex="classic", size=800, background="black",
                          foreground="white", initial_time=None,
                          time_unit='s', backend='auto', spacing='oct6',
                          title=None, verbose=None):
    """Plot SourceEstimate with PySurfer.

    By default this function uses :mod:`mayavi.mlab` to plot the source
    estimates. If Mayavi is not installed, the plotting is done with
    :mod:`matplotlib.pyplot` (much slower, decimated source space by default).

    Parameters
    ----------
    stc : SourceEstimate
        The source estimates to plot.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None stc.subject will be used. If that
        is None, the environment will be used.
    surface : str
        The type of surface (inflated, white etc.).
    hemi : str, 'lh' | 'rh' | 'split' | 'both'
        The hemisphere to display.
    %(colormap)s
        The default ('auto') uses 'hot' for one-sided data and
        'mne' for two-sided data.
    time_label : str | callable | None
        Format of the time label (a format string, a function that maps
        floating point time values to strings, or None for no label). The
        default is ``time=%%0.2f ms``.
    smoothing_steps : int
        The amount of smoothing
    %(transparent)s
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    time_viewer : bool
        Display time viewer GUI.
    %(subjects_dir)s
    figure : instance of mayavi.core.api.Scene | instance of matplotlib.figure.Figure | list | int | None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the Mayavi
        figure by it's id or create a new figure with the given id. If an
        instance of matplotlib figure, mpl backend is used for plotting.
    views : str | list
        View to use. See `surfer.Brain`. Supported views: ['lat', 'med', 'ros',
        'cau', 'dor' 'ven', 'fro', 'par']. Using multiple views is not
        supported for mpl backend.
    colorbar : bool
        If True, display colorbar on scene.
    %(clim)s
    cortex : str or tuple
        Specifies how binarized curvature values are rendered.
        Either the name of a preset PySurfer cortex colorscheme (one of
        'classic', 'bone', 'low_contrast', or 'high_contrast'), or the name of
        mayavi colormap, or a tuple with values (colormap, min, max, reverse)
        to fully specify the curvature colors. Has no effect with mpl backend.
    size : float or tuple of float
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
    title : str | None
        Title for the figure. If None, the subject name will be used.

        .. versionadded:: 0.17.0
    %(verbose)s

    Returns
    -------
    figure : instance of surfer.Brain | matplotlib.figure.Figure
        An instance of :class:`surfer.Brain` from PySurfer or
        matplotlib figure.
    """  # noqa: E501
    from .backends.renderer import get_3d_backend
    # import here to avoid circular import problem
    from ..source_estimate import SourceEstimate
    _validate_type(stc, SourceEstimate, "stc", "Surface Source Estimate")
    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    subject = _check_subject(stc.subject, subject, True)
    _check_option('backend', backend, ['auto', 'matplotlib', 'mayavi'])
    plot_mpl = backend == 'matplotlib'
    if not plot_mpl:
        try:
            from mayavi import mlab  # noqa: F401
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
                             spacing=spacing, time_viewer=time_viewer,
                             colorbar=colorbar, transparent=transparent)
    if get_3d_backend() == "mayavi":
        from surfer import Brain, TimeViewer
    else:
        from ._brain import _Brain as Brain
    _check_option('hemi', hemi, ['lh', 'rh', 'split', 'both'])

    time_label, times = _handle_time(time_label, time_unit, stc.times)
    # convert control points to locations in colormap
    colormap, scale_pts, diverging, transparent, _ = _limits_to_control_points(
        clim, stc.data, colormap, transparent)

    if hemi in ['both', 'split']:
        hemis = ['lh', 'rh']
    else:
        hemis = [hemi]

    if title is None:
        title = subject if len(hemis) > 1 else '%s - %s' % (subject, hemis[0])
    with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(subject, hemi=hemi, surf=surface,
                      title=title, cortex=cortex, size=size,
                      background=background, foreground=foreground,
                      figure=figure, subjects_dir=subjects_dir,
                      views=views)
    center = 0. if diverging else None
    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1
        data = getattr(stc, hemi + '_data')
        vertices = stc.vertices[hemi_idx]
        if len(data) > 0:
            kwargs = {
                "array": data, "colormap": colormap,
                "vertices": vertices,
                "smoothing_steps": smoothing_steps,
                "time": times, "time_label": time_label,
                "alpha": alpha, "hemi": hemi,
                "colorbar": colorbar, "initial_time": initial_time,
                "transparent": transparent, "center": center,
                "verbose": False
            }
            if get_3d_backend() == "mayavi":
                kwargs["min"] = scale_pts[0]
                kwargs["mid"] = scale_pts[1]
                kwargs["max"] = scale_pts[2]
            else:
                kwargs["fmin"] = scale_pts[0]
                kwargs["fmid"] = scale_pts[1]
                kwargs["fmax"] = scale_pts[2]
            with warnings.catch_warnings(record=True):  # traits warnings
                brain.add_data(**kwargs)
    if time_viewer:
        TimeViewer(brain)
    return brain


def _get_ps_kwargs(initial_time, diverging, mid, transparent):
    """Triage arguments based on PySurfer version."""
    import surfer
    surfer_version = LooseVersion(surfer.__version__)
    require = '0.8'
    if surfer_version < LooseVersion(require):
        raise ImportError("This function requires PySurfer %s (you are "
                          "running version %s). You can update PySurfer "
                          "using:\n\n    $ pip install -U pysurfer" %
                          (require, surfer.__version__))

    ad_kwargs = dict(verbose=False)
    sd_kwargs = dict(transparent=transparent, verbose=False)
    if initial_time is not None:
        ad_kwargs['initial_time'] = initial_time
    if surfer_version >= LooseVersion('0.9'):
        ad_kwargs.update(mid=mid, transparent=transparent)
        ad_kwargs['center'] = 0. if diverging else None
        sd_kwargs['center'] = 0. if diverging else None

    return ad_kwargs, sd_kwargs


def _glass_brain_crosshairs(params, x, y, z):
    for ax, a, b in ((params['ax_y'], x, z),
                     (params['ax_x'], y, z),
                     (params['ax_z'], x, y)):
        ax.axvline(a, color='0.75')
        ax.axhline(b, color='0.75')


def _cut_coords_to_ijk(cut_coords, img):
    ijk = apply_trans(linalg.inv(img.affine), cut_coords)
    ijk = np.clip(np.round(ijk).astype(int), 0, np.array(img.shape[:3]) - 1)
    return ijk


def _ijk_to_cut_coords(ijk, img):
    return apply_trans(img.affine, ijk)


@verbose
def plot_volume_source_estimates(stc, src, subject=None, subjects_dir=None,
                                 mode='stat_map', bg_img=None, colorbar=True,
                                 colormap='auto', clim='auto',
                                 transparent=None, show=True,
                                 initial_time=None, initial_pos=None,
                                 verbose=None):
    """Plot Nutmeg style volumetric source estimates using nilearn.

    Parameters
    ----------
    stc : VectorSourceEstimate
        The vector source estimate to plot.
    src : instance of SourceSpaces | instance of SourceMorph
        The source space. Can also be a SourceMorph to morph the STC to
        a new subject (see Examples).

        .. versionchanged:: 0.18
           Support for :class:`~nibabel.spatialimages.SpatialImage`.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None stc.subject will be used. If that
        is None, the environment will be used.
    %(subjects_dir)s
    mode : str
        The plotting mode to use. Either 'stat_map' (default) or 'glass_brain'.
        For "glass_brain", activation absolute values are displayed
        after being transformed to a standard MNI brain.
    bg_img : instance of SpatialImage | None
        The background image used in the nilearn plotting function.
        If None, it is the T1.mgz file that is found in the subjects_dir.
        Not used in "glass brain" plotting.
    colorbar : boolean, optional
        If True, display a colorbar on the right of the plots.
    %(colormap)s
    %(clim)s
    %(transparent)s
    show : bool
        Show figures if True. Defaults to True.
    initial_time : float | None
        The initial time to plot. Can be None (default) to use the time point
        with the maximal absolute value activation across all voxels
        or the ``initial_pos`` voxel (if ``initial_pos is None`` or not,
        respectively).

        .. versionadded:: 0.19
    initial_pos : ndarray, shape (3,) | None
        The initial position to use (in m). Can be None (default) to use the
        voxel with the maximum absolute value activation across all time points
        or at ``initial_time`` (if ``initial_time is None`` or not,
        respectively).

        .. versionadded:: 0.19
    %(verbose)s

    Notes
    -----
    Click on any of the anatomical slices to explore the time series.
    Clicking on any time point will bring up the corresponding anatomical map.

    The left and right arrow keys can be used to navigate in time.
    To move in time by larger steps, use shift+left and shift+right.

    In ``'glass_brain'`` mode, values are transformed to the standard MNI
    brain using the FreeSurfer Talairach transformation
    ``$SUBJECTS_DIR/$SUBJECT/mri/transforms/talairach.xfm``.

    .. versionadded:: 0.17

    .. versionchanged:: 0.19
       MRI volumes are automatically transformed to MNI space in
       ``'glass_brain'`` mode.

    Examples
    --------
    Passing a :class:`mne.SourceMorph` as the ``src``
    parameter can be useful for plotting in a different subject's space
    (here, a ``'sample'`` STC in ``'fsaverage'``'s space)::

    >>> morph = mne.compute_source_morph(src_sample, subject_to='fsaverage')  # doctest: +SKIP
    >>> fig = stc_vol_sample.plot(morph)  # doctest: +SKIP

    """  # noqa: E501
    from matplotlib import pyplot as plt, colors
    from matplotlib.cbook import mplDeprecation
    import nibabel as nib
    from ..source_estimate import VolSourceEstimate
    from ..morph import SourceMorph

    if not check_version('nilearn', '0.4'):
        raise RuntimeError('This function requires nilearn >= 0.4')

    from nilearn.plotting import plot_stat_map, plot_glass_brain
    from nilearn.image import index_img

    _check_option('mode', mode, ('stat_map', 'glass_brain'))
    plot_func = dict(stat_map=plot_stat_map,
                     glass_brain=plot_glass_brain)[mode]
    _validate_type(stc, VolSourceEstimate, 'stc')
    if isinstance(src, SourceMorph):
        img = src.apply(stc, 'nifti1', mri_resolution=False, mri_space=False)
        stc = src.apply(stc, mri_resolution=False, mri_space=False)
        kind, src_subject = 'morph.subject_to', src.subject_to
    else:
        src = _ensure_src(src, kind='volume', extra=' or SourceMorph')
        img = stc.as_volume(src, mri_resolution=False)
        kind, src_subject = 'src subject', src._subject
    _print_coord_trans(Transform('mri_voxel', 'ras', img.affine),
                       prefix='Image affine ', units='mm', level='debug')
    subject = _check_subject(src_subject, subject, True, kind=kind)
    stc_ijk = np.array(
        np.unravel_index(stc.vertices, img.shape[:3], order='F')).T
    assert stc_ijk.shape == (len(stc.vertices), 3)
    del src, kind

    # XXX this assumes zooms are uniform, should probably mult by zooms...
    dist_to_verts = _DistanceQuery(stc_ijk, allow_kdtree=True)

    def _cut_coords_to_idx(cut_coords, img):
        """Convert voxel coordinates to index in stc.data."""
        ijk = _cut_coords_to_ijk(cut_coords, img)
        del cut_coords
        logger.debug('    Affine remapped cut coords to [%d, %d, %d] idx'
                     % tuple(ijk))
        dist, loc_idx = dist_to_verts.query(ijk[np.newaxis])
        dist, loc_idx = dist[0], loc_idx[0]
        logger.debug('    Using vertex %d at a distance of %d voxels'
                     % (stc.vertices[loc_idx], dist))
        return loc_idx

    ax_name = dict(x='X (saggital)', y='Y (coronal)', z='Z (axial)')

    def _click_to_cut_coords(event, params):
        """Get voxel coordinates from mouse click."""
        if event.inaxes is params['ax_x']:
            ax = 'x'
            x = params['ax_z'].lines[0].get_xdata()[0]
            y, z = event.xdata, event.ydata
        elif event.inaxes is params['ax_y']:
            ax = 'y'
            y = params['ax_x'].lines[0].get_xdata()[0]
            x, z = event.xdata, event.ydata
        elif event.inaxes is params['ax_z']:
            ax = 'z'
            x, y = event.xdata, event.ydata
            z = params['ax_x'].lines[1].get_ydata()[0]
        else:
            logger.debug('    Click outside axes')
            return None
        cut_coords = np.array((x, y, z))
        logger.debug('')

        if params['mode'] == 'glass_brain':  # find idx for MIP
            # Figure out what XYZ in world coordinates is in our voxel data
            codes = ''.join(nib.aff2axcodes(params['img_idx'].affine))
            assert len(codes) == 3
            # We don't care about directionality, just which is which dim
            codes = codes.replace('L', 'R').replace('P', 'A').replace('I', 'S')
            idx = codes.index(dict(x='R', y='A', z='S')[ax])
            img_data = np.abs(_get_img_fdata(params['img_idx']))
            ijk = _cut_coords_to_ijk(cut_coords, params['img_idx'])
            if idx == 0:
                ijk[0] = np.argmax(img_data[:, ijk[1], ijk[2]])
                logger.debug('    MIP: i = %d idx' % (ijk[0],))
            elif idx == 1:
                ijk[1] = np.argmax(img_data[ijk[0], :, ijk[2]])
                logger.debug('    MIP: j = %d idx' % (ijk[1],))
            else:
                ijk[2] = np.argmax(img_data[ijk[0], ijk[1], :])
                logger.debug('    MIP: k = %d idx' % (ijk[2],))
            cut_coords = _ijk_to_cut_coords(ijk, params['img_idx'])

        logger.debug('    Cut coords for %s: (%0.1f, %0.1f, %0.1f) mm'
                     % ((ax_name[ax],) + tuple(cut_coords)))
        return cut_coords

    def _press(event, params):
        """Manage keypress on the plot."""
        pos = params['lx'].get_xdata()
        idx = params['stc'].time_as_index(pos)[0]
        if event.key == 'left':
            idx = max(0, idx - 2)
        elif event.key == 'shift+left':
            idx = max(0, idx - 10)
        elif event.key == 'right':
            idx = min(params['stc'].shape[1] - 1, idx + 2)
        elif event.key == 'shift+right':
            idx = min(params['stc'].shape[1] - 1, idx + 10)
        _update_timeslice(idx, params)
        params['fig'].canvas.draw()

    def _update_timeslice(idx, params):
        params['lx'].set_xdata(idx / params['stc'].sfreq +
                               params['stc'].tmin)
        ax_x, ax_y, ax_z = params['ax_x'], params['ax_y'], params['ax_z']
        plot_map_callback = params['plot_func']
        # Crosshairs are the first thing plotted in stat_map, and the last
        # in glass_brain
        idxs = [0, 0, 1] if mode == 'stat_map' else [-2, -2, -1]
        cut_coords = (
            ax_y.lines[idxs[0]].get_xdata()[0],
            ax_x.lines[idxs[1]].get_xdata()[0],
            ax_x.lines[idxs[2]].get_ydata()[0])
        ax_x.clear()
        ax_y.clear()
        ax_z.clear()
        params.update({'img_idx': index_img(img, idx)})
        params.update({'title': 'Activation (t=%.3f s.)'
                       % params['stc'].times[idx]})
        plot_map_callback(
            params['img_idx'], title='', cut_coords=cut_coords)

    @verbose_dec
    def _onclick(event, params, verbose=None):
        """Manage clicks on the plot."""
        ax_x, ax_y, ax_z = params['ax_x'], params['ax_y'], params['ax_z']
        plot_map_callback = params['plot_func']
        if event.inaxes is params['ax_time']:
            idx = params['stc'].time_as_index(
                event.xdata, use_rounding=True)[0]
            _update_timeslice(idx, params)

        cut_coords = _click_to_cut_coords(event, params)
        if cut_coords is None:
            return  # not in any axes

        ax_x.clear()
        ax_y.clear()
        ax_z.clear()
        plot_map_callback(params['img_idx'], title='',
                          cut_coords=cut_coords)
        loc_idx = _cut_coords_to_idx(cut_coords, params['img_idx'])
        ydata = stc.data[loc_idx]
        if loc_idx is not None:
            ax_time.lines[0].set_ydata(ydata)
        else:
            ax_time.lines[0].set_ydata(0.)
        params['fig'].canvas.draw()

    if mode == 'glass_brain':
        subject = _check_subject(stc.subject, subject, True)
        ras_mni_t = read_ras_mni_t(subject, subjects_dir)
        if not np.allclose(ras_mni_t['trans'], np.eye(4)):
            _print_coord_trans(
                ras_mni_t, prefix='Transforming subject ', units='mm')
            logger.info('')
            # To get from voxel coords to world coords (i.e., define affine)
            # we would apply img.affine, then also apply ras_mni_t, which
            # transforms from the subject's RAS to MNI RAS. So we left-multiply
            # these.
            img = nib.Nifti1Image(
                img.dataobj, np.dot(ras_mni_t['trans'], img.affine))
        bg_img = None  # not used
    else:  # stat_map
        if bg_img is None:
            subject = _check_subject(stc.subject, subject, True)
            subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                            raise_error=True)
            t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
            bg_img = nib.load(t1_fname)

    if initial_time is None:
        time_sl = slice(0, None)
    else:
        initial_time = float(initial_time)
        logger.info('Fixing initial time: %s sec' % (initial_time,))
        initial_time = np.argmin(np.abs(stc.times - initial_time))
        time_sl = slice(initial_time, initial_time + 1)
    if initial_pos is None:  # find max pos and (maybe) time
        loc_idx, time_idx = np.unravel_index(
            np.abs(stc.data[:, time_sl]).argmax(), stc.data[:, time_sl].shape)
        time_idx += time_sl.start
    else:  # position specified
        initial_pos = np.array(initial_pos, float)
        if initial_pos.shape != (3,):
            raise ValueError('initial_pos must be float ndarray with shape '
                             '(3,), got shape %s' % (initial_pos.shape,))
        initial_pos *= 1000
        logger.info('Fixing initial position: %s mm'
                    % (initial_pos.tolist(),))
        loc_idx = _cut_coords_to_idx(initial_pos, img)
        if initial_time is not None:  # time also specified
            time_idx = time_sl.start
        else:  # find the max
            time_idx = np.argmax(np.abs(stc.data[loc_idx]))
    img_idx = index_img(img, time_idx)
    assert img_idx.shape == img.shape[:3]
    del initial_time, initial_pos
    ijk = stc_ijk[loc_idx]
    cut_coords = _ijk_to_cut_coords(ijk, img_idx)
    np.testing.assert_allclose(_cut_coords_to_ijk(cut_coords, img_idx), ijk)
    logger.info('Showing: t = %0.3f s, (%0.1f, %0.1f, %0.1f) mm, '
                '[%d, %d, %d] vox, %d vertex'
                % ((stc.times[time_idx],) + tuple(cut_coords) + tuple(ijk) +
                   (stc.vertices[loc_idx],)))
    del ijk

    # Plot initial figure
    fig, (axes, ax_time) = plt.subplots(2)
    axes.set(xticks=[], yticks=[])
    marker = 'o' if len(stc.times) == 1 else None
    ydata = stc.data[loc_idx]
    ax_time.plot(stc.times, ydata, color='k', marker=marker)
    if len(stc.times) > 1:
        ax_time.set(xlim=stc.times[[0, -1]])
    ax_time.set(xlabel='Time (s)', ylabel='Activation')
    lx = ax_time.axvline(stc.times[time_idx], color='g')
    fig.tight_layout()

    allow_pos_lims = (mode != 'glass_brain')
    colormap, scale_pts, diverging, _, ticks = _limits_to_control_points(
        clim, stc.data, colormap, transparent, allow_pos_lims, linearize=True)
    ylim = [min((scale_pts[0], ydata.min())),
            max((scale_pts[-1], ydata.max()))]
    ylim = np.array(ylim) + np.array([-1, 1]) * 0.05 * np.diff(ylim)[0]
    dup_neg = False
    if stc.data.min() < 0:
        ax_time.axhline(0., color='0.5', ls='-', lw=0.5, zorder=2)
        dup_neg = not diverging  # glass brain with signed data
    yticks = list(ticks)
    if dup_neg:
        yticks += [0] + list(-np.array(ticks))
    yticks = np.unique(yticks)
    ax_time.set(yticks=yticks)
    ax_time.set(ylim=ylim)
    del yticks

    if not diverging:  # set eq above iff one-sided
        # there is a bug in nilearn where this messes w/transparency
        # Need to double the colormap
        if (scale_pts < 0).any():
            # XXX We should fix this, but it's hard to get nilearn to
            # use arbitrary bounds :(
            # Should get them to support non-mirrored colorbars, or
            # at least a proper `vmin` for one-sided things.
            # Hopefully this is a sufficiently rare use case!
            raise ValueError('Negative colormap limits for sequential '
                             'control points clim["lims"] not supported '
                             'currently, consider shifting or flipping the '
                             'sign of your data for visualization purposes')
        # due to nilearn plotting weirdness, extend this to go
        # -scale_pts[2]->scale_pts[2] instead of scale_pts[0]->scale_pts[2]
        colormap = plt.get_cmap(colormap)
        colormap = colormap(
            np.interp(np.linspace(-1, 1, 256),
                      scale_pts / scale_pts[2],
                      [0, 0.5, 1]))
        colormap = colors.ListedColormap(colormap)
    vmax = scale_pts[-1]

    # black_bg = True is needed because of some matplotlib
    # peculiarity. See: https://stackoverflow.com/a/34730204
    # Otherwise, event.inaxes does not work for ax_x and ax_z
    plot_kwargs = dict(
        threshold=None, axes=axes,
        resampling_interpolation='nearest', vmax=vmax, figure=fig,
        colorbar=colorbar, bg_img=bg_img, cmap=colormap, black_bg=True,
        symmetric_cbar=True)

    def plot_and_correct(*args, **kwargs):
        axes.clear()
        if params.get('fig_anat') is not None and plot_kwargs['colorbar']:
            params['fig_anat']._cbar.ax.clear()
        with warnings.catch_warnings(record=True):  # nilearn bug; ax recreated
            warnings.simplefilter('ignore', mplDeprecation)
            params['fig_anat'] = partial(
                plot_func, **plot_kwargs)(*args, **kwargs)
        params['fig_anat']._cbar.outline.set_visible(False)
        for key in 'xyz':
            params.update({'ax_' + key: params['fig_anat'].axes[key].ax})
        # Fix nilearn bug w/cbar background being white
        if plot_kwargs['colorbar']:
            params['fig_anat']._cbar.patch.set_facecolor('0.5')
            # adjust one-sided colorbars
            if not diverging:
                _crop_colorbar(params['fig_anat']._cbar, *scale_pts[[0, -1]])
            params['fig_anat']._cbar.set_ticks(params['cbar_ticks'])
        if mode == 'glass_brain':
            _glass_brain_crosshairs(params, *kwargs['cut_coords'])

    params = dict(stc=stc, ax_time=ax_time, plot_func=plot_and_correct,
                  img_idx=img_idx, fig=fig, lx=lx, mode=mode, cbar_ticks=ticks)

    plot_and_correct(stat_map_img=params['img_idx'], title='',
                     cut_coords=cut_coords)

    if show:
        plt.show()
    fig.canvas.mpl_connect('button_press_event',
                           partial(_onclick, params=params, verbose=verbose))
    fig.canvas.mpl_connect('key_press_event',
                           partial(_press, params=params))

    return fig


@fill_doc
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
    """Plot VectorSourceEstimate with PySurfer.

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
    %(colormap)s
        This should be a sequential colormap.
    time_label : str | callable | None
        Format of the time label (a format string, a function that maps
        floating point time values to strings, or None for no label). The
        default is ``time=%%0.2f ms``.
    smoothing_steps : int
        The amount of smoothing
    %(transparent)s
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
    figure : instance of mayavi.core.api.Scene | list | int | None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the Mayavi
        figure by it's id or create a new figure with the given id.
    views : str | list
        View to use. See `surfer.Brain`.
    colorbar : bool
        If True, display colorbar on scene.
    %(clim_onesided)s
    cortex : str or tuple
        specifies how binarized curvature values are rendered.
        either the name of a preset PySurfer cortex colorscheme (one of
        'classic', 'bone', 'low_contrast', or 'high_contrast'), or the
        name of mayavi colormap, or a tuple with values (colormap, min,
        max, reverse) to fully specify the curvature colors.
    size : float or tuple of float
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
    brain : surfer.Brain
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

    _validate_type(stc, VectorSourceEstimate, "stc", "Vector Source Estimate")
    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    subject = _check_subject(stc.subject, subject, True)
    _check_option('hemi', hemi, ['lh', 'rh', 'split', 'both'])
    time_label, times = _handle_time(time_label, time_unit, stc.times)

    # convert control points to locations in colormap
    colormap, scale_pts, _, transparent, _ = _limits_to_control_points(
        clim, stc.data, colormap, transparent, allow_pos_lims=False)

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
        brain = Brain(subject, hemi=hemi, surf='white',
                      title=title, cortex=cortex, size=size,
                      background=background, foreground=foreground,
                      figure=figure, subjects_dir=subjects_dir,
                      views=views, alpha=brain_alpha)

    ad_kwargs, sd_kwargs = _get_ps_kwargs(
        initial_time, False, scale_pts[1], transparent)
    del initial_time, transparent
    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1
        data = getattr(stc, hemi + '_data')
        vertices = stc.vertices[hemi_idx]
        if len(data) > 0:
            with warnings.catch_warnings(record=True):  # traits warnings
                brain.add_data(data, colormap=colormap, vertices=vertices,
                               smoothing_steps=smoothing_steps, time=times,
                               time_label=time_label, alpha=overlay_alpha,
                               hemi=hemi, colorbar=colorbar,
                               vector_alpha=vector_alpha,
                               scale_factor=scale_factor,
                               min=scale_pts[0], max=scale_pts[2],
                               **ad_kwargs)
        brain.scale_data_colormap(fmin=scale_pts[0], fmid=scale_pts[1],
                                  fmax=scale_pts[2], **sd_kwargs)

    if time_viewer:
        TimeViewer(brain)

    return brain


@verbose
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
    fig_name : str
        Mayavi figure name.
    fig_number : int
        Matplotlib figure number.
    labels : ndarray or list of ndarray
        Labels to show sources in clusters. Sources with the same
        label and the waveforms within each cluster are presented in
        the same color. labels should be a list of ndarrays when
        stcs is a list ie. one label for each stc.
    modes : list
        Should be a list, with each entry being ``'cone'`` or ``'sphere'``
        to specify how the dipoles should be shown.
    scale_factors : list
        List of floating point scale factors for the markers.
    %(verbose)s
    **kwargs : kwargs
        Keyword arguments to pass to mlab.triangular_mesh.

    Returns
    -------
    surface : instance of mayavi.mlab.pipeline.surface
        The triangular mesh surface.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ColorConverter
    # Update the backend
    from .backends.renderer import _Renderer

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
        colors = _get_color_list()

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

    renderer = _Renderer(bgcolor=bgcolor, size=(600, 600), name=fig_name)
    surface = renderer.mesh(x=points[:, 0], y=points[:, 1],
                            z=points[:, 2], triangles=use_faces,
                            color=brain_color, opacity=opacity,
                            backface_culling=True, shading=True,
                            **kwargs)

    # Show time courses
    fig = plt.figure(fig_number)
    fig.clf()
    ax = fig.add_subplot(111)

    colors = cycle(colors)

    logger.info("Total number of active sources: %d" % len(unique_vertnos))

    if labels is not None:
        colors = [next(colors) for _ in
                  range(np.unique(np.concatenate(labels).ravel()).size)]

    for idx, v in enumerate(unique_vertnos):
        # get indices of stcs it belongs to
        ind = [k for k, vertno in enumerate(vertnos) if v in vertno]
        is_common = len(ind) > 1

        if labels is None:
            c = next(colors)
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
        renderer.quiver3d(x=x, y=y, z=z, u=nx, v=ny, w=nz,
                          color=color_converter.to_rgb(c),
                          mode=mode, scale=scale_factor)

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

    renderer.show()
    return surface


@verbose
def plot_dipole_locations(dipoles, trans=None, subject=None, subjects_dir=None,
                          mode='orthoview', coord_frame='mri', idx='gof',
                          show_all=True, ax=None, block=False, show=True,
                          scale=5e-3, color=None, highlight_color='r',
                          fig=None, verbose=None):
    """Plot dipole locations.

    If mode is set to 'arrow' or 'sphere', only the location of the first
    time point of each dipole is shown else use the show_all parameter.

    The option mode='orthoview' was added in version 0.14.

    Parameters
    ----------
    dipoles : list of instances of Dipole | Dipole
        The dipoles to plot.
    trans : dict | None
        The mri to head trans.
        Can be None with mode set to '3d'.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT.
        Can be None with mode set to '3d'.
    subjects_dir : None | str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
        The default is None.
    mode : str
        Can be ``'arrow'``, ``'sphere'`` or ``'orthoview'``.

        .. versionadded:: 0.19.0
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
    scale: float
        The scale of the dipoles if ``mode`` is 'arrow' or 'sphere'.
    color : tuple
        The color of the dipoles.
        The default (None) will use ``'y'`` if mode is ``'orthoview'`` and
        ``show_all`` is True, else 'r'.

        .. versionchanged:: 0.19.0
           Color is now passed in orthoview mode.
    highlight_color : color
        The highlight color. Only used in orthoview mode with
        ``show_all=True``.

        .. versionadded:: 0.19.0
    fig : mayavi.mlab.Figure | None
        3D Scene in which to plot the alignment.
        If ``None``, creates a new 600x600 pixel figure with black background.

        .. versionadded:: 0.19.0
    %(verbose)s

    Returns
    -------
    fig : instance of mayavi.mlab.Figure or matplotlib.figure.Figure
        The mayavi figure or matplotlib Figure.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    if mode == 'orthoview':
        fig = _plot_dipole_mri_orthoview(
            dipoles, trans=trans, subject=subject, subjects_dir=subjects_dir,
            coord_frame=coord_frame, idx=idx, show_all=show_all,
            ax=ax, block=block, show=show, color=color,
            highlight_color=highlight_color)
    elif mode in ['arrow', 'sphere']:
        from .backends.renderer import _Renderer
        color = (1., 0., 0.) if color is None else color
        renderer = _Renderer(fig=fig, size=(600, 600))
        pos = dipoles.pos
        ori = dipoles.ori
        if coord_frame != 'head':
            trans = _get_trans(trans, fro='head', to=coord_frame)[0]
            pos = apply_trans(trans, pos)
            ori = apply_trans(trans, ori)

        renderer.sphere(center=pos, color=color, scale=scale)
        if mode == 'arrow':
            x, y, z = pos.T
            u, v, w = ori.T
            renderer.quiver3d(x, y, z, u, v, w, scale=3 * scale,
                              color=color, mode='arrow')

        fig = renderer.scene()
    else:
        raise ValueError('Mode must be "cone", "arrow" or orthoview", '
                         'got %s.' % (mode,))

    return fig


def snapshot_brain_montage(fig, montage, hide_sensors=True):
    """Take a snapshot of a Mayavi Scene and project channels onto 2d coords.

    Note that this will take the raw values for 3d coordinates of each channel,
    without applying any transforms. If brain images are flipped up/dn upon
    using `imshow`, check your matplotlib backend as this behavior changes.

    Parameters
    ----------
    fig : instance of ~mayavi.core.api.Scene
        The figure on which you've plotted electrodes using
        :func:`mne.viz.plot_alignment`.
    montage : instance of DigMontage or Info | dict
        The digital montage for the electrodes plotted in the scene. If `Info`,
        channel positions will be pulled from the `loc` field of `chs`.
        dict should have ch:xyz mappings.
    hide_sensors : bool
        Whether to remove the spheres in the scene before taking a snapshot.

    Returns
    -------
    xy : array, shape (n_channels, 2)
        The 2d location of each channel on the image of the current scene view.
    im : array, shape (m, n, 3)
        The screenshot of the current scene view
    """
    from ..channels import DigMontage
    from .. import Info
    # Update the backend
    from .backends.renderer import _Renderer

    if fig is None:
        raise ValueError('The figure must have a scene')
    if isinstance(montage, DigMontage):
        chs = montage._get_ch_pos()
        ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
    elif isinstance(montage, Info):
        xyz = [ich['loc'][:3] for ich in montage['chs']]
        ch_names = [ich['ch_name'] for ich in montage['chs']]
    elif isinstance(montage, dict):
        if not all(len(ii) == 3 for ii in montage.values()):
            raise ValueError('All electrode positions must be length 3')
        ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in montage.items()])
    else:
        raise TypeError('montage must be an instance of `DigMontage`, `Info`,'
                        ' or `dict`')

    # initialize figure
    renderer = _Renderer(fig, show=True)

    xyz = np.vstack(xyz)
    proj = renderer.project(xyz=xyz, ch_names=ch_names)
    if hide_sensors is True:
        proj.visible(False)

    im = renderer.screenshot()
    proj.visible(True)
    return proj.xy, im


@fill_doc
def plot_sensors_connectivity(info, con, picks=None):
    """Visualize the sensor connectivity in 3D.

    Parameters
    ----------
    info : dict | None
        The measurement info.
    con: array, shape (n_channels, n_channels)
        The computed connectivity measure(s).
    %(picks_good_data)s
        Indices of selected channels.

    Returns
    -------
    fig : instance of mayavi.mlab.Figure
        The mayavi figure.
    """
    _validate_type(info, "info")

    from .backends.renderer import _Renderer

    renderer = _Renderer(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

    picks = _picks_to_idx(info, picks)
    if len(picks) != len(con):
        raise ValueError('The number of channels picked (%s) does not '
                         'correspond the size of the connectivity data '
                         '(%s)' % (len(picks), len(con)))

    # Plot the sensor locations
    sens_loc = [info['chs'][k]['loc'][:3] for k in picks]
    sens_loc = np.array(sens_loc)

    renderer.sphere(np.c_[sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2]],
                    color=(1, 1, 1), opacity=1, scale=0.005)

    # Get the strongest connections
    n_con = 20  # show up to 20 connections
    min_dist = 0.05  # exclude sensors that are less than 5cm apart
    threshold = np.sort(con, axis=None)[-n_con]
    ii, jj = np.where(con >= threshold)

    # Remove close connections
    con_nodes = list()
    con_val = list()
    for i, j in zip(ii, jj):
        if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
            con_nodes.append((i, j))
            con_val.append(con[i, j])

    con_val = np.array(con_val)

    # Show the connections as tubes between sensors
    vmax = np.max(con_val)
    vmin = np.min(con_val)
    for val, nodes in zip(con_val, con_nodes):
        x1, y1, z1 = sens_loc[nodes[0]]
        x2, y2, z2 = sens_loc[nodes[1]]
        tube = renderer.tube(origin=np.c_[x1, y1, z1],
                             destination=np.c_[x2, y2, z2],
                             scalars=np.c_[val, val],
                             vmin=vmin, vmax=vmax, radius=0.001,
                             reverse_lut=True)

    renderer.scalarbar(source=tube, title='Phase Lag Index (PLI)')

    # Add the sensor names for the connections shown
    nodes_shown = list(set([n[0] for n in con_nodes] +
                           [n[1] for n in con_nodes]))

    for node in nodes_shown:
        x, y, z = sens_loc[node]
        renderer.text3d(x, y, z, text=info['ch_names'][picks[node]],
                        scale=0.005,
                        color=(0, 0, 0))

    renderer.set_camera(azimuth=-88.7, elevation=40.8,
                        distance=0.76,
                        focalpoint=np.array([-3.9e-4, -8.5e-3, -1e-2]))
    renderer.show()
    return renderer.scene()


def _plot_dipole_mri_orthoview(dipole, trans, subject, subjects_dir=None,
                               coord_frame='head', idx='gof', show_all=True,
                               ax=None, block=False, show=True, color=None,
                               highlight_color='r'):
    """Plot dipoles on top of MRI slices in 3-D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from .. import Dipole
    if not has_nibabel():
        raise ImportError('This function requires nibabel.')

    _check_option('coord_frame', coord_frame, ['head', 'mri'])

    if not isinstance(dipole, Dipole):
        from ..dipole import _concatenate_dipoles
        dipole = _concatenate_dipoles(dipole)
    if idx == 'gof':
        idx = np.argmax(dipole.gof)
    elif idx == 'amplitude':
        idx = np.argmax(np.abs(dipole.amplitude))
    else:
        idx = _ensure_int(idx, 'idx', 'an int or one of ["gof", "amplitude"]')

    vox, ori, pos, data = _get_dipole_loc(
        dipole, trans, subject, subjects_dir, coord_frame)

    dims = len(data)  # Symmetric size assumed.
    dd = dims // 2
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        _validate_type(ax, Axes3D, "ax", "Axes3D")
        fig = ax.get_figure()

    gridx, gridy = np.meshgrid(np.linspace(-dd, dd, dims),
                               np.linspace(-dd, dd, dims), indexing='ij')
    params = {'ax': ax, 'data': data, 'idx': idx, 'dipole': dipole,
              'vox': vox, 'gridx': gridx, 'gridy': gridy,
              'ori': ori, 'coord_frame': coord_frame,
              'show_all': show_all, 'pos': pos,
              'color': color, 'highlight_color': highlight_color}
    _plot_dipole(**params)
    ax.view_init(elev=30, azim=-140)

    callback_func = partial(_dipole_changed, params=params)
    fig.canvas.mpl_connect('scroll_event', callback_func)
    fig.canvas.mpl_connect('key_press_event', callback_func)

    plt_show(show, block=block)
    return fig


RAS_AFFINE = np.eye(4)
RAS_AFFINE[:3, 3] = [-128] * 3
RAS_SHAPE = (256, 256, 256)


def _get_dipole_loc(dipole, trans, subject, subjects_dir, coord_frame):
    """Get the dipole locations and orientations."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    _check_option('coord_frame', coord_frame, ['head', 'mri'])

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    t1 = nib.load(t1_fname)
    # Do everything in mm here to make life slightly easier
    vox_ras_t, _, mri_ras_t, _, _ = _read_mri_info(
        t1_fname, units='mm')
    head_mri_t = _get_trans(trans, fro='head', to='mri')[0].copy()
    head_mri_t['trans'][:3, 3] *= 1000  # m→mm
    del trans
    pos = dipole.pos * 1e3  # m→mm
    ori = dipole.ori
    # Figure out how to always resample to an identity, 256x256x256 RAS:
    #
    # 1. Resample to head or MRI surface RAS (the conditional), but also
    # 2. Resample to what will work for the standard 1mm** RAS_AFFINE (resamp)
    #
    # We could do this with two resample_from_to calls, but it's cleaner,
    # faster, and we get fewer boundary artifacts if we do it in one shot.
    # So first olve usamp s.t. ``upsamp @ vox_ras_t == RAS_AFFINE``` (2):
    upsamp = np.linalg.solve(vox_ras_t['trans'].T, RAS_AFFINE.T).T
    # Now figure out how we would resample from RAS to head or MRI coords:
    if coord_frame == 'head':
        dest_ras_t = combine_transforms(
            head_mri_t, mri_ras_t, 'head', 'ras')['trans']
    else:
        pos = apply_trans(head_mri_t, pos)
        ori = apply_trans(head_mri_t, dipole.ori, move=False)
        dest_ras_t = mri_ras_t['trans']
    # The order here is wacky because we need `resample_from_to` to operate
    # in a reverse order
    affine = np.dot(np.dot(dest_ras_t, upsamp), vox_ras_t['trans'])
    t1 = resample_from_to(t1, (RAS_SHAPE, affine), order=0)
    # Now we could do:
    #
    #    t1 = SpatialImage(t1.dataobj, RAS_AFFINE)
    #
    # And t1 would be in our destination (mri or head) space. But we don't
    # need to construct the image -- let's just get our voxel coords and data:
    vox = apply_trans(np.linalg.inv(RAS_AFFINE), pos)
    t1_data = _get_img_fdata(t1)
    return vox, ori, pos, t1_data


def _plot_dipole(ax, data, vox, idx, dipole, gridx, gridy, ori, coord_frame,
                 show_all, pos, color, highlight_color):
    """Plot dipoles."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ColorConverter
    color_converter = ColorConverter()
    xidx, yidx, zidx = np.round(vox[idx]).astype(int)
    xslice = data[xidx]
    yslice = data[:, yidx]
    zslice = data[:, :, zidx]

    ori = ori[idx]
    if color is None:
        color = 'y' if show_all else 'r'
    color = np.array(color_converter.to_rgba(color))
    highlight_color = np.array(color_converter.to_rgba(highlight_color))
    if show_all:
        colors = np.repeat(color[np.newaxis], len(vox), axis=0)
        colors[idx] = highlight_color
        size = np.repeat(5, len(vox))
        size[idx] = 20
        visible = np.arange(len(vox))
    else:
        colors = color
        size = 20
        visible = idx

    offset = np.min(gridx)
    xyz = pos
    ax.scatter(xs=xyz[visible, 0], ys=xyz[visible, 1],
               zs=xyz[visible, 2], zorder=2, s=size, facecolor=colors)
    xx = np.linspace(offset, xyz[idx, 0], xidx)
    yy = np.linspace(offset, xyz[idx, 1], yidx)
    zz = np.linspace(offset, xyz[idx, 2], zidx)
    ax.plot(xx, np.repeat(xyz[idx, 1], len(xx)), zs=xyz[idx, 2], zorder=1,
            linestyle='-', color=highlight_color)
    ax.plot(np.repeat(xyz[idx, 0], len(yy)), yy, zs=xyz[idx, 2], zorder=1,
            linestyle='-', color=highlight_color)
    ax.plot(np.repeat(xyz[idx, 0], len(zz)),
            np.repeat(xyz[idx, 1], len(zz)), zs=zz, zorder=1,
            linestyle='-', color=highlight_color)
    ax.quiver(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], ori[0], ori[1],
              ori[2], length=50, color=highlight_color,
              pivot='tail')
    dims = np.array([(len(data) / -2.), (len(data) / 2.)])
    ax.set(xlim=-dims, ylim=-dims, zlim=dims)

    # Plot slices.
    ax.contourf(xslice, gridx, gridy, offset=offset, zdir='x',
                cmap='gray', zorder=0, alpha=.5)
    ax.contourf(gridx, yslice, gridy, offset=offset, zdir='y',
                cmap='gray', zorder=0, alpha=.5)
    ax.contourf(gridx, gridy, zslice, offset=offset, zdir='z',
                cmap='gray', zorder=0, alpha=.5)

    # These are the only two options
    coord_frame_name = 'Head' if coord_frame == 'head' else 'MRI'
    plt.suptitle('Dipole #%s / %s @ %.3fs, GOF: %.1f%%, %.1fnAm\n%s: ' % (
        idx + 1, len(dipole.times), dipole.times[idx], dipole.gof[idx],
        dipole.amplitude[idx] * 1e9, coord_frame_name) +
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
    _plot_dipole(**params)


def _update_coord_frame(obj, rr, nn, mri_trans, head_trans):
    if obj['coord_frame'] == FIFF.FIFFV_COORD_MRI:
        rr = apply_trans(mri_trans, rr)
        nn = apply_trans(mri_trans, nn, move=False)
    elif obj['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
        rr = apply_trans(head_trans, rr)
        nn = apply_trans(head_trans, nn, move=False)
    return rr, nn


@fill_doc
def plot_brain_colorbar(ax, clim, colormap='auto', transparent=True,
                        orientation='vertical', label='Activation',
                        bgcolor='0.5'):
    """Plot a colorbar that corresponds to a brain activation map.

    Parameters
    ----------
    ax : instance of Axes
        The Axes to plot into.
    %(clim)s
    %(colormap)s
    %(transparent)s
    orientation : str
        Orientation of the colorbar, can be "vertical" or "horizontal".
    label : str
        The colorbar label.
    bgcolor : color
        The color behind the colorbar (for alpha blending).

    Returns
    -------
    cbar : instance of ColorbarBase
        The colorbar.

    Notes
    -----
    .. versionadded:: 0.19
    """
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    cmap, scale_pts, diverging, _, ticks = _limits_to_control_points(
        clim, 0., colormap, transparent=True, linearize=True)
    norm = Normalize(vmin=scale_pts[0], vmax=scale_pts[-1])
    cbar = ColorbarBase(ax, cmap, norm=norm, ticks=ticks,
                        label=label, orientation=orientation)
    # make the colorbar background match the brain color
    cbar.patch.set(facecolor=bgcolor)
    # remove the colorbar frame except for the line containing the ticks
    cbar.outline.set_visible(False)
    cbar.ax.set_frame_on(True)
    for key in ('left', 'top',
                'bottom' if orientation == 'vertical' else 'right'):
        ax.spines[key].set_visible(False)
    return cbar
