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

import numpy as np
from scipy import linalg

from ..defaults import DEFAULTS
from ..externals.six import BytesIO, string_types, advance_iterator
from ..io import _loc_to_coil_trans, Info
from ..io.pick import pick_types
from ..io.constants import FIFF
from ..io.meas_info import read_fiducials
from ..source_space import SourceSpaces
from ..surface import (_get_head_surface, get_meg_helmet_surf, read_surface,
                       transform_surface_to, _project_onto_surface,
                       complete_surface_info)
from ..transforms import (read_trans, _find_trans, apply_trans,
                          combine_transforms, _get_trans, _ensure_trans,
                          invert_transform, Transform)
from ..utils import (get_subjects_dir, logger, _check_subject, verbose, warn,
                     _import_mlab, SilenceStdout)
from .utils import mne_analyze_colormap, _prepare_trellis, COLORS, plt_show


FIDUCIAL_ORDER = (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION,
                  FIFF.FIFFV_POINT_RPA)


def _fiducial_coords(points, coord_frame=None):
    """Generate 3x3 array of fiducial coordinates."""
    if coord_frame is not None:
        points = (p for p in points if p['coord_frame'] == coord_frame)
    points_ = dict((p['ident'], p) for p in points if
                   p['kind'] == FIFF.FIFFV_POINT_CARDINAL)
    return np.array([points_[i]['r'] for i in FIDUCIAL_ORDER])


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


@verbose
def plot_trans(info, trans='auto', subject=None, subjects_dir=None,
               ch_type=None, source=('bem', 'head', 'outer_skin'),
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
    ch_type : None | 'eeg' | 'meg'
        This argument is deprecated. Use meg_sensors and eeg_sensors instead.
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
    if ch_type is not None:
        if ch_type not in ['eeg', 'meg']:
            raise ValueError('Argument ch_type must be None | eeg | meg. Got '
                             '%s.' % ch_type)
        warnings.warn('the ch_type argument is deprecated and will be removed '
                      'in 0.14. Use meg_sensors and eeg_sensors instead.')
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
        trans = read_trans(trans)
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

    if 'helmet' in meg_sensors and len(meg_picks) > 0 and \
            (ch_type is None or ch_type == 'meg'):
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
    if len(eeg_sensors) > 0 and (ch_type is None or ch_type == 'eeg'):
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
        else:
            # only warn if EEG explicitly requested, or EEG channels exist but
            # no locations are provided
            if (ch_type is not None or
                    len(pick_types(info, meg=False, eeg=True)) > 0):
                warn('EEG electrode locations not found. Cannot plot EEG '
                     'electrodes.')
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


def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='auto', time_label='auto',
                          smoothing_steps=10, transparent=None, alpha=1.0,
                          time_viewer=False, subjects_dir=None, figure=None,
                          views='lat', colorbar=True, clim='auto',
                          cortex="classic", size=800, background="black",
                          foreground="white", initial_time=None,
                          time_unit='s'):
    """Plot SourceEstimates with PySurfer.

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
    time_label : str | callable | None
        Format of the time label (a format string, a function that maps
        floating point time values to strings, or None for no label). The
        default is ``time=%0.2f ms``.
    smoothing_steps : int
        The amount of smoothing
    transparent : bool | None
        If True, use a linear transparency between fmin and fmid.
        None will choose automatically based on colormap type.
    alpha : float
        Alpha value to apply globally to the overlay.
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
        A instance of surfer.viz.Brain from PySurfer.
    """
    import surfer
    from surfer import Brain, TimeViewer
    import mayavi

    # import here to avoid circular import problem
    from ..source_estimate import SourceEstimate

    surfer_version = LooseVersion(surfer.__version__)
    v06 = LooseVersion('0.6')
    if surfer_version < v06:
        raise ImportError("This function requires PySurfer 0.6 (you are "
                          "running version %s). You can update PySurfer "
                          "using:\n\n    $ pip install -U pysurfer" %
                          surfer.__version__)

    if time_unit not in ('s', 'ms'):
        raise ValueError("time_unit needs to be 's' or 'ms', got %r" %
                         (time_unit,))

    if initial_time is not None and surfer_version > v06:
        kwargs = {'initial_time': initial_time}
        initial_time = None  # don't set it twice
    else:
        kwargs = {}

    if time_label == 'auto':
        if time_unit == 'ms':
            time_label = 'time=%0.2f ms'
        else:
            def time_label(t):
                return 'time=%0.2f ms' % (t * 1e3)

    if not isinstance(stc, SourceEstimate):
        raise ValueError('stc has to be a surface source estimate')

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
    with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(subject, hemi=hemi, surf=surface, curv=True,
                      title=title, cortex=cortex, size=size,
                      background=background, foreground=foreground,
                      figure=figure, subjects_dir=subjects_dir,
                      views=views)

    if time_unit == 's':
        times = stc.times
    else:  # time_unit == 'ms'
        times = 1e3 * stc.times

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
                               colorbar=colorbar, **kwargs)

        # scale colormap and set time (index) to display
        brain.scale_data_colormap(fmin=scale_pts[0], fmid=scale_pts[1],
                                  fmax=scale_pts[2], transparent=transparent)

    if initial_time is not None:
        brain.set_time(initial_time)
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
                          bgcolor=(1, 1, 1), opacity=0.3,
                          brain_color=(1, 1, 0), fig_name=None,
                          fig_size=(600, 600), mode='cone',
                          scale_factor=0.1e-1, colors=None, verbose=None):
    """Plot dipole locations.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig : instance of mlab.Figure
        The mayavi figure.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    mlab = _import_mlab()
    from matplotlib.colors import ColorConverter
    color_converter = ColorConverter()

    trans = _get_trans(trans)[0]
    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir,
                                    raise_error=True)
    fname = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')
    surf = complete_surface_info(read_surface(fname, return_dict=True,
                                              verbose=False)[2], copy=False)
    surf['rr'] = apply_trans(trans['trans'], surf['rr'] * 1e-3)

    from .. import Dipole
    if isinstance(dipoles, Dipole):
        dipoles = [dipoles]

    if mode not in ['cone', 'sphere']:
        raise ValueError('mode must be in "cone" or "sphere"')

    if colors is None:
        colors = cycle(COLORS)

    fig = mlab.figure(size=fig_size, bgcolor=bgcolor, fgcolor=(0, 0, 0))
    _toggle_mlab_render(fig, False)
    mesh = _create_mesh_surf(surf, fig=fig)
    with warnings.catch_warnings(record=True):  # traits
        surface = mlab.pipeline.surface(mesh, color=brain_color,
                                        opacity=opacity)
    surface.actor.property.backface_culling = True
    for dip, color in zip(dipoles, colors):
        rgb_color = color_converter.to_rgb(color)
        with warnings.catch_warnings(record=True):  # FutureWarning in traits
            mlab.quiver3d(dip.pos[0, 0], dip.pos[0, 1], dip.pos[0, 2],
                          dip.ori[0, 0], dip.ori[0, 1], dip.ori[0, 2],
                          opacity=1., mode=mode, color=rgb_color,
                          scalars=dip.amplitude.max(),
                          scale_factor=scale_factor)
    if fig_name is not None:
        with warnings.catch_warnings(record=True):  # traits
            mlab.title(fig_name)
    if fig.scene is not None:  # safe for Travis
        fig.scene.x_plus_view()
    _toggle_mlab_render(fig, True)
    return fig


def snapshot_brain_montage(fig, montage, hide_sensors=True):
    """Take a snapshot of a Mayavi Scene and project channels onto 2d coords.

    Note that this will take the raw values for 3d coordinates of each channel,
    without applying any transforms. If brain images are flipped up/dn upon
    using `imshow`, check your matplotlib backend as this behavior changes.

    Parameters
    ----------
    fig : instance of Mayavi Scene
        The figure on which you've plotted electrodes using `plot_trans`.
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
