# -*- coding: utf-8 -*-
"""Single-dipole functions and classes."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy
from functools import partial
import re

import numpy as np
from scipy import linalg

from .cov import read_cov, compute_whitener
from .io.constants import FIFF
from .io.pick import pick_types
from .io.proj import make_projector, _needs_eeg_average_ref_proj
from .bem import _fit_sphere
from .evoked import _read_evoked, _aspect_rev, _write_evokeds
from .transforms import _print_coord_trans, _coord_frame_name, apply_trans
from .viz.evoked import _plot_evoked
from .forward._make_forward import (_get_trans, _setup_bem,
                                    _prep_meg_channels, _prep_eeg_channels)
from .forward._compute_forward import (_compute_forwards_meeg,
                                       _prep_field_computation)

from .surface import (transform_surface_to, _compute_nearest,
                      _points_outside_surface)
from .bem import _bem_find_surface, _surf_name
from .source_space import _make_volume_source_space, SourceSpaces
from .parallel import parallel_func
from .utils import (logger, verbose, _time_mask, warn, _check_fname,
                    check_fname, _pl, fill_doc, _check_option, ShiftTimeMixin,
                    _svd_lwork, _repeated_svd, ddot, dgemv, dgemm)


@fill_doc
class Dipole(object):
    u"""Dipole class for sequential dipole fits.

    .. note:: This class should usually not be instantiated directly,
              instead :func:`mne.read_dipole` should be used.

    Used to store positions, orientations, amplitudes, times, goodness of fit
    of dipoles, typically obtained with Neuromag/xfit, mne_dipole_fit
    or certain inverse solvers. Note that dipole position vectors are given in
    the head coordinate frame.

    Parameters
    ----------
    times : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted (sec).
    pos : array, shape (n_dipoles, 3)
        The dipoles positions (m) in head coordinates.
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles (Am).
    ori : array, shape (n_dipoles, 3)
        The dipole orientations (normalized to unit length).
    gof : array, shape (n_dipoles,)
        The goodness of fit.
    name : str | None
        Name of the dipole.
    conf : dict
        Confidence limits in dipole orientation for "vol" in m^3 (volume),
        "depth" in m (along the depth axis), "long" in m (longitudinal axis),
        "trans" in m (transverse axis), "qlong" in Am, and "qtrans" in Am
        (currents). The current confidence limit in the depth direction is
        assumed to be zero (although it can be non-zero when a BEM is used).

        .. versionadded:: 0.15
    khi2 : array, shape (n_dipoles,)
        The χ^2 values for the fits.

        .. versionadded:: 0.15
    nfree : array, shape (n_dipoles,)
        The number of free parameters for each fit.

        .. versionadded:: 0.15
    %(verbose)s

    See Also
    --------
    fit_dipole
    DipoleFixed
    read_dipole

    Notes
    -----
    This class is for sequential dipole fits, where the position
    changes as a function of time. For fixed dipole fits, where the
    position is fixed as a function of time, use :class:`mne.DipoleFixed`.
    """

    @verbose
    def __init__(self, times, pos, amplitude, ori, gof,
                 name=None, conf=None, khi2=None, nfree=None,
                 verbose=None):  # noqa: D102
        self.times = np.array(times)
        self.pos = np.array(pos)
        self.amplitude = np.array(amplitude)
        self.ori = np.array(ori)
        self.gof = np.array(gof)
        self.name = name
        self.conf = deepcopy(conf) if conf is not None else dict()
        self.khi2 = np.array(khi2) if khi2 is not None else None
        self.nfree = np.array(nfree) if nfree is not None else None
        self.verbose = verbose

    def __repr__(self):  # noqa: D105
        s = "n_times : %s" % len(self.times)
        s += ", tmin : %0.3f" % np.min(self.times)
        s += ", tmax : %0.3f" % np.max(self.times)
        return "<Dipole | %s>" % s

    def save(self, fname, overwrite=False):
        """Save dipole in a .dip or .bdip file.

        Parameters
        ----------
        fname : str
            The name of the .dip or .bdip file.
        overwrite : bool
            If True, overwrite the file (if it exists).

            .. versionadded:: 0.20

        Notes
        -----
        .. versionchanged:: 0.20
           Support for writing bdip (Xfit binary) files.
        """
        # obligatory fields
        fname = _check_fname(fname, overwrite=overwrite)
        if fname.endswith('.bdip'):
            _write_dipole_bdip(fname, self)
        else:
            _write_dipole_text(fname, self)

    @fill_doc
    def crop(self, tmin=None, tmax=None, include_tmax=True):
        """Crop data to a given time interval.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        %(include_tmax)s

        Returns
        -------
        self : instance of Dipole
            The cropped instance.
        """
        sfreq = None
        if len(self.times) > 1:
            sfreq = 1. / np.median(np.diff(self.times))
        mask = _time_mask(self.times, tmin, tmax, sfreq=sfreq,
                          include_tmax=include_tmax)
        for attr in ('times', 'pos', 'gof', 'amplitude', 'ori',
                     'khi2', 'nfree'):
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[mask])
        for key in self.conf.keys():
            self.conf[key] = self.conf[key][mask]
        return self

    def copy(self):
        """Copy the Dipoles object.

        Returns
        -------
        dip : instance of Dipole
            The copied dipole instance.
        """
        return deepcopy(self)

    @verbose
    def plot_locations(self, trans, subject, subjects_dir=None,
                       mode='orthoview', coord_frame='mri', idx='gof',
                       show_all=True, ax=None, block=False, show=True,
                       scale=5e-3, color=(1.0, 0.0, 0.0), fig=None,
                       verbose=None, title=None):
        """Plot dipole locations in 3d.

        Parameters
        ----------
        trans : dict
            The mri to head trans.
        subject : str
            The subject name corresponding to FreeSurfer environment
            variable SUBJECT.
        %(subjects_dir)s
        mode : str
            Can be ``'arrow'``, ``'sphere'`` or ``'orthoview'``.

            .. versionadded:: 0.14.0
        coord_frame : str
            Coordinate frame to use, 'head' or 'mri'. Defaults to 'mri'.

            .. versionadded:: 0.14.0
        idx : int | 'gof' | 'amplitude'
            Index of the initially plotted dipole. Can also be 'gof' to plot
            the dipole with highest goodness of fit value or 'amplitude' to
            plot the dipole with the highest amplitude. The dipoles can also be
            browsed through using up/down arrow keys or mouse scroll. Defaults
            to 'gof'. Only used if mode equals 'orthoview'.

            .. versionadded:: 0.14.0
        show_all : bool
            Whether to always plot all the dipoles. If True (default), the
            active dipole is plotted as a red dot and it's location determines
            the shown MRI slices. The the non-active dipoles are plotted as
            small blue dots. If False, only the active dipole is plotted.
            Only used if mode equals 'orthoview'.

            .. versionadded:: 0.14.0
        ax : instance of matplotlib Axes3D | None
            Axes to plot into. If None (default), axes will be created.
            Only used if mode equals 'orthoview'.

            .. versionadded:: 0.14.0
        block : bool
            Whether to halt program execution until the figure is closed.
            Defaults to False. Only used if mode equals 'orthoview'.

            .. versionadded:: 0.14.0
        show : bool
            Show figure if True. Defaults to True.
            Only used if mode equals 'orthoview'.

        scale : float
            The scale of the dipoles if ``mode`` is 'arrow' or 'sphere'.
        color : tuple
            The color of the dipoles if ``mode`` is 'arrow' or 'sphere'.
        fig : mayavi.mlab.Figure | None
            Mayavi Scene in which to plot the alignment.
            If ``None``, creates a new 600x600 pixel figure with black
            background.

            .. versionadded:: 0.14.0
        %(verbose_meth)s
        %(dipole_locs_fig_title)s

            .. versionadded:: 0.21.0

        Returns
        -------
        fig : instance of mayavi.mlab.Figure or matplotlib.figure.Figure
            The mayavi figure or matplotlib Figure.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        _check_option('mode', mode, [None, 'arrow', 'sphere', 'orthoview'])

        from .viz import plot_dipole_locations
        return plot_dipole_locations(
            self, trans, subject, subjects_dir, mode, coord_frame, idx,
            show_all, ax, block, show, scale=scale, color=color, fig=fig,
            title=title)

    def plot_amplitudes(self, color='k', show=True):
        """Plot the dipole amplitudes as a function of time.

        Parameters
        ----------
        color : matplotlib color
            Color to use for the trace.
        show : bool
            Show figure if True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        from .viz import plot_dipole_amplitudes
        return plot_dipole_amplitudes([self], [color], show)

    def __getitem__(self, item):
        """Get a time slice.

        Parameters
        ----------
        item : array-like or slice
            The slice of time points to use.

        Returns
        -------
        dip : instance of Dipole
            The sliced dipole.
        """
        if isinstance(item, int):  # make sure attributes stay 2d
            item = [item]

        selected_times = self.times[item].copy()
        selected_pos = self.pos[item, :].copy()
        selected_amplitude = self.amplitude[item].copy()
        selected_ori = self.ori[item, :].copy()
        selected_gof = self.gof[item].copy()
        selected_name = self.name
        selected_conf = dict()
        for key in self.conf.keys():
            selected_conf[key] = self.conf[key][item]
        selected_khi2 = self.khi2[item] if self.khi2 is not None else None
        selected_nfree = self.nfree[item] if self.nfree is not None else None
        return Dipole(
            selected_times, selected_pos, selected_amplitude, selected_ori,
            selected_gof, selected_name, selected_conf, selected_khi2,
            selected_nfree)

    def __len__(self):
        """Return the number of dipoles.

        Returns
        -------
        len : int
            The number of dipoles.

        Examples
        --------
        This can be used as::

            >>> len(dipoles)  # doctest: +SKIP
            10

        """
        return self.pos.shape[0]


def _read_dipole_fixed(fname):
    """Read a fixed dipole FIF file."""
    logger.info('Reading %s ...' % fname)
    info, nave, aspect_kind, comment, times, data = _read_evoked(fname)
    return DipoleFixed(info, data, times, nave, aspect_kind, comment=comment)


@fill_doc
class DipoleFixed(ShiftTimeMixin):
    """Dipole class for fixed-position dipole fits.

    .. note:: This class should usually not be instantiated directly,
              instead :func:`mne.read_dipole` should be used.

    Parameters
    ----------
    info : instance of Info
        The measurement info.
    data : array, shape (n_channels, n_times)
        The dipole data.
    times : array, shape (n_times,)
        The time points.
    nave : int
        Number of averages.
    aspect_kind : int
        The kind of data.
    comment : str
        The dipole comment.
    %(verbose)s

    See Also
    --------
    read_dipole
    Dipole
    fit_dipole

    Notes
    -----
    This class is for fixed-position dipole fits, where the position
    (and maybe orientation) is static over time. For sequential dipole fits,
    where the position can change a function of time, use :class:`mne.Dipole`.

    .. versionadded:: 0.12
    """

    @verbose
    def __init__(self, info, data, times, nave, aspect_kind,
                 comment='', verbose=None):  # noqa: D102
        self.info = info
        self.nave = nave
        self._aspect_kind = aspect_kind
        self.kind = _aspect_rev.get(aspect_kind, 'unknown')
        self.comment = comment
        self.times = times
        self.data = data
        self.verbose = verbose
        self.preload = True
        self._update_first_last()

    def __repr__(self):  # noqa: D105
        s = "n_times : %s" % len(self.times)
        s += ", tmin : %s" % np.min(self.times)
        s += ", tmax : %s" % np.max(self.times)
        return "<DipoleFixed | %s>" % s

    def copy(self):
        """Copy the DipoleFixed object.

        Returns
        -------
        inst : instance of DipoleFixed
            The copy.

        Notes
        -----
        .. versionadded:: 0.16
        """
        return deepcopy(self)

    @property
    def ch_names(self):
        """Channel names."""
        return self.info['ch_names']

    @verbose
    def save(self, fname, verbose=None):
        """Save dipole in a .fif file.

        Parameters
        ----------
        fname : str
            The name of the .fif file. Must end with ``'.fif'`` or
            ``'.fif.gz'`` to make it explicit that the file contains
            dipole information in FIF format.
        %(verbose_meth)s
        """
        check_fname(fname, 'DipoleFixed', ('-dip.fif', '-dip.fif.gz',
                                           '_dip.fif', '_dip.fif.gz',),
                    ('.fif', '.fif.gz'))
        _write_evokeds(fname, self, check=False)

    def plot(self, show=True, time_unit='s'):
        """Plot dipole data.

        Parameters
        ----------
        show : bool
            Call pyplot.show() at the end or not.
        time_unit : str
            The units for the time axis, can be "ms" or "s" (default).

            .. versionadded:: 0.16

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure containing the time courses.
        """
        return _plot_evoked(self, picks=None, exclude=(), unit=True, show=show,
                            ylim=None, xlim='tight', proj=False, hline=None,
                            units=None, scalings=None, titles=None, axes=None,
                            gfp=False, window_title=None, spatial_colors=False,
                            plot_type="butterfly", selectable=False,
                            time_unit=time_unit)


# #############################################################################
# IO
@verbose
def read_dipole(fname, verbose=None):
    """Read .dip file from Neuromag/xfit or MNE.

    Parameters
    ----------
    fname : str
        The name of the .dip or .fif file.
    %(verbose)s

    Returns
    -------
    dipole : instance of Dipole or DipoleFixed
        The dipole.

    See Also
    --------
    Dipole
    DipoleFixed
    fit_dipole

    Notes
    -----
    .. versionchanged:: 0.20
       Support for reading bdip (Xfit binary) format.
    """
    fname = _check_fname(fname, overwrite='read', must_exist=True)
    if fname.endswith('.fif') or fname.endswith('.fif.gz'):
        return _read_dipole_fixed(fname)
    elif fname.endswith('.bdip'):
        return _read_dipole_bdip(fname)
    else:
        return _read_dipole_text(fname)


def _read_dipole_text(fname):
    """Read a dipole text file."""
    # Figure out the special fields
    need_header = True
    def_line = name = None
    # There is a bug in older np.loadtxt regarding skipping fields,
    # so just read the data ourselves (need to get name and header anyway)
    data = list()
    with open(fname, 'r') as fid:
        for line in fid:
            if not (line.startswith('%') or line.startswith('#')):
                need_header = False
                data.append(line.strip().split())
            else:
                if need_header:
                    def_line = line
                if line.startswith('##') or line.startswith('%%'):
                    m = re.search('Name "(.*) dipoles"', line)
                    if m:
                        name = m.group(1)
        del line
    data = np.atleast_2d(np.array(data, float))
    if def_line is None:
        raise IOError('Dipole text file is missing field definition '
                      'comment, cannot parse %s' % (fname,))
    # actually parse the fields
    def_line = def_line.lstrip('%').lstrip('#').strip()
    # MNE writes it out differently than Elekta, let's standardize them...
    fields = re.sub(r'([X|Y|Z] )\(mm\)',  # "X (mm)", etc.
                    lambda match: match.group(1).strip() + '/mm', def_line)
    fields = re.sub(r'\((.*?)\)',  # "Q(nAm)", etc.
                    lambda match: '/' + match.group(1), fields)
    fields = re.sub('(begin|end) ',  # "begin" and "end" with no units
                    lambda match: match.group(1) + '/ms', fields)
    fields = fields.lower().split()
    required_fields = ('begin/ms',
                       'x/mm', 'y/mm', 'z/mm',
                       'q/nam', 'qx/nam', 'qy/nam', 'qz/nam',
                       'g/%')
    optional_fields = ('khi^2', 'free',  # standard ones
                       # now the confidence fields (up to 5!)
                       'vol/mm^3', 'depth/mm', 'long/mm', 'trans/mm',
                       'qlong/nam', 'qtrans/nam')
    conf_scales = [1e-9, 1e-3, 1e-3, 1e-3, 1e-9, 1e-9]
    missing_fields = sorted(set(required_fields) - set(fields))
    if len(missing_fields) > 0:
        raise RuntimeError('Could not find necessary fields in header: %s'
                           % (missing_fields,))
    handled_fields = set(required_fields) | set(optional_fields)
    assert len(handled_fields) == len(required_fields) + len(optional_fields)
    ignored_fields = sorted(set(fields) -
                            set(handled_fields) -
                            {'end/ms'})
    if len(ignored_fields) > 0:
        warn('Ignoring extra fields in dipole file: %s' % (ignored_fields,))
    if len(fields) != data.shape[1]:
        raise IOError('More data fields (%s) found than data columns (%s): %s'
                      % (len(fields), data.shape[1], fields))

    logger.info("%d dipole(s) found" % len(data))

    if 'end/ms' in fields:
        if np.diff(data[:, [fields.index('begin/ms'),
                            fields.index('end/ms')]], 1, -1).any():
            warn('begin and end fields differed, but only begin will be used '
                 'to store time values')

    # Find the correct column in our data array, then scale to proper units
    idx = [fields.index(field) for field in required_fields]
    assert len(idx) >= 9
    times = data[:, idx[0]] / 1000.
    pos = 1e-3 * data[:, idx[1:4]]  # put data in meters
    amplitude = data[:, idx[4]]
    norm = amplitude.copy()
    amplitude /= 1e9
    norm[norm == 0] = 1
    ori = data[:, idx[5:8]] / norm[:, np.newaxis]
    gof = data[:, idx[8]]
    # Deal with optional fields
    optional = [None] * 2
    for fi, field in enumerate(optional_fields[:2]):
        if field in fields:
            optional[fi] = data[:, fields.index(field)]
    khi2, nfree = optional
    conf = dict()
    for field, scale in zip(optional_fields[2:], conf_scales):  # confidence
        if field in fields:
            conf[field.split('/')[0]] = scale * data[:, fields.index(field)]
    return Dipole(times, pos, amplitude, ori, gof, name, conf, khi2, nfree)


def _write_dipole_text(fname, dip):
    fmt = '  %7.1f %7.1f %8.2f %8.2f %8.2f %8.3f %8.3f %8.3f %8.3f %6.2f'
    header = ('#   begin     end   X (mm)   Y (mm)   Z (mm)'
              '   Q(nAm)  Qx(nAm)  Qy(nAm)  Qz(nAm)    g/%')
    t = dip.times[:, np.newaxis] * 1000.
    gof = dip.gof[:, np.newaxis]
    amp = 1e9 * dip.amplitude[:, np.newaxis]
    out = (t, t, dip.pos / 1e-3, amp, dip.ori * amp, gof)

    # optional fields
    fmts = dict(khi2=('    khi^2', ' %8.1f', 1.),
                nfree=('  free', ' %5d', 1),
                vol=('  vol/mm^3', ' %9.3f', 1e9),
                depth=('  depth/mm', ' %9.3f', 1e3),
                long=('  long/mm', ' %8.3f', 1e3),
                trans=('  trans/mm', ' %9.3f', 1e3),
                qlong=('  Qlong/nAm', ' %10.3f', 1e9),
                qtrans=('  Qtrans/nAm', ' %11.3f', 1e9),
                )
    for key in ('khi2', 'nfree'):
        data = getattr(dip, key)
        if data is not None:
            header += fmts[key][0]
            fmt += fmts[key][1]
            out += (data[:, np.newaxis] * fmts[key][2],)
    for key in ('vol', 'depth', 'long', 'trans', 'qlong', 'qtrans'):
        data = dip.conf.get(key)
        if data is not None:
            header += fmts[key][0]
            fmt += fmts[key][1]
            out += (data[:, np.newaxis] * fmts[key][2],)
    out = np.concatenate(out, axis=-1)

    # NB CoordinateSystem is hard-coded as Head here
    with open(fname, 'wb') as fid:
        fid.write('# CoordinateSystem "Head"\n'.encode('utf-8'))
        fid.write((header + '\n').encode('utf-8'))
        np.savetxt(fid, out, fmt=fmt)
        if dip.name is not None:
            fid.write(('## Name "%s dipoles" Style "Dipoles"'
                       % dip.name).encode('utf-8'))


_BDIP_ERROR_KEYS = ('depth', 'long', 'trans', 'qlong', 'qtrans')


def _read_dipole_bdip(fname):
    name = None
    nfree = 0
    with open(fname, 'rb') as fid:
        # Which dipole in a multi-dipole set
        times = list()
        pos = list()
        amplitude = list()
        ori = list()
        gof = list()
        conf = dict(vol=list())
        khi2 = list()
        has_errors = None
        while True:
            num = np.frombuffer(fid.read(4), '>i4')
            if len(num) == 0:
                break
            times.append(np.frombuffer(fid.read(4), '>f4')[0])
            fid.read(4)  # end
            fid.read(12)  # r0
            pos.append(np.frombuffer(fid.read(12), '>f4'))
            Q = np.frombuffer(fid.read(12), '>f4')
            amplitude.append(np.linalg.norm(Q))
            ori.append(Q / amplitude[-1])
            gof.append(100 * np.frombuffer(fid.read(4), '>f4')[0])
            this_has_errors = bool(np.frombuffer(fid.read(4), '>i4')[0])
            if has_errors is None:
                has_errors = this_has_errors
                for key in _BDIP_ERROR_KEYS:
                    conf[key] = list()
            assert has_errors == this_has_errors
            fid.read(4)  # Noise level used for error computations
            limits = np.frombuffer(fid.read(20), '>f4')  # error limits
            for key, lim in zip(_BDIP_ERROR_KEYS, limits):
                conf[key].append(lim)
            fid.read(100)  # (5, 5) fully describes the conf. ellipsoid
            conf['vol'].append(np.frombuffer(fid.read(4), '>f4')[0])
            khi2.append(np.frombuffer(fid.read(4), '>f4')[0])
            fid.read(4)  # prob
            fid.read(4)  # total noise estimate
    return Dipole(times, pos, amplitude, ori, gof, name, conf, khi2, nfree)


def _write_dipole_bdip(fname, dip):
    with open(fname, 'wb+') as fid:
        for ti, t in enumerate(dip.times):
            fid.write(np.zeros(1, '>i4').tobytes())  # int dipole
            fid.write(np.array([t, 0]).astype('>f4').tobytes())
            fid.write(np.zeros(3, '>f4').tobytes())  # r0
            fid.write(dip.pos[ti].astype('>f4').tobytes())  # pos
            Q = dip.amplitude[ti] * dip.ori[ti]
            fid.write(Q.astype('>f4').tobytes())
            fid.write(np.array(dip.gof[ti] / 100., '>f4').tobytes())
            has_errors = int(bool(len(dip.conf)))
            fid.write(np.array(has_errors, '>i4').tobytes())  # has_errors
            fid.write(np.zeros(1, '>f4').tobytes())  # noise level
            for key in _BDIP_ERROR_KEYS:
                val = dip.conf[key][ti] if key in dip.conf else 0.
                assert val.shape == ()
                fid.write(np.array(val, '>f4').tobytes())
            fid.write(np.zeros(25, '>f4').tobytes())
            conf = dip.conf['vol'][ti] if 'vol' in dip.conf else 0.
            fid.write(np.array(conf, '>f4').tobytes())
            khi2 = dip.khi2[ti] if dip.khi2 is not None else 0
            fid.write(np.array(khi2, '>f4').tobytes())
            fid.write(np.zeros(1, '>f4').tobytes())  # prob
            fid.write(np.zeros(1, '>f4').tobytes())  # total noise est


# #############################################################################
# Fitting

def _dipole_forwards(fwd_data, whitener, rr, n_jobs=1):
    """Compute the forward solution and do other nice stuff."""
    B = _compute_forwards_meeg(rr, fwd_data, n_jobs, silent=True)
    B = np.concatenate(B, axis=1)
    assert np.isfinite(B).all()
    B_orig = B.copy()

    # Apply projection and whiten (cov has projections already)
    B = dgemm(1., B, whitener.T)

    # column normalization doesn't affect our fitting, so skip for now
    # S = np.sum(B * B, axis=1)  # across channels
    # scales = np.repeat(3. / np.sqrt(np.sum(np.reshape(S, (len(rr), 3)),
    #                                        axis=1)), 3)
    # B *= scales[:, np.newaxis]
    scales = np.ones(3)
    return B, B_orig, scales


@verbose
def _make_guesses(surf, grid, exclude, mindist, n_jobs=1, verbose=None):
    """Make a guess space inside a sphere or BEM surface."""
    if 'rr' in surf:
        logger.info('Guess surface (%s) is in %s coordinates'
                    % (_surf_name[surf['id']],
                       _coord_frame_name(surf['coord_frame'])))
    else:
        logger.info('Making a spherical guess space with radius %7.1f mm...'
                    % (1000 * surf['R']))
    logger.info('Filtering (grid = %6.f mm)...' % (1000 * grid))
    src = _make_volume_source_space(surf, grid, exclude, 1000 * mindist,
                                    do_neighbors=False, n_jobs=n_jobs)[0]
    assert 'vertno' in src
    # simplify the result to make things easier later
    src = dict(rr=src['rr'][src['vertno']], nn=src['nn'][src['vertno']],
               nuse=src['nuse'], coord_frame=src['coord_frame'],
               vertno=np.arange(src['nuse']), type='discrete')
    return SourceSpaces([src])


def _fit_eval(rd, B, B2, fwd_svd=None, fwd_data=None, whitener=None,
              lwork=None):
    """Calculate the residual sum of squares."""
    if fwd_svd is None:
        fwd = _dipole_forwards(fwd_data, whitener, rd[np.newaxis, :])[0]
        uu, sing, vv = _repeated_svd(fwd, lwork, overwrite_a=True)
    else:
        uu, sing, vv = fwd_svd
    gof = _dipole_gof(uu, sing, vv, B, B2)[0]
    # mne-c uses fitness=B2-Bm2, but ours (1-gof) is just a normalized version
    return 1. - gof


def _dipole_gof(uu, sing, vv, B, B2):
    """Calculate the goodness of fit from the forward SVD."""
    ncomp = 3 if sing[2] / (sing[0] if sing[0] > 0 else 1.) > 0.2 else 2
    one = dgemv(1., vv[:ncomp], B)  # np.dot(vv[:ncomp], B)
    Bm2 = ddot(one, one)  # np.sum(one * one)
    gof = Bm2 / B2
    return gof, one


def _fit_Q(fwd_data, whitener, B, B2, B_orig, rd, ori=None):
    """Fit the dipole moment once the location is known."""
    if 'fwd' in fwd_data:
        # should be a single precomputed "guess" (i.e., fixed position)
        assert rd is None
        fwd = fwd_data['fwd']
        assert fwd.shape[0] == 3
        fwd_orig = fwd_data['fwd_orig']
        assert fwd_orig.shape[0] == 3
        scales = fwd_data['scales']
        assert scales.shape == (3,)
        fwd_svd = fwd_data['fwd_svd'][0]
    else:
        fwd, fwd_orig, scales = _dipole_forwards(fwd_data, whitener,
                                                 rd[np.newaxis, :])
        fwd_svd = None
    if ori is None:
        if fwd_svd is None:
            fwd_svd = linalg.svd(fwd, full_matrices=False)
        uu, sing, vv = fwd_svd
        gof, one = _dipole_gof(uu, sing, vv, B, B2)
        ncomp = len(one)
        one /= sing[:ncomp]
        Q = np.dot(one, uu.T[:ncomp])
    else:
        fwd = np.dot(ori[np.newaxis], fwd)
        sing = np.linalg.norm(fwd)
        one = np.dot(fwd / sing, B)
        gof = (one * one)[0] / B2
        Q = ori * np.sum(one / sing)
        ncomp = 3
    # Counteract the effect of column normalization
    Q *= scales[0]
    B_residual_noproj = B_orig - np.dot(fwd_orig.T, Q)
    return Q, gof, B_residual_noproj, ncomp


def _fit_dipoles(fun, min_dist_to_inner_skull, data, times, guess_rrs,
                 guess_data, fwd_data, whitener, ori, n_jobs, rank):
    """Fit a single dipole to the given whitened, projected data."""
    from scipy.optimize import fmin_cobyla
    parallel, p_fun, _ = parallel_func(fun, n_jobs)
    # parallel over time points
    res = parallel(p_fun(min_dist_to_inner_skull, B, t, guess_rrs,
                         guess_data, fwd_data, whitener,
                         fmin_cobyla, ori, rank)
                   for B, t in zip(data.T, times))
    pos = np.array([r[0] for r in res])
    amp = np.array([r[1] for r in res])
    ori = np.array([r[2] for r in res])
    gof = np.array([r[3] for r in res]) * 100  # convert to percentage
    conf = None
    if res[0][4] is not None:
        conf = np.array([r[4] for r in res])
        keys = ['vol', 'depth', 'long', 'trans', 'qlong', 'qtrans']
        conf = {key: conf[:, ki] for ki, key in enumerate(keys)}
    khi2 = np.array([r[5] for r in res])
    nfree = np.array([r[6] for r in res])
    residual_noproj = np.array([r[7] for r in res]).T

    return pos, amp, ori, gof, conf, khi2, nfree, residual_noproj


'''Simplex code in case we ever want/need it for testing

def _make_tetra_simplex():
    """Make the initial tetrahedron"""
    #
    # For this definition of a regular tetrahedron, see
    #
    # http://mathworld.wolfram.com/Tetrahedron.html
    #
    x = np.sqrt(3.0) / 3.0
    r = np.sqrt(6.0) / 12.0
    R = 3 * r
    d = x / 2.0
    simplex = 1e-2 * np.array([[x, 0.0, -r],
                               [-d, 0.5, -r],
                               [-d, -0.5, -r],
                               [0., 0., R]])
    return simplex


def try_(p, y, psum, ndim, fun, ihi, neval, fac):
    """Helper to try a value"""
    ptry = np.empty(ndim)
    fac1 = (1.0 - fac) / ndim
    fac2 = fac1 - fac
    ptry = psum * fac1 - p[ihi] * fac2
    ytry = fun(ptry)
    neval += 1
    if ytry < y[ihi]:
        y[ihi] = ytry
        psum[:] += ptry - p[ihi]
        p[ihi] = ptry
    return ytry, neval


def _simplex_minimize(p, ftol, stol, fun, max_eval=1000):
    """Minimization with the simplex algorithm

    Modified from Numerical recipes"""
    y = np.array([fun(s) for s in p])
    ndim = p.shape[1]
    assert p.shape[0] == ndim + 1
    mpts = ndim + 1
    neval = 0
    psum = p.sum(axis=0)

    loop = 1
    while(True):
        ilo = 1
        if y[1] > y[2]:
            ihi = 1
            inhi = 2
        else:
            ihi = 2
            inhi = 1
        for i in range(mpts):
            if y[i] < y[ilo]:
                ilo = i
            if y[i] > y[ihi]:
                inhi = ihi
                ihi = i
            elif y[i] > y[inhi]:
                if i != ihi:
                    inhi = i

        rtol = 2 * np.abs(y[ihi] - y[ilo]) / (np.abs(y[ihi]) + np.abs(y[ilo]))
        if rtol < ftol:
            break
        if neval >= max_eval:
            raise RuntimeError('Maximum number of evaluations exceeded.')
        if stol > 0:  # Has the simplex collapsed?
            dsum = np.sqrt(np.sum((p[ilo] - p[ihi]) ** 2))
            if loop > 5 and dsum < stol:
                break

        ytry, neval = try_(p, y, psum, ndim, fun, ihi, neval, -1.)
        if ytry <= y[ilo]:
            ytry, neval = try_(p, y, psum, ndim, fun, ihi, neval, 2.)
        elif ytry >= y[inhi]:
            ysave = y[ihi]
            ytry, neval = try_(p, y, psum, ndim, fun, ihi, neval, 0.5)
            if ytry >= ysave:
                for i in range(mpts):
                    if i != ilo:
                        psum[:] = 0.5 * (p[i] + p[ilo])
                        p[i] = psum
                        y[i] = fun(psum)
                neval += ndim
                psum = p.sum(axis=0)
        loop += 1
'''


def _fit_confidence(rd, Q, ori, whitener, fwd_data):
    # As describedd in the Xfit manual, confidence intervals can be calculated
    # by examining a linearization of model at the best-fitting location,
    # i.e. taking the Jacobian and using the whitener:
    #
    #     J = [∂b/∂x ∂b/∂y ∂b/∂z ∂b/∂Qx ∂b/∂Qy ∂b/∂Qz]
    #     C = (J.T C^-1 J)^-1
    #
    # And then the confidence interval is the diagonal of C, scaled by 1.96
    # (for 95% confidence).
    direction = np.empty((3, 3))
    # The coordinate system has the x axis aligned with the dipole orientation,
    direction[0] = ori
    # the z axis through the origin of the sphere model
    rvec = rd - fwd_data['inner_skull']['r0']
    direction[2] = rvec - ori * np.dot(ori, rvec)  # orthogonalize
    direction[2] /= np.linalg.norm(direction[2])
    # and the y axis perpendical with these forming a right-handed system.
    direction[1] = np.cross(direction[2], direction[0])
    assert np.allclose(np.dot(direction, direction.T), np.eye(3))
    # Get spatial deltas in dipole coordinate directions
    deltas = (-1e-4, 1e-4)
    J = np.empty((whitener.shape[0], 6))
    for ii in range(3):
        fwds = []
        for delta in deltas:
            this_r = rd[np.newaxis] + delta * direction[ii]
            fwds.append(
                np.dot(Q, _dipole_forwards(fwd_data, whitener, this_r)[0]))
        J[:, ii] = np.diff(fwds, axis=0)[0] / np.diff(deltas)[0]
    # Get current (Q) deltas in the dipole directions
    deltas = np.array([-0.01, 0.01]) * np.linalg.norm(Q)
    this_fwd = _dipole_forwards(fwd_data, whitener, rd[np.newaxis])[0]
    for ii in range(3):
        fwds = []
        for delta in deltas:
            fwds.append(np.dot(Q + delta * direction[ii], this_fwd))
        J[:, ii + 3] = np.diff(fwds, axis=0)[0] / np.diff(deltas)[0]
    # J is already whitened, so we don't need to do np.dot(whitener, J).
    # However, the units in the Jacobian are potentially quite different,
    # so we need to do some normalization during inversion, then revert.
    direction_norm = np.linalg.norm(J[:, :3])
    Q_norm = np.linalg.norm(J[:, 3:5])  # omit possible zero Z
    norm = np.array([direction_norm] * 3 + [Q_norm] * 3)
    J /= norm
    J = np.dot(J.T, J)
    C = linalg.pinvh(J, rcond=1e-14)
    C /= norm
    C /= norm[:, np.newaxis]
    conf = 1.96 * np.sqrt(np.diag(C))
    # The confidence volume of the dipole location is obtained from by
    # taking the eigenvalues of the upper left submatrix and computing
    # v = 4π/3 √(c^3 λ1 λ2 λ3) with c = 7.81, or:
    vol_conf = 4 * np.pi / 3. * np.sqrt(
        476.379541 * np.prod(linalg.eigh(C[:3, :3], eigvals_only=True)))
    conf = np.concatenate([conf, [vol_conf]])
    # Now we reorder and subselect the proper columns:
    # vol, depth, long, trans, Qlong, Qtrans (discard Qdepth, assumed zero)
    conf = conf[[6, 2, 0, 1, 3, 4]]
    return conf


def _surface_constraint(rd, surf, min_dist_to_inner_skull):
    """Surface fitting constraint."""
    dist = _compute_nearest(surf['rr'], rd[np.newaxis, :],
                            return_dists=True)[1][0]
    if _points_outside_surface(rd[np.newaxis, :], surf, 1)[0]:
        dist *= -1.
    # Once we know the dipole is below the inner skull,
    # let's check if its distance to the inner skull is at least
    # min_dist_to_inner_skull. This can be enforced by adding a
    # constrain proportional to its distance.
    dist -= min_dist_to_inner_skull
    return dist


def _sphere_constraint(rd, r0, R_adj):
    """Sphere fitting constraint."""
    return R_adj - np.sqrt(np.sum((rd - r0) ** 2))


def _fit_dipole(min_dist_to_inner_skull, B_orig, t, guess_rrs,
                guess_data, fwd_data, whitener, fmin_cobyla, ori, rank):
    """Fit a single bit of data."""
    B = np.dot(whitener, B_orig)

    # make constraint function to keep the solver within the inner skull
    if 'rr' in fwd_data['inner_skull']:  # bem
        surf = fwd_data['inner_skull']
        constraint = partial(_surface_constraint, surf=surf,
                             min_dist_to_inner_skull=min_dist_to_inner_skull)
    else:  # sphere
        surf = None
        constraint = partial(
            _sphere_constraint, r0=fwd_data['inner_skull']['r0'],
            R_adj=fwd_data['inner_skull']['R'] - min_dist_to_inner_skull)

    # Find a good starting point (find_best_guess in C)
    B2 = np.dot(B, B)
    if B2 == 0:
        warn('Zero field found for time %s' % t)
        return np.zeros(3), 0, np.zeros(3), 0, B

    idx = np.argmin([_fit_eval(guess_rrs[[fi], :], B, B2, fwd_svd)
                     for fi, fwd_svd in enumerate(guess_data['fwd_svd'])])
    x0 = guess_rrs[idx]
    lwork = _svd_lwork((3, B.shape[0]))
    fun = partial(_fit_eval, B=B, B2=B2, fwd_data=fwd_data, whitener=whitener,
                  lwork=lwork)

    # Tested minimizers:
    #    Simplex, BFGS, CG, COBYLA, L-BFGS-B, Powell, SLSQP, TNC
    # Several were similar, but COBYLA won for having a handy constraint
    # function we can use to ensure we stay inside the inner skull /
    # smallest sphere
    rd_final = fmin_cobyla(fun, x0, (constraint,), consargs=(),
                           rhobeg=5e-2, rhoend=5e-5, disp=False)

    # simplex = _make_tetra_simplex() + x0
    # _simplex_minimize(simplex, 1e-4, 2e-4, fun)
    # rd_final = simplex[0]

    # Compute the dipole moment at the final point
    Q, gof, residual_noproj, n_comp = _fit_Q(
        fwd_data, whitener, B, B2, B_orig, rd_final, ori=ori)
    khi2 = (1 - gof) * B2
    nfree = rank - n_comp
    amp = np.sqrt(np.dot(Q, Q))
    norm = 1. if amp == 0. else amp
    ori = Q / norm

    conf = _fit_confidence(rd_final, Q, ori, whitener, fwd_data)

    msg = '---- Fitted : %7.1f ms' % (1000. * t)
    if surf is not None:
        dist_to_inner_skull = _compute_nearest(
            surf['rr'], rd_final[np.newaxis, :], return_dists=True)[1][0]
        msg += (", distance to inner skull : %2.4f mm"
                % (dist_to_inner_skull * 1000.))

    logger.info(msg)
    return rd_final, amp, ori, gof, conf, khi2, nfree, residual_noproj


def _fit_dipole_fixed(min_dist_to_inner_skull, B_orig, t, guess_rrs,
                      guess_data, fwd_data, whitener,
                      fmin_cobyla, ori, rank):
    """Fit a data using a fixed position."""
    B = np.dot(whitener, B_orig)
    B2 = np.dot(B, B)
    if B2 == 0:
        warn('Zero field found for time %s' % t)
        return np.zeros(3), 0, np.zeros(3), 0, np.zeros(6)
    # Compute the dipole moment
    Q, gof, residual_noproj = _fit_Q(guess_data, whitener, B, B2, B_orig,
                                     rd=None, ori=ori)[:3]
    if ori is None:
        amp = np.sqrt(np.dot(Q, Q))
        norm = 1. if amp == 0. else amp
        ori = Q / norm
    else:
        amp = np.dot(Q, ori)
    rd_final = guess_rrs[0]
    # This will be slow, and we don't use it anyway, so omit it for now:
    # conf = _fit_confidence(rd_final, Q, ori, whitener, fwd_data)
    conf = khi2 = nfree = None
    # No corresponding 'logger' message here because it should go *very* fast
    return rd_final, amp, ori, gof, conf, khi2, nfree, residual_noproj


@verbose
def fit_dipole(evoked, cov, bem, trans=None, min_dist=5., n_jobs=1,
               pos=None, ori=None, rank=None, verbose=None):
    """Fit a dipole.

    Parameters
    ----------
    evoked : instance of Evoked
        The dataset to fit.
    cov : str | instance of Covariance
        The noise covariance.
    bem : str | instance of ConductorModel
        The BEM filename (str) or conductor model.
    trans : str | None
        The head<->MRI transform filename. Must be provided unless BEM
        is a sphere model.
    min_dist : float
        Minimum distance (in millimeters) from the dipole to the inner skull.
        Must be positive. Note that because this is a constraint passed to
        a solver it is not strict but close, i.e. for a ``min_dist=5.`` the
        fits could be 4.9 mm from the inner skull.
    %(n_jobs)s
        It is used in field computation and fitting.
    pos : ndarray, shape (3,) | None
        Position of the dipole to use. If None (default), sequential
        fitting (different position and orientation for each time instance)
        is performed. If a position (in head coords) is given as an array,
        the position is fixed during fitting.

        .. versionadded:: 0.12
    ori : ndarray, shape (3,) | None
        Orientation of the dipole to use. If None (default), the
        orientation is free to change as a function of time. If an
        orientation (in head coordinates) is given as an array, ``pos``
        must also be provided, and the routine computes the amplitude and
        goodness of fit of the dipole at the given position and orientation
        for each time instant.

        .. versionadded:: 0.12
    %(rank_None)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    dip : instance of Dipole or DipoleFixed
        The dipole fits. A :class:`mne.DipoleFixed` is returned if
        ``pos`` and ``ori`` are both not None, otherwise a
        :class:`mne.Dipole` is returned.
    residual : instance of Evoked
        The M-EEG data channels with the fitted dipolar activity removed.

    See Also
    --------
    mne.beamformer.rap_music
    Dipole
    DipoleFixed
    read_dipole

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    # This could eventually be adapted to work with other inputs, these
    # are what is needed:

    evoked = evoked.copy()

    # Determine if a list of projectors has an average EEG ref
    if _needs_eeg_average_ref_proj(evoked.info):
        raise ValueError('EEG average reference is mandatory for dipole '
                         'fitting.')
    if min_dist < 0:
        raise ValueError('min_dist should be positive. Got %s' % min_dist)
    if ori is not None and pos is None:
        raise ValueError('pos must be provided if ori is not None')

    data = evoked.data
    if not np.isfinite(data).all():
        raise ValueError('Evoked data must be finite')
    info = evoked.info
    times = evoked.times.copy()
    comment = evoked.comment

    # Convert the min_dist to meters
    min_dist_to_inner_skull = min_dist / 1000.
    del min_dist

    # Figure out our inputs
    neeg = len(pick_types(info, meg=False, eeg=True, ref_meg=False,
                          exclude=[]))
    if isinstance(bem, str):
        bem_extra = bem
    else:
        bem_extra = repr(bem)
        logger.info('BEM               : %s' % bem_extra)
    mri_head_t, trans = _get_trans(trans)
    logger.info('MRI transform     : %s' % trans)
    bem = _setup_bem(bem, bem_extra, neeg, mri_head_t, verbose=False)
    if not bem['is_sphere']:
        # Find the best-fitting sphere
        inner_skull = _bem_find_surface(bem, 'inner_skull')
        inner_skull = inner_skull.copy()
        R, r0 = _fit_sphere(inner_skull['rr'], disp=False)
        # r0 back to head frame for logging
        r0 = apply_trans(mri_head_t['trans'], r0[np.newaxis, :])[0]
        inner_skull['r0'] = r0
        logger.info('Head origin       : '
                    '%6.1f %6.1f %6.1f mm rad = %6.1f mm.'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2], 1000 * R))
        del R, r0
    else:
        r0 = bem['r0']
        if len(bem.get('layers', [])) > 0:
            R = bem['layers'][0]['rad']
            kind = 'rad'
        else:  # MEG-only
            # Use the minimum distance to the MEG sensors as the radius then
            R = np.dot(linalg.inv(info['dev_head_t']['trans']),
                       np.hstack([r0, [1.]]))[:3]  # r0 -> device
            R = R - [info['chs'][pick]['loc'][:3]
                     for pick in pick_types(info, meg=True, exclude=[])]
            if len(R) == 0:
                raise RuntimeError('No MEG channels found, but MEG-only '
                                   'sphere model used')
            R = np.min(np.sqrt(np.sum(R * R, axis=1)))  # use dist to sensors
            kind = 'max_rad'
        logger.info('Sphere model      : origin at (% 7.2f % 7.2f % 7.2f) mm, '
                    '%s = %6.1f mm'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2], kind, R))
        inner_skull = dict(R=R, r0=r0)  # NB sphere model defined in head frame
        del R, r0
    accurate = False  # can be an option later (shouldn't make big diff)

    # Deal with DipoleFixed cases here
    if pos is not None:
        fixed_position = True
        pos = np.array(pos, float)
        if pos.shape != (3,):
            raise ValueError('pos must be None or a 3-element array-like,'
                             ' got %s' % (pos,))
        logger.info('Fixed position    : %6.1f %6.1f %6.1f mm'
                    % tuple(1000 * pos))
        if ori is not None:
            ori = np.array(ori, float)
            if ori.shape != (3,):
                raise ValueError('oris must be None or a 3-element array-like,'
                                 ' got %s' % (ori,))
            norm = np.sqrt(np.sum(ori * ori))
            if not np.isclose(norm, 1):
                raise ValueError('ori must be a unit vector, got length %s'
                                 % (norm,))
            logger.info('Fixed orientation  : %6.4f %6.4f %6.4f mm'
                        % tuple(ori))
        else:
            logger.info('Free orientation   : <time-varying>')
        fit_n_jobs = 1  # only use 1 job to do the guess fitting
    else:
        fixed_position = False
        # Eventually these could be parameters, but they are just used for
        # the initial grid anyway
        guess_grid = 0.02  # MNE-C uses 0.01, but this is faster w/similar perf
        guess_mindist = max(0.005, min_dist_to_inner_skull)
        guess_exclude = 0.02

        logger.info('Guess grid        : %6.1f mm' % (1000 * guess_grid,))
        if guess_mindist > 0.0:
            logger.info('Guess mindist     : %6.1f mm'
                        % (1000 * guess_mindist,))
        if guess_exclude > 0:
            logger.info('Guess exclude     : %6.1f mm'
                        % (1000 * guess_exclude,))
        logger.info('Using %s MEG coil definitions.'
                    % ("accurate" if accurate else "standard"))
        fit_n_jobs = n_jobs
    if isinstance(cov, str):
        logger.info('Noise covariance  : %s' % (cov,))
        cov = read_cov(cov, verbose=False)
    logger.info('')

    _print_coord_trans(mri_head_t)
    _print_coord_trans(info['dev_head_t'])
    logger.info('%d bad channels total' % len(info['bads']))

    # Forward model setup (setup_forward_model from setup.c)
    ch_types = evoked.get_channel_types()

    megcoils, compcoils, megnames, meg_info = [], [], [], None
    eegels, eegnames = [], []
    if 'grad' in ch_types or 'mag' in ch_types:
        megcoils, compcoils, megnames, meg_info = \
            _prep_meg_channels(info, exclude='bads',
                               accurate=accurate, verbose=verbose)
    if 'eeg' in ch_types:
        eegels, eegnames = _prep_eeg_channels(info, exclude='bads',
                                              verbose=verbose)

    # Ensure that MEG and/or EEG channels are present
    if len(megcoils + eegels) == 0:
        raise RuntimeError('No MEG or EEG channels found.')

    # Whitener for the data
    logger.info('Decomposing the sensor noise covariance matrix...')
    picks = pick_types(info, meg=True, eeg=True, ref_meg=False)

    # In case we want to more closely match MNE-C for debugging:
    # from .io.pick import pick_info
    # from .cov import prepare_noise_cov
    # info_nb = pick_info(info, picks)
    # cov = prepare_noise_cov(cov, info_nb, info_nb['ch_names'], verbose=False)
    # nzero = (cov['eig'] > 0)
    # n_chan = len(info_nb['ch_names'])
    # whitener = np.zeros((n_chan, n_chan), dtype=np.float64)
    # whitener[nzero, nzero] = 1.0 / np.sqrt(cov['eig'][nzero])
    # whitener = np.dot(whitener, cov['eigvec'])

    whitener, _, rank = compute_whitener(cov, info, picks=picks,
                                         rank=rank, return_rank=True)

    # Proceed to computing the fits (make_guess_data)
    if fixed_position:
        guess_src = dict(nuse=1, rr=pos[np.newaxis], inuse=np.array([True]))
        logger.info('Compute forward for dipole location...')
    else:
        logger.info('\n---- Computing the forward solution for the guesses...')
        guess_src = _make_guesses(inner_skull, guess_grid, guess_exclude,
                                  guess_mindist, n_jobs=n_jobs)[0]
        # grid coordinates go from mri to head frame
        transform_surface_to(guess_src, 'head', mri_head_t)
        logger.info('Go through all guess source locations...')

    # inner_skull goes from mri to head frame
    if 'rr' in inner_skull:
        transform_surface_to(inner_skull, 'head', mri_head_t)
    if fixed_position:
        if 'rr' in inner_skull:
            check = _surface_constraint(pos, inner_skull,
                                        min_dist_to_inner_skull)
        else:
            check = _sphere_constraint(
                pos, inner_skull['r0'],
                R_adj=inner_skull['R'] - min_dist_to_inner_skull)
        if check <= 0:
            raise ValueError('fixed position is %0.1fmm outside the inner '
                             'skull boundary' % (-1000 * check,))

    # C code computes guesses w/sphere model for speed, don't bother here
    fwd_data = dict(coils_list=[megcoils, eegels], infos=[meg_info, None],
                    ccoils_list=[compcoils, None], coil_types=['meg', 'eeg'],
                    inner_skull=inner_skull)
    # fwd_data['inner_skull'] in head frame, bem in mri, confusing...
    _prep_field_computation(guess_src['rr'], bem, fwd_data, n_jobs,
                            verbose=False)
    guess_fwd, guess_fwd_orig, guess_fwd_scales = _dipole_forwards(
        fwd_data, whitener, guess_src['rr'], n_jobs=fit_n_jobs)
    # decompose ahead of time
    guess_fwd_svd = [linalg.svd(fwd, overwrite_a=False, full_matrices=False)
                     for fwd in np.array_split(guess_fwd,
                                               len(guess_src['rr']))]
    guess_data = dict(fwd=guess_fwd, fwd_svd=guess_fwd_svd,
                      fwd_orig=guess_fwd_orig, scales=guess_fwd_scales)
    del guess_fwd, guess_fwd_svd, guess_fwd_orig, guess_fwd_scales  # destroyed
    logger.info('[done %d source%s]' % (guess_src['nuse'],
                                        _pl(guess_src['nuse'])))

    # Do actual fits
    data = data[picks]
    ch_names = [info['ch_names'][p] for p in picks]
    proj_op = make_projector(info['projs'], ch_names, info['bads'])[0]
    fun = _fit_dipole_fixed if fixed_position else _fit_dipole
    out = _fit_dipoles(
        fun, min_dist_to_inner_skull, data, times, guess_src['rr'],
        guess_data, fwd_data, whitener, ori, n_jobs, rank)
    assert len(out) == 8
    if fixed_position and ori is not None:
        # DipoleFixed
        data = np.array([out[1], out[3]])
        out_info = deepcopy(info)
        loc = np.concatenate([pos, ori, np.zeros(6)])
        out_info['chs'] = [
            dict(ch_name='dip 01', loc=loc, kind=FIFF.FIFFV_DIPOLE_WAVE,
                 coord_frame=FIFF.FIFFV_COORD_UNKNOWN, unit=FIFF.FIFF_UNIT_AM,
                 coil_type=FIFF.FIFFV_COIL_DIPOLE,
                 unit_mul=0, range=1, cal=1., scanno=1, logno=1),
            dict(ch_name='goodness', loc=np.full(12, np.nan),
                 kind=FIFF.FIFFV_GOODNESS_FIT, unit=FIFF.FIFF_UNIT_AM,
                 coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                 coil_type=FIFF.FIFFV_COIL_NONE,
                 unit_mul=0, range=1., cal=1., scanno=2, logno=100)]
        for key in ['hpi_meas', 'hpi_results', 'projs']:
            out_info[key] = list()
        for key in ['acq_pars', 'acq_stim', 'description', 'dig',
                    'experimenter', 'hpi_subsystem', 'proj_id', 'proj_name',
                    'subject_info']:
            out_info[key] = None
        out_info['bads'] = []
        out_info._update_redundant()
        out_info._check_consistency()
        dipoles = DipoleFixed(out_info, data, times, evoked.nave,
                              evoked._aspect_kind, comment=comment)
    else:
        dipoles = Dipole(times, out[0], out[1], out[2], out[3], comment,
                         out[4], out[5], out[6])
    residual = evoked.copy().apply_proj()  # set the projs active
    residual.data[picks] = np.dot(proj_op, out[-1])
    logger.info('%d time points fitted' % len(dipoles.times))
    return dipoles, residual


def get_phantom_dipoles(kind='vectorview'):
    """Get standard phantom dipole locations and orientations.

    Parameters
    ----------
    kind : str
        Get the information for the given system:

            ``vectorview`` (default)
              The Neuromag VectorView phantom.
            ``otaniemi``
              The older Neuromag phantom used at Otaniemi.

    Returns
    -------
    pos : ndarray, shape (n_dipoles, 3)
        The dipole positions.
    ori : ndarray, shape (n_dipoles, 3)
        The dipole orientations.

    Notes
    -----
    The Elekta phantoms have a radius of 79.5mm, and HPI coil locations
    in the XY-plane at the axis extrema (e.g., (79.5, 0), (0, -79.5), ...).
    """
    _check_option('kind', kind, ['vectorview', 'otaniemi'])
    if kind == 'vectorview':
        # these values were pulled from a scanned image provided by
        # Elekta folks
        a = np.array([59.7, 48.6, 35.8, 24.8, 37.2, 27.5, 15.8, 7.9])
        b = np.array([46.1, 41.9, 38.3, 31.5, 13.9, 16.2, 20.0, 19.3])
        x = np.concatenate((a, [0] * 8, -b, [0] * 8))
        y = np.concatenate(([0] * 8, -a, [0] * 8, b))
        c = [22.9, 23.5, 25.5, 23.1, 52.0, 46.4, 41.0, 33.0]
        d = [44.4, 34.0, 21.6, 12.7, 62.4, 51.5, 39.1, 27.9]
        z = np.concatenate((c, c, d, d))
        signs = ([1, -1] * 4 + [-1, 1] * 4) * 2
    elif kind == 'otaniemi':
        # these values were pulled from an Neuromag manual
        # (NM20456A, 13.7.1999, p.65)
        a = np.array([56.3, 47.6, 39.0, 30.3])
        b = np.array([32.5, 27.5, 22.5, 17.5])
        c = np.zeros(4)
        x = np.concatenate((a, b, c, c, -a, -b, c, c))
        y = np.concatenate((c, c, -a, -b, c, c, b, a))
        z = np.concatenate((b, a, b, a, b, a, a, b))
        signs = [-1] * 8 + [1] * 16 + [-1] * 8
    pos = np.vstack((x, y, z)).T / 1000.
    # Locs are always in XZ or YZ, and so are the oris. The oris are
    # also in the same plane and tangential, so it's easy to determine
    # the orientation.
    ori = list()
    for pi, this_pos in enumerate(pos):
        this_ori = np.zeros(3)
        idx = np.where(this_pos == 0)[0]
        # assert len(idx) == 1
        idx = np.setdiff1d(np.arange(3), idx[0])
        this_ori[idx] = (this_pos[idx][::-1] /
                         np.linalg.norm(this_pos[idx])) * [1, -1]
        this_ori *= signs[pi]
        # Now we have this quality, which we could uncomment to
        # double-check:
        # np.testing.assert_allclose(np.dot(this_ori, this_pos) /
        #                            np.linalg.norm(this_pos), 0,
        #                            atol=1e-15)
        ori.append(this_ori)
    ori = np.array(ori)
    return pos, ori


def _concatenate_dipoles(dipoles):
    """Concatenate a list of dipoles."""
    times, pos, amplitude, ori, gof = [], [], [], [], []
    for dipole in dipoles:
        times.append(dipole.times)
        pos.append(dipole.pos)
        amplitude.append(dipole.amplitude)
        ori.append(dipole.ori)
        gof.append(dipole.gof)

    return Dipole(np.concatenate(times), np.concatenate(pos),
                  np.concatenate(amplitude), np.concatenate(ori),
                  np.concatenate(gof), name=None)
