"""Single-dipole functions and classes."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy
from functools import partial
import re

import numpy as np
from scipy import linalg

from .cov import read_cov, _get_whitener_data
from .io.constants import FIFF
from .io.pick import pick_types, channel_type
from .io.proj import make_projector, _needs_eeg_average_ref_proj
from .bem import _fit_sphere
from .evoked import _read_evoked, _aspect_rev, _write_evokeds
from .transforms import (_print_coord_trans, _coord_frame_name,
                         apply_trans, invert_transform, Transform)
from .viz.evoked import _plot_evoked

from .forward._make_forward import (_get_trans, _setup_bem,
                                    _prep_meg_channels, _prep_eeg_channels)
from .forward._compute_forward import (_compute_forwards_meeg,
                                       _prep_field_computation)

from .externals.six import string_types
from .surface import (transform_surface_to, _normalize_vectors,
                      _get_ico_surface, _compute_nearest)
from .bem import _bem_find_surface, _bem_explain_surface
from .source_space import (_make_volume_source_space, SourceSpaces,
                           _points_outside_surface)
from .parallel import parallel_func
from .utils import (logger, verbose, _time_mask, warn, _check_fname,
                    check_fname, _pl)


class Dipole(object):
    """Dipole class for sequential dipole fits.

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
        The amplitude of the dipoles (nAm).
    ori : array, shape (n_dipoles, 3)
        The dipole orientations (normalized to unit length).
    gof : array, shape (n_dipoles,)
        The goodness of fit.
    name : str | None
        Name of the dipole.

    See Also
    --------
    read_dipole
    DipoleFixed

    Notes
    -----
    This class is for sequential dipole fits, where the position
    changes as a function of time. For fixed dipole fits, where the
    position is fixed as a function of time, use :class:`mne.DipoleFixed`.
    """

    def __init__(self, times, pos, amplitude, ori, gof,
                 name=None):  # noqa: D102
        self.times = np.array(times)
        self.pos = np.array(pos)
        self.amplitude = np.array(amplitude)
        self.ori = np.array(ori)
        self.gof = np.array(gof)
        self.name = name

    def __repr__(self):  # noqa: D105
        s = "n_times : %s" % len(self.times)
        s += ", tmin : %s" % np.min(self.times)
        s += ", tmax : %s" % np.max(self.times)
        return "<Dipole  |  %s>" % s

    def save(self, fname):
        """Save dipole in a .dip file.

        Parameters
        ----------
        fname : str
            The name of the .dip file.
        """
        fmt = "  %7.1f %7.1f %8.2f %8.2f %8.2f %8.3f %8.3f %8.3f %8.3f %6.1f"
        # NB CoordinateSystem is hard-coded as Head here
        with open(fname, 'wb') as fid:
            fid.write('# CoordinateSystem "Head"\n'.encode('utf-8'))
            fid.write('#   begin     end   X (mm)   Y (mm)   Z (mm)'
                      '   Q(nAm)  Qx(nAm)  Qy(nAm)  Qz(nAm)    g/%\n'
                      .encode('utf-8'))
            t = self.times[:, np.newaxis] * 1000.
            gof = self.gof[:, np.newaxis]
            amp = 1e9 * self.amplitude[:, np.newaxis]
            out = np.concatenate((t, t, self.pos / 1e-3, amp,
                                  self.ori * amp, gof), axis=-1)
            np.savetxt(fid, out, fmt=fmt)
            if self.name is not None:
                fid.write(('## Name "%s dipoles" Style "Dipoles"'
                           % self.name).encode('utf-8'))

    def crop(self, tmin=None, tmax=None):
        """Crop data to a given time interval.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        """
        sfreq = None
        if len(self.times) > 1:
            sfreq = 1. / np.median(np.diff(self.times))
        mask = _time_mask(self.times, tmin, tmax, sfreq=sfreq)
        for attr in ('times', 'pos', 'gof', 'amplitude', 'ori'):
            setattr(self, attr, getattr(self, attr)[mask])

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
                       bgcolor=(1, 1, 1), opacity=0.3,
                       brain_color=(1, 1, 0), fig_name=None,
                       fig_size=(600, 600), mode='cone',
                       scale_factor=0.1e-1, colors=None, verbose=None):
        """Plot dipole locations as arrows.

        Parameters
        ----------
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
        fig_name : tuple of length 2
            Mayavi figure name.
        fig_size : tuple of length 2
            Mayavi figure size.
        mode : str
            Should be ``'cone'`` or ``'sphere'`` to specify how the
            dipoles should be shown.
        scale_factor : float
            The scaling applied to amplitudes for the plot.
        colors: list of colors | None
            Color to plot with each dipole. If None defaults colors are used.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        fig : instance of mlab.Figure
            The mayavi figure.
        """
        from .viz import plot_dipole_locations
        dipoles = []
        for t in self.times:
            dipoles.append(self.copy())
            dipoles[-1].crop(t, t)
        return plot_dipole_locations(
            dipoles, trans, subject, subjects_dir, bgcolor, opacity,
            brain_color, fig_name, fig_size, mode, scale_factor,
            colors)

    def plot_amplitudes(self, color='k', show=True):
        """Plot the dipole amplitudes as a function of time.

        Parameters
        ----------
        color: matplotlib Color
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
        return Dipole(
            selected_times, selected_pos, selected_amplitude, selected_ori,
            selected_gof, selected_name)

    def __len__(self):
        """The number of dipoles.

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
    """Helper to read a fixed dipole FIF file."""
    logger.info('Reading %s ...' % fname)
    _check_fname(fname, overwrite=True, must_exist=True)
    info, nave, aspect_kind, first, last, comment, times, data = \
        _read_evoked(fname)
    return DipoleFixed(info, data, times, nave, aspect_kind, first, last,
                       comment)


class DipoleFixed(object):
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
    first : int
        First sample.
    last : int
        Last sample.
    comment : str
        The dipole comment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    read_dipole
    Dipole

    Notes
    -----
    This class is for fixed-position dipole fits, where the position
    (and maybe orientation) is static over time. For sequential dipole fits,
    where the position can change a function of time, use :class:`mne.Dipole`.

    .. versionadded:: 0.12
    """

    @verbose
    def __init__(self, info, data, times, nave, aspect_kind, first, last,
                 comment, verbose=None):  # noqa: D102
        self.info = info
        self.nave = nave
        self._aspect_kind = aspect_kind
        self.kind = _aspect_rev.get(str(aspect_kind), 'Unknown')
        self.first = first
        self.last = last
        self.comment = comment
        self.times = times
        self.data = data
        self.verbose = verbose

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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).
        """
        check_fname(fname, 'DipoleFixed', ('-dip.fif', '-dip.fif.gz'),
                    ('.fif', '.fif.gz'))
        _write_evokeds(fname, self, check=False)

    def plot(self, show=True):
        """Plot dipole data.

        Parameters
        ----------
        show : bool
            Call pyplot.show() at the end or not.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure containing the time courses.
        """
        return _plot_evoked(self, picks=None, exclude=(), unit=True, show=show,
                            ylim=None, xlim='tight', proj=False, hline=None,
                            units=None, scalings=None, titles=None, axes=None,
                            gfp=False, window_title=None, spatial_colors=False,
                            plot_type="butterfly", selectable=False)


# #############################################################################
# IO
@verbose
def read_dipole(fname, verbose=None):
    """Read .dip file from Neuromag/xfit or MNE.

    Parameters
    ----------
    fname : str
        The name of the .dip or .fif file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    dipole : instance of Dipole or DipoleFixed
        The dipole.

    See Also
    --------
    mne.Dipole
    mne.DipoleFixed
    """
    _check_fname(fname, overwrite=True, must_exist=True)
    if fname.endswith('.fif') or fname.endswith('.fif.gz'):
        return _read_dipole_fixed(fname)
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
    fields = re.sub('([X|Y|Z] )\(mm\)',  # "X (mm)", etc.
                    lambda match: match.group(1).strip() + '/mm', def_line)
    fields = re.sub('\((.*?)\)',  # "Q(nAm)", etc.
                    lambda match: '/' + match.group(1), fields)
    fields = re.sub('(begin|end) ',  # "begin" and "end" with no units
                    lambda match: match.group(1) + '/ms', fields)
    fields = fields.lower().split()
    used_fields = ('begin/ms',
                   'x/mm', 'y/mm', 'z/mm',
                   'q/nam',
                   'qx/nam', 'qy/nam', 'qz/nam',
                   'g/%')
    missing_fields = sorted(set(used_fields) - set(fields))
    if len(missing_fields) > 0:
        raise RuntimeError('Could not find necessary fields in header: %s'
                           % (missing_fields,))
    ignored_fields = sorted(set(fields) - set(used_fields) - set(['end/ms']))
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
    idx = [fields.index(field) for field in used_fields]
    assert len(idx) == 9
    times = data[:, idx[0]] / 1000.
    pos = 1e-3 * data[:, idx[1:4]]  # put data in meters
    amplitude = data[:, idx[4]]
    norm = amplitude.copy()
    amplitude /= 1e9
    norm[norm == 0] = 1
    ori = data[:, idx[5:8]] / norm[:, np.newaxis]
    gof = data[:, idx[8]]
    return Dipole(times, pos, amplitude, ori, gof, name)


# #############################################################################
# Fitting

def _dipole_forwards(fwd_data, whitener, rr, n_jobs=1):
    """Compute the forward solution and do other nice stuff."""
    B = _compute_forwards_meeg(rr, fwd_data, n_jobs, verbose=False)
    B = np.concatenate(B, axis=1)
    B_orig = B.copy()

    # Apply projection and whiten (cov has projections already)
    B = np.dot(B, whitener.T)

    # column normalization doesn't affect our fitting, so skip for now
    # S = np.sum(B * B, axis=1)  # across channels
    # scales = np.repeat(3. / np.sqrt(np.sum(np.reshape(S, (len(rr), 3)),
    #                                        axis=1)), 3)
    # B *= scales[:, np.newaxis]
    scales = np.ones(3)
    return B, B_orig, scales


def _make_guesses(surf_or_rad, r0, grid, exclude, mindist, n_jobs):
    """Make a guess space inside a sphere or BEM surface."""
    if isinstance(surf_or_rad, dict):
        surf = surf_or_rad
        logger.info('Guess surface (%s) is in %s coordinates'
                    % (_bem_explain_surface(surf['id']),
                       _coord_frame_name(surf['coord_frame'])))
    else:
        radius = surf_or_rad[0]
        logger.info('Making a spherical guess space with radius %7.1f mm...'
                    % (1000 * radius))
        surf = _get_ico_surface(3)
        _normalize_vectors(surf['rr'])
        surf['rr'] *= radius
        surf['rr'] += r0
    logger.info('Filtering (grid = %6.f mm)...' % (1000 * grid))
    src = _make_volume_source_space(surf, grid, exclude, 1000 * mindist,
                                    do_neighbors=False, n_jobs=n_jobs)
    # simplify the result to make things easier later
    src = dict(rr=src['rr'][src['vertno']], nn=src['nn'][src['vertno']],
               nuse=src['nuse'], coord_frame=src['coord_frame'],
               vertno=np.arange(src['nuse']))
    return SourceSpaces([src])


def _fit_eval(rd, B, B2, fwd_svd=None, fwd_data=None, whitener=None):
    """Calculate the residual sum of squares."""
    if fwd_svd is None:
        fwd = _dipole_forwards(fwd_data, whitener, rd[np.newaxis, :])[0]
        uu, sing, vv = linalg.svd(fwd, overwrite_a=True, full_matrices=False)
    else:
        uu, sing, vv = fwd_svd
    gof = _dipole_gof(uu, sing, vv, B, B2)[0]
    # mne-c uses fitness=B2-Bm2, but ours (1-gof) is just a normalized version
    return 1. - gof


def _dipole_gof(uu, sing, vv, B, B2):
    """Calculate the goodness of fit from the forward SVD."""
    ncomp = 3 if sing[2] / sing[0] > 0.2 else 2
    one = np.dot(vv[:ncomp], B)
    Bm2 = np.sum(one * one)
    gof = Bm2 / B2
    return gof, one


def _fit_Q(fwd_data, whitener, proj_op, B, B2, B_orig, rd, ori=None):
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
        # Counteract the effect of column normalization
        Q = scales[0] * np.sum(uu.T[:ncomp] *
                               (one / sing[:ncomp])[:, np.newaxis], axis=0)
    else:
        fwd = np.dot(ori[np.newaxis], fwd)
        sing = np.linalg.norm(fwd)
        one = np.dot(fwd / sing, B)
        gof = (one * one)[0] / B2
        Q = ori * (scales[0] * np.sum(one / sing))
    B_residual = _compute_residual(proj_op, B_orig, fwd_orig, Q)
    return Q, gof, B_residual


def _compute_residual(proj_op, B_orig, fwd_orig, Q):
    """Compute the residual."""
    # apply the projector to both elements
    return np.dot(proj_op, B_orig) - np.dot(np.dot(Q, fwd_orig), proj_op.T)


def _fit_dipoles(fun, min_dist_to_inner_skull, data, times, guess_rrs,
                 guess_data, fwd_data, whitener, proj_op, ori, n_jobs):
    """Fit a single dipole to the given whitened, projected data."""
    from scipy.optimize import fmin_cobyla
    parallel, p_fun, _ = parallel_func(fun, n_jobs)
    # parallel over time points
    res = parallel(p_fun(min_dist_to_inner_skull, B, t, guess_rrs,
                         guess_data, fwd_data, whitener, proj_op,
                         fmin_cobyla, ori)
                   for B, t in zip(data.T, times))
    pos = np.array([r[0] for r in res])
    amp = np.array([r[1] for r in res])
    ori = np.array([r[2] for r in res])
    gof = np.array([r[3] for r in res]) * 100  # convert to percentage
    residual = np.array([r[4] for r in res]).T

    return pos, amp, ori, gof, residual


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
                guess_data, fwd_data, whitener, proj_op,
                fmin_cobyla, ori):
    """Fit a single bit of data."""
    B = np.dot(whitener, B_orig)

    # make constraint function to keep the solver within the inner skull
    if isinstance(fwd_data['inner_skull'], dict):  # bem
        surf = fwd_data['inner_skull']
        constraint = partial(_surface_constraint, surf=surf,
                             min_dist_to_inner_skull=min_dist_to_inner_skull)
    else:  # sphere
        surf = None
        R, r0 = fwd_data['inner_skull']
        constraint = partial(_sphere_constraint, r0=r0,
                             R_adj=R - min_dist_to_inner_skull)
        del R, r0

    # Find a good starting point (find_best_guess in C)
    B2 = np.dot(B, B)
    if B2 == 0:
        warn('Zero field found for time %s' % t)
        return np.zeros(3), 0, np.zeros(3), 0, B

    idx = np.argmin([_fit_eval(guess_rrs[[fi], :], B, B2, fwd_svd)
                     for fi, fwd_svd in enumerate(guess_data['fwd_svd'])])
    x0 = guess_rrs[idx]
    fun = partial(_fit_eval, B=B, B2=B2, fwd_data=fwd_data, whitener=whitener)

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
    Q, gof, residual = _fit_Q(fwd_data, whitener, proj_op, B, B2, B_orig,
                              rd_final, ori=ori)
    amp = np.sqrt(np.dot(Q, Q))
    norm = 1. if amp == 0. else amp
    ori = Q / norm

    msg = '---- Fitted : %7.1f ms' % (1000. * t)
    if surf is not None:
        dist_to_inner_skull = _compute_nearest(
            surf['rr'], rd_final[np.newaxis, :], return_dists=True)[1][0]
        msg += (", distance to inner skull : %2.4f mm"
                % (dist_to_inner_skull * 1000.))

    logger.info(msg)
    return rd_final, amp, ori, gof, residual


def _fit_dipole_fixed(min_dist_to_inner_skull, B_orig, t, guess_rrs,
                      guess_data, fwd_data, whitener, proj_op,
                      fmin_cobyla, ori):
    """Fit a data using a fixed position."""
    B = np.dot(whitener, B_orig)
    B2 = np.dot(B, B)
    if B2 == 0:
        warn('Zero field found for time %s' % t)
        return np.zeros(3), 0, np.zeros(3), 0
    # Compute the dipole moment
    Q, gof, residual = _fit_Q(guess_data, whitener, proj_op, B, B2, B_orig,
                              rd=None, ori=ori)
    if ori is None:
        amp = np.sqrt(np.dot(Q, Q))
        norm = 1. if amp == 0. else amp
        ori = Q / norm
    else:
        amp = np.dot(Q, ori)
    # No corresponding 'logger' message here because it should go *very* fast
    return guess_rrs[0], amp, ori, gof, residual


@verbose
def fit_dipole(evoked, cov, bem, trans=None, min_dist=5., n_jobs=1,
               pos=None, ori=None, verbose=None):
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
        Minimum distance (in milimeters) from the dipole to the inner skull.
        Must be positive. Note that because this is a constraint passed to
        a solver it is not strict but close, i.e. for a ``min_dist=5.`` the
        fits could be 4.9 mm from the inner skull.
    n_jobs : int
        Number of jobs to run in parallel (used in field computation
        and fitting).
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

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    dip : instance of Dipole or DipoleFixed
        The dipole fits. A :class:`mne.DipoleFixed` is returned if
        ``pos`` and ``ori`` are both not None.
    residual : ndarray, shape (n_meeg_channels, n_times)
        The good M-EEG data channels with the fitted dipolar activity
        removed.

    See Also
    --------
    mne.beamformer.rap_music

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
    if isinstance(bem, string_types):
        bem_extra = bem
    else:
        bem_extra = repr(bem)
        logger.info('BEM               : %s' % bem_extra)
    if trans is not None:
        logger.info('MRI transform     : %s' % trans)
        mri_head_t, trans = _get_trans(trans)
    else:
        mri_head_t = Transform('head', 'mri')
    bem = _setup_bem(bem, bem_extra, neeg, mri_head_t, verbose=False)
    if not bem['is_sphere']:
        if trans is None:
            raise ValueError('mri must not be None if BEM is provided')
        # Find the best-fitting sphere
        inner_skull = _bem_find_surface(bem, 'inner_skull')
        inner_skull = inner_skull.copy()
        R, r0 = _fit_sphere(inner_skull['rr'], disp=False)
        # r0 back to head frame for logging
        r0 = apply_trans(mri_head_t['trans'], r0[np.newaxis, :])[0]
        logger.info('Head origin       : '
                    '%6.1f %6.1f %6.1f mm rad = %6.1f mm.'
                    % (1000 * r0[0], 1000 * r0[1], 1000 * r0[2], 1000 * R))
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
        inner_skull = [R, r0]  # NB sphere model defined in head frame
    r0_mri = apply_trans(invert_transform(mri_head_t)['trans'],
                         r0[np.newaxis, :])[0]
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
    if isinstance(cov, string_types):
        logger.info('Noise covariance  : %s' % (cov,))
        cov = read_cov(cov, verbose=False)
    logger.info('')

    _print_coord_trans(mri_head_t)
    _print_coord_trans(info['dev_head_t'])
    logger.info('%d bad channels total' % len(info['bads']))

    # Forward model setup (setup_forward_model from setup.c)
    ch_types = [channel_type(info, idx) for idx in range(info['nchan'])]

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
    # whitener = np.zeros((n_chan, n_chan), dtype=np.float)
    # whitener[nzero, nzero] = 1.0 / np.sqrt(cov['eig'][nzero])
    # whitener = np.dot(whitener, cov['eigvec'])

    whitener = _get_whitener_data(info, cov, picks, verbose=False)

    # Proceed to computing the fits (make_guess_data)
    if fixed_position:
        guess_src = dict(nuse=1, rr=pos[np.newaxis], inuse=np.array([True]))
        logger.info('Compute forward for dipole location...')
    else:
        logger.info('\n---- Computing the forward solution for the guesses...')
        guess_src = _make_guesses(inner_skull, r0_mri,
                                  guess_grid, guess_exclude, guess_mindist,
                                  n_jobs=n_jobs)[0]
        # grid coordinates go from mri to head frame
        transform_surface_to(guess_src, 'head', mri_head_t)
        logger.info('Go through all guess source locations...')

    # inner_skull goes from mri to head frame
    if isinstance(inner_skull, dict):
        transform_surface_to(inner_skull, 'head', mri_head_t)
    if fixed_position:
        if isinstance(inner_skull, dict):
            check = _surface_constraint(pos, inner_skull,
                                        min_dist_to_inner_skull)
        else:
            check = _sphere_constraint(pos, r0,
                                       R_adj=R - min_dist_to_inner_skull)
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
        guess_data, fwd_data, whitener, proj_op, ori, n_jobs)
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
            dict(ch_name='goodness', loc=np.zeros(12),
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
        out_info._update_redundant()
        out_info._check_consistency()
        dipoles = DipoleFixed(out_info, data, times, evoked.nave,
                              evoked._aspect_kind, evoked.first, evoked.last,
                              comment)
    else:
        dipoles = Dipole(times, out[0], out[1], out[2], out[3], comment)
    residual = out[4]
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
    _valid_types = ('vectorview', 'otaniemi')
    if not isinstance(kind, string_types) or kind not in _valid_types:
        raise ValueError('kind must be one of %s, got %s'
                         % (_valid_types, kind,))
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
    elif kind == 'otaniemi':
        # these values were pulled from an Neuromag manual
        # (NM20456A, 13.7.1999, p.65)
        a = np.array([56.3, 47.6, 39.0, 30.3])
        b = np.array([32.5, 27.5, 22.5, 17.5])
        c = np.zeros(4)
        x = np.concatenate((a, b, c, c, -a, -b, c, c))
        y = np.concatenate((c, c, -a, -b, c, c, b, a))
        z = np.concatenate((b, a, b, a, b, a, a, b))
    pos = np.vstack((x, y, z)).T / 1000.
    # Locs are always in XZ or YZ, and so are the oris. The oris are
    # also in the same plane and tangential, so it's easy to determine
    # the orientation.
    ori = list()
    for this_pos in pos:
        this_ori = np.zeros(3)
        idx = np.where(this_pos == 0)[0]
        # assert len(idx) == 1
        idx = np.setdiff1d(np.arange(3), idx[0])
        this_ori[idx] = (this_pos[idx][::-1] /
                         np.linalg.norm(this_pos[idx])) * [1, -1]
        # Now we have this quality, which we could uncomment to
        # double-check:
        # np.testing.assert_allclose(np.dot(this_ori, this_pos) /
        #                            np.linalg.norm(this_pos), 0,
        #                            atol=1e-15)
        ori.append(this_ori)
    ori = np.array(ori)
    return pos, ori
