# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>

# This code was adapted and relicensed (with BSD form) with permission from
# Jussi Nurminen

# License: BSD (3-clause)

# Note, there are an absurd number of different notations for spherical
# harmonics (partly because there is also no accepted standard for spherical
# coordinates). Here, we purposefully stay away from shorthand notation in
# both and use explicit terms to avoid confusion.

# TODO: write in equation numbers from Samu's paper

import numpy as np
from scipy.special import sph_harm, lpmv
from ..forward._compute_forward import _concatenate_coils
from ..forward._lead_dots import (_get_legen_table, _get_legen_lut_fast,
                                  _get_legen_lut_accurate)
#from ..fixes import partial
from scipy.misc import factorial as fact


def maxwell_filter(raw, origin, int_order=8, ext_order=3):
    """Apply Maxwell filter to data using spherical harmonics.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    origin : tuple, shape (3,)
        Origin of head
    in_order : int
        Order of internal component of spherical expansion
    out_order : int
        Order of external component of spherical expansion

    Returns
    -------
    raw_sss : instance of mne.io.Raw
        The raw data with Maxwell filtering applied
    """

    # TODO: Figure out all parameters required
    # TODO: Error checks on input parameters

    # TODO: Compute spherical harmonics
    # TODO: Project data into spherical harmonics space
    # TODO: Reconstruct and return Raw file object


def _sss_basis(origin, coils, int_order=8, ext_order=3):
    """Compute SSS basis for given conditions.

    Parameters
    ----------
    origin : ndarray, shape (3,)
        Origin of the multipolar moment space.
    coils : list
        List of MEG coils. Each should contain coil information dict
    int_order : int
        Order of the internal multipolar moment space
    ext_order : int
        Order of the external multipolar moment space

    Returns
    -------
    list
        List of length 2 containing internal and external basis sets
    """
    r_int_pts, ncoils, wcoils, int_pts = _concatenate_coils(coils)
    n_sens = len(int_pts)
    n_bases = (int_order - 1) ** 2 + (ext_order - 1) ** 2 - 2
    n_int_pts = len(r_int_pts)
    int_lens = np.insert(np.cumsum(int_pts), obj=0, values=0)

    S_in = np.empty(((int_order + 1) ** 2, n_sens))
    S_out = np.empty(((ext_order + 1) ** 2, n_sens))
    S_in.fill(np.nan)
    S_out.fill(np.nan)

    assert n_bases <= n_sens, ('Number of requested bases (%s) exceeds number '
                               'of sensors (%s)' % (str(n_bases), str(n_sens)))

    # Compute internal basis vectors
    for deg in range(0, int_order + 1):
        for order in range(-deg, deg + 1):

            # Compute position vector between origin and coil integration pts
            cvec = r_int_pts - origin * np.ones((n_int_pts, 1))
            # Compute gradient for integration point position vectors
            grads = -1 * _grad_in_components(deg, order, cvec[:, 0],
                                             cvec[:, 1], cvec[:, 2])

            # Gradients dotted with integration point normals and weighted
            a1_all = wcoils * np.einsum('ij,ij->i', cvec, grads)

            # For order and degree, sum across integration pts for each sensor
            for pt_i in range(0, len(int_lens) - 1):
                int_pts_sum = np.sum(a1_all[int_lens[pt_i]:int_lens[pt_i + 1]])
                S_in[deg ** 2 + deg + order, pt_i] = int_pts_sum

    # Compute external basis vectors
    for deg in range(0, ext_order + 1):
        for order in range(-deg, deg + 1):

            # Compute position vector between origin and coil integration pts
            cvec = r_int_pts - origin * np.ones((n_int_pts, 1))
            # Compute gradient for integration point position vectors
            grads = -1 * _grad_out_components(deg, order, cvec[:, 0],
                                              cvec[:, 1], cvec[:, 2])

            # Gradients dotted with integration point normals and weighted
            b1_all = wcoils * np.einsum('ij,ij->i', cvec, grads)

            # For order and degree, sum across integration pts for each sensor
            for pt_i in range(0, len(int_lens) - 1):
                int_pts_sum = np.sum(b1_all[int_lens[pt_i]:int_lens[pt_i + 1]])
                S_out[deg ** 2 + deg + order, pt_i] = int_pts_sum

    return [S_in, S_out]


def _sph_harmonic(degree, order, az, pol):
    """Evaluate point in specified multipolar moment.

    When using, pay close attention to inputs. Spherical harmonic notation for
    order/degree, and theta/phi are both reversed in original SSS work compared
    to many other sources.

    Parameters
    ----------
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    az : float
        Azimuthal (longitudinal) spherical coordinate [0, 2*pi]. 0 is aligned
        with x-axis.
    pol : float
        Polar (or colatitudinal) spherical coordinate [0, pi]. 0 is aligned
        with z-axis.

    Returns
    -------
    y : complex float
        The spherical harmonic value at the specified azimuth and polar angles
    """
    assert np.abs(order) <= degree, ('Absolute value of expansion coefficient'
                                     ' must be <= degree')

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # TODO: Decide on notation for spherical coords
    # TODO: Should factorial function use floating or long precision?
    # TODO: Check that [-m to m] convention is correct for all equations

    #Ensure that polar and azimuth angles are arrays
    azimuth = np.array(az)
    polar = np.array(pol)

    # Real valued spherical expansion as given by Wikso, (11)
    base = np.sqrt((2 * degree + 1) / (4 * np.pi) * fact(degree - order) /
                   fact(degree + order)) * lpmv(order, degree, np.cos(polar))
    if order < 0:
        return base * np.sin(order * azimuth)
    else:
        return base * np.cos(order * azimuth)

    # TODO: Check speed of taking real part of scipy's sph harmonic function
    # Note reversal in notation order between scipy and original SSS papers
    # Degree/order and theta/phi reversed
    #return np.real(sph_harm(order, degree, az, pol))


def _alegendre_deriv(degree, order, val):
    """Compute the derivative of the associated Legendre polynomial at a value.

    Parameters
    ----------
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    val : float
        Value to evaluate the derivative at

    Returns
    -------
    dPlm
        Associated Legendre function derivative
    """

    # TODO: Eventually, probably want to switch to look up table but optimize
    # later
    C = 1
    if order < 0:
        order = abs(order)
        C = (-1) ** order * fact(degree - order) / fact(1 + order)
    return C * (order * val * lpmv(order, degree, val) + (degree + order) *
                (degree - order + 1) * np.sqrt(1 - val ** 2) *
                lpmv(order, degree - 1, val)) / (1 - val ** 2)


def _grad_in_components(degree, order, rad, az, pol, lut_fun=None):
    """Compute gradient of internal component of V(r) spherical expansion.

    Internal component has form: Ylm(pol, az) / (rad ** (degree + 1))

    Parameters
    ----------
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    rad : ndarray, shape (n_samples,)
        Array of radii
    az : ndarray, shape (n_samples,)
        Array of azimuthal (longitudinal) spherical coordinates [0, 2*pi]. 0 is
        aligned with x-axis.
    pol : ndarray, shape (n_samples,)
        Array of polar (or colatitudinal) spherical coordinates [0, pi]. 0 is
        aligned with z-axis.

    Returns
    -------
    grad_vec
        Gradient at the spherical harmonic and vector specified in rectangular
        coordinates
    """
    # TODO: add check/warning if az or pol outside appropriate ranges

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for all spherical coordinates
    g_rad = -(degree + 1) / rad ** (degree + 2) * _sph_harmonic(degree, order,
                                                                az, pol)

    g_theta = 1 / rad ** (degree + 2) * np.sqrt((2 * degree + 1) *
                                                fact(degree - order) /
                                                (4 * np.pi *
                                                 fact(degree + order))) * \
        -np.sin(pol) * _alegendre_deriv(degree, order, np.cos(pol)) * \
        np.exp(1j * order * az)

    g_phi = 1 / (rad ** (degree + 2) * np.sin(pol)) * 1j * order * \
        _sph_harmonic(degree, order, az, pol)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.c_[g_rad, g_theta, g_phi], order)


def _grad_out_components(degree, order, rad, az, pol, lut_fun=None):
    """Compute gradient of internal component of V(r) spherical expansion.

    Internal component has form: Ylm(azimuth, polar) * (radius ** degree)

    Parameters
    ----------
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'
    rad : ndarray, shape (n_samples,)
        Array of radii
    az : ndarray, shape (n_samples,)
        Array of azimuthal (longitudinal) spherical coordinates [0, 2*pi]. 0 is
        aligned with x-axis.
    pol : ndarray, shape (n_samples,)
        Array of polar (or colatitudinal) spherical coordinates [0, pi]. 0 is
        aligned with z-axis.

    Returns
    -------
    grad_vec
        Gradient at the spherical harmonic and vector specified in rectangular
        coordinates
    """
    # TODO: add check/warning if az or pol outside appropriate ranges

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for all spherical coordinates
    g_rad = degree * rad ** (degree - 1) * _sph_harmonic(degree, order, az,
                                                         pol)

    g_theta = rad ** (degree - 1) * np.sqrt((2 * degree + 1) *
                                            fact(degree - order) /
                                           (4 * np.pi *
                                            fact(degree + order))) * \
        -np.sin(pol) * _alegendre_deriv(degree, order, np.cos(pol)) * \
        np.exp(1j * order * az)

    g_phi = rad ** (degree - 1) / np.sin(pol) * 1j * order * \
        _sph_harmonic(degree, order, az, pol)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.c_[g_rad, g_theta, g_phi], order)


def _to_real_and_cart(grad_vec_raw, order):
    """Helper function to take real component of gradient vector and convert
    from spherical to cartesian coords.

    Parameters
    ----------
    grad_vec_raw :
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    """

    # TODO: Check if there's away to not have to explicitly take real part
    if order > 0:
        grad_vec = np.real(np.sqrt(2) * np.real(grad_vec_raw))
    elif order < 0:
        grad_vec = np.real(np.sqrt(2) * np.imag(grad_vec_raw))
    else:
        grad_vec = np.real(grad_vec_raw)

    # Convert to rectanglar coords
    return _spherical_to_cartesian(grad_vec[:, 0], grad_vec[:, 1],
                                   grad_vec[:, 2])


def get_num_harmonics(in_order, out_order):
    """Compute total number of spherical harmonics.

    Parameters
    ---------
    in_order : int
        Internal expansion order
    out_order : int
        External expansion order

    Returns
    -------
    M : int
        Total number of spherical harmonics
    """

    # TODO: Eventually, reuse code in field_interpolation

    M = in_order ** 2 + 2 * in_order + out_order ** 2 + 2 * out_order
    return M


# TODO: confirm that this equation follows the correct convention
def _spherical_to_cartesian(r, az, pol):
    """Convert spherical coords to cartesian coords.

    Parameters
    ----------
    r : ndarray
        Radius
    az : ndarray
        Azimuth angle in radians. 0 is along X-axis
    pol : ndarray
        Polar (or inclination) angle in radians. 0 is along z-axis.
        (Note, this is NOT the elevation angle)
    Returns
    -------
    tuple
        Cartesian coordinate triplet
    """

    rsin_phi = r * np.sin(pol)

    x = rsin_phi * np.cos(az)
    y = rsin_phi * np.sin(az)
    z = r * np.cos(pol)

    return np.c_[x, y, z]


import os
from os import path as op

from ..io import read_info
from ..transforms import _get_mri_head_t, _print_coord_trans
from ..source_space import read_source_spaces, SourceSpaces
from ..externals.six import string_types
from mne.forward._make_forward import _prep_channels


# TODO: Find cleaner way to get channel info than reusing forward soln code
def _make_coils(info, trans, src, bem, fname=None, meg=True, eeg=True,
                mindist=0.0, ignore_ref=False, overwrite=False, n_jobs=1):
    """Prepare dict of coil information

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info | str
        If str, then it should be a filename to a Raw, Epochs, or Evoked
        file with measurement information. If dict, should be an info
        dict (such as one from Raw, Epochs, or Evoked).
    trans : dict | str | None
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option). Can be None to use the identity transform.
    src : str | instance of SourceSpaces
        If string, should be a source space filename. Can also be an
        instance of loaded or generated SourceSpaces.
    bem : dict | str
        Filename of the BEM (e.g., "sample-5120-5120-5120-bem-sol.fif") to
        use, or a loaded sphere model (dict).
    fname : str | None
        Destination forward solution filename. If None, the solution
        will not be saved.
    meg : bool
        If True (Default), include MEG computations.
    eeg : bool
        If True (Default), include EEG computations.
    mindist : float
        Minimum distance of sources from inner skull surface (in mm).
    ignore_ref : bool
        If True, do not include reference channels in compensation. This
        option should be True for KIT files, since forward computation
        with reference channels is not currently supported.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    megcoils : dict
        MEG coil information dict
    """

    # read the transformation from MRI to HEAD coordinates
    # (could also be HEAD to MRI)
    mri_head_t, trans = _get_mri_head_t(trans)

    if not isinstance(src, string_types):
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be a string or SourceSpaces')
        src_extra = 'list'
    else:
        src_extra = src
        if not op.isfile(src):
            raise IOError('Source space file "%s" not found' % src)
    if isinstance(bem, dict):
        bem_extra = 'dict'
    else:
        bem_extra = bem
        if not op.isfile(bem):
            raise IOError('BEM file "%s" not found' % bem)
    if fname is not None and op.isfile(fname) and not overwrite:
        raise IOError('file "%s" exists, consider using overwrite=True'
                      % fname)
    if not isinstance(info, (dict, string_types)):
        raise TypeError('info should be a dict or string')
    if isinstance(info, string_types):
        info_extra = op.split(info)[1]
        info_extra_long = info
        info = read_info(info, verbose=False)
    else:
        info_extra = 'info dict'
        info_extra_long = info_extra
    verbose = False
    arg_list = [info_extra, trans, src_extra, bem_extra, fname,  meg, eeg,
                mindist, overwrite, n_jobs, verbose]
    cmd = 'make_forward_solution(%s)' % (', '.join([str(a) for a in arg_list]))

    if isinstance(src, string_types):
        src = read_source_spaces(src, verbose=False)
    else:
        # let's make a copy in case we modify something
        src = src.copy()
    nsource = sum(s['nuse'] for s in src)
    if nsource == 0:
        raise RuntimeError('No sources are active in these source spaces. '
                           '"do_all" option should be used.')

    # Read the MRI -> head coordinate transformation
    _print_coord_trans(mri_head_t)

    # make a new dict with the relevant information
    mri_id = dict(machid=np.zeros(2, np.int32), version=0, secs=0, usecs=0)
    info = dict(nchan=info['nchan'], chs=info['chs'], comps=info['comps'],
                ch_names=info['ch_names'], dev_head_t=info['dev_head_t'],
                mri_file=trans, mri_id=mri_id, meas_file=info_extra_long,
                meas_id=None, working_dir=os.getcwd(),
                command_line=cmd, bads=info['bads'])

    megcoils, compcoils, eegels, megnames, eegnames, meg_info = \
        _prep_channels(info, meg, eeg, ignore_ref)

    return megcoils
