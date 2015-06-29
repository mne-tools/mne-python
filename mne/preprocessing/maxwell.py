# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>

# This code was adapted and relicensed (with BSD form) with permission from
# Jussi Nurminen

# License: BSD (3-clause)

# Note, there are an absurd number of different possible notations for
# spherical coordinates, which confounds the notation for spherical harmonics.
# Here, we purposefully stay away from shorthand notation in both and use
# explicit terms (like 'azimuth' and 'polar') to avoid confusion.

# TODO: write in equation numbers from Samu's paper

import numpy as np
from scipy.special import lpmv
from scipy.linalg import pinv
from ..forward._compute_forward import _concatenate_coils
#from ..fixes import partial
from scipy.misc import factorial as fact


def maxwell_filter(raw, coils, origin, int_order=8, ext_order=3, n_jobs=1):
    """Apply Maxwell filter to data using spherical harmonics.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    coils : list
        List of MEG coils. Each element must a contain coil information dict
        continaining 'rmag', 'cosmag', and 'w' keys
    origin : ndarray or tuple, shape (3,)
        Origin of internal and external multipolar moment space in millimeters
    int_order : int
        Order of internal component of spherical expansion
    ext_order : int
        Order of external component of spherical expansion
    n_jobs : int
        Number of jobs to run in parallel

    Returns
    -------
    raw_sss : instance of mne.io.Raw
        The raw data with Maxwell filtering applied
    """

    # TODO: Add error checks on input parameters
    picks = [raw.info['ch_names'].index(ch) for ch in [coil['chname']
                                                       for coil in coils]]
    data, times = raw[picks, :]

    # Magnetometers (with coil_class == 1.0) must be scaled by 100 to # improve
    # numerical stability as they have different scales than gradiometers
    coil_scale = np.ones((len(picks)))
    coil_scale[np.array([coil['coil_class'] == 1.0 for coil in coils])] = 100.

    # Compute spherical harmonics
    origin = np.array(origin) / 1000.  # Convert scale from mm to m
    S_in, S_out = _sss_basis(origin, coils, int_order, ext_order)
    S_tot = np.c_[S_in, S_out]

    # Pseudo-inverse of total multipolar moment basis set
    pS_tot = pinv(S_tot, cond=1e-15)
    # Compute multipolar moments of (magnetometer scaled) data
    mm = np.dot(pS_tot, data * coil_scale[:, np.newaxis])
    # Reconstruct data from internal space
    recon = np.dot(S_in, mm[:S_in.shape[1], :])

    # Return reconstructed raw file object
    raw_sss = raw.copy()
    raw_sss[:, :] = recon / coil_scale[:, np.newaxis]

    return raw_sss


def _sss_basis(origin, coils, int_order, ext_order):
    """Compute SSS basis for given conditions.

    Parameters
    ----------
    origin : ndarray, shape (3,)
        Origin of the multipolar moment space in millimeters
    coils : list
        List of MEG coils. Each should contain coil information dict. All
        position info must be in the same coordinate frame as 'origin'
    int_order : int
        Order of the internal multipolar moment space
    ext_order : int
        Order of the external multipolar moment space

    Returns
    -------
    tuple, len (2)
        Internal and external basis sets ndarrays with shape
        (n_coils, n_mult_moments)
    """
    r_int_pts, ncoils, wcoils, int_pts = _concatenate_coils(coils)
    n_sens = len(int_pts)
    n_bases = (int_order - 1) ** 2 + (ext_order - 1) ** 2 - 2
    n_int_pts = len(r_int_pts)
    int_lens = np.insert(np.cumsum(int_pts), obj=0, values=0)

    S_in = np.empty((n_sens, (int_order + 1) ** 2 - 1))
    S_out = np.empty((n_sens, (ext_order + 1) ** 2 - 1))
    S_in.fill(np.nan)
    S_out.fill(np.nan)

    # Set all magnetometers (with 'coil_type' == 1.0) to be scaled by 100
    coil_scale = np.ones((len(coils)))
    coil_scale[np.array([coil['coil_class'] == 1.0 for coil in coils])] = 100.

    assert n_bases <= n_sens, ('Number of requested bases (%s) exceeds number '
                               'of sensors (%s)' % (str(n_bases), str(n_sens)))

    # Compute position vector between origin and coil integration pts
    cvec_cart = r_int_pts - origin * np.ones((n_int_pts, 1))
    # Convert points to spherical coordinates
    cvec_sph = _cart_to_sph(cvec_cart)

    # Compute internal basis vectors (exclude deg, order 0)
    for deg in range(1, int_order + 1):
        for order in range(-deg, deg + 1):

            # Compute gradient for integration point position vectors
            grads = -1 * _grad_in_components(deg, order, cvec_sph[:, 0],
                                             cvec_sph[:, 1], cvec_sph[:, 2])

            # Gradients dotted with integration point normals and weighted
            a1_all = wcoils * np.einsum('ij,ij->i', grads, ncoils)

            # For order and degree, sum across integration pts for each sensor
            for pt_i in range(0, len(int_lens) - 1):
                int_pts_sum = np.sum(a1_all[int_lens[pt_i]:int_lens[pt_i + 1]])
                S_in[pt_i, deg ** 2 + deg + order - 1] = int_pts_sum

    # Compute external basis vectors (exclude deg, order 0)
    for deg in range(1, ext_order + 1):
        for order in range(-deg, deg + 1):

            # Compute gradient for integration point position vectors
            grads = -1 * _grad_out_components(deg, order, cvec_sph[:, 0],
                                              cvec_sph[:, 1], cvec_sph[:, 2])

            # Gradients dotted with integration point normals and weighted
            b1_all = wcoils * np.einsum('ij,ij->i', grads, ncoils)

            # For order and degree, sum across integration pts for each sensor
            for pt_i in range(0, len(int_lens) - 1):
                int_pts_sum = np.sum(b1_all[int_lens[pt_i]:int_lens[pt_i + 1]])
                S_out[pt_i, deg ** 2 + deg + order - 1] = int_pts_sum

    # Scale and normalize each basis vector to have unity magnitude
    S_in *= coil_scale[:, np.newaxis]
    S_out *= coil_scale[:, np.newaxis]
    S_in = np.divide(S_in, np.linalg.norm(S_in, axis=0))
    S_out = np.divide(S_out, np.linalg.norm(S_out, axis=0))

    return (S_in, S_out)


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
    base : complex float
        The spherical harmonic value at the specified azimuth and polar angles
    """
    # Error checks
    assert np.abs(order) <= degree, ('Absolute value of expansion coefficient'
                                     ' must be <= degree')
    assert ((az >= -2 * np.pi).all() and (az <= 2 * np.pi).all(),
            'Azimuth coord outside [-2*pi, 2*pi]')
    assert ((pol >= 0).all() and (pol <= np.pi).all()), ('Polar coord outside '
                                                         '[0, pi]')

    #Ensure that polar and azimuth angles are arrays
    azimuth = np.array(az)
    polar = np.array(pol)

    base = np.sqrt((2 * degree + 1) / (4 * np.pi) * fact(degree - order) /
                   fact(degree + order)) * lpmv(order, degree, np.cos(polar)) \
        * np.exp(1j * order * azimuth)
    return base

    # TODO: Check speed of taking real part of scipy's sph harmonic function


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

    C = 1
    if order < 0:
        order = abs(order)
        C = (-1) ** order * fact(degree - order) / fact(degree + order)
    return C * (order * val * lpmv(order, degree, val) + (degree + order) *
                (degree - order + 1) * np.sqrt(1 - val ** 2) *
                lpmv(order - 1, degree, val)) / (1 - val ** 2)


def _grad_in_components(degree, order, rad, az, pol):
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
    ndarray, shape (n_samples, 3)
        Gradient of the spherical harmonic and vector specified in rectangular
        coordinates
    """
    # Compute gradients for all spherical coordinates
    g_rad = -(degree + 1) / rad ** (degree + 2) * _sph_harmonic(degree, order,
                                                                az, pol)

    g_az = 1. / (rad ** (degree + 2) * np.sin(pol)) * 1j * order * \
        _sph_harmonic(degree, order, az, pol)

    g_pol = 1. / rad ** (degree + 2) * np.sqrt((2 * degree + 1) *
                                               fact(degree - order) /
                                               (4 * np.pi *
                                                fact(degree + order))) * \
        -np.sin(pol) * _alegendre_deriv(degree, order, np.cos(pol)) * \
        np.exp(1j * order * az)

    # Get real component of vectors, convert to cartesian coords, and return
    real_grads = _get_real_grad(np.c_[g_rad, g_az, g_pol], order)
    return _sph_to_cart_partials(np.c_[rad, az, pol], real_grads)


def _grad_out_components(degree, order, rad, az, pol):
    """Compute gradient of external component of V(r) spherical expansion.

    External component has form: Ylm(azimuth, polar) * (radius ** degree)

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
    ndarray, shape (n_samples, 3)
        Gradient of the spherical harmonic and vector specified in rectangular
        coordinates
    """
    # Compute gradients for all spherical coordinates
    g_rad = degree * rad ** (degree - 1) * _sph_harmonic(degree, order, az,
                                                         pol)

    g_az = rad ** (degree - 1) / np.sin(pol) * 1j * order * \
        _sph_harmonic(degree, order, az, pol)

    g_pol = rad ** (degree - 1) * np.sqrt((2 * degree + 1) *
                                          fact(degree - order) /
                                          (4 * np.pi *
                                           fact(degree + order))) * \
        -np.sin(pol) * _alegendre_deriv(degree, order, np.cos(pol)) * \
        np.exp(1j * order * az)

    # Get real component of vectors, convert to cartesian coords, and return
    real_grads = _get_real_grad(np.c_[g_rad, g_az, g_pol], order)
    return _sph_to_cart_partials(np.c_[rad, az, pol], real_grads)


def _get_real_grad(grad_vec_raw, order):
    """Helper function to convert gradient vector to to real basis functions.

    Parameters
    ----------
    grad_vec_raw : ndarray, shape (n_gradients, 3)
        Gradient array with columns for radius, azimuth, polar points
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    grad_vec : ndarray, shape (n_gradients, 3)
        Gradient vectors with only real componnet
    """

    if order > 0:
        grad_vec = np.sqrt(2) * np.real(grad_vec_raw)
    elif order < 0:
        grad_vec = np.sqrt(2) * np.imag(grad_vec_raw)
    else:
        grad_vec = grad_vec_raw

    return np.real_if_close(grad_vec)


def get_num_harmonics(int_order, ext_order):
    """Compute total number of spherical harmonics.

    Parameters
    ---------
    int_order : int
        Internal expansion order
    ext_order : int
        External expansion order

    Returns
    -------
    M : int
        Total number of spherical harmonics
    """

    # TODO: Eventually, reuse code in field_interpolation

    M = int_order ** 2 + 2 * int_order + ext_order ** 2 + 2 * ext_order
    return M


def _sph_to_cart_partials(sph_pts, sph_grads):
    """Convert spherical partial derivatives to cartesian coords.

    Note: Because we are dealing with partial derivatives, this calculation is
    not a static transformation. The transformation matrix itself is dependent
    on azimuth and polar coord.
    See mathworld.wolfram.com/SphericalCoordinates.html Eq. 96

    Parameters
    ----------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing spherical coordinates points (rad, azimuth, polar)
    sph_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives at each spherical coordinate

    Returns
    -------
    cart_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives in Cartesian coordinates (x, y, z)
    """

    cart_grads = np.zeros_like(sph_grads)

    # TODO: needs vectorization, currently matching Jussi's code for debugging
    for pt_i, (sph_pt, sph_grad) in enumerate(zip(sph_pts, sph_grads)):
        # get cosine and sine of azimuth and polar coord
        c_a, s_a = np.cos(sph_pt[1]), np.sin(sph_pt[1])
        c_p, s_p = np.cos(sph_pt[2]), np.sin(sph_pt[2])

        trans = np.array([[c_a * s_p, -s_a, c_a * c_p],
                          [s_a * s_p, c_a, c_p * s_a],
                          [c_p, 0, -s_p]])

        cart_grads[pt_i, :] = np.dot(trans, sph_grad)

    return cart_grads


def _cart_to_sph(cart_pts):
    """Convert Cartesian coordinates to spherical coordinates

    Parameters
    ----------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    ndarray, shape (n_points, 3)
        Array containing points in spherical coordinates (rad, azimuth, polar)
    """

    rad = np.linalg.norm(cart_pts, axis=1)
    az = np.arctan2(cart_pts[:, 1], cart_pts[:, 0])
    pol = np.arccos(cart_pts[:, 2] / rad)

    return np.c_[rad, az, pol]


# TODO: Find cleaner way to get channel info than reusing forward soln code
import os
from os import path as op

from ..io import read_info
from ..transforms import _get_mri_head_t, _print_coord_trans
from ..source_space import read_source_spaces, SourceSpaces
from ..externals.six import string_types
from mne.forward._make_forward import _prep_channels


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
