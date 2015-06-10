# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>

# This code was adapted and relicensed (with BSD form) with permission from
# Jussi Nurminen

# License: BSD (3-clause)

# Note, there are an absurd number of different notations for spherical
# harmonics (partly because there is also no accepted standard for spherical
# coordinates). Here, we purposefully stay away from shorthand notation in
# both and use explicit terms to avoid confusion.

#TODO: write in equation numbers from Samu's paper

import numpy as np
from scipy.special import sph_harm, lpmv
from scipy.misc import factorial
from ..forward._lead_dots import (_get_legen_table, _get_legen_lut_fast,
                                  _get_legen_lut_accurate)
#from ..fixes import partial
from ..transforms import _sphere_to_cartesian as sph_to_cart
from scipy.misc import factorial as fact


def maxwell_filter(raw, origin, int_order=8, ext_order=3):
    """
    Apply Maxwell filter to data using spherical harmonics.

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

    #TODO: Figure out all parameters required
    #TODO: Error checks on input parameters

    #TODO: Compute spherical harmonics
    #TODO: Project data into spherical harmonics space
    #TODO: Reconstruct and return Raw file object


def _sss_basis(origin_int, origin_ext, int_order, ext_order, int_pts, int_ws,
               int_norms):
    """Compute SSS basis for given conditions

    Parameters
    ----------
    origin_int : ndarray, shape(3,)
        Origin of the internal space.
    origin_ext : ndarray, shape(3,)
        Origin of the external space.
    int_order : int
        Order of internal space
    ext_order : int
        Order of external space
    int_pts : list, len(n_sensors)
        3D position of integration points for each sensor
    int_ws :
        Weights for each sensors integration points
    int_norms :
        Unit normal vector for each integration point

    Returns
    -------
    list
        List of length 2 containing internal and external basis sets
    """
    n_sens = len(int_pts)
    n_bases = (int_order - 1) ** 2 + (ext_order - 1) ** 2 - 2
    n_int_pts = len(int_pts)
    S_in = np.empty((n_sens, (int_order + 1) ** 2 - 1)).fill(np.nan)
    S_out = np.empty((n_sens, (ext_order + 1) ** 2 - 1)).fill(np.nan)

    assert n_bases >= n_sens, ('Number of requested bases (%s) exceeds number '
                               'of sensors (%s)' % (str(n_bases), str(n_sens)))

    # Compute internal basis vectors
    for deg in range(int_order):
        for order in range(- deg, deg + 1):
            cvec = int_pts - origin_int * np.ones(n_int_pts, 1)
            grads = -1 * _grad_in_components(deg, order, cvec[:, 0],
                                             cvec[:, 1], cvec[:, 2])
            #a1_all = int_ws * np.dot(grads, int_norms, 2)
            a1_all = np.einsum('ij,kj->ik', cvec, grads)

            #XXX: sum all signals correctly
            S_in[:, deg ** 2 + deg + ord] = np.sum(a1_all, 1)

    # Compute external basis vectors
    for deg in range(int_order):
        for order in range(- deg, deg + 1):
            cvec = int_pts - origin_int * np.ones(n_int_pts, 1)
            grads = -1 * _grad_out_components(deg, order, cvec[:, 0],
                                              cvec[:, 1], cvec[:, 2])
            #b1_all = int_ws * np.dot(grads, int_norms, 2)
            b1_all = np.einsum('ij,kj->ik', cvec, grads)

            #XXX: sum all signals correctly
            S_out[:, deg ** 2 + deg + ord] = np.sum(b1_all, 1)

    return [S_in, S_out]


def _sph_harmonic(degree, order, az, pol):
    """
    Compute the spherical harmonic function at point in spherical coordinates.
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

    #TODO: Decide on notation for spherical coords
    #TODO: Should factorial function use floating or long precision?
    #TODO: Check that [-m to m] convention is correct for all equations

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

    #TODO: Check speed of taking real part of scipy's sph harmonic function
    # Note reversal in notation order between scipy and original SSS papers
    # Degree/order and theta/phi reversed
    #return np.real(sph_harm(order, degree, az, pol))


def _alegendre_deriv(degree, order, val):
    """
    Compute the derivative of the associated Legendre polynomial at a value.

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

    #TODO: Eventually, probably want to switch to look up table but optimize
    # later
    C = 1
    if order < 0:
        order = abs(order)
        C = (-1) ** order * fact(degree - order) / fact(1 + order)
    return C * (order * val * lpmv(order, degree, val) + (degree + order) *
                (degree - order + 1) * np.sqrt(1 - val ** 2) *
                lpmv(order, degree - 1, val)) / (1 - val ** 2)


def _grad_in_components(degree, order, rad, az, pol, lut_fun=None):
    """
    Compute gradient of in-component of V(r) spherical expansion having form
    Ylm(pol, az) / (rad ** (degree + 1))

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
    #TODO: add check/warning if az or pol outside appropriate ranges

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for all spherical coordinates
    r1 = -(degree + 1) / rad ** (degree + 2) * _sph_harmonic(degree, order, az,
                                                             pol)

    theta1 = 1 / rad ** (degree + 2) * np.sqrt((2 * degree + 1) *
                                               factorial(degree - order) /
                                               (4 * np.pi *
                                                factorial(degree + order))) * \
        -np.sin(pol) * _alegendre_deriv(degree, order, np.cos(pol)) * \
        np.exp(1j * order * az)

    phi1 = 1 / (rad ** (degree + 2) * np.sin(pol)) * 1j * order * \
        _sph_harmonic(degree, order, az, pol)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.concatenate((r1, theta1, phi1)), order)


def _grad_out_components(degree, order, rad, az, pol, lut_fun=None):
    """
    Compute gradient of RHS of V(r) spherical expansion having form
    Ylm(azimuth, polar) * (radius ** degree)

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
    #TODO: add check/warning if az or pol outside appropriate ranges

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for all spherical coordinates
    r1 = degree * rad ** (degree - 1) * _sph_harmonic(degree, order, az, pol)

    theta1 = rad ** (degree - 1) * np.sqrt((2 * degree + 1) *
                                           factorial(degree - order) /
                                           (4 * np.pi *
                                            factorial(degree + order))) * \
        -np.sin(pol) * _alegendre_deriv(degree, order, np.cos(pol)) * \
        np.exp(1j * order * az)

    phi1 = rad ** (degree - 1) / np.sin(pol) * 1j * order * \
        _sph_harmonic(degree, order, az, pol)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.concatenate((r1, theta1, phi1)), order)


def _to_real_and_cart(grad_vec_raw, order):
    """
    Helper function to take real component of gradient vector and convert from
    spherical to cartesian coords
    """

    if order > 0:
        grad_vec = np.sqrt(2) * np.real(grad_vec_raw)
    elif order < 0:
        grad_vec = np.sqrt(2) * np.imag(grad_vec_raw)
    else:
        grad_vec = grad_vec_raw

    # Convert to rectanglar coords
    # TODO: confirm that this equation follows the correct convention
    return sph_to_cart(grad_vec[:, 0], grad_vec[:, 1], grad_vec[:, 2])


def get_num_harmonics(in_order, out_order):
    """
    Compute total number of spherical harmonics.

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

    #TODO: Eventually, reuse code in field_interpolation

    M = in_order ** 2 + 2 * in_order + out_order ** 2 + 2 * out_order
    return M


def _spherical_to_cartesian(r, az, pol):
    """ Convert spherical coords to cartesian coords.

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

    return np.array([x, y, z])
