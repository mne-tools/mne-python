# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Jussi Nurminen
#
# License: BSD (3-clause)

#TODO: write in equation numbers from Samu's paper

import numpy as np
from scipy.special import sph_harm, lpmv
from scipy.misc import factorial
from ..forward._lead_dots import (_get_legen_table, _get_legen_lut_fast,
                                  _get_legen_lut_accurate)
from ..fixes import partial
from ..transforms import _cartesian_to_sphere as cart_to_sph
from ..transforms import _sphere_to_cartesian as sph_to_cart
from scipy.misc import factorial as fact


def maxwell_filter(Raw):
    pass


def sph_harmonic(l, m, polar, azimuth):
    """
    Compute the spherical harmonic function at point in spherical coordinates.
    When using, pay close attention to inputs. Spherical harmonic notation for
    order/degree, and theta/phi are both reversed in original SSS work compared
    to most other sources.

    Parameters
    ----------
    l : int
        Degree of spherical harmonic
    m : int
        Order of spherical harmonic
    polar : float
        Polar (or elevation) spherical coordinate [0, pi]
    azimuth : float
        Elevation spherical coordinate [0, 2*pi]

    Returns
    -------
    y : complex float
        The spherical harmonic value at the specified polar and azimuth angles
    """
    assert np.abs(m) <= l, 'Absolute value of expansion coefficient must be <= l'

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    #TODO: Decide on notation for spherical coords
    #TODO: Should factorial function use floating or long precision?
    #TODO: Ensure that polar and azimuth angles are arrays in efficient manner
    polar = np.array(polar)
    azimuth = np.array(azimuth)

    # Real valued spherical expansion as given by Wikso, (11)
    base = np.sqrt(((2 * l + 1) / (4 * np.pi) *
                    factorial(l - m) / factorial(l + m)) *
                   lpmv(m, l, np.cos(polar)))
    if m < 0:
        return base * np.sin(m * azimuth)
    else:
        return base * np.cos(m * azimuth)

    #TODO: Check how fast taking the real part of scipy's sph harmonic function is
    # Degree/order and theta/phi reversed

    # Note reversal in notation order between scipy and original SSS papers
    #return np.real(sph_harm(m, l, azimuth, polar))


def get_num_harmonics(L_in, L_out):
    """
    Compute total number of spherical harmonics.

    Parameters
    ---------
    L_in : int
        Internal expansion order
    L_out : int
        External expansion order

    Returns
    -------
    M : int
        Total number of spherical harmonics
    """

    return L_in ** 2 + 2 * L_in + L_out ** 2 + 2 * L_out


def _alegendre_deriv(l, m, x):
    """
    Compute the derivative of the associated Legendre polynomial at x.

    Parameters
    ----------
    l : int
        Degree of spherical harmonic
    m : int
        Order of spherical harmonic
    x : float
        Value to evaluate the derivative at
    """

    C = 1
    if m < 0:
        m = abs(m)
        C = (-1) ** m * fact(l - m) / fact(1 + m)
    return C * (m * x * lpmv(m, l, x) + (l + m) * (l - m + 1) *
                np.sqrt(1 - x ** 2) * lpmv(m, l - 1, x)) / (1 - x ** 2)


def grad_in_comp(l, m, r, theta, phi, lut_func):
    """
    Compute gradient of LHS of V(r) spherical expansion having form
    Ylm(theta, phi) / (r ** (l+1))
    """
    #TODO: add check/warning if theta or phi outside appropriate ranges

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for r, theta, and phi
    r1 = -(l + 1) / r ** (l + 2) * sph_harmonic(l, m, theta, phi)

    theta1 = 1 / r ** (l + 2) * np.sqrt((2 * l + 1) * factorial(l - m) /
                                        (4 * np.pi * factorial(l + m))) * \
        -np.sin(theta) * _alegendre_deriv(l, m, np.cos(theta)) * np.exp(1j * m * phi)

    phi1 = 1 / (r ** (l + 2) * np.sin(theta)) * 1j * m * \
        sph_harmonic(l, m, theta, phi)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.concatenate((r1, theta1, phi1)), m)


def grad_out_comp(l, m, r, theta, phi):
    """
    Compute gradient of RHS of V(r) spherical expansion having form
    Ylm(theta, phi) * (r ** l)
    """
    #TODO: add check/warning if theta or phi outside appropriate ranges

    # Get function for Legendre derivatives
    #lut, n_fact = _get_legen_table('meg', False, 100)
    #lut_deriv_fun = partial(_get_legen_lut_accurate, lut=lut)

    # Compute gradients for r, theta, and phi
    r1 = l * r ** (l - 1) * sph_harmonic(l, m, theta, phi)

    theta1 = r ** (l - 1) * np.sqrt((2 * l + 1) * factorial(l - m) /
                                    (4 * np.pi * factorial(1 + m))) * \
        -np.sin(theta) * _alegendre_deriv(l, m, np.cos(theta)) * np.exp(1j * m * phi)

    phi1 = r ** (l - 1) / np.sin(theta) * 1j * m * sph_harmonic(l, m, theta, phi)

    # Get real component of vectors, convert to cartesian coords, and return
    return _to_real_and_cart(np.concatenate((r1, theta1, phi1)), m)


def _to_real_and_cart(grad_vec_raw, m):
    """
    Helper function to take real component of gradient vector and convert from
    spherical to cartesian coords
    """

    if m > 0:
        grad_vec = np.sqrt(2) * np.real(grad_vec_raw)
    elif m < 0:
        grad_vec = np.sqrt(2) * np.imag(grad_vec_raw)
    else:
        grad_vec = grad_vec_raw

    # Convert to rectanglar coords
    # TODO: confirm that this equation follows the correct convention
    return sph_to_cart(grad_vec[:, 0], grad_vec[:, 1], grad_vec[:, 2])
