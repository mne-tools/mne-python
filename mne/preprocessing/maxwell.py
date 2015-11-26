# -*- coding: utf-8 -*-
# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>


# License: BSD (3-clause)

from copy import deepcopy
import numpy as np
from scipy import linalg
from math import factorial
from os import path as op

from .. import __version__
from ..bem import fit_sphere_to_headshape
from ..transforms import _str_to_frame, _get_trans
from ..forward._compute_forward import _concatenate_coils
from ..forward._make_forward import _prep_meg_channels
from ..surface import _normalize_vectors
from ..io.constants import FIFF
from ..io.proc_history import _read_ctc
from ..io.write import _generate_meas_id, _date_now
from ..io import _loc_to_coil_trans, _BaseRaw
from ..io.pick import pick_types, pick_info, pick_channels
from ..utils import verbose, logger, _clean_names
from ..fixes import _get_args
from ..externals.six import string_types
from ..channels.channels import _get_T1T2_mag_inds


# Note: Elekta uses single precision and some algorithms might use
# truncated versions of constants (e.g., μ0), which could lead to small
# differences between algorithms


@verbose
def maxwell_filter(raw, origin='auto', int_order=8, ext_order=3,
                   calibration=None, cross_talk=None, st_duration=None,
                   st_correlation=0.98, coord_frame='head', destination=None,
                   regularize='in', verbose=None):
    """Apply Maxwell filter to data using multipole moments

    .. warning:: Automatic bad channel detection is not currently implemented.
                 It is critical to mark bad channels before running Maxwell
                 filtering, so data should be inspected and marked accordingly
                 prior to running this algorithm.

    .. warning:: Not all features of Elekta MaxFilter™ are currently
                 implemented (see Notes).

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    origin : array-like, shape (3,) | str
        Origin of internal and external multipolar moment space in head coords
        and in meters. The default is ``'auto'``, which means
        ``(0., 0., 0.)`` for ``coord_frame='meg'``, and a
        head-digitization-based origin fit for ``coord_frame='head'``.
    int_order : int
        Order of internal component of spherical expansion
    ext_order : int
        Order of external component of spherical expansion
    calibration : str | None
        Path to the ``'.dat'`` file with fine calibration coefficients.
        File can have 1D or 3D gradiometer imbalance correction.
        This file is machine/site-specific.
    cross_talk : str | None
        Path to the FIF file with cross-talk correction information.
    st_duration : float | None
        If not None, apply spatiotemporal SSS with specified buffer duration
        (in seconds). Elekta's default is 10.0 seconds in MaxFilter™ v2.2.
        Spatiotemporal SSS acts as implicitly as a high-pass filter where the
        cut-off frequency is 1/st_dur Hz. For this (and other) reasons, longer
        buffers are generally better as long as your system can handle the
        higher memory usage. To ensure that each window is processed
        identically, choose a buffer length that divides evenly into your data.
        Any data at the trailing edge that doesn't fit evenly into a whole
        buffer window will be lumped into the previous buffer.
    st_correlation : float
        Correlation limit between inner and outer subspaces used to reject
        ovwrlapping intersecting inner/outer signals during spatiotemporal SSS.
    coord_frame : str
        The coordinate frame that the ``origin`` is specified in, either
        ``'meg'`` or ``'head'``. For empty-room recordings that do not have
        a head<->meg transform ``info['dev_head_t']``, the MEG coordinate
        frame should be used.
    destination : str | array-like, shape (3,) | None
        The destination location for the head. Can be ``None``, which
        will not change the head position, or a string path to a FIF file
        containing a MEG device<->head transformation, or a 3-element array
        giving the coordinates to translate to (with no rotations).
        For example, ``destination=(0, 0, 0.04)`` would translate the bases
        as ``--trans default`` would in MaxFilter™ (i.e., to the default
        head location).
    regularize : str | None
        Basis regularization type, must be "in" or None.
        "in" is the same algorithm as the "-regularize in" option in
        MaxFilter™.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose)

    Returns
    -------
    raw_sss : instance of mne.io.Raw
        The raw data with Maxwell filtering applied

    Notes
    -----
    .. versionadded:: 0.11

    Equation numbers refer to Taulu and Kajola, 2005 [1]_ unless otherwise
    noted.

    Some of this code was adapted and relicensed (with BSD form) with
    permission from Jussi Nurminen.

    Compared to Elekta's MaxFilter™ software, our algorithm
    currently provides the following features:

        * Basic Maxwell filtering
        * Cross-talk cancellation
        * tSSS
        * Bad channel reconstruction
        * Coordinate frame translation
        * Regularization of internal components using information theory

    The following features are not yet implemented:

        * Movement compensation
        * Automatic bad channel detection
        * cHPI subtraction

    Our algorithm has the following enhancements:

        * Double floating point precision
        * Handling of 3D (in additon to 1D) fine calibration files
        * Processing of data from (un-compensated) non-Elekta systems
        * Automated processing of split (-1.fif) and concatenated files

    .. note:: Various Maxwell filtering algorithm components are covered by
              patents owned by Elekta Oy, Helsinki, Finland.
              These patents include, but may not be limited to:

                  - US2006031038 (Signal Space Separation)
                  - US6876196 (Head position determination)
                  - WO2005067789 (DC fields)
                  - WO2005078467 (MaxShield)
                  - WO2006114473 (Temporal Signal Space Separation)

              These patents likely preclude the use of Maxwell filtering code
              in commercial applications. Consult a lawyer if necessary.

    References
    ----------
    .. [1] Taulu S. and Kajola M. "Presentation of electromagnetic
           multichannel data: The signal space separation method,"
           Journal of Applied Physics, vol. 97, pp. 124905 1-10, 2005.

           http://lib.tkk.fi/Diss/2008/isbn9789512295654/article2.pdf

    .. [2] Taulu S. and Simola J. "Spatiotemporal signal space separation
           method for rejecting nearby interference in MEG measurements,"
           Physics in Medicine and Biology, vol. 51, pp. 1759-1768, 2006.

           http://lib.tkk.fi/Diss/2008/isbn9789512295654/article3.pdf
    """
    # There are an absurd number of different possible notations for spherical
    # coordinates, which confounds the notation for spherical harmonics.  Here,
    # we purposefully stay away from shorthand notation in both and use
    # explicit terms (like 'azimuth' and 'polar') to avoid confusion.
    # See mathworld.wolfram.com/SphericalHarmonic.html for more discussion.
    # Our code follows the same standard that ``scipy`` uses for ``sph_harm``.

    # triage inputs ASAP to avoid late-thrown errors

    _check_raw(raw)
    if raw.proj:
        raise RuntimeError('Projectors cannot be applied to raw data.')
    if len(raw.info.get('comps', [])) > 0:
        raise RuntimeError('Maxwell filter cannot handle compensated '
                           'channels.')
    st_correlation = float(st_correlation)
    if st_correlation <= 0. or st_correlation > 1.:
        raise ValueError('Need 0 < st_correlation <= 1., got %s'
                         % st_correlation)
    if coord_frame not in ('head', 'meg'):
        raise ValueError('coord_frame must be either "head" or "meg", not "%s"'
                         % coord_frame)
    head_frame = True if coord_frame == 'head' else False
    if not (regularize is None or (isinstance(regularize, string_types) and
                                   regularize in ('in',))):
        raise ValueError('regularize must be None or "in"')
    if destination is not None:
        if not head_frame:
            raise RuntimeError('destination can only be set if using the '
                               'head coordinate frame')
        if isinstance(destination, string_types):
            recon_trans = _get_trans(destination, 'meg', 'head')[0]['trans']
        else:
            destination = np.array(destination, float)
            if destination.shape != (3,):
                raise ValueError('destination must be a 3-element vector, '
                                 'str, or None')
            recon_trans = np.eye(4)
            recon_trans[:3, 3] = destination
    if st_duration is not None:
        st_duration = float(st_duration)
        if not 0. < st_duration <= raw.times[-1]:
            raise ValueError('st_duration (%0.1fs) must be between 0 and the '
                             'duration of the data (%0.1fs).'
                             % (st_duration, raw.times[-1]))
        st_correlation = float(st_correlation)
        if not 0. < st_correlation <= 1:
            raise ValueError('st_correlation must be between 0. and 1.')

    # Now we can actually get moving

    logger.info('Maxwell filtering raw data')
    raw_sss = raw.copy().load_data(verbose=False)
    del raw
    info, times = raw_sss.info, raw_sss.times

    # Check for T1/T2 mag types
    mag_inds_T1T2 = _get_T1T2_mag_inds(info)
    if len(mag_inds_T1T2) > 0:
        logger.warning('%d T1/T2 magnetometer channel types found. If using '
                       ' SSS, it is advised to replace coil types using '
                       ' `fix_mag_coil_types`.' % len(mag_inds_T1T2))
    meg_picks = pick_types(info, meg=True, exclude=[])
    recons = [ch for ch in info['bads']
              if info['ch_names'].index(ch) in meg_picks]
    if len(recons) > 0:
        logger.info('    Bad MEG channels being reconstructed: %s' % recons)
    else:
        logger.info('    No bad MEG channels')
    #
    # Fine calibration processing (load fine cal and overwrite sensor geometry)
    #
    if calibration is not None:
        grad_imbalances, mag_cals, sss_cal = \
            _update_sensor_geometry(info, calibration)
    else:
        sss_cal = dict()

    # Get indices of channels to use in multipolar moment calculation
    good_picks = pick_types(info, meg=True, exclude='bads')
    n_bases = _get_n_moments([int_order, ext_order]).sum()
    if n_bases > len(good_picks):
        raise ValueError('Number of requested bases (%s) exceeds number of '
                         'good sensors (%s)' % (str(n_bases), len(good_picks)))

    # Get indices of MEG channels
    mag_picks = pick_types(info, meg='mag', exclude=[])
    grad_picks = pick_types(info, meg='grad', exclude=[])
    grad_info = pick_info(info, grad_picks)
    if info['dev_head_t'] is None and coord_frame == 'head':
        raise RuntimeError('coord_frame cannot be "head" because '
                           'info["dev_head_t"] is None; if this is an '
                           'empty room recording, consider using '
                           'coord_frame="meg"')
    meg_coils = _prep_meg_channels(info, accurate=True, elekta_defs=True,
                                   head_frame=head_frame, verbose=False)[0]

    # Magnetometers are scaled by 100 to improve numerical stability
    coil_scale = np.ones((len(meg_picks), 1))
    coil_scale[mag_picks] = 100.

    # Compute multipolar moment bases
    if isinstance(origin, string_types):
        # XXX eventually we could add "auto" mode here
        if origin != 'auto':
            raise ValueError('origin must be a numerical array, or "auto", '
                             'not %s' % (origin,))
        if coord_frame == 'head':
            R, origin = fit_sphere_to_headshape(raw_sss.info,
                                                verbose=False)[:2]
            origin /= 1000.
            logger.info('    Automatic origin fit: head of radius %0.1f mm'
                        % R)
            del R
        else:
            origin = (0., 0., 0.)
    origin = np.array(origin, float)
    if origin.shape != (3,):
        raise ValueError('origin must be a 3-element array')
    origin_str = ', '.join(['%0.1f' % (o * 1000) for o in origin])
    logger.info('    Using origin %s mm in the %s frame'
                % (origin_str, coord_frame))
    # Compute in/out bases and create copies containing only good chs
    S_decomp = _sss_basis(origin, meg_coils, int_order, ext_order)
    # We always want to reconstruct with non-corrected defs
    n_in, n_out = _get_n_moments([int_order, ext_order])
    S_recon = S_decomp.copy()

    #
    # Cross-talk processing
    #
    if cross_talk is not None:
        sss_ctc = _read_ctc(cross_talk)
        ctc_chs = sss_ctc['proj_items_chs']
        if set(info['ch_names'][p] for p in meg_picks) != set(ctc_chs):
            raise RuntimeError('ctc channels and raw channels do not match')
        ctc_picks = pick_channels(ctc_chs,
                                  [info['ch_names'][c] for c in good_picks])
        ctc = sss_ctc['decoupler'][ctc_picks][:, ctc_picks]
        # I have no idea why, but MF transposes this for storage..
        sss_ctc['decoupler'] = sss_ctc['decoupler'].T.tocsc()
    else:
        sss_ctc = dict()

    #
    # Fine calibration processing (point-like magnetometers and calib. coeffs)
    #
    if calibration is not None:
        # Compute point-like mags to incorporate gradiometer imbalance
        S_fine = _sss_basis_point(origin, grad_info, int_order, ext_order,
                                  grad_imbalances, head_frame=head_frame)
        # Add point like magnetometer data to bases.
        S_decomp[grad_picks, :] += S_fine
        # Scale magnetometers by calibration coefficient
        S_decomp[mag_picks, :] /= mag_cals

    #
    # Regularization
    #
    if regularize is not None:  # regularize='in'
        logger.info('    Computing regularization')
        in_removes, out_removes = _regularize_in(
            int_order, ext_order, coil_scale, S_decomp[good_picks])
        logger.info('        Using %s/%s inside and %s/%s outside harmonic '
                    'components' % (n_in - len(in_removes), n_in,
                                    n_out - len(out_removes), n_out))
        reg_in_moments = np.setdiff1d(np.arange(n_in), in_removes)
        n_use_in = len(reg_in_moments)
        reg_out_moments = np.arange(n_in, n_in + n_out)
        reg_moments = np.concatenate((reg_in_moments, reg_out_moments))
        S_decomp = S_decomp.take(reg_moments, axis=1)
    else:
        reg_in_moments = slice(None, None, None)
        n_use_in = n_in
        reg_moments = slice(None, None, None)

    #
    # Translate to destination frame
    #
    if destination is not None:
        info_recon = deepcopy(info)
        info_recon['dev_head_t']['trans'] = recon_trans
        recon_coils = _prep_meg_channels(info_recon,
                                         accurate=True, elekta_defs=True,
                                         head_frame=head_frame,
                                         verbose=False)[0]
        S_recon = _sss_basis(origin, recon_coils, int_order, ext_order)
        # warn if we have translated too far
        diff = 1000 * (info['dev_head_t']['trans'][:3, 3] -
                       info_recon['dev_head_t']['trans'][:3, 3])
        dist = np.sqrt(np.sum(_sq(diff)))
        if dist > 25.:
            logger.warning('Head position change is over 25 mm (%s) = %0.1f mm'
                           % (', '.join('%0.1f' % x for x in diff), dist))

    #
    # Do the heavy lifting
    #

    # Pseudo-inverse of total multipolar moment basis set (Part of Eq. 37)
    pS_decomp_good = _col_norm_pinv(S_decomp[good_picks])

    # Build in our data scaling here
    pS_decomp_good *= coil_scale[good_picks].T

    # Split into inside and outside versions
    pS_decomp_in = pS_decomp_good[:n_use_in]
    pS_decomp_out = pS_decomp_good[n_use_in:]
    del pS_decomp_good

    # Reconstruct data from internal space only (Eq. 38), first rescale S_recon
    S_recon = S_recon[:, reg_moments]
    S_recon /= coil_scale

    # Reconstruct raw file object with spatiotemporal processed data
    max_st = dict()
    if st_duration is not None:
        max_st.update(job=10, subspcorr=st_correlation, buflen=st_duration)
        logger.info('    Processing data using tSSS with st_duration=%s'
                    % st_duration)
    else:
        st_duration = min(raw_sss.times[-1], 10.)  # chunk size
        st_correlation = None

    # Generate time points to break up data in to windows
    lims = raw_sss.time_as_index(np.arange(times[0], times[-1],
                                           st_duration))
    len_last_buf = raw_sss.times[-1] - raw_sss.index_as_time(lims[-1])[0]
    if len_last_buf == st_duration:
        lims = np.concatenate([lims, [len(raw_sss.times)]])
    else:
        # len_last_buf < st_dur so fold it into the previous buffer
        lims[-1] = len(raw_sss.times)
        if st_correlation is not None:
            logger.info('    Spatiotemporal window did not fit evenly into '
                        'raw object. The final %0.2f seconds were lumped '
                        'onto the previous window.' % len_last_buf)

    S_decomp /= coil_scale
    S_decomp = S_decomp[good_picks]
    logger.info('    Processing data in chunks of %0.1f sec' % st_duration)
    # Loop through buffer windows of data
    for start, stop in zip(lims[:-1], lims[1:]):
        # Compute multipolar moments of (magnetometer scaled) data (Eq. 37)
        orig_data = raw_sss._data[good_picks, start:stop]
        if cross_talk is not None:
            orig_data = ctc.dot(orig_data)
        mm_in = np.dot(pS_decomp_in, orig_data)
        in_data = np.dot(S_recon[:, :n_use_in], mm_in)

        if st_correlation is not None:
            # Reconstruct data using original location from external
            # and internal spaces and compute residual
            mm_out = np.dot(pS_decomp_out, orig_data)
            resid = orig_data  # we will operate inplace but it's safe
            orig_in_data = np.dot(S_decomp[:, :n_use_in], mm_in)
            orig_out_data = np.dot(S_decomp[:, n_use_in:], mm_out)
            resid -= orig_in_data
            resid -= orig_out_data
            _check_finite(resid)

            # Compute SSP-like projection vectors based on minimal correlation
            _check_finite(orig_in_data)
            t_proj = _overlap_projector(orig_in_data, resid, st_correlation)

            # Apply projector according to Eq. 12 in [2]_
            logger.info('        Projecting %s intersecting tSSS components '
                        'for %0.3f-%0.3f sec'
                        % (t_proj.shape[1], start / raw_sss.info['sfreq'],
                           stop / raw_sss.info['sfreq']))
            in_data -= np.dot(np.dot(in_data, t_proj), t_proj.T)
        raw_sss._data[meg_picks, start:stop] = in_data

    # Update info
    _update_sss_info(raw_sss, origin, int_order, ext_order, len(good_picks),
                     coord_frame, sss_ctc, sss_cal, max_st, reg_moments)
    logger.info('[done]')
    return raw_sss


def _col_norm_pinv(x):
    """Compute the pinv with column-normalization to stabilize calculation

    Note: will modify/overwrite x.
    """
    norm = np.sqrt(np.sum(x * x, axis=0))
    x /= norm
    u, s, v = linalg.svd(x, full_matrices=False, overwrite_a=True,
                         **check_disable)
    v /= norm
    return np.dot(v.T * 1. / s, u.T)


def _sq(x):
    """Helper to square"""
    return x * x


def _check_finite(data):
    """Helper to ensure data is finite"""
    if not np.isfinite(data).all():
        raise RuntimeError('data contains non-finite numbers')


def _sph_harm_norm(order, degree):
    """Normalization factor for spherical harmonics"""
    # we could use scipy.special.poch(degree + order + 1, -2 * order)
    # here, but it's slower for our fairly small degree
    norm = np.sqrt((2 * degree + 1.) / (4 * np.pi))
    if order != 0:
        norm *= np.sqrt(factorial(degree - order) /
                        float(factorial(degree + order)))
    return norm


def _sph_harm(order, degree, az, pol, norm=True):
    """Evaluate point in specified multipolar moment. [1]_ Equation 4.

    When using, pay close attention to inputs. Spherical harmonic notation for
    order/degree, and theta/phi are both reversed in original SSS work compared
    to many other sources. See mathworld.wolfram.com/SphericalHarmonic.html for
    more discussion.

    Note that scipy has ``scipy.special.sph_harm``, but that function is
    too slow on old versions (< 0.15) for heavy use.

    Parameters
    ----------
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'.
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'.
    az : float
        Azimuthal (longitudinal) spherical coordinate [0, 2*pi]. 0 is aligned
        with x-axis.
    pol : float
        Polar (or colatitudinal) spherical coordinate [0, pi]. 0 is aligned
        with z-axis.
    norm : bool
        If True, include normalization factor.

    Returns
    -------
    base : complex float
        The spherical harmonic value.
    """
    from scipy.special import lpmv

    # Error checks
    if np.abs(order) > degree:
        raise ValueError('Absolute value of order must be <= degree')
    # Ensure that polar and azimuth angles are arrays
    az = np.asarray(az)
    pol = np.asarray(pol)
    if (np.abs(az) > 2 * np.pi).any():
        raise ValueError('Azimuth coords must lie in [-2*pi, 2*pi]')
    if(pol < 0).any() or (pol > np.pi).any():
        raise ValueError('Polar coords must lie in [0, pi]')
    # This is the "seismology" convention on Wikipedia, w/o Condon-Shortley
    if norm:
        norm = _sph_harm_norm(order, degree)
    else:
        norm = 1.
    return norm * lpmv(order, degree, np.cos(pol)) * np.exp(1j * order * az)


def _concatenate_sph_coils(coils):
    """Helper to concatenate MEG coil parameters for spherical harmoncs."""
    rs = np.concatenate([coil['r0_exey'] for coil in coils])
    wcoils = np.concatenate([coil['w'] for coil in coils])
    ezs = np.concatenate([np.tile(coil['ez'][np.newaxis, :],
                                  (len(coil['rmag']), 1))
                          for coil in coils])
    bins = np.repeat(np.arange(len(coils)),
                     [len(coil['rmag']) for coil in coils])
    return rs, wcoils, ezs, bins


_mu_0 = 4e-7 * np.pi  # magnetic permeability


def _sss_basis(origin, coils, int_order, ext_order, mag_scale=100.,
               method='standard'):
    """Compute SSS basis for given conditions.

    Parameters
    ----------
    origin : ndarray, shape (3,)
        Origin of the multipolar moment space in millimeters
    coils : list
        List of MEG coils. Each should contain coil information dict specifying
        position, normals, weights, number of integration points and channel
        type. All coil geometry must be in the same coordinate frame
        as ``origin`` (``head`` or ``meg``).
    int_order : int
        Order of the internal multipolar moment space
    ext_order : int
        Order of the external multipolar moment space
    scale : float
        Scale factor for magnetometers.
    method : str
        Basis calculation method. Used only for testing.

    Returns
    -------
    bases : ndarray, shape (n_coils, n_mult_moments)
        Internal and external basis sets as a single ndarray.

    Notes
    -----
    Incorporates magnetometer scaling factor. Does not normalize spaces.
    """
    # Compute vector between origin and coil, convert to spherical coords
    if method == 'standard':
        # Get position, normal, weights, and number of integration pts.
        rmags, cosmags, wcoils, bins = _concatenate_coils(coils)
        rmags -= origin
        # Convert points to spherical coordinates
        rad, az, pol = _cart_to_sph(rmags).T
        cosmags *= wcoils[:, np.newaxis]
        del rmags, wcoils
        out_type = np.float64
    else:  # testing equivalence method
        rs, wcoils, ezs, bins = _concatenate_sph_coils(coils)
        rs -= origin
        rad, az, pol = _cart_to_sph(rs).T
        ezs *= wcoils[:, np.newaxis]
        del rs, wcoils
        out_type = np.complex128
    del origin

    # Set up output matrices
    n_in, n_out = _get_n_moments([int_order, ext_order])
    S_tot = np.empty((len(coils), n_in + n_out), out_type)
    S_in = S_tot[:, :n_in]
    S_out = S_tot[:, n_in:]

    # Set all magnetometers (with 'coil_class' == 1.0) to be scaled by 100
    coil_scale = np.ones((len(coils)))
    coil_scale[np.array([coil['coil_class'] == FIFF.FWD_COILC_MAG
                         for coil in coils])] = mag_scale

    # Compute internal/external basis vectors (exclude degree 0; L/RHS Eq. 5)
    for degree in range(1, max(int_order, ext_order) + 1):
        # Only loop over positive orders, negative orders are handled
        # for efficiency within
        for order in range(degree + 1):
            S_in_out = list()
            grads_in_out = list()
            # Same spherical harmonic is used for both internal and external
            sph = _sph_harm(order, degree, az, pol, norm=False)
            sph_norm = _sph_harm_norm(order, degree)
            sph *= sph_norm
            # Compute complex gradient for all integration points
            # in spherical coordinates (Eq. 6). The gradient for rad, az, pol
            # is obtained by taking the partial derivative of Eq. 4 w.r.t. each
            # coordinate.
            az_factor = 1j * order * sph / np.sin(np.maximum(pol, 1e-16))
            pol_factor = (-sph_norm * np.sin(pol) * np.exp(1j * order * az) *
                          _alegendre_deriv(order, degree, np.cos(pol)))
            if degree <= int_order:
                S_in_out.append(S_in)
                in_norm = _mu_0 * rad ** -(degree + 2)
                g_rad = in_norm * (-(degree + 1.) * sph)
                g_az = in_norm * az_factor
                g_pol = in_norm * pol_factor
                grads_in_out.append(_sph_to_cart_partials(az, pol,
                                                          g_rad, g_az, g_pol))
            if degree <= ext_order:
                S_in_out.append(S_out)
                out_norm = _mu_0 * rad ** (degree - 1)
                g_rad = out_norm * degree * sph
                g_az = out_norm * az_factor
                g_pol = out_norm * pol_factor
                grads_in_out.append(_sph_to_cart_partials(az, pol,
                                                          g_rad, g_az, g_pol))
            for spc, grads in zip(S_in_out, grads_in_out):
                # We could convert to real at the end, but it's more efficient
                # to do it now
                if method == 'standard':
                    grads_pos_neg = [_sh_complex_to_real(grads, order)]
                    orders_pos_neg = [order]
                    # Deal with the negative orders
                    if order > 0:
                        # it's faster to use the conjugation property for
                        # our normalized spherical harmonics than recalculate
                        grads_pos_neg.append(_sh_complex_to_real(
                            _sh_negate(grads, order), -order))
                        orders_pos_neg.append(-order)
                    for gr, oo in zip(grads_pos_neg, orders_pos_neg):
                        # Gradients dotted w/integration point weighted normals
                        gr = np.einsum('ij,ij->i', gr, cosmags)
                        vals = np.bincount(bins, gr, len(coils))
                        spc[:, _deg_order_idx(degree, oo)] = -vals
                else:
                    grads = np.einsum('ij,ij->i', grads, ezs)
                    v = (np.bincount(bins, grads.real, len(coils)) +
                         1j * np.bincount(bins, grads.imag, len(coils)))
                    spc[:, _deg_order_idx(degree, order)] = -v
                    if order > 0:
                        spc[:, _deg_order_idx(degree, -order)] = \
                            -_sh_negate(v, order)

    # Scale magnetometers
    S_tot *= coil_scale[:, np.newaxis]
    if method != 'standard':
        # Eventually we could probably refactor this for 2x mem (and maybe CPU)
        # savings by changing how spc/S_tot is assigned above (real only)
        S_tot = _bases_complex_to_real(S_tot, int_order, ext_order)
    return S_tot


def _get_degrees_orders(order):
    """Helper to get the set of degrees used in our basis functions"""
    degrees = np.zeros(_get_n_moments(order), int)
    orders = np.zeros_like(degrees)
    for degree in range(1, order + 1):
        # Only loop over positive orders, negative orders are handled
        # for efficiency within
        for order in range(degree + 1):
            ii = _deg_order_idx(degree, order)
            degrees[ii] = degree
            orders[ii] = order
            ii = _deg_order_idx(degree, -order)
            degrees[ii] = degree
            orders[ii] = -order
    return degrees, orders


def _deg_order_idx(deg, order):
    """Helper to get the index into S_in or S_out given a degree and order"""
    return _sq(deg) + deg + order - 1


def _alegendre_deriv(order, degree, val):
    """Compute the derivative of the associated Legendre polynomial at a value.

    Parameters
    ----------
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'.
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'.
    val : float
        Value to evaluate the derivative at.

    Returns
    -------
    dPlm : float
        Associated Legendre function derivative
    """
    from scipy.special import lpmv
    assert order >= 0
    return (order * val * lpmv(order, degree, val) + (degree + order) *
            (degree - order + 1.) * np.sqrt(1. - val * val) *
            lpmv(order - 1, degree, val)) / (1. - val * val)


def _sh_negate(sh, order):
    """Helper to get the negative spherical harmonic from a positive one"""
    assert order >= 0
    return sh.conj() * (-1. if order % 2 else 1.)  # == (-1) ** order


def _sh_complex_to_real(sh, order):
    """Helper function to convert complex to real basis functions.

    Parameters
    ----------
    sh : array-like
        Spherical harmonics. Must be from order >=0 even if negative orders
        are used.
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    real_sh : array-like
        The real version of the spherical harmonics.

    Notes
    -----
    This does not include the Condon-Shortely phase.
    """

    if order == 0:
        return np.real(sh)
    else:
        return np.sqrt(2.) * (np.real if order > 0 else np.imag)(sh)


def _sh_real_to_complex(shs, order):
    """Convert real spherical harmonic pair to complex

    Parameters
    ----------
    shs : ndarray, shape (2, ...)
        The real spherical harmonics at ``[order, -order]``.
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    sh : array-like, shape (...)
        The complex version of the spherical harmonics.
    """
    if order == 0:
        return shs[0]
    else:
        return (shs[0] + 1j * np.sign(order) * shs[1]) / np.sqrt(2.)


def _bases_complex_to_real(complex_tot, int_order, ext_order):
    """Convert complex spherical harmonics to real"""
    n_in, n_out = _get_n_moments([int_order, ext_order])
    complex_in = complex_tot[:, :n_in]
    complex_out = complex_tot[:, n_in:]
    real_tot = np.empty(complex_tot.shape, np.float64)
    real_in = real_tot[:, :n_in]
    real_out = real_tot[:, n_in:]
    for comp, real, exp_order in zip([complex_in, complex_out],
                                     [real_in, real_out],
                                     [int_order, ext_order]):
        for deg in range(1, exp_order + 1):
            for order in range(deg + 1):
                idx_pos = _deg_order_idx(deg, order)
                idx_neg = _deg_order_idx(deg, -order)
                real[:, idx_pos] = _sh_complex_to_real(comp[:, idx_pos], order)
                if order != 0:
                    # This extra mult factor baffles me a bit, but it works
                    # in round-trip testing, so we'll keep it :(
                    mult = (-1 if order % 2 == 0 else 1)
                    real[:, idx_neg] = mult * _sh_complex_to_real(
                        comp[:, idx_neg], -order)
    return real_tot


def _bases_real_to_complex(real_tot, int_order, ext_order):
    """Convert real spherical harmonics to complex"""
    n_in, n_out = _get_n_moments([int_order, ext_order])
    real_in = real_tot[:, :n_in]
    real_out = real_tot[:, n_in:]
    comp_tot = np.empty(real_tot.shape, np.complex128)
    comp_in = comp_tot[:, :n_in]
    comp_out = comp_tot[:, n_in:]
    for real, comp, exp_order in zip([real_in, real_out],
                                     [comp_in, comp_out],
                                     [int_order, ext_order]):
        for deg in range(1, exp_order + 1):
            # only loop over positive orders, figure out neg from pos
            for order in range(deg + 1):
                idx_pos = _deg_order_idx(deg, order)
                idx_neg = _deg_order_idx(deg, -order)
                this_comp = _sh_real_to_complex([real[:, idx_pos],
                                                 real[:, idx_neg]], order)
                comp[:, idx_pos] = this_comp
                comp[:, idx_neg] = _sh_negate(this_comp, order)
    return comp_tot


def _get_n_moments(order):
    """Compute the number of multipolar moments.

    Equivalent to [1]_ Eq. 32.

    Parameters
    ----------
    order : array-like
        Expansion orders, often ``[int_order, ext_order]``.

    Returns
    -------
    M : ndarray
        Number of moments due to each order.
    """
    order = np.asarray(order, int)
    return (order + 2) * order


def _sph_to_cart_partials(az, pol, g_rad, g_az, g_pol):
    """Convert spherical partial derivatives to cartesian coords.

    Note: Because we are dealing with partial derivatives, this calculation is
    not a static transformation. The transformation matrix itself is dependent
    on azimuth and polar coord.

    See the 'Spherical coordinate sytem' section here:
    wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    Parameters
    ----------
    az : ndarray, shape (n_points,)
        Array containing spherical coordinates points (azimuth).
    pol : ndarray, shape (n_points,)
        Array containing spherical coordinates points (polar).
    sph_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives at each spherical coordinate
        (radius, azimuth, polar).

    Returns
    -------
    cart_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives in Cartesian coordinates (x, y, z)
    """
    sph_grads = np.c_[g_rad, g_az, g_pol]
    cart_grads = np.zeros_like(sph_grads)
    c_as, s_as = np.cos(az), np.sin(az)
    c_ps, s_ps = np.cos(pol), np.sin(pol)
    trans = np.array([[c_as * s_ps, -s_as, c_as * c_ps],
                      [s_as * s_ps, c_as, c_ps * s_as],
                      [c_ps, np.zeros_like(c_as), -s_ps]])
    cart_grads = np.einsum('ijk,kj->ki', trans, sph_grads)
    return cart_grads


def _cart_to_sph(cart_pts):
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing points in spherical coordinates (rad, azimuth, polar)
    """
    rad = np.sqrt(np.sum(cart_pts * cart_pts, axis=1))
    az = np.arctan2(cart_pts[:, 1], cart_pts[:, 0])
    pol = np.arccos(cart_pts[:, 2] / rad)
    return np.array([rad, az, pol]).T


def _check_raw(raw):
    """Ensure that Maxwell filtering has not been applied yet"""
    if not isinstance(raw, _BaseRaw):
        raise TypeError('raw must be Raw, not %s' % type(raw))
    for ent in raw.info.get('proc_history', []):
        for msg, key in (('SSS', 'sss_info'),
                         ('tSSS', 'max_st'),
                         ('fine calibration', 'sss_cal'),
                         ('cross-talk cancellation',  'sss_ctc')):
            if len(ent['max_info'][key]) > 0:
                raise RuntimeError('Maxwell filtering %s step has already '
                                   'been applied' % msg)


def _update_sss_info(raw, origin, int_order, ext_order, nchan, coord_frame,
                     sss_ctc, sss_cal, max_st, reg_moments):
    """Helper function to update info inplace after Maxwell filtering

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    origin : array-like, shape (3,)
        Origin of internal and external multipolar moment space in head coords
        and in millimeters
    int_order : int
        Order of internal component of spherical expansion
    ext_order : int
        Order of external component of spherical expansion
    nchan : int
        Number of sensors
    sss_ctc : dict
        The cross talk information.
    sss_cal : dict
        The calibration information.
    max_st : dict
        The tSSS information.
    reg_moments : ndarray | slice
        The moments that were used.
    """
    n_in, n_out = _get_n_moments([int_order, ext_order])
    raw.info['maxshield'] = False
    components = np.zeros(n_in + n_out).astype('int32')
    components[reg_moments] = 1
    sss_info_dict = dict(in_order=int_order, out_order=ext_order,
                         nchan=nchan, origin=origin.astype('float32'),
                         job=np.array([2]), nfree=np.sum(components[:n_in]),
                         frame=_str_to_frame[coord_frame],
                         components=components)
    max_info_dict = dict(sss_info=sss_info_dict, max_st=max_st,
                         sss_cal=sss_cal, sss_ctc=sss_ctc)
    block_id = _generate_meas_id()
    proc_block = dict(max_info=max_info_dict, block_id=block_id,
                      creator='mne-python v%s' % __version__,
                      date=_date_now(), experimentor='')
    raw.info['proc_history'] = [proc_block] + raw.info.get('proc_history', [])
    # Reset 'bads' for any MEG channels since they've been reconstructed
    meg_picks = pick_types(raw.info, meg=True, exclude=[])
    raw.info['bads'] = [bad for bad in raw.info['bads']
                        if raw.ch_names.index(bad) not in meg_picks]


check_disable = dict()  # not available on really old versions of SciPy
if 'check_finite' in _get_args(linalg.svd):
    check_disable['check_finite'] = False


def _orth_overwrite(A):
    """Helper to create a slightly more efficient 'orth'"""
    # adapted from scipy/linalg/decomp_svd.py
    u, s = linalg.svd(A, overwrite_a=True, full_matrices=False,
                      **check_disable)[:2]
    M, N = A.shape
    eps = np.finfo(float).eps
    tol = max(M, N) * np.amax(s) * eps
    num = np.sum(s > tol, dtype=int)
    return u[:, :num]


def _overlap_projector(data_int, data_res, corr):
    """Calculate projector for removal of subspace intersection in tSSS"""
    # corr necessary to deal with noise when finding identical signal
    # directions in the subspace. See the end of the Results section in [2]_

    # Note that the procedure here is an updated version of [2]_ (and used in
    # Elekta's tSSS) that uses residuals instead of internal/external spaces
    # directly. This provides more degrees of freedom when analyzing for
    # intersections between internal and external spaces.

    # Normalize data, then compute orth to get temporal bases. Matrices
    # must have shape (n_samps x effective_rank) when passed into svd
    # computation
    n = np.sqrt(np.sum(data_int * data_int))
    Q_int = linalg.qr(_orth_overwrite((data_int / n).T),
                      overwrite_a=True, mode='economic', **check_disable)[0].T
    n = np.sqrt(np.sum(data_res * data_res))
    Q_res = linalg.qr(_orth_overwrite((data_res / n).T),
                      overwrite_a=True, mode='economic', **check_disable)[0]
    assert data_int.shape[1] > 0
    C_mat = np.dot(Q_int, Q_res)
    del Q_int

    # Compute angles between subspace and which bases to keep
    S_intersect, Vh_intersect = linalg.svd(C_mat, overwrite_a=True,
                                           full_matrices=False,
                                           **check_disable)[1:]
    del C_mat
    intersect_mask = (S_intersect >= corr)
    del S_intersect

    # Compute projection operator as (I-LL_T) Eq. 12 in [2]_
    # V_principal should be shape (n_time_pts x n_retained_inds)
    Vh_intersect = Vh_intersect[intersect_mask].T
    V_principal = np.dot(Q_res, Vh_intersect)
    return V_principal


def _read_fine_cal(fine_cal):
    """Read sensor locations and calib. coeffs from fine calibration file."""

    # Read new sensor locations
    cal_chs = list()
    cal_ch_numbers = list()
    with open(fine_cal, 'r') as fid:
        lines = [line for line in fid if line[0] not in '#\n']
        for line in lines:
            # `vals` contains channel number, (x, y, z), x-norm 3-vec, y-norm
            # 3-vec, z-norm 3-vec, and (1 or 3) imbalance terms
            vals = np.fromstring(line, sep=' ').astype(np.float64)

            # Check for correct number of items
            if len(vals) not in [14, 16]:
                raise RuntimeError('Error reading fine calibration file')

            ch_name = 'MEG' + '%04d' % vals[0]  # Zero-pad names to 4 char
            cal_ch_numbers.append(vals[0])

            # Get orientation information for coil transformation
            loc = vals[1:13].copy()  # Get orientation information for 'loc'
            calib_coeff = vals[13:].copy()  # Get imbalance/calibration coeff
            cal_chs.append(dict(ch_name=ch_name,
                                loc=loc, calib_coeff=calib_coeff,
                                coord_frame=FIFF.FIFFV_COORD_DEVICE))
    return cal_chs, cal_ch_numbers


def _skew_symmetric_cross(a):
    """The skew-symmetric cross product of a vector"""
    return np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])


def _find_vector_rotation(a, b):
    """Find the rotation matrix that maps unit vector a to b"""
    # Rodrigues' rotation formula:
    #   https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    #   http://math.stackexchange.com/a/476311
    R = np.eye(3)
    v = np.cross(a, b)
    if np.allclose(v, 0.):  # identical
        return R
    s = np.sqrt(np.sum(v * v))  # sine of the angle between them
    c = np.sqrt(np.sum(a * b))  # cosine of the angle between them
    vx = _skew_symmetric_cross(v)
    R += vx + np.dot(vx, vx) * (1 - c) / s
    return R


def _update_sensor_geometry(info, fine_cal):
    """Helper to replace sensor geometry information and reorder cal_chs"""
    logger.info('    Using fine calibration %s' % op.basename(fine_cal))
    cal_chs, cal_ch_numbers = _read_fine_cal(fine_cal)

    # Check that we ended up with correct channels
    meg_info = pick_info(info, pick_types(info, meg=True, exclude=[]))
    clean_meg_names = _clean_names(meg_info['ch_names'],
                                   remove_whitespace=True)
    order = pick_channels([c['ch_name'] for c in cal_chs], clean_meg_names)
    if not (len(cal_chs) == meg_info['nchan'] == len(order)):
        raise RuntimeError('Number of channels in fine calibration file (%i) '
                           'does not equal number of channels in info (%i)' %
                           (len(cal_chs), meg_info['nchan']))
    # ensure they're ordered like our data
    cal_chs = [cal_chs[ii] for ii in order]

    # Replace sensor locations (and track differences) for fine calibration
    ang_shift = np.zeros((len(cal_chs), 3))
    used = np.zeros(len(info['chs']), bool)
    cal_corrs = list()
    coil_types = list()
    grad_picks = pick_types(meg_info, meg='grad')
    adjust_logged = False
    clean_info_names = _clean_names(info['ch_names'], remove_whitespace=True)
    for ci, cal_ch in enumerate(cal_chs):
        idx = clean_info_names.index(cal_ch['ch_name'])
        assert not used[idx]
        used[idx] = True
        info_ch = info['chs'][idx]
        coil_types.append(info_ch['coil_type'])

        # Some .dat files might only rotate EZ, so we must check first that
        # EX and EY are orthogonal to EZ. If not, we find the rotation between
        # the original and fine-cal ez, and rotate EX and EY accordingly:
        ch_coil_rot = _loc_to_coil_trans(info_ch['loc'])[:3, :3]
        cal_loc = cal_ch['loc'].copy()
        cal_coil_rot = _loc_to_coil_trans(cal_loc)[:3, :3]
        if np.max([np.abs(np.dot(cal_coil_rot[:, ii], cal_coil_rot[:, 2]))
                   for ii in range(2)]) > 1e-6:  # X or Y not orthogonal
            if not adjust_logged:
                logger.info('        Adjusting non-orthogonal EX and EY')
                adjust_logged = True
            # find the rotation matrix that goes from one to the other
            this_trans = _find_vector_rotation(ch_coil_rot[:, 2],
                                               cal_coil_rot[:, 2])
            cal_loc[3:] = np.dot(this_trans, ch_coil_rot).T.ravel()

        # calculate shift angle
        v1 = _loc_to_coil_trans(cal_ch['loc'])[:3, :3]
        _normalize_vectors(v1)
        v2 = _loc_to_coil_trans(info_ch['loc'])[:3, :3]
        _normalize_vectors(v2)
        ang_shift[ci] = np.sum(v1 * v2, axis=0)
        if idx in grad_picks:
            extra = [1., cal_ch['calib_coeff'][0]]
        else:
            extra = [cal_ch['calib_coeff'][0], 0.]
        cal_corrs.append(np.concatenate([extra, cal_loc]))
        # Adjust channel normal orientations with those from fine calibration
        # Channel positions are not changed
        info_ch['loc'][3:] = cal_loc[3:]
        assert (info_ch['coord_frame'] == cal_ch['coord_frame'] ==
                FIFF.FIFFV_COORD_DEVICE)
    cal_chans = [[sc, ct] for sc, ct in zip(cal_ch_numbers, coil_types)]
    sss_cal = dict(cal_corrs=np.array(cal_corrs),
                   cal_chans=np.array(cal_chans))

    # Deal with numerical precision giving absolute vals slightly more than 1.
    np.clip(ang_shift, -1., 1., ang_shift)
    np.rad2deg(np.arccos(ang_shift), ang_shift)  # Convert to degrees

    # Log quantification of sensor changes
    logger.info('        Adjusted coil positions by (μ ± σ): '
                '%0.1f° ± %0.1f° (max: %0.1f°)' %
                (np.mean(ang_shift), np.std(ang_shift),
                 np.max(np.abs(ang_shift))))

    # Determine gradiometer imbalances and magnetometer calibrations
    grad_picks = pick_types(info, meg='grad', exclude=[])
    mag_picks = pick_types(info, meg='mag', exclude=[])
    grad_imbalances = np.array([cal_chs[ii]['calib_coeff']
                                for ii in grad_picks]).T
    mag_cals = np.array([cal_chs[ii]['calib_coeff'] for ii in mag_picks])
    return grad_imbalances, mag_cals, sss_cal


def _sss_basis_point(origin, info, int_order, ext_order, imbalances,
                     head_frame=True):
    """Compute multipolar moments for point-like magnetometers (in fine cal)"""

    # Construct 'coils' with r, weights, normal vecs, # integration pts, and
    # channel type.
    if imbalances.shape[0] not in [1, 3]:
        raise ValueError('Must have 1 (x) or 3 (x, y, z) point-like ' +
                         'magnetometers. Currently have %i' %
                         imbalances.shape[0])

    # Coil_type values for x, y, z point magnetometers
    # Note: 1D correction files only have x-direction corrections
    pt_types = [FIFF.FIFFV_COIL_POINT_MAGNETOMETER_X,
                FIFF.FIFFV_COIL_POINT_MAGNETOMETER_Y,
                FIFF.FIFFV_COIL_POINT_MAGNETOMETER]

    # Loop over all coordinate directions desired and create point mags
    S_tot = 0.
    for imb, pt_type in zip(imbalances, pt_types):
        temp_info = deepcopy(info)
        for ch in temp_info['chs']:
            ch['coil_type'] = pt_type
        coils_add = _prep_meg_channels(
            temp_info, accurate=True, elekta_defs=True, head_frame=head_frame,
            verbose=False)[0]
        # Scale spaces by gradiometer imbalance
        S_add = _sss_basis(origin, coils_add, int_order,
                           ext_order) * imb[:, np.newaxis]
        S_tot += S_add

    # Return point-like mag bases
    return S_tot


def _regularize_in(int_order, ext_order, coil_scale, S_decomp):
    """Regularize basis set using idealized SNR measure"""
    n_in, n_out = _get_n_moments([int_order, ext_order])

    # The "signal" terms depend only on the inner expansion order
    # (i.e., not sensor geometry or head position / expansion origin)
    a_lm_sq, rho_i = _compute_sphere_activation_in(
        np.arange(int_order + 1))
    degrees, orders = _get_degrees_orders(int_order)
    a_lm_sq = a_lm_sq[degrees]

    I_tots = np.empty(n_in)
    in_keepers = list(range(n_in))
    out_keepers = list(range(n_in, n_in + n_out))
    remove_order = []
    S_decomp = S_decomp.copy()
    use_norm = np.sqrt(np.sum(S_decomp * S_decomp, axis=0))
    S_decomp /= use_norm
    eigs = np.zeros((n_in, 2))

    # plot = False  # for debugging
    # if plot:
    #     import matplotlib.pyplot as plt
    #     fig, axs = plt.subplots(3, figsize=[6, 12])
    #     plot_ord = np.empty(n_in, int)
    #     plot_ord.fill(-1)
    #     count = 0
    #     # Reorder plot to match MF
    #     for degree in range(1, int_order + 1):
    #         for order in range(0, degree + 1):
    #             assert plot_ord[count] == -1
    #             plot_ord[count] = _deg_order_idx(degree, order)
    #             count += 1
    #             if order > 0:
    #                 assert plot_ord[count] == -1
    #                 plot_ord[count] = _deg_order_idx(degree, -order)
    #                 count += 1
    #     assert count == n_in
    #     assert (plot_ord >= 0).all()
    #     assert len(np.unique(plot_ord)) == n_in
    noise_lev = 5e-13  # noise level in T/m
    noise_lev *= noise_lev  # effectively what would happen by earlier multiply
    for ii in range(n_in):
        this_S = S_decomp.take(in_keepers + out_keepers, axis=1)
        u, s, v = linalg.svd(this_S, full_matrices=False, overwrite_a=True,
                             **check_disable)
        eigs[ii] = s[[0, -1]]
        v = v.T[:len(in_keepers)]
        v /= use_norm[in_keepers][:, np.newaxis]
        eta_lm_sq = np.dot(v * 1. / s, u.T)
        del u, s, v
        eta_lm_sq *= eta_lm_sq
        eta_lm_sq = eta_lm_sq.sum(axis=1)
        eta_lm_sq *= noise_lev

        # Mysterious scale factors to match Elekta, likely due to differences
        # in the basis normalizations...
        eta_lm_sq[orders[in_keepers] == 0] *= 2
        eta_lm_sq *= 0.0025
        snr = a_lm_sq[in_keepers] / eta_lm_sq
        I_tots[ii] = 0.5 * np.log2(snr + 1.).sum()
        remove_order.append(in_keepers[np.argmin(snr)])
        in_keepers.pop(in_keepers.index(remove_order[-1]))
        # if plot and ii == 0:
        #     axs[0].semilogy(snr[plot_ord[in_keepers]], color='k')
    # if plot:
    #     axs[0].set(ylabel='SNR', ylim=[0.1, 500], xlabel='Component')
    #     axs[1].plot(I_tots)
    #     axs[1].set(ylabel='Information', xlabel='Iteration')
    #     axs[2].plot(eigs[:, 0] / eigs[:, 1])
    #     axs[2].set(ylabel='Condition', xlabel='Iteration')
    # Pick the components that give at least 98% of max info
    # This is done because the curves can be quite flat, and we err on the
    # side of including rather than excluding components
    max_info = np.max(I_tots)
    lim_idx = np.where(I_tots >= 0.98 * max_info)[0][0]
    in_removes = remove_order[:lim_idx]
    for ii, ri in enumerate(in_removes):
        logger.debug('            Condition %0.3f/%0.3f = %03.1f, '
                     'Removing in component %s: l=%s, m=%+0.0f'
                     % (tuple(eigs[ii]) + (eigs[ii, 0] / eigs[ii, 1],
                        ri, degrees[ri], orders[ri])))
    logger.info('        Resulting information: %0.1f bits/sample '
                '(%0.1f%% of peak %0.1f)'
                % (I_tots[lim_idx], 100 * I_tots[lim_idx] / max_info,
                   max_info))
    return in_removes, list()


def _compute_sphere_activation_in(degrees):
    """Helper to compute the "in" power from random currents in a sphere

    Parameters
    ----------
    degrees : ndarray
        The degrees to evaluate.

    Returns
    -------
    a_power : ndarray
        The a_lm associated for the associated degrees.
    rho_i : float
        The current density.

    Notes
    -----
    See also:

        A 122-channel whole-cortex SQUID system for measuring the brain’s
        magnetic fields. Knuutila et al. IEEE Transactions on Magnetics,
        Vol 29 No 6, Nov 1993.
    """
    r_in = 0.080  # radius of the randomly-activated sphere

    # set the observation point r=r_s, az=el=0, so we can just look at m=0 term
    # compute the resulting current density rho_i

    # This is the "surface" version of the equation:
    # b_r_in = 100e-15  # fixed radial field amplitude at distance r_s = 100 fT
    # r_s = 0.13  # 5 cm from the surface
    # rho_degrees = np.arange(1, 100)
    # in_sum = (rho_degrees * (rho_degrees + 1.) /
    #           ((2. * rho_degrees + 1.)) *
    #           (r_in / r_s) ** (2 * rho_degrees + 2)).sum() * 4. * np.pi
    # rho_i = b_r_in * 1e7 / np.sqrt(in_sum)
    # rho_i = 5.21334885574e-07  # value for r_s = 0.125
    rho_i = 5.91107375632e-07  # deterministic from above, so just store it
    a_power = _sq(rho_i) * (degrees * r_in ** (2 * degrees + 4) /
                            (_sq(2. * degrees + 1.) *
                            (degrees + 1.)))
    return a_power, rho_i
