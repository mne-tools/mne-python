"""Dynamic Imaging of Coherent Sources (DICS)."""

# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#          Susanna Aro <susanna.aro@aalto.fi>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD-3-Clause
import numpy as np

from ..channels import equalize_channels
from ..io.pick import pick_info, pick_channels
from ..utils import (logger, verbose, _check_one_ch_type,
                     _check_channels_spatial_filter, _check_rank,
                     _check_option, _validate_type, warn)
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import combine_xyz, _check_reference, _check_depth
from ..rank import compute_rank
from ..source_estimate import _make_stc, _get_src_type
from ..time_frequency import EpochsTFR
from ..time_frequency.tfr import _check_tfr_complex
from ._compute_beamformer import (_prepare_beamformer_input,
                                  _compute_beamformer, _check_src_type,
                                  Beamformer, _compute_power,
                                  _proj_whiten_data)


@verbose
def make_dics(info, forward, csd, reg=0.05, noise_csd=None, label=None,
              pick_ori=None, rank=None, weight_norm=None,
              reduce_rank=False, depth=1., real_filter=True,
              inversion='matrix', verbose=None):
    """Compute a Dynamic Imaging of Coherent Sources (DICS) spatial filter.

    This is a beamformer filter that can be used to estimate the source power
    at a specific frequency range :footcite:`GrossEtAl2001`. It does this by
    constructing a spatial filter for each source point.
    The computation of these filters is very similar to those of the LCMV
    beamformer (:func:`make_lcmv`), but instead of operating on a covariance
    matrix, the CSD matrix is used. When applying these filters to a CSD matrix
    (see :func:`apply_dics_csd`), the source power can be estimated for each
    source point.

    Parameters
    ----------
    %(info_not_none)s
    forward : instance of Forward
        Forward operator.
    csd : instance of CrossSpectralDensity
        The data cross-spectral density (CSD) matrices. A source estimate is
        performed for each frequency or frequency-bin defined in the CSD
        object.
    reg : float
        The regularization to apply to the cross-spectral density before
        computing the inverse.
    noise_csd : instance of CrossSpectralDensity | None
        Noise cross-spectral density (CSD) matrices. If provided, whitening
        will be done. The noise CSDs need to have been computed for the same
        frequencies as the data CSDs. Providing noise CSDs is mandatory if you
        mix sensor types, e.g. gradiometers with magnetometers or EEG with
        MEG.

        .. versionadded:: 0.20
    label : Label | None
        Restricts the solution to a given label.
    %(pick_ori_bf)s
    %(rank_none)s

        .. versionadded:: 0.17
    %(weight_norm)s

        Defaults to ``None``, in which case no normalization is performed.
    %(reduce_rank)s
    %(depth)s
    real_filter : bool
        If ``True``, take only the real part of the cross-spectral-density
        matrices to compute real filters.

        .. versionchanged:: 0.23
            Version 0.23 an earlier used ``real_filter=False`` as the default,
            as of version 0.24 ``True`` is the default.
    %(inversion_bf)s

        .. versionchanged:: 0.21
           Default changed to ``'matrix'``.
    %(verbose)s

    Returns
    -------
    filters : instance of Beamformer
        Dictionary containing filter weights from DICS beamformer.
        Contains the following keys:

            'kind' : str
                The type of beamformer, in this case 'DICS'.
            'weights' : ndarray, shape (n_frequencies, n_weights)
                For each frequency, the filter weights of the beamformer.
            'csd' : instance of CrossSpectralDensity
                The data cross-spectral density matrices used to compute the
                beamformer.
            'ch_names' : list of str
                Channels used to compute the beamformer.
            'proj' : ndarray, shape (n_channels, n_channels)
                Projections used to compute the beamformer.
            'vertices' : list of ndarray
                Vertices for which the filter weights were computed.
            'n_sources' : int
                Number of source location for which the filter weight were
                computed.
            'subject' : str
                The subject ID.
            'pick-ori' : None | 'max-power' | 'normal' | 'vector'
                The orientation in which the beamformer filters were computed.
            'inversion' : 'single' | 'matrix'
                Whether the spatial filters were computed for each dipole
                separately or jointly for all dipoles at each vertex using a
                matrix inversion.
            'weight_norm' : None | 'unit-noise-gain'
                The normalization of the weights.
            'src_type' : str
                Type of source space.
            'source_nn' : ndarray, shape (n_sources, 3)
                For each source location, the surface normal.
            'is_free_ori' : bool
                Whether the filter was computed in a fixed direction
                (pick_ori='max-power', pick_ori='normal') or not.
            'whitener' : None | ndarray, shape (n_channels, n_channels)
                Whitening matrix, provided if whitening was applied to the
                covariance matrix and leadfield during computation of the
                beamformer weights.
            'max-power-ori' : ndarray, shape (n_sources, 3) | None
                When pick_ori='max-power', this fields contains the estimated
                direction of maximum power at each source location.

    See Also
    --------
    apply_dics_csd

    Notes
    -----
    The original reference is :footcite:`GrossEtAl2001`. See
    :footcite:`vanVlietEtAl2018` for a tutorial style paper on the topic.

    The DICS beamformer is very similar to the LCMV (:func:`make_lcmv`)
    beamformer and many of the parameters are shared. However,
    :func:`make_dics` and :func:`make_lcmv` currently have different defaults
    for these parameters, which were settled on separately through extensive
    practical use case testing (but not necessarily exhaustive parameter space
    searching), and it remains to be seen how functionally interchangeable they
    could be.

    The default setting reproduce the DICS beamformer as described in
    :footcite:`vanVlietEtAl2018`::

        inversion='single', weight_norm=None, depth=1.

    To use the :func:`make_lcmv` defaults, use::

        inversion='matrix', weight_norm='unit-noise-gain-invariant', depth=None

    For more information about ``real_filter``, see the
    supplemental information from :footcite:`HippEtAl2011`.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    rank = _check_rank(rank)
    _check_option('pick_ori', pick_ori,
                  [None, 'vector', 'normal', 'max-power'])
    _check_option('inversion', inversion, ['single', 'matrix'])
    _validate_type(weight_norm, (str, None), 'weight_norm')

    frequencies = [np.mean(freq_bin) for freq_bin in csd.frequencies]
    n_freqs = len(frequencies)

    _, _, allow_mismatch = _check_one_ch_type('dics', info, forward, csd,
                                              noise_csd)
    # remove bads so that equalize_channels only keeps all good
    info = pick_info(info, pick_channels(info['ch_names'], [], info['bads']))
    info, forward, csd = equalize_channels([info, forward, csd])

    csd, noise_csd = _prepare_noise_csd(csd, noise_csd, real_filter)

    depth = _check_depth(depth, 'depth_sparse')
    if inversion == 'single':
        depth['combine_xyz'] = False

    is_free_ori, info, proj, vertices, G, whitener, nn, orient_std = \
        _prepare_beamformer_input(
            info, forward, label, pick_ori, noise_cov=noise_csd, rank=rank,
            pca=False, **depth)

    # Compute ranks
    csd_int_rank = []
    if not allow_mismatch:
        noise_rank = compute_rank(noise_csd, info=info, rank=rank)
    for i in range(len(frequencies)):
        csd_rank = compute_rank(csd.get_data(index=i, as_cov=True),
                                info=info, rank=rank)
        if not allow_mismatch:
            for key in csd_rank:
                if key not in noise_rank or csd_rank[key] != noise_rank[key]:
                    raise ValueError('%s data rank (%s) did not match the '
                                     'noise rank (%s)'
                                     % (key, csd_rank[key],
                                        noise_rank.get(key, None)))
        csd_int_rank.append(sum(csd_rank.values()))

    del noise_csd
    ch_names = list(info['ch_names'])

    logger.info('Computing DICS spatial filters...')
    Ws = []
    max_oris = []
    for i, freq in enumerate(frequencies):
        if n_freqs > 1:
            logger.info('    computing DICS spatial filter at '
                        f'{round(freq, 2)} Hz ({i + 1}/{n_freqs})')

        Cm = csd.get_data(index=i)

        # XXX: Weird that real_filter happens *before* whitening, which could
        # make things complex again...?
        if real_filter:
            Cm = Cm.real

        # compute spatial filter
        n_orient = 3 if is_free_ori else 1
        W, max_power_ori = _compute_beamformer(
            G, Cm, reg, n_orient, weight_norm, pick_ori, reduce_rank,
            rank=csd_int_rank[i], inversion=inversion, nn=nn,
            orient_std=orient_std, whitener=whitener)
        Ws.append(W)
        max_oris.append(max_power_ori)

    Ws = np.array(Ws)
    if pick_ori == 'max-power':
        max_oris = np.array(max_oris)
    else:
        max_oris = None

    src_type = _get_src_type(forward['src'], vertices)
    subject = _subject_from_forward(forward)
    is_free_ori = is_free_ori if pick_ori in [None, 'vector'] else False
    n_sources = np.sum([len(v) for v in vertices])

    filters = Beamformer(
        kind='DICS', weights=Ws, csd=csd, ch_names=ch_names, proj=proj,
        vertices=vertices, n_sources=n_sources, subject=subject,
        pick_ori=pick_ori, inversion=inversion, weight_norm=weight_norm,
        src_type=src_type, source_nn=forward['source_nn'].copy(),
        is_free_ori=is_free_ori, whitener=whitener, max_power_ori=max_oris)

    return filters


def _prepare_noise_csd(csd, noise_csd, real_filter):
    if noise_csd is not None:
        csd, noise_csd = equalize_channels([csd, noise_csd])
        # Use the same noise CSD for all frequencies
        if len(noise_csd.frequencies) > 1:
            noise_csd = noise_csd.mean()
        noise_csd = noise_csd.get_data(as_cov=True)
        if real_filter:
            noise_csd['data'] = noise_csd['data'].real
    return csd, noise_csd


def _apply_dics(data, filters, info, tmin, tfr=False):
    """Apply DICS spatial filter to data for source reconstruction."""
    if isinstance(data, np.ndarray) and data.ndim == (2 + tfr):
        data = [data]
        one_epoch = True
    else:
        one_epoch = False

    Ws = filters['weights']
    one_freq = len(Ws) == 1

    subject = filters['subject']
    # compatibility with 0.16, add src_type as None if not present:
    filters, warn_text = _check_src_type(filters)

    for i, M in enumerate(data):
        if not one_epoch:
            logger.info("Processing epoch : %d" % (i + 1))

        # Apply SSPs
        if not tfr:  # save computation, only compute once
            M_w = _proj_whiten_data(M, info['projs'], filters)

        stcs = []
        for j, W in enumerate(Ws):

            if tfr:  # must compute for each frequency
                M_w = _proj_whiten_data(M[:, j], info['projs'], filters)

            # project to source space using beamformer weights
            sol = np.dot(W, M_w)

            if filters['is_free_ori'] and filters['pick_ori'] != 'vector':
                logger.info('combining the current components...')
                sol = combine_xyz(sol)

            tstep = 1.0 / info['sfreq']

            stcs.append(_make_stc(sol, vertices=filters['vertices'],
                                  src_type=filters['src_type'], tmin=tmin,
                                  tstep=tstep, subject=subject,
                                  vector=(filters['pick_ori'] == 'vector'),
                                  source_nn=filters['source_nn'],
                                  warn_text=warn_text))
        if one_freq:
            yield stcs[0]
        else:
            yield stcs

    logger.info('[done]')


@verbose
def apply_dics(evoked, filters, verbose=None):
    """Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights.

    Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights
    on evoked data.

    .. warning:: The result of this function is meant as an intermediate step
                 for further processing (such as computing connectivity). If
                 you are interested in estimating source time courses, use an
                 LCMV beamformer (:func:`make_lcmv`, :func:`apply_lcmv`)
                 instead. If you are interested in estimating spectral power at
                 the source level, use :func:`apply_dics_csd`.
    .. warning:: This implementation has not been heavily tested so please
                 report any issues or suggestions.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to apply the DICS beamformer weights to.
    filters : instance of Beamformer
        DICS spatial filter (beamformer weights)
        Filter weights returned from :func:`make_dics`.
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate | list
        Source time courses. If the DICS beamformer has been computed for more
        than one frequency, a list is returned containing for each frequency
        the corresponding time courses.

    See Also
    --------
    apply_dics_epochs
    apply_dics_tfr_epochs
    apply_dics_csd
    """  # noqa: E501
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    sel = _check_channels_spatial_filter(evoked.ch_names, filters)
    data = data[sel]

    stc = _apply_dics(data=data, filters=filters, info=info, tmin=tmin)

    return next(stc)


@verbose
def apply_dics_epochs(epochs, filters, return_generator=False, verbose=None):
    """Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights.

    Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights
    on single trial data.

    .. warning:: The result of this function is meant as an intermediate step
                 for further processing (such as computing connectivity). If
                 you are interested in estimating source time courses, use an
                 LCMV beamformer (:func:`make_lcmv`, :func:`apply_lcmv`)
                 instead. If you are interested in estimating spectral power at
                 the source level, use :func:`apply_dics_csd`.
    .. warning:: This implementation has not been heavily tested so please
                 report any issue or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    filters : instance of Beamformer
        DICS spatial filter (beamformer weights)
        Filter weights returned from :func:`make_dics`. The DICS filters must
        have been computed for a single frequency only.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    %(verbose)s

    Returns
    -------
    stc: list | generator of (SourceEstimate | VolSourceEstimate)
        The source estimates for all epochs.

    See Also
    --------
    apply_dics
    apply_dics_tfr_epochs
    apply_dics_csd
    """
    _check_reference(epochs)

    if len(filters['weights']) > 1:
        raise ValueError(
            'This function only works on DICS beamformer weights that have '
            'been computed for a single frequency. When calling make_dics(), '
            'make sure to use a CSD object with only a single frequency (or '
            'frequency-bin) defined.'
        )

    info = epochs.info
    tmin = epochs.times[0]

    sel = _check_channels_spatial_filter(epochs.ch_names, filters)
    data = epochs.get_data()[:, sel, :]

    stcs = _apply_dics(data=data, filters=filters, info=info, tmin=tmin)

    if not return_generator:
        stcs = list(stcs)

    return stcs


@verbose
def apply_dics_tfr_epochs(epochs_tfr, filters, return_generator=False,
                          verbose=None):
    """Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights.

    Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights
    on single trial time-frequency data.

    Parameters
    ----------
    epochs_tfr : EpochsTFR
        Single trial time-frequency epochs.
    filters : instance of Beamformer
        DICS spatial filter (beamformer weights)
        Filter weights returned from :func:`make_dics`.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    %(verbose)s

    Returns
    -------
    stcs : list of list of (SourceEstimate | VectorSourceEstimate | VolSourceEstimate)
        The source estimates for all epochs (outside list) and for
        all frequencies (inside list).

    See Also
    --------
    apply_dics
    apply_dics_epochs
    apply_dics_csd
    """ # noqa E501
    _validate_type(epochs_tfr, EpochsTFR)
    _check_tfr_complex(epochs_tfr)

    if filters['pick_ori'] == 'vector':
        warn('Using a vector solution to compute power will lead to '
             'inaccurate directions (only in the first quadrent) '
             'because power is a strictly positive (squared) metric. '
             'Using singular value decomposition (SVD) to determine '
             'the direction is not yet supported in MNE.')

    sel = _check_channels_spatial_filter(epochs_tfr.ch_names, filters)
    data = epochs_tfr.data[:, sel, :, :]

    stcs = _apply_dics(data, filters, epochs_tfr.info,
                       epochs_tfr.tmin, tfr=True)
    if not return_generator:
        stcs = [[stc for stc in tfr_stcs] for tfr_stcs in stcs]
    return stcs


@verbose
def apply_dics_csd(csd, filters, verbose=None):
    """Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights.

    Apply a previously computed DICS beamformer to a cross-spectral density
    (CSD) object to estimate source power in time and frequency windows
    specified in the CSD object :footcite:`GrossEtAl2001`.

    .. note:: Only power can computed from the cross-spectral density, not
              complex phase-amplitude, so vector DICS filters will be
              converted to scalar source estimates since power is strictly
              positive and so 3D directions cannot be combined meaningfully
              (the direction would be confined to the positive quadrant).

    Parameters
    ----------
    csd : instance of CrossSpectralDensity
        The data cross-spectral density (CSD) matrices. A source estimate is
        performed for each frequency or frequency-bin defined in the CSD
        object.
    filters : instance of Beamformer
        DICS spatial filter (beamformer weights)
        Filter weights returned from `make_dics`.
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate
        Source power with frequency instead of time.
    frequencies : list of float
        The frequencies for which the source power has been computed. If the
        data CSD object defines frequency-bins instead of exact frequencies,
        the mean of each bin is returned.

    See Also
    --------
    apply_dics
    apply_dics_epochs
    apply_dics_tfr_epochs

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    ch_names = filters['ch_names']
    vertices = filters['vertices']
    n_orient = 3 if filters['is_free_ori'] else 1
    subject = filters['subject']
    whitener = filters['whitener']
    n_sources = filters['n_sources']

    # If CSD is summed over multiple frequencies, take the average frequency
    frequencies = [np.mean(dfreq) for dfreq in csd.frequencies]
    n_freqs = len(frequencies)

    source_power = np.zeros((n_sources, len(csd.frequencies)))

    # Ensure the CSD is in the same order as the weights
    csd_picks = [csd.ch_names.index(ch) for ch in ch_names]

    logger.info('Computing DICS source power...')
    for i, freq in enumerate(frequencies):
        if n_freqs > 1:
            logger.info('    applying DICS spatial filter at '
                        f'{round(freq, 2)} Hz ({i + 1}/{n_freqs})')

        Cm = csd.get_data(index=i)
        Cm = Cm[csd_picks, :][:, csd_picks]
        W = filters['weights'][i]

        # Whiten the CSD
        Cm = np.dot(whitener, np.dot(Cm, whitener.conj().T))

        source_power[:, i] = _compute_power(Cm, W, n_orient)

    logger.info('[done]')

    # compatibility with 0.16, add src_type as None if not present:
    filters, warn_text = _check_src_type(filters)

    return (_make_stc(source_power, vertices=vertices,
                      src_type=filters['src_type'], tmin=0., tstep=1.,
                      subject=subject, warn_text=warn_text),
            frequencies)
