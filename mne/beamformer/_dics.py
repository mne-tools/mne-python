"""Dynamic Imaging of Coherent Sources (DICS)."""

# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#          Susanna Aro <susanna.aro@aalto.fi>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)
import numpy as np

from ..io.pick import pick_info
from ..utils import (logger, verbose, warn, _check_one_ch_type,
                     _check_channels_spatial_filter, _check_rank,
                     _check_option, _check_info_inv)
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import combine_xyz, _check_reference
from ..source_estimate import _make_stc, _get_src_type
from ..time_frequency import csd_fourier, csd_multitaper, csd_morlet
from ._compute_beamformer import (_check_proj_match, _prepare_beamformer_input,
                                  _compute_beamformer, _check_src_type,
                                  Beamformer, _compute_power)


@verbose
def make_dics(info, forward, csd, reg=0.05, noise_csd=None, label=None,
              pick_ori=None, rank=None, inversion='single', weight_norm=None,
              normalize_fwd=True, real_filter=False, reduce_rank=False,
              verbose=None):
    """Compute a Dynamic Imaging of Coherent Sources (DICS) spatial filter.

    This is a beamformer filter that can be used to estimate the source power
    at a specific frequency range [1]_. It does this by constructing a spatial
    filter for each source point.  The computation of these filters is very
    similar to those of the LCMV beamformer (:func:`make_lcmv`), but instead of
    operating on a covariance matrix, the CSD matrix is used. When applying
    these filters to a CSD matrix (see :func:`apply_dics_csd`), the source
    power can be estimated for each source point.

    Parameters
    ----------
    info : instance of Info
        Measurement info, e.g. ``epochs.info``.
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
    pick_ori : None | 'normal' | 'max-power'
        The source orientation to compute the filter for:

            ``None`` :
                orientations are pooled (Default)
            'normal' :
                filters are computed for the orientation tangential to the
                cortical surface
            'max-power' :
                filters are computer for the orientation that maximizes
                spectral power.

    rank : None | int | 'full'
        This controls the effective rank of the covariance matrix when
        computing the inverse. The rank can be set explicitly by specifying an
        integer value. If ``None``, the rank will be automatically estimated.
        Since applying regularization will always make the covariance matrix
        full rank, the rank is estimated before regularization in this case. If
        'full', the rank will be estimated after regularization and hence
        will mean using the full rank, unless ``reg=0`` is used.
        The default is None.

        .. versionadded:: 0.17
    inversion : 'single' | 'matrix'
        This determines how the beamformer deals with source spaces in "free"
        orientation. Such source spaces define three orthogonal dipoles at each
        source point. When ``inversion='single'``, each dipole is considered
        as an individual source and the corresponding spatial filter is
        computed for each dipole separately. When ``inversion='matrix'``, all
        three dipoles at a source vertex are considered as a group and the
        spatial filters are computed jointly using a matrix inversion. While
        ``inversion='single'`` is more stable, ``inversion='matrix'`` is more
        precise. See section 5 of [5]_.  Defaults to 'single'.
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        If 'nai', the Neural Activity Index [4]_ will be computed.
        Defaults to ``None``, in which case no normalization is performed.
    normalize_fwd : bool
        Whether to normalize the forward solution. Defaults to ``True``. Note
        that this normalization is not required when weight normalization
        (``weight_norm``) is used.
    real_filter : bool
        If ``True``, take only the real part of the cross-spectral-density
        matrices to compute real filters. Defaults to ``False``.
    reduce_rank : bool
        If ``True``, the rank of the forward operator will be reduced by 1 for
        each spatial location, prior to inversion. This may be necessary when
        you use a single sphere model for MEG and ``mode='vertex'``.
        Defaults to ``False``.
    %(verbose)s

    Returns
    -------
    filters : instance of Beamformer
        Dictionary containing filter weights from DICS beamformer.
        Contains the following keys:

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
            'inversion' : 'single' | 'matrix'
                Whether the spatial filters were computed for each dipole
                separately or jointly for all dipoles at each vertex using a
                matrix inversion.
            'weight_norm' : None | 'unit-noise-gain'
                The normalization of the weights.
            'normalize_fwd' : bool
                Whether the forward solution was normalized
            'n_orient' : int
                Number of source orientations defined in the forward model.
            'subject' : str
                The subject ID.
            'src_type' : str
                Type of source space.

    See Also
    --------
    apply_dics_csd, tf_dics

    Notes
    -----
    The original reference is [1]_. See [5]_ for a tutorial style paper on the
    topic.

    The DICS beamformer is very similar to the LCMV (:func:`make_lcmv`)
    beamformer and many of the parameters are shared. However,
    :func:`make_dics` and :func:`make_lcmv` currently have different defaults
    for these parameters, which were settled on separately through extensive
    practical use case testing (but not necessarily exhaustive parameter space
    searching), and it remains to be seen how functionally interchangeable they
    could be.

    The default setting reproduce the DICS beamformer as described in [5]_::

        inversion='single', weight_norm=None, normalize_fwd=True

    To use the :func:`make_lcmv` defaults, use::

        inversion='matrix', weight_norm='unit-gain', normalize_fwd=False

    For more information about ``real_filter``, see the
    supplemental information from [3]_.

    References
    ----------
    .. [1] Gross et al. (2001) Dynamic imaging of coherent sources: Studying
           neural interactions in the human brain. PNAS vol. 98 (2)
           pp. 694-699. https://doi.org/10.1073/pnas.98.2.694
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    .. [3] Hipp JF, Engel AK, Siegel M (2011) Oscillatory Synchronization
           in Large-Scale Cortical Networks Predicts Perception.
           Neuron (2011) vol 69 pp. 387-396.
           https://doi.org/10.1016/j.neuron.2010.12.027
    .. [4] Van Veen et al. Localization of brain electrical activity via
           linearly constrained minimum variance spatial filtering.
           Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    .. [5] van Vliet, et al. (2018) Analysis of functional connectivity and
           oscillatory power using DICS: from raw MEG data to group-level
           statistics in Python. bioRxiv, 245530.
           https://doi.org/10.1101/245530
    """  # noqa: E501
    rank = _check_rank(rank)
    _check_option('pick_ori', pick_ori, [None, 'normal', 'max-power'])
    _check_option('inversion', inversion, ['single', 'matrix'])
    _check_option('weight_norm', weight_norm, ['unit-noise-gain', 'nai', None])

    picks = _check_info_inv(info, forward)
    info = pick_info(info, picks)

    # Leadfield rank and optional rank reduction
    # (to deal with problems with complex eigenvalues within the computation
    # of the optimal orientation when using pinv if the leadfield was only
    # rank 2 (e.g., with the spherical headmodel of the phantom data),
    # see gh-4568 and gh-4628.
    if reduce_rank and not (pick_ori == 'max-power' and inversion == 'matrix'):
        raise NotImplementedError(
            'The computation of spatial filters with rank reduction using '
            'reduce_rank=True is only implemented with pick_ori=="max-power" '
            'and inversion="matrix".'
        )

    frequencies = [np.mean(freq_bin) for freq_bin in csd.frequencies]
    n_freqs = len(frequencies)
    n_orient = forward['sol']['ncol'] // forward['nsource']

    # Determine how to normalize the leadfield
    if normalize_fwd:
        if inversion == 'single':
            if weight_norm == 'unit-noise-gain':
                raise ValueError('The computation of a unit-noise-gain '
                                 'beamformer with inversion="single" is not '
                                 'stable with depth normalization, set  '
                                 'normalize_fwd to False.')
            combine_xyz = False
        else:
            combine_xyz = 'fro'
        exp = 1.  # turn on depth weighting with exponent 1
    else:
        exp = None  # turn off depth weighting entirely
        combine_xyz = False

    _check_one_ch_type('dics', info, forward, csd, noise_csd)

    # pick info, get gain matrix, etc.
    subject = _subject_from_forward(forward)
    ch_names = list(info['ch_names'])

    csd_picks = [csd.ch_names.index(ch) for ch in ch_names]

    if noise_csd is not None:
        if len(noise_csd.frequencies) > 1:
            noise_csd = noise_csd.mean()
        noise_csd = noise_csd.get_data(as_cov=True)
        if real_filter:
            noise_csd['data'] = noise_csd['data'].real

    _, _, proj, vertices, G, whitener, nn, orient_std = \
        _prepare_beamformer_input(
            info, forward, label, pick_ori, noise_csd,
            combine_xyz=combine_xyz, exp=exp)

    logger.info('Computing DICS spatial filters...')
    Ws = []
    whiteners = []
    for i, freq in enumerate(frequencies):
        if n_freqs > 1:
            logger.info('    computing DICS spatial filter at %sHz (%d/%d)' %
                        (freq, i + 1, n_freqs))

        Cm = csd.get_data(index=i)
        if real_filter:
            Cm = Cm.real

        # Ensure the CSD is in the same order as the leadfield
        Cm = Cm[csd_picks, :][:, csd_picks]

        # Whiten the CSD
        Cm = np.dot(whitener, np.dot(Cm, whitener.conj().T))

        # compute spatial filter
        W = _compute_beamformer(G, Cm, reg, n_orient, weight_norm, pick_ori,
                                reduce_rank, rank=rank, inversion=inversion,
                                nn=nn, orient_std=orient_std)
        Ws.append(W)
        whiteners.append(whitener)

    Ws = np.array(Ws)
    whiteners = np.array(whiteners)

    src_type = _get_src_type(forward['src'], vertices)
    filters = Beamformer(
        kind='DICS', weights=Ws, csd=csd, ch_names=ch_names, proj=proj,
        vertices=vertices, subject=subject, pick_ori=pick_ori,
        inversion=inversion, weight_norm=weight_norm,
        normalize_fwd=bool(normalize_fwd), src_type=src_type,
        n_orient=n_orient if pick_ori is None else 1, whiteners=whiteners)

    return filters


def _apply_dics(data, filters, info, tmin):
    """Apply DICS spatial filter to data for source reconstruction."""
    if isinstance(data, np.ndarray) and data.ndim == 2:
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
        if info['projs']:
            _check_proj_match(info, filters)
            M = np.dot(filters['proj'], M)

        stcs = []
        for W in Ws:
            # project to source space using beamformer weights
            sol = np.dot(W, M)

            if filters['n_orient'] > 1:
                logger.info('combining the current components...')
                sol = combine_xyz(sol)

            tstep = 1.0 / info['sfreq']

            stcs.append(_make_stc(sol, vertices=filters['vertices'],
                                  src_type=filters['src_type'], tmin=tmin,
                                  tstep=tstep, subject=subject,
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
def apply_dics_csd(csd, filters, verbose=None):
    """Apply Dynamic Imaging of Coherent Sources (DICS) beamformer weights.

    Apply a previously computed DICS beamformer to a cross-spectral density
    (CSD) object to estimate source power in time and frequency windows
    specified in the CSD object [1]_.

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

    References
    ----------
    .. [1] Gross et al. Dynamic imaging of coherent sources: Studying neural
           interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """  # noqa: E501
    ch_names = filters['ch_names']
    vertices = filters['vertices']
    n_orient = filters['n_orient']
    subject = filters['subject']
    n_sources = np.sum([len(v) for v in vertices])

    # If CSD is summed over multiple frequencies, take the average frequency
    frequencies = [np.mean(dfreq) for dfreq in csd.frequencies]
    n_freqs = len(frequencies)

    source_power = np.zeros((n_sources, len(csd.frequencies)))

    # Ensure the CSD is in the same order as the weights
    csd_picks = [csd.ch_names.index(ch) for ch in ch_names]

    logger.info('Computing DICS source power...')
    for i, freq in enumerate(frequencies):
        if n_freqs > 1:
            logger.info('    applying DICS spatial filter at %sHz (%d/%d)' %
                        (freq, i + 1, n_freqs))

        Cm = csd.get_data(index=i)
        Cm = Cm[csd_picks, :][:, csd_picks]
        W = filters['weights'][i]

        # Whiten the CSD
        whitener = filters['whiteners'][i]
        Cm = np.dot(whitener, np.dot(Cm, whitener.conj().T))

        source_power[:, i] = _compute_power(Cm, W, n_orient)

    logger.info('[done]')

    # compatibility with 0.16, add src_type as None if not present:
    filters, warn_text = _check_src_type(filters)

    return (_make_stc(source_power, vertices=vertices,
                      src_type=filters['src_type'], tmin=0., tstep=1.,
                      subject=subject, warn_text=warn_text),
            frequencies)


@verbose
def tf_dics(epochs, forward, noise_csds, tmin, tmax, tstep, win_lengths,
            subtract_evoked=False, mode='fourier', freq_bins=None,
            frequencies=None, n_ffts=None, mt_bandwidths=None,
            mt_adaptive=False, mt_low_bias=True, cwt_n_cycles=7, decim=1,
            reg=0.05, label=None, pick_ori=None, rank=None, inversion='single',
            weight_norm=None, normalize_fwd=True, real_filter=False,
            reduce_rank=False, verbose=None):
    """5D time-frequency beamforming based on DICS.

    Calculate source power in time-frequency windows using a spatial filter
    based on the Dynamic Imaging of Coherent Sources (DICS) beamforming
    approach [1]_. For each time window and frequency bin combination,
    cross-spectral density (CSD) is computed and used to create a DICS
    beamformer spatial filter.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_csds : list of instances of CrossSpectralDensity | None
        Noise cross-spectral density for each frequency bin. If these are
        specified, the DICS filters will be applied to both the signal and
        noise CSDs. The source power estimates for each frequency bin will be
        scaled by the estimated noise power (signal / noise).
        Specifying ``None`` will disable performing noise normalization.
    tmin : float
        Minimum time instant to consider.
    tmax : float
        Maximum time instant to consider.
    tstep : float
        Spacing between consecutive time windows, should be smaller than or
        equal to the shortest time window length.
    win_lengths : list of float
        Time window lengths in seconds. One time window length should be
        provided for each frequency bin.
    subtract_evoked : bool
        If True, subtract the averaged evoked response prior to computing the
        tf source grid. Defaults to False.
    mode : 'fourier' | 'multitaper' | 'cwt_morlet'
        Spectrum estimation mode. Defaults to 'fourier'.
    freq_bins : list of tuple of float
        Start and end point of frequency bins of interest.
        Only used in 'multitaper' or 'fourier' mode. For 'cwt_morlet' mode, use
        the ``frequencies`` parameter instead.
    frequencies : list of float | list of list of float
        The frequencies to compute the source power for. If you want to compute
        the average power for multiple frequency bins, specify a list of
        lists: each list containing the frequencies for the corresponding bin.
        Only used in 'cwt_morlet' mode. In other modes, use the ``freq_bins``
        parameter instead.
    n_ffts : list | None
        Length of the FFT for each frequency bin. If ``None`` (the default),
        the exact number of samples between ``tmin`` and ``tmax`` will be used.
        Only used in 'multitaper' or 'fourier' mode.
    mt_bandwidths : list of float
        The bandwidths of the multitaper windowing function in Hz. Only used in
        'multitaper' mode. One value should be provided for each frequency bin.
        Defaults to None.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into CSD. Only used
        in 'multitaper' mode. Defaults to False.
    mt_low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth. Only used in 'multitaper' mode. Defaults to True.
    cwt_n_cycles : float | list of float | None
        Number of cycles to use when constructing Morlet wavelets. Fixed number
        or one per frequency. Defaults to 7.
        Only used in 'cwt_morlet' mode.
    decim : int | slice
        To reduce memory usage, decimation factor during time-frequency
        decomposition. Defaults to 1 (no decimation).
        Only used in 'cwt_morlet' mode.

        If `int`, uses tfr[..., ::decim].
        If `slice`, uses tfr[..., decim].
    reg : float
        Regularization to use for the DICS beamformer computation.
        Defaults to 0.05.
    label : Label | None
        Restricts the solution to a given label. Defaults to None.
    pick_ori : None | 'normal' | 'max-power'
        The source orientation to estimate source power for:

            ``None`` :
                orientations are pooled. (Default)
            'normal' :
                filters are computed for the orientation tangential to the
                cortical surface
            'max-power' :
                filters are computer for the orientation that maximizes
                spectral power.

        Defaults to ``None``.

    rank : None | int | 'full'
        This controls the effective rank of the covariance matrix when
        computing the inverse. The rank can be set explicitly by specifying an
        integer value. If ``None``, the rank will be automatically estimated.
        Since applying regularization will always make the covariance matrix
        full rank, the rank is estimated before regularization in this case. If
        'full', the rank will be estimated after regularization and hence
        will mean using the full rank, unless ``reg=0`` is used.
        The default is None.

        .. versionadded:: 0.17
    inversion : 'single' | 'matrix'
        This determines how the beamformer deals with source spaces in "free"
        orientation. Such source spaces define three orthogonal dipoles at each
        source point. When ``inversion='single'``, each dipole is considered
        as an individual source and the corresponding spatial filter is
        computed for each dipole separately. When ``inversion='matrix'``, all
        three dipoles at a source vertex are considered as a group and the
        spatial filters are computed jointly using a matrix inversion. While
        ``inversion='single'`` is more stable, ``inversion='matrix'`` is more
        precise. See Notes of :func:`make_dics`.  Defaults to 'single'.
    weight_norm : None | 'unit-noise-gain'
        How to normalize the beamformer weights. None means no normalization is
        performed.  If 'unit-noise-gain', the unit-noise gain minimum variance
        beamformer will be computed (Borgiotti-Kaplan beamformer) [2]_.
        Defaults to ``None``.
    normalize_fwd : bool
        Whether to normalize the forward solution. Defaults to ``True``. Note
        that this normalization is not required when weight normalization
        (``weight_norm``) is used.
    real_filter : bool
        If ``True``, take only the real part of the cross-spectral-density
        matrices to compute real filters. Defaults to ``False``.
    reduce_rank : bool
        If ``True``, the rank of the forward operator will be reduced by 1 for
        each spatial location, prior to inversion. This may be necessary when
        you use a single sphere model for MEG and ``mode='vertex'``.
        Defaults to ``False``.
    %(verbose)s

    Returns
    -------
    stcs : list of SourceEstimate | VolSourceEstimate
        Source power at each time window. One SourceEstimate object is returned
        for each frequency bin.

    Notes
    -----
    Dalal et al. [1]_ used a synthetic aperture magnetometry beamformer (SAM)
    in each time-frequency window instead of DICS.

    An alternative to using noise CSDs is to normalize the forward solution
    (``normalize_fwd``) or the beamformer weights (``weight_norm``). In
    this case, ``noise_csds`` may be set to ``None``.

    References
    ----------
    .. [1] Dalal et al. Five-dimensional neuroimaging: Localization of the
           time-frequency dynamics of cortical activity.
           NeuroImage (2008) vol. 40 (4) pp. 1686-1700
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    _check_reference(epochs)
    rank = _check_rank(rank)

    if mode == 'cwt_morlet' and frequencies is None:
        raise ValueError('In "cwt_morlet" mode, the "frequencies" parameter '
                         'should be used.')
    elif mode != 'cwt_morlet' and freq_bins is None:
        raise ValueError('In "%s" mode, the "freq_bins" parameter should be '
                         'used.' % mode)

    if frequencies is not None:
        # Make sure frequencies are always in the form of a list of lists
        frequencies = [np.atleast_1d(f) for f in frequencies]
        n_freq_bins = len(frequencies)
    else:
        n_freq_bins = len(freq_bins)

    if len(win_lengths) != n_freq_bins:
        raise ValueError('One time window length expected per frequency bin')
    if any(win_length < tstep for win_length in win_lengths):
        raise ValueError('Time step should not be larger than any of the '
                         'window lengths')

    if noise_csds is not None and len(noise_csds) != n_freq_bins:
        raise ValueError('One noise CSD object expected per frequency bin')

    if n_ffts is not None and len(n_ffts) != n_freq_bins:
        raise ValueError('When specifying number of FFT samples, one value '
                         'must be provided per frequency bin')
    if mt_bandwidths is not None and len(mt_bandwidths) != n_freq_bins:
        raise ValueError('When using multitaper mode and specifying '
                         'multitaper transform bandwidth, one value must be '
                         'provided per frequency bin')

    # Multiplying by 1e3 to avoid numerical issues, e.g. 0.3 // 0.05 == 5
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))

    # Subtract evoked response
    if subtract_evoked:
        epochs = epochs.copy().subtract_evoked()

    sol_final = []

    # Compute source power for each frequency bin
    for i_freq in range(n_freq_bins):
        win_length = win_lengths[i_freq]
        n_overlap = int((win_length * 1e3) // (tstep * 1e3))

        # Scale noise CSD to allow data and noise CSDs to have different length
        if noise_csds is not None:
            noise_csd = noise_csds[i_freq].copy()
            noise_csd._data /= noise_csd.n_fft

        if mode == 'cwt_morlet':
            freq_bin = frequencies[i_freq]
            fmin = np.min(freq_bin)
            fmax = np.max(freq_bin)
        else:
            fmin, fmax = freq_bins[i_freq]
            if n_ffts is None:
                n_fft = None
            else:
                n_fft = n_ffts[i_freq]
            if mt_bandwidths is None:
                mt_bandwidth = None
            else:
                mt_bandwidth = mt_bandwidths[i_freq]

        sol_single = []
        sol_overlap = []
        for i_time in range(n_time_steps):
            win_tmin = tmin + i_time * tstep
            win_tmax = win_tmin + win_length

            # If in the last step the last time point was not covered in
            # previous steps and will not be covered now, a solution needs to
            # be calculated for an additional time window
            if (i_time == n_time_steps - 1 and
                    win_tmax - tstep < tmax and
                    win_tmax >= tmax + (epochs.times[-1] - epochs.times[-2])):
                warn('Adding a time window to cover last time points')
                win_tmin = tmax - win_length
                win_tmax = tmax

            if win_tmax < tmax + (epochs.times[-1] - epochs.times[-2]):
                # Counteracts unsafe floating point arithmetic ensuring all
                # relevant samples will be taken into account when selecting
                # data in time windows
                logger.info(
                    'Computing time-frequency DICS beamformer for time '
                    'window %d to %d ms, in frequency range %d to %d Hz' %
                    (win_tmin * 1e3, win_tmax * 1e3, fmin, fmax)
                )

                # Calculating data CSD in current time window
                if mode == 'fourier':
                    csd = csd_fourier(
                        epochs, fmin=fmin, fmax=fmax, tmin=win_tmin,
                        tmax=win_tmax, n_fft=n_fft, verbose=False)
                elif mode == 'multitaper':
                    csd = csd_multitaper(
                        epochs, fmin=fmin, fmax=fmax, tmin=win_tmin,
                        tmax=win_tmax, n_fft=n_fft, bandwidth=mt_bandwidth,
                        low_bias=mt_low_bias, verbose=False)
                elif mode == 'cwt_morlet':
                    csd = csd_morlet(
                        epochs, frequencies=freq_bin, tmin=win_tmin,
                        tmax=win_tmax, n_cycles=cwt_n_cycles, decim=decim,
                        verbose=False)
                else:
                    raise ValueError('Invalid mode, choose either '
                                     "'fourier' or 'multitaper'")

                csd = csd.sum()

                # Scale data CSD to allow data and noise CSDs to have different
                # length
                csd._data /= csd.n_fft

                filters = make_dics(epochs.info, forward, csd, reg=reg,
                                    label=label, pick_ori=pick_ori,
                                    rank=rank, inversion=inversion,
                                    weight_norm=weight_norm,
                                    normalize_fwd=normalize_fwd,
                                    reduce_rank=reduce_rank,
                                    real_filter=real_filter, verbose=False)
                stc, _ = apply_dics_csd(csd, filters, verbose=False)

                if noise_csds is not None:
                    # Scale signal power by noise power
                    noise_stc, _ = apply_dics_csd(noise_csd, filters,
                                                  verbose=False)
                    stc /= noise_stc

                sol_single.append(stc.data[:, 0])

            # Average over all time windows that contain the current time
            # point, which is the current time window along with
            # n_overlap - 1 previous ones
            if i_time - n_overlap < 0:
                curr_sol = np.mean(sol_single[0:i_time + 1], axis=0)
            else:
                curr_sol = np.mean(sol_single[i_time - n_overlap + 1:
                                              i_time + 1], axis=0)

            # The final result for the current time point in the current
            # frequency bin
            sol_overlap.append(curr_sol)

        # Gathering solutions for all time points for current frequency bin
        sol_final.append(sol_overlap)

    sol_final = np.array(sol_final)

    # Creating stc objects containing all time points for each frequency bin
    stcs = []
    # compatibility with 0.16, add src_type as None if not present:
    filters, warn_text = _check_src_type(filters)

    for i_freq in range(n_freq_bins):
        stc = _make_stc(sol_final[i_freq, :, :].T, vertices=stc.vertices,
                        src_type=filters['src_type'], tmin=tmin, tstep=tstep,
                        subject=stc.subject, warn_text=warn_text)
        stcs.append(stc)

    return stcs
