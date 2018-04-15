"""Dynamic Imaging of Coherent Sources (DICS)."""

# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Britta Westner
#          Susanna Aro <susanna.aro@aalto.fi>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..utils import logger, verbose, warn, deprecated
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import combine_xyz, _check_reference
from ..source_estimate import _make_stc
from ..time_frequency import csd_fourier, csd_multitaper, csd_morlet
from ._compute_beamformer import (_reg_pinv, _eig_inv, _setup_picks,
                                  _pick_channels_spatial_filter,
                                  _check_proj_match, _prepare_beamformer_input)
from ..externals import six


deprecation_text = ('This function will be removed in 0.17, please use the '
                    '`make_dics` and `apply_dics_*` functions instead. These '
                    'functions use a new implementation of DICS that has been '
                    'more thoroughly tested.')


@verbose
def make_dics(info, forward, csd, reg=0.05, label=None, pick_ori=None,
              inversion='single', weight_norm=None, normalize_fwd=True,
              real_filter=False, reduce_rank=False, verbose=None):
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

    inversion : 'single' | 'matrix'
        This determines how the beamformer deals with source spaces in "free"
        orientation. Such source spaces define three orthogonal dipoles at each
        source point. When ``inversion='single'``, each dipole is considered
        as an individual source and the corresponding spatial filter is
        computed for each dipole separately. When ``inversion='matrix'``, all
        three dipoles at a source vertex are considered as a group and the
        spatial filters are computed jointly using a matrix inversion. While
        ``inversion='single'`` is more stable, ``inversion='matrix'`` is more
        precise. See section 5 of [4]_.  Defaults to 'single'.
    weight_norm : None | 'unit-noise-gain'
        How to normalize the beamformer weights. None means no normalization is
        performed. If 'unit-noise-gain', the unit-noise gain minimum variance
        beamformer will be computed (Borgiotti-Kaplan beamformer) [2]_.
        Defaults to ``None''.
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
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    filters : dict
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

    See Also
    --------
    apply_dics_csd, tf_dics

    Notes
    -----
    The original reference is [1]_. See [4]_ for a tutorial style paper on the
    topic.

    The DICS beamformer is very similar to the LCMV (:func:`make_lcmv`)
    beamformer and many of the parameters are shared. However,
    :func:`make_dics` and :func:`make_lcmv` currently have different defaults
    for these parameters, which were settled on separately through extensive
    practical use case testing (but not necessarily exhaustive parameter space
    searching), and it remains to be seen how functionally interchangeable they
    could be.

    The default setting reproduce the DICS beamformer as described in [4]_::

        inversion='single', weight_norm=None, normalize_fwd=True

    To use the :func:`make_lcmv` defaults, use::

        inversion='matrix', weight_norm='unit-gain', normalize_fwd=False

    For more information about ``real_filter``, see the
    `supplemental information <http://www.cell.com/cms/attachment/616681/4982593/mmc1.pdf>`_
    from [3]_.

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
    .. [4] van Vliet, et al. (2018) Analysis of functional connectivity and
           oscillatory power using DICS: from raw MEG data to group-level
           statistics in Python. bioRxiv, 245530.
           https://doi.org/10.1101/245530
    """  # noqa: E501
    allowed_ori = [None, 'normal', 'max-power']
    if pick_ori not in allowed_ori:
        raise ValueError('"pick_ori" should be one of %s.' % allowed_ori)

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

    # Inversion mode
    if inversion not in ['single', 'matrix']:
        raise ValueError("The inversion parameter should be either 'single' "
                         "or 'matrix'.")

    frequencies = [np.mean(freq_bin) for freq_bin in csd.frequencies]
    n_freqs = len(frequencies)
    n_orient = forward['sol']['ncol'] // forward['nsource']

    # Determine how to normalize the leadfield
    if normalize_fwd:
        if inversion == 'single':
            fwd_norm = 'dipole'
        else:
            fwd_norm = 'vertex'
    else:
        fwd_norm = None  # No normalization

    picks = _setup_picks(info=info, forward=forward)
    _, ch_names, proj, vertices, G = _prepare_beamformer_input(
        info, forward, label, picks=picks, pick_ori=pick_ori,
        fwd_norm=fwd_norm,
    )
    csd_picks = [csd.ch_names.index(ch) for ch in ch_names]

    n_sources = G.shape[1] // n_orient

    logger.info('Computing DICS spatial filters...')
    Ws = []
    for i, freq in enumerate(frequencies):
        if n_freqs > 1:
            logger.info('    computing DICS spatial filter at %sHz (%d/%d)' %
                        (freq, i + 1, n_freqs))

        Cm = csd.get_data(index=i)

        if real_filter:
            Cm = Cm.real

        # Ensure the CSD is in the same order as the leadfield
        Cm = Cm[csd_picks, :][:, csd_picks]

        # Tikhonov regularization using reg parameter to control for
        # trade-off between spatial resolution and noise sensitivity
        # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
        Cm_inv, _ = _reg_pinv(Cm, reg, rcond='auto')

        if (pick_ori == 'max-power' and weight_norm == 'unit-noise-gain' and
                inversion == 'matrix'):
            Cm_inv_sq = Cm_inv.dot(Cm_inv)

        # Compute spatial filters
        W = np.dot(G.T, Cm_inv)

        for k in range(n_sources):
            Wk = W[n_orient * k: n_orient * k + n_orient]
            Gk = G[:, n_orient * k: n_orient * k + n_orient]

            # Compute power at the source
            Ck = np.dot(Wk, Gk)

            # XXX This should be de-duplicated with LCMV

            # Normalize the spatial filters
            if Wk.ndim == 2 and len(Wk) > 1:
                # Free source orientation
                if inversion == 'single':
                    # Invert for each dipole separately using plain division
                    Wk /= np.diag(Ck)[:, np.newaxis]
                elif inversion == 'matrix':
                    # Invert for all dipoles simultaneously using matrix
                    # inversion.
                    Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
            else:
                # Fixed source orientation
                Wk /= Ck

            if pick_ori == 'max-power':
                # Compute spectral power, so we can pick the orientation that
                # maximizes this power.
                if inversion == 'matrix' and weight_norm == 'unit-noise-gain':
                    # Use Eq. 4.47 from [2]_
                    norm_inv = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
                    if reduce_rank:
                        # Use pseudo inverse computation setting smallest
                        # component to zero if the leadfield is not full rank
                        norm = _eig_inv(norm_inv, norm_inv.shape[0] - 1)
                    else:
                        # Use straight inverse with full rank leadfield
                        try:
                            norm = linalg.inv(norm_inv)
                        except np.linalg.linalg.LinAlgError:
                            raise ValueError(
                                'Singular matrix detected when estimating '
                                'DICS filters. Consider reducing the rank '
                                'of the forward operator by using '
                                'reduce_rank=True.'
                            )
                    power = np.dot(norm, np.dot(Wk, Gk))

                elif (inversion == 'single' and
                      weight_norm == 'unit-noise-gain'):
                    # First make the filters unit gain, then apply them to the
                    # CSD matrix to compute power.
                    norm = 1 / np.sqrt(np.sum(Wk ** 2, axis=1))
                    Wk_norm = Wk / norm[:, np.newaxis]
                    power = Wk_norm.dot(Cm).dot(Wk_norm.T)
                else:
                    # Compute spectral power by applying the spatial filters to
                    # the CSD matrix.
                    power = Wk.dot(Cm).dot(Wk.T)

                # Compute the direction of max power
                u, s, _ = np.linalg.svd(power.real)
                max_power_ori = u[:, 0]

                # Re-compute the filter in the direction of max power
                Wk[:] = max_power_ori.dot(Wk)

        if pick_ori == 'normal':
            W = W[2::3]
        elif pick_ori == 'max-power':
            W = W[0::3]

        if weight_norm == 'unit-noise-gain':
            # Scale weights so that W @ I @ W.T == I
            if pick_ori is None and n_orient > 1:
                # Compute the norm for each set of 3 dipoles
                W = W.reshape(-1, 3, W.shape[1])
                norm = np.sqrt(np.sum(W ** 2, axis=(1, 2)))
                W /= norm[:, np.newaxis, np.newaxis]
                W = W.reshape(-1, W.shape[2])
            else:
                # Compute the norm for each dipole
                norm = np.sqrt(np.sum(W ** 2, axis=1))
                W /= norm[:, np.newaxis]

        Ws.append(W)

    Ws = np.array(Ws)

    subject = _subject_from_forward(forward)
    filters = dict(weights=Ws, csd=csd, ch_names=ch_names, proj=proj,
                   vertices=vertices, subject=subject,
                   pick_ori=pick_ori, inversion=inversion,
                   weight_norm=weight_norm,
                   normalize_fwd=normalize_fwd, n_orient=n_orient
                   if pick_ori is None else 1)

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

            stcs.append(_make_stc(sol, vertices=filters['vertices'], tmin=tmin,
                                  tstep=tstep, subject=subject))
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
    filters : dict
        DICS spatial filter (beamformer weights)
        Filter weights returned from :func:`make_dics`.
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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

    sel = _pick_channels_spatial_filter(evoked.ch_names, filters)
    data = data[sel]

    stc = _apply_dics(data=data, filters=filters, info=info, tmin=tmin)

    return six.advance_iterator(stc)


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
    filters : dict
        DICS spatial filter (beamformer weights)
        Filter weights returned from :func:`make_dics`. The DICS filters must
        have been computed for a single frequency only.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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

    sel = _pick_channels_spatial_filter(epochs.ch_names, filters)
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
    filters : dict
        DICS spatial filter (beamformer weights)
        Filter weights returned from `make_dics`.
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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

        for k in range(n_sources):
            Wk = W[n_orient * k: n_orient * k + n_orient]
            power = Wk.dot(Cm).dot(Wk.T)

            if n_orient > 1:  # Pool the orientations
                source_power[k, i] = np.abs(power.trace())
            else:
                source_power[k, i] = np.abs(power)

    logger.info('[done]')

    return (_make_stc(source_power.reshape(-1, n_freqs), vertices=vertices,
                      tmin=0, tstep=1, subject=subject), frequencies)


def _apply_old_dics(data, info, tmin, forward, noise_csd, data_csd, reg,
                    label=None, picks=None, pick_ori=None, real_filter=False,
                    verbose=None):
    """Old implementation of Dynamic Imaging of Coherent Sources (DICS)."""
    from scipy import linalg  # Local import to keep 'import mne' fast

    if len(noise_csd.frequencies) > 1 or len(data_csd.frequencies) > 1:
        raise ValueError('CSD matrix object should only contain one '
                         'frequency.')

    is_free_ori, _, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    Cm = data_csd.get_data(index=0)
    Cm_noise = noise_csd.get_data(index=0)

    # Take real part of Cm to compute real filters
    if real_filter:
        Cm = Cm.real

    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, _ = _reg_pinv(Cm, reg)
    del Cm

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient

    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        # Noise normalization
        noise_norm = np.dot(np.dot(Wk.conj(), Cm_noise), Wk.T)
        noise_norm = np.abs(noise_norm).trace()
        Wk /= np.sqrt(noise_norm)

    # Pick source orientation normal to cortical surface
    if pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    subject = _subject_from_forward(forward)
    for i, M in enumerate(data):
        if len(M) != len(picks):
            raise ValueError('data and picks must have the same length')

        if not return_single:
            logger.info("Processing epoch : %d" % (i + 1))

        # Apply SSPs
        if info['projs']:
            M = np.dot(proj, M)

        # project to source space using beamformer weights
        if is_free_ori:
            sol = np.dot(W, M)
            logger.info('combining the current components...')
            sol = combine_xyz(sol)
        else:
            # Linear inverse: do not delay compuation due to non-linear abs
            sol = np.dot(W, M)

        tstep = 1.0 / info['sfreq']
        if np.iscomplexobj(sol):
            sol = np.abs(sol)  # XXX : STC cannot contain (yet?) complex values
        yield _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                        subject=subject)

    logger.info('[done]')


@deprecated(deprecation_text)
@verbose
def dics(evoked, forward, noise_csd, data_csd, reg=0.05, label=None,
         pick_ori=None, real_filter=False, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Compute a Dynamic Imaging of Coherent Sources (DICS) [1]_ beamformer
    on evoked data and return estimates of source time courses.

    .. note:: Fixed orientation forward operators with ``real_filter=False``
              will result in complex time courses, in which case absolute
              values will be returned.

    .. warning:: This implementation has not been heavily tested and may be
                 incorrect.

    Parameters
    ----------
    evoked : Evoked
        Evoked data.
    forward : dict
        Forward operator.
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    real_filter : bool
        If True, take only the real part of the cross-spectral-density matrices
        to compute real filters as in [2]_. Default is False.
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses

    See Also
    --------
    dics_epochs

    Notes
    -----
    For more information about ``real_filter``, see the
    `supplemental information <http://www.cell.com/cms/attachment/616681/4982593/mmc1.pdf>`_
    from [2]_.

    References
    ----------
    .. [1] Gross et al. Dynamic imaging of coherent sources: Studying neural
           interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    .. [2] Hipp JF, Engel AK, Siegel M (2011) Oscillatory Synchronization
           in Large-Scale Cortical Networks Predicts Perception.
           Neuron 69:387-396.
    """  # noqa: E501
    _check_reference(evoked)
    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    picks = _setup_picks(info=info, forward=forward)
    data = data[picks]

    stc = _apply_old_dics(data, info, tmin, forward, noise_csd, data_csd,
                          reg=reg, label=label, pick_ori=pick_ori, picks=picks,
                          real_filter=real_filter)
    return six.advance_iterator(stc)


@deprecated(deprecation_text)
@verbose
def dics_epochs(epochs, forward, noise_csd, data_csd, reg=0.05, label=None,
                pick_ori=None, return_generator=False, real_filter=False,
                verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Compute a Dynamic Imaging of Coherent Sources (DICS) beamformer
    on single trial data and return estimates of source time courses.

    .. note:: Fixed orientation forward operators with ``real_filter=False``
              will result in complex time courses, in which case absolute
              values will be returned.

    .. warning:: This implementation has not been heavily tested and may be
                 incorrect.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    real_filter : bool
        If True, take only the real part of the cross-spectral-density matrices
        to compute real filters as in [1]_. Default is False.
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc: list | generator of SourceEstimate | VolSourceEstimate
        The source estimates for all epochs

    See Also
    --------
    dics

    References
    ----------
    .. [1] Hipp JF, Engel AK, Siegel M (2011) Oscillatory Synchronization
           in Large-Scale Cortical Networks Predicts Perception.
           Neuron 69:387-396.
    """
    _check_reference(epochs)

    info = epochs.info
    tmin = epochs.times[0]

    picks = _setup_picks(info=info, forward=forward)
    data = epochs.get_data()[:, picks, :]

    stcs = _apply_old_dics(data, info, tmin, forward, noise_csd, data_csd,
                           reg=reg, label=label, pick_ori=pick_ori,
                           picks=picks, real_filter=real_filter)

    if not return_generator:
        stcs = list(stcs)

    return stcs


@deprecated(deprecation_text)
@verbose
def dics_source_power(info, forward, noise_csds, data_csds, reg=0.05,
                      label=None, pick_ori=None, real_filter=False,
                      verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Calculate source power in time and frequency windows specified in the
    calculation of the data cross-spectral density matrix or matrices. Source
    power is normalized by noise power.

    .. warning:: This implementation has not been heavily tested and may be
                 incorrect.

    Parameters
    ----------
    info : instance of Info
        Measurement info, e.g. ``epochs.info``.
    forward : instance of Forward
        Forward operator.
    noise_csds : CrossSpectralDensity
        The noise cross-spectral density matrices for a single frequency or
        multiple frequencies.
    data_csds : CrossSpectralDensity
        The data cross-spectral density matrix for a frequency or multiple
        frequencies.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    real_filter : bool
        If True, take only the real part of the cross-spectral-density matrices
        to compute real filters.
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source power with frequency instead of time.

    Notes
    -----
    The original reference is:
    Gross et al. Dynamic imaging of coherent sources: Studying neural
    interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """
    from scipy import linalg
    if data_csds.n_channels != noise_csds.n_channels:
        raise ValueError('One noise CSD matrix should be provided for each '
                         'data CSD matrix and vice versa. All CSD matrices '
                         'should have identical shape.')

    if not np.allclose(data_csds.frequencies, noise_csds.frequencies):
        raise ValueError('Data and noise CSDs should be calculated at '
                         'identical frequencies')

    # If CSD is summed over multiple frequencies, take the average frequency
    frequencies = [np.mean(dfreq) for dfreq in data_csds.frequencies]
    n_freqs = len(frequencies)
    fmin = frequencies[0]

    if n_freqs > 2:
        fstep = np.diff(frequencies)
        if np.var(fstep) > 1e-5:
            warn('Uneven frequency spacing in CSD object, frequencies in the '
                 'resulting stc file will be inaccurate.')
        fstep = fstep[0]
    elif n_freqs > 1:
        fstep = frequencies[1] - frequencies[0]
    else:
        fstep = 1  # dummy value

    picks = _setup_picks(info=info, forward=forward)

    is_free_ori, _, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks=picks,
                                  pick_ori=pick_ori)

    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    source_power = np.zeros((n_sources, n_freqs))

    logger.info('Computing DICS source power...')
    for i in range(n_freqs):
        if n_freqs > 1:
            logger.info('    computing DICS spatial filter %d out of %d' %
                        (i + 1, n_freqs))

        Cm = data_csds.get_data(index=i)

        # Take real part of Cm to compute real filters
        if real_filter:
            Cm = Cm.real

        # Tikhonov regularization using reg parameter to control for
        # trade-off between spatial resolution and noise sensitivity
        # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
        Cm_inv, _ = _reg_pinv(Cm, reg)
        del Cm

        # Compute spatial filters
        W = np.dot(G.T, Cm_inv)

        # Make new copies of the CSDs that are not converted to real values
        data_Cm = data_csds.get_data(index=i)
        noise_Cm = noise_csds.get_data(index=i)

        # Apply spatial filters to CSDs
        for k in range(n_sources):
            Wk = W[n_orient * k: n_orient * k + n_orient]
            Gk = G[:, n_orient * k: n_orient * k + n_orient]
            Ck = np.dot(Wk, Gk)

            if is_free_ori:
                # Free source orientation
                Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
            else:
                # Fixed source orientation
                Wk /= Ck

            # Noise normalization
            noise_norm = np.dot(np.dot(Wk.conj(), noise_Cm), Wk.T)
            noise_norm = np.abs(noise_norm).trace()

            # Calculating source power
            sp_temp = np.dot(np.dot(Wk.conj(), data_Cm), Wk.T)
            sp_temp /= max(noise_norm, 1e-40)  # Avoid division by 0

            if pick_ori == 'normal':
                source_power[k, i] = np.abs(sp_temp)[2, 2]
            else:
                source_power[k, i] = np.abs(sp_temp).trace()

    logger.info('[done]')

    subject = _subject_from_forward(forward)
    return _make_stc(source_power, vertices=vertno, tmin=fmin / 1000.,
                     tstep=fstep / 1000., subject=subject)


@verbose
def tf_dics(epochs, forward, noise_csds, tmin, tmax, tstep, win_lengths,
            subtract_evoked=False, mode='fourier', freq_bins=None,
            frequencies=None, n_ffts=None, mt_bandwidths=None,
            mt_adaptive=False, mt_low_bias=True, cwt_n_cycles=7, decim=1,
            reg=0.05, label=None, pick_ori=None, inversion='single',
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
    freq_bins : list of tuples of float
        Start and end point of frequency bins of interest.
        Only used in 'multitaper' or 'fourier' mode. For 'cwt_morlet' mode, use
        the ``frequencies`` parameter instead.
    frequencies : list of float | list of lists of float
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
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode. Defaults to True.
    cwt_n_cycles: float | list of float | None
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
    inversion : 'single' | 'matrix'
        This determines how the beamformer deals with source spaces in "free"
        orientation. Such source spaces define three orthogonal dipoles at each
        source point. When ``inversion='single'``, each dipole is considered
        as an individual source and the corresponding spatial filter is
        computed for each dipole separately. When ``inversion='matrix'``, all
        three dipoles at a source vertex are considered as a group and the
        spatial filters are computed jointly using a matrix inversion. While
        ``inversion='single'`` is more stable, ``inversion='matrix'`` is more
        precise. See section 5 of [4]_.  Defaults to 'single'.
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
    verbose : bool, str, int, None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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
                win_tmin = win_tmin
                win_tmax = win_tmax

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
                                    inversion=inversion,
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
    for i_freq in range(n_freq_bins):
        stc = _make_stc(sol_final[i_freq, :, :].T, vertices=stc.vertices,
                        tmin=tmin, tstep=tstep, subject=stc.subject)
        stcs.append(stc)

    return stcs
