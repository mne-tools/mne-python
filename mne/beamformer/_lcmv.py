"""Compute Linearly constrained minimum variance (LCMV) beamformer."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.meas_info import _simplify_info
from .._fiff.pick import pick_channels_cov, pick_info
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import _check_depth, _check_reference, combine_xyz
from ..rank import compute_rank
from ..source_estimate import _get_src_type, _make_stc
from ..utils import (
    _check_channels_spatial_filter,
    _check_info_inv,
    _check_one_ch_type,
    logger,
    verbose_static,
)
from ._compute_beamformer import (
    Beamformer,
    _check_src_type,
    _compute_beamformer,
    _compute_power,
    _prepare_beamformer_input,
    _proj_whiten_data,
)


@verbose_static(
    "info_not_none",
    "pick_ori_bf",
    "rank_info",
    "weight_norm",
    "reduce_rank",
    "depth",
    "inversion_bf",
)
def make_lcmv(
    info,
    forward,
    data_cov,
    reg=0.05,
    noise_cov=None,
    label=None,
    pick_ori=None,
    rank="info",
    weight_norm="unit-noise-gain-invariant",
    reduce_rank=False,
    depth=None,
    inversion="matrix",
    verbose=None,
):
    """Compute LCMV spatial filter.

    Parameters
    ----------
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
        Specifies the channels to include. Bad channels (in ``info['bads']``)
        are not used.
    forward : instance of Forward
        Forward operator.
    data_cov : instance of Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    noise_cov : instance of Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.

        .. note::
            If ``noise_cov`` is ``None`` and ``weight_norm='unit-noise-gain'``,
            the unit noise is assumed to be 1 in SI units, e.g., 1 T for
            magnetometers, 1 V for EEG, so resulting amplitudes will be tiny.
            Consider using :func:`mne.make_ad_hoc_cov` to provide a
            ``noise_cov`` to set noise values that are more reasonable for
            neural data or using ``weight_norm='nai'`` for weight-normalized
            beamformer output that is scaled by a noise estimate.
    label : instance of Label
        Restricts the LCMV solution to a given label.
    pick_ori : None | str
        For forward solutions with fixed orientation, None (default) must be
        used and a scalar beamformer is computed. For free-orientation forward
        solutions, a vector beamformer is computed and:

        - ``None``
            Orientations are pooled after computing a vector beamformer (Default).
        - ``'normal'``
            Filters are computed for the orientation tangential to the
            cortical surface.
        - ``'max-power'``
            Filters are computed for the orientation that maximizes power.

        - ``'vector'``
            Keeps the currents for each direction separate
    rank : None | 'info' | 'full' | dict
        This controls the rank computation that can be read from the
        measurement info or estimated from the data. When a noise covariance
        is used for whitening, this should reflect the rank of that covariance,
        otherwise amplification of noise components can occur in whitening (e.g.,
        often during source localization).

        :data:`python:None`
            The rank will be estimated from the data after proper scaling of
            different channel types.
        ``'info'``
            The rank is inferred from ``info``. If data have been processed
            with Maxwell filtering, the Maxwell filtering header is used.
            Otherwise, the channel counts themselves are used.
            In both cases, the number of projectors is subtracted from
            the (effective) number of channels in the data.
            For example, if Maxwell filtering reduces the rank to 68, with
            two projectors the returned value will be 66.
        ``'full'``
            The rank is assumed to be full, i.e. equal to the
            number of good channels. If a `~mne.Covariance` is passed, this can
            make sense if it has been (possibly improperly) regularized without
            taking into account the true data rank.
        :class:`dict`
            Calculate the rank only for a subset of channel types, and explicitly
            specify the rank for the remaining channel types. This can be
            extremely useful if you already **know** the rank of (part of) your
            data, for instance in case you have calculated it earlier.

            This parameter must be a dictionary whose **keys** correspond to
            channel types in the data (e.g. ``'meg'``, ``'mag'``, ``'grad'``,
            ``'eeg'``), and whose **values** are integers representing the
            respective ranks. For example, ``{'mag': 90, 'eeg': 45}`` will assume
            a rank of ``90`` and ``45`` for magnetometer data and EEG data,
            respectively.

            The ranks for all channel types present in the data, but
            **not** specified in the dictionary will be estimated empirically.
            That is, if you passed a dataset containing magnetometer, gradiometer,
            and EEG data together with the dictionary from the previous example,
            only the gradiometer rank would be determined, while the specified
            magnetometer and EEG ranks would be taken for granted.

        The default is ``'info'``.
    weight_norm : str | None
        Can be:

        - ``None``
            The unit-gain LCMV beamformer :footcite:`SekiharaNagarajan2008` will be
            computed.
        - ``'unit-noise-gain'``
            The unit-noise gain minimum variance beamformer will be computed
            (Borgiotti-Kaplan beamformer) :footcite:`SekiharaNagarajan2008`,
            which is not rotation invariant when ``pick_ori='vector'``.
            This should be combined with
            :meth:`stc.project('pca') <mne.VectorSourceEstimate.project>` to follow
            the definition in :footcite:`SekiharaNagarajan2008`.
        - ``'nai'``
            The Neural Activity Index :footcite:`VanVeenEtAl1997` will be computed,
            which simply scales all values from ``'unit-noise-gain'`` by a fixed
            value.
        - ``'unit-noise-gain-invariant'``
            Compute a rotation-invariant normalization using the matrix square
            root. This differs from ``'unit-noise-gain'`` only when
            ``pick_ori='vector'``, creating a solution that:

            1. Is rotation invariant (``'unit-noise-gain'`` is not);
            2. Satisfies the first requirement from
               :footcite:`SekiharaNagarajan2008` that ``w @ w.conj().T == I``,
               whereas ``'unit-noise-gain'`` has non-zero off-diagonals; but
            3. Does not satisfy the second requirement that ``w @ G.T = θI``,
               which arguably does not make sense for a rotation-invariant
               solution.

        Defaults to ``'unit-noise-gain-invariant'``.
    reduce_rank : bool
        If True, the rank of the denominator of the beamformer formula (i.e.,
        during pseudo-inversion) will be reduced by one for each spatial location.
        Setting ``reduce_rank=True`` is typically necessary if you use a single
        sphere model with MEG data.

        .. versionchanged:: 0.20
            Support for reducing rank in all modes (previously only supported
            ``pick='max_power'`` with weight normalization).
    depth : None | float | dict
        How to weight (or normalize) the forward using a depth prior.
        If float (default 0.8), it acts as the depth weighting exponent (``exp``)
        to use. None is equivalent to 0, meaning no depth weighting is performed.
        It can also be a :class:`dict` containing keyword arguments to pass to
        :func:`mne.forward.compute_depth_prior` (see docstring for details and
        defaults).

        .. versionadded:: 0.18
    inversion : 'single' | 'matrix'
        This determines how the beamformer deals with source spaces in "free"
        orientation. Such source spaces define three orthogonal dipoles at each
        source point. When ``inversion='single'``, each dipole is considered
        as an individual source and the corresponding spatial filter is
        computed for each dipole separately. When ``inversion='matrix'``, all
        three dipoles at a source vertex are considered as a group and the
        spatial filters are computed jointly using a matrix inversion. While
        ``inversion='single'`` is more stable, ``inversion='matrix'`` is more
        precise. See section 5 of :footcite:`vanVlietEtAl2018`.
        Defaults to ``'matrix'``.

        .. versionadded:: 0.21
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    filters : instance of Beamformer
        Dictionary containing filter weights from LCMV beamformer.
        Contains the following keys:

            'kind' : str
                The type of beamformer, in this case 'LCMV'.
            'weights' : array
                The filter weights of the beamformer.
            'data_cov' : instance of Covariance
                The data covariance matrix used to compute the beamformer.
            'noise_cov' : instance of Covariance | None
                The noise covariance matrix used to compute the beamformer.
            'whitener' : None | ndarray, shape (n_channels, n_channels)
                Whitening matrix, provided if whitening was applied to the
                covariance matrix and leadfield during computation of the
                beamformer weights.
            'weight_norm' : str | None
                Type of weight normalization used to compute the filter
                weights.
            'pick-ori' : None | 'max-power' | 'normal' | 'vector'
                The orientation in which the beamformer filters were computed.
            'ch_names' : list of str
                Channels used to compute the beamformer.
            'proj' : array
                Projections used to compute the beamformer.
            'is_ssp' : bool
                If True, projections were applied prior to filter computation.
            'vertices' : list
                Vertices for which the filter weights were computed.
            'is_free_ori' : bool
                If True, the filter was computed with free source orientation.
            'n_sources' : int
                Number of source location for which the filter weight were
                computed.
            'src_type' : str
                Type of source space.
            'source_nn' : ndarray, shape (n_sources, 3)
                For each source location, the surface normal.
            'proj' : ndarray, shape (n_channels, n_channels)
                Projections used to compute the beamformer.
            'subject' : str
                The subject ID.
            'rank' : int
                The rank of the data covariance matrix used to compute the
                beamformer weights.
            'max-power-ori' : ndarray, shape (n_sources, 3) | None
                When pick_ori='max-power', this fields contains the estimated
                direction of maximum power at each source location.
            'inversion' : 'single' | 'matrix'
                Whether the spatial filters were computed for each dipole
                separately or jointly for all dipoles at each vertex using a
                matrix inversion.

    Notes
    -----
    The original reference is :footcite:`VanVeenEtAl1997`.

    To obtain the Sekihara unit-noise-gain vector beamformer, you should use
    ``weight_norm='unit-noise-gain', pick_ori='vector'`` followed by
    :meth:`vec_stc.project('pca', src) <mne.VectorSourceEstimate.project>`.

    .. versionchanged:: 0.21
       The computations were extensively reworked, and the default for
       ``weight_norm`` was set to ``'unit-noise-gain-invariant'``.

    References
    ----------
    .. footbibliography::
    """
    # check number of sensor types present in the data and ensure a noise cov
    info = _simplify_info(info, keep=("proc_history",))
    noise_cov, _, allow_mismatch = _check_one_ch_type(
        "lcmv", info, forward, data_cov, noise_cov
    )
    # XXX we need this extra picking step (can't just rely on minimum norm's
    # because there can be a mismatch. Should probably add an extra arg to
    # _prepare_beamformer_input at some point (later)
    picks = _check_info_inv(info, forward, data_cov, noise_cov)
    with info._skip_checks():  # info is already consistent
        info = pick_info(info, picks)
    data_rank = compute_rank(data_cov, rank=rank, info=info)
    noise_rank = compute_rank(noise_cov, rank=rank, info=info)
    for key in data_rank:
        if (
            key not in noise_rank or data_rank[key] != noise_rank[key]
        ) and not allow_mismatch:
            raise ValueError(
                f"{key} data rank ({data_rank[key]}) did not match the noise rank ("
                f"{noise_rank.get(key, None)})"
            )
    del noise_rank
    rank = data_rank
    logger.info(f"Making LCMV beamformer with rank {rank}")
    del data_rank
    depth = _check_depth(depth, "depth_sparse")
    if inversion == "single":
        depth["combine_xyz"] = False

    (
        is_free_ori,
        info,
        proj,
        vertno,
        G,
        whitener,
        nn,
        orient_std,
    ) = _prepare_beamformer_input(
        info,
        forward,
        label,
        pick_ori,
        noise_cov=noise_cov,
        rank=rank,
        pca=False,
        **depth,
    )
    ch_names = list(info["ch_names"])

    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov._get_square()
    if "estimator" in data_cov:
        del data_cov["estimator"]
    rank_int = sum(rank.values())
    del rank

    # compute spatial filter
    n_orient = 3 if is_free_ori else 1
    W, max_power_ori = _compute_beamformer(
        G,
        Cm,
        reg,
        n_orient,
        weight_norm,
        pick_ori,
        reduce_rank,
        rank_int,
        inversion=inversion,
        nn=nn,
        orient_std=orient_std,
        whitener=whitener,
    )

    # get src type to store with filters for _make_stc
    src_type = _get_src_type(forward["src"], vertno)

    # get subject to store with filters
    subject_from = _subject_from_forward(forward)

    # Is the computed beamformer a scalar or vector beamformer?
    is_free_ori = is_free_ori if pick_ori in [None, "vector"] else False
    is_ssp = bool(info["projs"])

    filters = Beamformer(
        kind="LCMV",
        weights=W,
        data_cov=data_cov,
        noise_cov=noise_cov,
        whitener=whitener,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        ch_names=ch_names,
        proj=proj,
        is_ssp=is_ssp,
        vertices=vertno,
        is_free_ori=is_free_ori,
        n_sources=forward["nsource"],
        src_type=src_type,
        source_nn=forward["source_nn"].copy(),
        subject=subject_from,
        rank=rank_int,
        max_power_ori=max_power_ori,
        inversion=inversion,
    )

    return filters


def _apply_lcmv(data, filters, info, tmin):
    """Apply LCMV spatial filter to data for source reconstruction."""
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    W = filters["weights"]

    for i, M in enumerate(data):
        if len(M) != len(filters["ch_names"]):
            raise ValueError("data and picks must have the same length")

        if not return_single:
            logger.info(f"Processing epoch : {i + 1}")

        M = _proj_whiten_data(M, info["projs"], filters)

        # project to source space using beamformer weights
        vector = False
        if filters["is_free_ori"]:
            sol = np.dot(W, M)
            if filters["pick_ori"] == "vector":
                vector = True
            else:
                logger.info("combining the current components...")
                sol = combine_xyz(sol)
        else:
            # Linear inverse: do computation here or delayed
            if M.shape[0] < W.shape[0] and filters["pick_ori"] != "max-power":
                sol = (W, M)
            else:
                sol = np.dot(W, M)

        tstep = 1.0 / info["sfreq"]

        # compatibility with 0.16, add src_type as None if not present:
        filters, warn_text = _check_src_type(filters)

        yield _make_stc(
            sol,
            vertices=filters["vertices"],
            tmin=tmin,
            tstep=tstep,
            subject=filters["subject"],
            vector=vector,
            source_nn=filters["source_nn"],
            src_type=filters["src_type"],
            warn_text=warn_text,
        )

    logger.info("[done]")


@verbose_static()
def apply_lcmv(evoked, filters, *, verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on evoked data.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert.
    filters : instance of Beamformer
        LCMV spatial filter (beamformer weights).
        Filter weights returned from :func:`make_lcmv`.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate | VectorSourceEstimate
        Source time courses.

    See Also
    --------
    make_lcmv, apply_lcmv_raw, apply_lcmv_epochs, apply_lcmv_cov

    Notes
    -----
    .. versionadded:: 0.18
    """
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    sel = _check_channels_spatial_filter(evoked.ch_names, filters)
    data = data[sel]

    stc = _apply_lcmv(data=data, filters=filters, info=info, tmin=tmin)

    return next(stc)


@verbose_static()
def apply_lcmv_epochs(epochs, filters, *, return_generator=False, verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on single trial data.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    filters : instance of Beamformer
        LCMV spatial filter (beamformer weights)
        Filter weights returned from :func:`make_lcmv`.
    return_generator : bool
         Return a generator object instead of a list. This allows iterating
         over the stcs without having to keep them all in memory.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    stc: list | generator of (SourceEstimate | VolSourceEstimate)
        The source estimates for all epochs.

    See Also
    --------
    make_lcmv, apply_lcmv_raw, apply_lcmv, apply_lcmv_cov
    """
    _check_reference(epochs)

    info = epochs.info
    tmin = epochs.times[0]

    sel = _check_channels_spatial_filter(epochs.ch_names, filters)
    data = epochs.get_data(sel)
    stcs = _apply_lcmv(data=data, filters=filters, info=info, tmin=tmin)

    if not return_generator:
        stcs = [s for s in stcs]

    return stcs


@verbose_static()
def apply_lcmv_raw(raw, filters, start=None, stop=None, *, verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data to invert.
    filters : instance of Beamformer
        LCMV spatial filter (beamformer weights).
        Filter weights returned from :func:`make_lcmv`.
    start : int
        Index of first time sample (index not time is seconds).
    stop : int
        Index of first time sample not to include (index not time is seconds).
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses.

    See Also
    --------
    make_lcmv, apply_lcmv_epochs, apply_lcmv, apply_lcmv_cov
    """
    _check_reference(raw)

    info = raw.info

    sel = _check_channels_spatial_filter(raw.ch_names, filters)
    data, times = raw[sel, start:stop]
    tmin = times[0]

    stc = _apply_lcmv(data=data, filters=filters, info=info, tmin=tmin)

    return next(stc)


@verbose_static()
def apply_lcmv_cov(data_cov, filters, verbose=None):
    """Apply Linearly Constrained  Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    to a data covariance matrix to estimate source power.

    Parameters
    ----------
    data_cov : instance of Covariance
        Data covariance matrix.
    filters : instance of Beamformer
        LCMV spatial filter (beamformer weights).
        Filter weights returned from :func:`make_lcmv`.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source power.

    See Also
    --------
    make_lcmv, apply_lcmv, apply_lcmv_epochs, apply_lcmv_raw
    """
    sel = _check_channels_spatial_filter(data_cov.ch_names, filters)
    sel_names = [data_cov.ch_names[ii] for ii in sel]
    data_cov = pick_channels_cov(data_cov, sel_names)

    n_orient = filters["weights"].shape[0] // filters["n_sources"]
    # Need to project and whiten along both dimensions
    data = _proj_whiten_data(data_cov["data"].T, data_cov["projs"], filters)
    data = _proj_whiten_data(data.T, data_cov["projs"], filters)
    del data_cov
    source_power = _compute_power(data, filters["weights"], n_orient)

    # compatibility with 0.16, add src_type as None if not present:
    filters, warn_text = _check_src_type(filters)

    return _make_stc(
        source_power,
        vertices=filters["vertices"],
        src_type=filters["src_type"],
        tmin=0.0,
        tstep=1.0,
        subject=filters["subject"],
        source_nn=filters["source_nn"],
        warn_text=warn_text,
    )
