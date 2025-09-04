# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.constants import FIFF
from .._fiff.pick import pick_info
from ..baseline import _log_rescale, rescale
from ..epochs import Epochs
from ..event import make_fixed_length_events
from ..evoked import EvokedArray
from ..fixes import _safe_svd
from ..label import BiHemiLabel, Label
from ..parallel import parallel_func
from ..source_estimate import _make_stc
from ..time_frequency.multitaper import (
    _compute_mt_params,
    _mt_spectra,
    _psd_from_mt,
    _psd_from_mt_adaptive,
)
from ..time_frequency.tfr import cwt, morlet
from ..utils import ProgressBar, _check_option, _pl, _validate_type, logger, verbose
from .inverse import (
    INVERSE_METHODS,
    _assemble_kernel,
    _check_or_prepare,
    _check_ori,
    _pick_channels_inverse_operator,
    _subject_from_inverse,
    combine_xyz,
)


def _restrict_K_to_lbls(labels, K, noise_norm, vertno, pick_ori):
    """Use labels to choose desired sources in the kernel."""
    verts_to_use = [[], []]
    # create mask for K by compiling original vertices from vertno in labels
    for ii in range(len(labels)):
        lab = labels[ii]
        # handle BiHemi labels; ok so long as no overlap w/ single hemi labels
        if lab.hemi == "both":
            l_verts = np.intersect1d(vertno[0], lab.lh.vertices)
            r_verts = np.intersect1d(vertno[1], lab.rh.vertices)  # output sorted
            verts_to_use[0] += list(l_verts)
            verts_to_use[1] += list(r_verts)
        else:
            hidx = 0 if lab.hemi == "lh" else 1
            verts = np.intersect1d(vertno[hidx], lab.vertices)
            verts_to_use[hidx] += list(verts)

    # check that we don't have overlapping vertices in our labels
    for ii in range(2):
        if len(np.unique(verts_to_use[ii])) != len(verts_to_use[ii]):
            raise RuntimeError(
                "Labels cannot have overlapping vertices. "
                "Please select labels with unique vertices "
                "and try again."
            )

    # turn original vertex numbers from vertno into indices for K
    K_mask = np.searchsorted(vertno[0], verts_to_use[0])
    r_kmask = np.searchsorted(vertno[1], verts_to_use[1]) + len(vertno[0])
    K_mask = np.hstack((K_mask, r_kmask))

    # record which original vertices are at each index in out_K
    hemis = ("lh", "rh")
    ki_keys = [
        (hemis[hi], verts_to_use[hi][ii])
        for hi in range(2)
        for ii in range(len(verts_to_use[hi]))
    ]
    ki_vals = list(range(len(K_mask)))
    k_idxs = dict(zip(ki_keys, ki_vals))

    # mask K, handling the orientation issue
    len_allverts = len(vertno[0]) + len(vertno[1])
    if len(K) == len_allverts:
        assert pick_ori == "normal"
        out_K = K[K_mask]
    else:
        # here, K = [x0, y0, z0, x1, y1, z1 ...]
        # we need to drop x, y and z of unused vertices
        assert not pick_ori == "normal", pick_ori
        assert len(K) == 3 * len_allverts, (len(K), len_allverts)
        out_len = len(K_mask) * 3
        out_K = K[0:out_len]  # get the correct-shaped array
        for di in range(3):
            K_pick = K[di::3]
            out_K[di::3] = K_pick[K_mask]  # set correct values for out

    out_vertno = verts_to_use
    if noise_norm is not None:
        out_nn = noise_norm[K_mask]
    else:
        out_nn = None

    return out_K, out_nn, out_vertno, k_idxs


def _prepare_source_params(
    inst,
    inverse_operator,
    label=None,
    lambda2=1.0 / 9.0,
    method="dSPM",
    nave=1,
    pca=True,
    pick_ori="normal",
    prepared=False,
    method_params=None,
    use_cps=True,
):
    """Prepare inverse operator and params for spectral / TFR analysis."""
    inv = _check_or_prepare(
        inverse_operator, nave, lambda2, method, method_params, prepared
    )

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(inst.ch_names, inv)
    logger.info("Picked %d channels from the data", len(sel))
    logger.info("Computing inverse...")
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    # K shape: (3 x n_sources, n_channels) or (n_sources, n_channels)
    # noise_norm shape: (n_sources, 1)
    # vertno: [lh_verts, rh_verts]

    k_idxs = None
    if not isinstance(label, Label | BiHemiLabel):
        whole_K, whole_noise_norm, whole_vertno, _ = _assemble_kernel(
            inv, None, method, pick_ori, use_cps=use_cps
        )
        if isinstance(label, list):
            K, noise_norm, vertno, k_idxs = _restrict_K_to_lbls(
                label, whole_K, whole_noise_norm, whole_vertno, pick_ori
            )
        else:
            assert not label
            K, noise_norm, vertno = whole_K, whole_noise_norm, whole_vertno
    elif isinstance(label, Label | BiHemiLabel):
        K, noise_norm, vertno, _ = _assemble_kernel(
            inv, label, method, pick_ori, use_cps=use_cps
        )

    if pca:
        U, s, Vh = _safe_svd(K, full_matrices=False)
        rank = np.sum(s > 1e-8 * s[0])
        K = s[:rank] * U[:, :rank]
        Vh = Vh[:rank]
        logger.info("Reducing data rank %d -> %d", len(s), rank)
    else:
        Vh = None
    is_free_ori = inverse_operator["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI

    return K, sel, Vh, vertno, is_free_ori, noise_norm, k_idxs


@verbose
def source_band_induced_power(
    epochs,
    inverse_operator,
    bands,
    label=None,
    lambda2=1.0 / 9.0,
    method="dSPM",
    nave=1,
    n_cycles=5,
    df=1,
    use_fft=False,
    decim=1,
    baseline=None,
    baseline_mode="logratio",
    pca=True,
    n_jobs=None,
    prepared=False,
    method_params=None,
    use_cps=True,
    *,
    verbose=None,
):
    """Compute source space induced power in given frequency bands.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    bands : dict
        Example : bands = dict(alpha=[8, 9]).
    label : Label | list of Label
        Restricts the source estimates to a given label or list of labels. If
        labels are provided in a list, power will be averaged over vertices.
    lambda2 : float
        The regularization parameter of the minimum norm.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    df : float
        Delta frequency within bands.
    use_fft : bool
        Do convolutions in time or frequency domain with FFT.
    decim : int
        Temporal decimation factor.
    baseline : None (default) or tuple, shape (2,)
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used.
    baseline_mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by

        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')

    pca : bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    %(n_jobs)s
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    method_params : dict | None
        Additional options for eLORETA. See Notes of :func:`apply_inverse`.

        .. versionadded:: 0.16
    %(use_cps_restricted)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    stcs : dict of SourceEstimate (or VolSourceEstimate)
        The estimated source space induced power estimates in shape
        (n_vertices, n_frequencies, n_samples) if label=None or label=label.
        For lists of one or more labels, the induced power estimate has shape
        (n_labels, n_frequencies, n_samples).
    """  # noqa: E501
    _check_option("method", method, INVERSE_METHODS)

    freqs = np.concatenate(
        [np.arange(band[0], band[1] + df / 2.0, df) for _, band in bands.items()]
    )

    powers, _, vertno = _source_induced_power(
        epochs,
        inverse_operator,
        freqs,
        label=label,
        lambda2=lambda2,
        method=method,
        nave=nave,
        n_cycles=n_cycles,
        decim=decim,
        use_fft=use_fft,
        pca=pca,
        n_jobs=n_jobs,
        with_plv=False,
        prepared=prepared,
        method_params=method_params,
        use_cps=use_cps,
    )

    Fs = epochs.info["sfreq"]  # sampling in Hz
    stcs = dict()

    subject = _subject_from_inverse(inverse_operator)
    _log_rescale(baseline, baseline_mode)  # for early failure
    for name, band in bands.items():
        idx = [k for k, f in enumerate(freqs) if band[0] <= f <= band[1]]

        # average power in band + mean over epochs
        power = np.mean(powers[:, idx, :], axis=1)

        # Run baseline correction
        power = rescale(
            power,
            epochs.times[::decim],
            baseline,
            baseline_mode,
            copy=False,
            verbose=False,
        )

        tmin = epochs.times[0]
        tstep = float(decim) / Fs
        stc = _make_stc(
            power,
            vertices=vertno,
            tmin=tmin,
            tstep=tstep,
            subject=subject,
            src_type=inverse_operator["src"].kind,
        )
        stcs[name] = stc

        logger.info("[done]")

    return stcs


def _prepare_tfr(data, decim, pick_ori, Ws, K, source_ori):
    """Prepare TFR source localization."""
    n_times = data[:, :, ::decim].shape[2]
    n_freqs = len(Ws)
    n_sources = K.shape[0]
    is_free_ori = False
    if source_ori == FIFF.FIFFV_MNE_FREE_ORI and pick_ori is None:
        is_free_ori = True
        n_sources //= 3

    shape = (n_sources, n_freqs, n_times)
    return shape, is_free_ori


@verbose
def _compute_pow_plv(
    data,
    K,
    sel,
    Ws,
    source_ori,
    use_fft,
    Vh,
    with_power,
    with_plv,
    pick_ori,
    decim,
    noise_norm=None,
    verbose=None,
):
    """Aux function for induced power and PLV."""
    shape, is_free_ori = _prepare_tfr(data, decim, pick_ori, Ws, K, source_ori)
    power = np.zeros(shape, dtype=np.float64)  # power or raw TFR
    # phase lock
    plv = np.zeros(shape, dtype=np.complex128) if with_plv else None

    for epoch in data:
        epoch = epoch[sel]  # keep only selected channels

        if Vh is not None:
            epoch = np.dot(Vh, epoch)  # reducing data rank

        power_e, plv_e = _single_epoch_tfr(
            data=epoch,
            is_free_ori=is_free_ori,
            K=K,
            Ws=Ws,
            use_fft=use_fft,
            decim=decim,
            shape=shape,
            with_plv=with_plv,
            with_power=with_power,
        )

        power += power_e
        if with_plv:
            plv += plv_e

    if noise_norm is not None:
        power *= noise_norm[:, :, np.newaxis] ** 2

    return power, plv


def _single_epoch_tfr(
    data, is_free_ori, K, Ws, use_fft, decim, shape, with_plv, with_power
):
    """Compute single trial TFRs, either ITC, power or raw TFR."""
    tfr_e = np.zeros(shape, dtype=np.float64)  # power or raw TFR
    # phase lock
    plv_e = np.zeros(shape, dtype=np.complex128) if with_plv else None
    n_sources, _, n_times = shape
    for f, w in enumerate(Ws):
        tfr_ = cwt(data, [w], use_fft=use_fft, decim=decim)
        tfr_ = np.asfortranarray(tfr_.reshape(len(data), -1))

        # phase lock and power at freq f
        if with_plv:
            plv_f = np.zeros((n_sources, n_times), dtype=np.complex128)

        tfr_f = np.zeros((n_sources, n_times), dtype=np.float64)

        for k, t in enumerate([np.real(tfr_), np.imag(tfr_)]):
            sol = np.dot(K, t)

            sol_pick_normal = sol
            if is_free_ori:
                sol_pick_normal = sol[2::3]

            if with_plv:
                if k == 0:  # real
                    plv_f += sol_pick_normal
                else:  # imag
                    plv_f += 1j * sol_pick_normal

            if is_free_ori:
                logger.debug("combining the current components...")
                sol = combine_xyz(sol, square=with_power)
            elif with_power:
                sol *= sol
            tfr_f += sol
            del sol

        tfr_e[:, f, :] += tfr_f
        del tfr_f

        if with_plv:
            plv_f /= np.abs(plv_f)
            plv_e[:, f, :] += plv_f
            del plv_f

    return tfr_e, plv_e


def _get_label_power(power, labels, vertno, k_idxs):
    """Average power across vertices in labels."""
    (_, ps1, ps2) = power.shape
    # construct out array with correct shape
    out_power = np.zeros(shape=(len(labels), ps1, ps2))

    # for each label, compile list of vertices we want
    for li in np.arange(len(labels)):
        lab = labels[li]
        hemis = ("lh", "rh")
        all_vnums = [[], []]
        if lab.hemi == "both":
            all_vnums[0] = np.intersect1d(lab.lh.vertices, vertno[0])
            all_vnums[1] = np.intersect1d(lab.rh.vertices, vertno[1])
        else:
            assert lab.hemi == "lh" or lab.hemi == "rh"
            h_id = 0 if lab.hemi == "lh" else 1
            all_vnums[h_id] = np.intersect1d(vertno[h_id], lab.vertices)

        verts = [(hemis[hi], vn) for hi in range(2) for vn in all_vnums[hi]]

        # restrict power to relevant vertices in label
        lab_mask = np.array([False] * len(power))
        for vert in verts:
            lab_mask[k_idxs[vert]] = True  # k_idxs[vert] gives power row index
        lab_power = power[lab_mask]  # only pass through rows we want
        assert lab_power.shape == (len(verts), ps1, ps2)

        # set correct out values for label
        out_power[li, :, :] = np.mean(lab_power, axis=0)

    assert out_power.shape == (len(labels), ps1, ps2)
    return out_power


@verbose
def _source_induced_power(
    epochs,
    inverse_operator,
    freqs,
    label=None,
    lambda2=1.0 / 9.0,
    method="dSPM",
    nave=1,
    n_cycles=5,
    decim=1,
    use_fft=False,
    pca=True,
    pick_ori="normal",
    n_jobs=None,
    with_plv=True,
    zero_mean=False,
    prepared=False,
    method_params=None,
    use_cps=True,
    verbose=None,
):
    """Aux function for source induced power."""
    if label:
        _validate_type(
            label,
            types=(Label, BiHemiLabel, list, tuple, None),
            type_name=("Label or BiHemiLabel", "list of labels", "None"),
        )
        if isinstance(label, list | tuple):
            for item in label:
                _validate_type(
                    item,
                    types=(Label, BiHemiLabel),
                    type_name=("Label or BiHemiLabel"),
                )
            if len(label) > 1 and with_plv:
                raise RuntimeError(
                    "Phase-locking value cannot be calculated "
                    "when averaging induced power within "
                    "labels. Please set `with_plv` to False, pass a "
                    "single `label=label`, or set `label=None`."
                )

    epochs_data = epochs.get_data(copy=False)
    K, sel, Vh, vertno, is_free_ori, noise_norm, k_id = _prepare_source_params(
        inst=epochs,
        inverse_operator=inverse_operator,
        label=label,
        lambda2=lambda2,
        method=method,
        nave=nave,
        pca=pca,
        pick_ori=pick_ori,
        prepared=prepared,
        method_params=method_params,
        use_cps=use_cps,
    )

    inv = inverse_operator
    parallel, my_compute_source_tfrs, n_jobs = parallel_func(
        _compute_pow_plv, n_jobs, max_jobs=len(epochs_data)
    )
    Fs = epochs.info["sfreq"]  # sampling in Hz

    logger.info("Computing source power ...")

    Ws = morlet(Fs, freqs, n_cycles=n_cycles, zero_mean=zero_mean)

    out = parallel(
        my_compute_source_tfrs(
            data=data,
            K=K,
            sel=sel,
            Ws=Ws,
            source_ori=inv["source_ori"],
            use_fft=use_fft,
            Vh=Vh,
            with_plv=with_plv,
            with_power=True,
            pick_ori=pick_ori,
            decim=decim,
            noise_norm=noise_norm,
        )
        for data in np.array_split(epochs_data, n_jobs)
    )
    power = sum(o[0] for o in out)  # power shape: (n_verts, n_freqs, n_samps)
    power /= len(epochs_data)  # average power over epochs

    if isinstance(label, Label | BiHemiLabel):
        logger.info(
            f"Outputting power for {len(power)} vertices in label {label.name}."
        )
    elif isinstance(label, list):
        power = _get_label_power(power, label, vertno, k_id)
        logger.info(
            "Averaging induced power across vertices within labels "
            f"for {len(label)} label{_pl(label)}."
        )
    else:
        assert not label

    if with_plv:
        plv = sum(o[1] for o in out)
        plv = np.abs(plv)
        plv /= len(epochs_data)  # average power over epochs
    else:
        plv = None

    return power, plv, vertno


@verbose
def source_induced_power(
    epochs,
    inverse_operator,
    freqs,
    label=None,
    lambda2=1.0 / 9.0,
    method="dSPM",
    nave=1,
    n_cycles=5,
    decim=1,
    use_fft=False,
    pick_ori=None,
    baseline=None,
    baseline_mode="logratio",
    pca=True,
    n_jobs=None,
    *,
    return_plv=True,
    zero_mean=False,
    prepared=False,
    method_params=None,
    use_cps=True,
    verbose=None,
):
    """Compute induced power and phase lock.

    Computation can optionally be restricted in a label.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    freqs : array
        Array of frequencies of interest.
    label : Label | list of Label
        Restricts the source estimates to a given label or list of labels. If
        labels are provided in a list, power will be averaged over vertices within each
        label.
    lambda2 : float
        The regularization parameter of the minimum norm.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim : int
        Temporal decimation factor.
    use_fft : bool
        Do convolutions in time or frequency domain with FFT.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    baseline_mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by

        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')

    pca : bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    %(n_jobs)s
    return_plv : bool
        If True, return the phase-locking value array. Else, only return power.

        .. versionadded:: 1.6
    zero_mean : bool
        Make sure the wavelets are zero mean.
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    method_params : dict | None
        Additional options for eLORETA. See Notes of :func:`apply_inverse`.
    %(use_cps_restricted)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    power : array
        The induced power array with shape (n_sources, n_freqs, n_samples) if
        label=None or label=label. For lists of one or more labels, the induced
        power estimate has shape (n_labels, n_frequencies, n_samples).
    plv : array
        The phase-locking value array with shape (n_sources, n_freqs,
        n_samples). Only returned if ``return_plv=True``.
    """  # noqa: E501
    _check_option("method", method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator["source_ori"], inverse_operator["src"])

    power, plv, vertno = _source_induced_power(
        epochs,
        inverse_operator,
        freqs,
        label=label,
        lambda2=lambda2,
        method=method,
        nave=nave,
        n_cycles=n_cycles,
        decim=decim,
        use_fft=use_fft,
        pick_ori=pick_ori,
        pca=pca,
        n_jobs=n_jobs,
        with_plv=return_plv,
        method_params=method_params,
        zero_mean=zero_mean,
        prepared=prepared,
        use_cps=use_cps,
    )

    # Run baseline correction
    power = rescale(power, epochs.times[::decim], baseline, baseline_mode, copy=False)

    outs = (power, plv) if return_plv else power
    return outs


@verbose
def compute_source_psd(
    raw,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method="dSPM",
    tmin=0.0,
    tmax=None,
    fmin=0.0,
    fmax=200.0,
    n_fft=2048,
    overlap=0.5,
    pick_ori=None,
    label=None,
    nave=1,
    pca=True,
    prepared=False,
    method_params=None,
    inv_split=None,
    bandwidth="hann",
    adaptive=False,
    low_bias=False,
    n_jobs=None,
    return_sensor=False,
    dB=False,
    *,
    verbose=None,
):
    """Compute source power spectral density (PSD).

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    tmin : float
        The beginning of the time interval of interest (in seconds).
        Use 0. for the beginning of the file.
    tmax : float | None
        The end of the time interval of interest (in seconds). If None
        stop at the end of the file.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    n_fft : int
        Window size for the FFT. Should be a power of 2.
    overlap : float
        The overlap fraction between windows. Should be between 0 and 1.
        0 means no overlap.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    label : Label
        Restricts the source estimates to a given label.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    pca : bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    method_params : dict | None
        Additional options for eLORETA. See Notes of :func:`apply_inverse`.

        .. versionadded:: 0.16
    inv_split : int or None
        Split inverse operator into inv_split parts in order to save memory.

        .. versionadded:: 0.17
    bandwidth : float | str
        The bandwidth of the multi taper windowing function in Hz.
        Can also be a string (e.g., 'hann') to use a single window.

        For backward compatibility, the default is 'hann'.

        .. versionadded:: 0.17
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).

        .. versionadded:: 0.17
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.

        .. versionadded:: 0.17
    %(n_jobs)s
        It is only used if adaptive=True.

        .. versionadded:: 0.17
    return_sensor : bool
        If True, return the sensor PSDs as an EvokedArray.

        .. versionadded:: 0.17
    dB : bool
        If True (default False), return output it decibels.

        .. versionadded:: 0.17
    %(verbose)s

    Returns
    -------
    stc_psd : instance of SourceEstimate | VolSourceEstimate
        The PSD of each of the sources.
    sensor_psd : instance of EvokedArray
        The PSD of each sensor. Only returned if ``return_sensor`` is True.

    See Also
    --------
    compute_source_psd_epochs

    Notes
    -----
    Each window is multiplied by a window before processing, so
    using a non-zero overlap is recommended.

    This function is different from :func:`compute_source_psd_epochs` in that:

    1. ``bandwidth='hann'`` by default, skipping multitaper estimation
    2. For convenience it wraps
       :func:`mne.make_fixed_length_events` and :class:`mne.Epochs`.

    Otherwise the two should produce identical results.
    """
    tmin = 0.0 if tmin is None else float(tmin)
    overlap = float(overlap)
    if not 0 <= overlap < 1:
        raise ValueError(f"Overlap must be at least 0 and less than 1, got {overlap}")
    n_fft = int(n_fft)
    duration = ((1.0 - overlap) * n_fft) / raw.info["sfreq"]
    events = make_fixed_length_events(raw, 1, tmin, tmax, duration)
    epochs = Epochs(raw, events, 1, 0, (n_fft - 1) / raw.info["sfreq"], baseline=None)
    out = compute_source_psd_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method,
        fmin,
        fmax,
        pick_ori,
        label,
        nave,
        pca,
        inv_split,
        bandwidth,
        adaptive,
        low_bias,
        True,
        n_jobs,
        prepared,
        method_params,
        return_sensor=True,
    )
    source_data = 0.0
    sensor_data = 0.0
    count = 0
    for stc, evoked in out:
        source_data += stc.data
        sensor_data += evoked.data
        count += 1
    assert count > 0  # should be guaranteed by make_fixed_length_events
    sensor_data /= count
    source_data /= count
    if dB:
        np.log10(sensor_data, out=sensor_data)
        sensor_data *= 10.0
        np.log10(source_data, out=source_data)
        source_data *= 10.0
    evoked.data = sensor_data
    evoked.nave = count
    stc.data = source_data
    out = stc
    if return_sensor:
        out = (out, evoked)
    return out


def _compute_source_psd_epochs(
    epochs,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method="dSPM",
    fmin=0.0,
    fmax=200.0,
    pick_ori=None,
    label=None,
    nave=1,
    pca=True,
    inv_split=None,
    bandwidth=4.0,
    adaptive=False,
    low_bias=True,
    n_jobs=None,
    prepared=False,
    method_params=None,
    return_sensor=False,
    use_cps=True,
):
    """Generate compute_source_psd_epochs."""
    logger.info(f"Considering frequencies {fmin} ... {fmax} Hz")

    if label:
        # TODO: add multi-label support
        # since `_prepare_source_params` can handle a list of labels now,
        # multi-label support should be within reach for psd calc as well
        _validate_type(
            label,
            types=(Label, BiHemiLabel, None),
            type_name=("Label or BiHemiLabel", "None"),
        )

    K, sel, Vh, vertno, is_free_ori, noise_norm, _ = _prepare_source_params(
        inst=epochs,
        inverse_operator=inverse_operator,
        label=label,
        lambda2=lambda2,
        method=method,
        nave=nave,
        pca=pca,
        pick_ori=pick_ori,
        prepared=prepared,
        method_params=method_params,
        use_cps=use_cps,
    )
    # Simplify code with a tiny (rel. to other computations) penalty for eye
    # mult
    Vh = np.eye(K.shape[1]) if Vh is None else Vh

    # split the inverse operator
    if inv_split is not None:
        K_split = np.array_split(K, inv_split)
    else:
        K_split = [K]

    # compute DPSS windows
    n_times = len(epochs.times)
    sfreq = epochs.info["sfreq"]

    dpss, eigvals, adaptive = _compute_mt_params(
        n_times, sfreq, bandwidth, low_bias, adaptive, verbose=False
    )

    n_tapers = len(dpss)
    try:
        n_epochs = len(epochs)
    except RuntimeError:
        n_epochs = len(epochs.events)
        extra = f"on at most {n_epochs} epochs"
    else:
        extra = f"on {n_epochs} epochs"
    if isinstance(bandwidth, str):
        bandwidth = f"{bandwidth} windowing"
    else:
        bandwidth = f"{n_tapers} tapers with bandwidth {bandwidth:0.1f} Hz"
    logger.info(f"Using {bandwidth} {extra}")

    if adaptive:
        parallel, my_psd_from_mt_adaptive, n_jobs = parallel_func(
            _psd_from_mt_adaptive, n_jobs
        )
    else:
        weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]

    subject = _subject_from_inverse(inverse_operator)
    iter_epochs = ProgressBar(epochs, max_value=n_epochs)
    evoked_info = pick_info(epochs.info, sel, verbose=False)
    for k, e in enumerate(iter_epochs):
        data = np.dot(Vh, e[sel])  # reducing data rank

        # compute tapered spectra in sensor space
        x_mt, freqs = _mt_spectra(data, dpss, sfreq)

        if k == 0:
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            fstep = np.mean(np.diff(freqs))
            with evoked_info._unlock():
                evoked_info["sfreq"] = 1.0 / fstep
        freqs = freqs[freq_mask]

        # sensor space PSD
        x_mt_sensor = np.empty(
            (len(sel), x_mt.shape[1], x_mt.shape[2]), dtype=x_mt.dtype
        )
        for i in range(n_tapers):
            x_mt_sensor[:, i, :] = np.dot(Vh.T, x_mt[:, i, :])
        if adaptive:
            out = parallel(
                my_psd_from_mt_adaptive(x, eigvals, freq_mask)
                for x in np.array_split(x_mt_sensor, min(n_jobs, len(x_mt_sensor)))
            )
            sensor_psd = np.concatenate(out)
        else:
            x_mt_sensor = x_mt_sensor[:, :, freq_mask]
            sensor_psd = _psd_from_mt(x_mt_sensor, weights)

        # allocate space for output
        psd = np.empty((K.shape[0], np.sum(freq_mask)))

        # Optionally, we split the inverse operator into parts to save memory.
        # Without splitting the tapered spectra in source space have size
        # (n_vertices x n_tapers x n_times / 2)
        pos = 0
        for K_part in K_split:
            # allocate space for tapered spectra in source space
            x_mt_src = np.empty(
                (K_part.shape[0], x_mt.shape[1], x_mt.shape[2]), dtype=x_mt.dtype
            )

            # apply inverse to each taper (faster than equiv einsum)
            for i in range(n_tapers):
                x_mt_src[:, i, :] = np.dot(K_part, x_mt[:, i, :])

            # compute the psd
            if adaptive:
                out = parallel(
                    my_psd_from_mt_adaptive(x, eigvals, freq_mask)
                    for x in np.array_split(x_mt_src, min(n_jobs, len(x_mt_src)))
                )
                this_psd = np.concatenate(out)
            else:
                x_mt_src = x_mt_src[:, :, freq_mask]
                this_psd = _psd_from_mt(x_mt_src, weights)

            psd[pos : pos + K_part.shape[0], :] = this_psd
            pos += K_part.shape[0]

        # combine orientations
        if is_free_ori and pick_ori is None:
            psd = combine_xyz(psd, square=False)

        if noise_norm is not None:
            psd *= noise_norm**2

        out = _make_stc(
            psd,
            tmin=freqs[0],
            tstep=fstep,
            vertices=vertno,
            subject=subject,
            src_type=inverse_operator["src"].kind,
        )

        if return_sensor:
            comment = f"Epoch {k} PSD"
            out = (
                out,
                EvokedArray(sensor_psd, evoked_info.copy(), freqs[0], comment, nave),
            )

        # we return a generator object for "stream processing"
        yield out

    iter_epochs.update(n_epochs)  # in case some were skipped


@verbose
def compute_source_psd_epochs(
    epochs,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method="dSPM",
    fmin=0.0,
    fmax=200.0,
    pick_ori=None,
    label=None,
    nave=1,
    pca=True,
    inv_split=None,
    bandwidth=4.0,
    adaptive=False,
    low_bias=True,
    return_generator=False,
    n_jobs=None,
    prepared=False,
    method_params=None,
    return_sensor=False,
    use_cps=True,
    verbose=None,
):
    """Compute source power spectral density (PSD) from Epochs.

    This uses the multi-taper method to compute the PSD for each epoch.

    Parameters
    ----------
    epochs : instance of Epochs
        The raw data.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    label : Label
        Restricts the source estimates to a given label.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    pca : bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    inv_split : int or None
        Split inverse operator into inv_split parts in order to save memory.
    bandwidth : float | str
        The bandwidth of the multi taper windowing function in Hz.
        Can also be a string (e.g., 'hann') to use a single window.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    %(n_jobs)s
        It is only used if adaptive=True.
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    method_params : dict | None
        Additional options for eLORETA. See Notes of :func:`apply_inverse`.

        .. versionadded:: 0.16
    return_sensor : bool
        If True, also return the sensor PSD for each epoch as an EvokedArray.

        .. versionadded:: 0.17
    %(use_cps_restricted)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    out : list (or generator object)
        A list (or generator) for the source space PSD (and optionally the
        sensor PSD) for each epoch.

    See Also
    --------
    compute_source_psd
    """
    # use an auxiliary function so we can either return a generator or a list
    stcs_gen = _compute_source_psd_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        fmin=fmin,
        fmax=fmax,
        pick_ori=pick_ori,
        label=label,
        nave=nave,
        pca=pca,
        inv_split=inv_split,
        bandwidth=bandwidth,
        adaptive=adaptive,
        low_bias=low_bias,
        n_jobs=n_jobs,
        prepared=prepared,
        method_params=method_params,
        return_sensor=return_sensor,
        use_cps=use_cps,
    )

    if return_generator:
        # return generator object
        return stcs_gen
    else:
        # return a list
        stcs = list()
        for stc in stcs_gen:
            stcs.append(stc)

        return stcs
