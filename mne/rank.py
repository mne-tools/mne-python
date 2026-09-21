"""Some utility functions for rank estimation."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy import linalg

from ._fiff.meas_info import Info, _simplify_info
from ._fiff.pick import _picks_by_type, _picks_to_idx, pick_channels_cov, pick_info
from ._fiff.proj import make_projector
from .defaults import _handle_default
from .utils import (
    _apply_scaling_cov,
    _check_on_missing,
    _check_rank,
    _compute_row_norms,
    _on_missing,
    _pl,
    _scaled_array,
    _undo_scaling_cov,
    _validate_type,
    _verbose_control,
    fill_doc_static,
    logger,
    verbose_static,
    warn,
)


@_verbose_control
@fill_doc_static("tol_rank", "tol_kind_rank")
def estimate_rank(
    data,
    tol="auto",
    return_singular=False,
    norm=True,
    tol_kind="absolute",
    verbose=None,
):
    """Estimate the rank of data.

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one.

    Parameters
    ----------
    data : array
        Data to estimate the rank of (should be 2-dimensional).
    tol : float | 'auto'
        Tolerance for singular values to consider non-zero in
        calculating the rank. The singular values are calculated
        in this method such that independent data are expected to
        have singular value around one. Can be 'auto' to use the
        same thresholding as :func:`scipy.linalg.orth`.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    norm : bool
        If True, data will be scaled by their estimated row-wise norm.
        Else data are assumed to be scaled. Defaults to True.
    tol_kind : str
        Can be: "absolute" (default) or "relative". Only used if ``tol`` is a
        float, because when ``tol`` is a string the mode is implicitly relative.
        After applying the chosen scale factors / normalization to the data,
        the singular values are computed, and the rank is then taken as:

        - ``'absolute'``
            The number of singular values ``s`` greater than ``tol``.
            This mode can fail if your data do not adhere to typical
            data scalings.
        - ``'relative'``
            The number of singular values ``s`` greater than ``tol * s.max()``.
            This mode can fail if you have one or more large components in the
            data (e.g., artifacts).

        .. versionadded:: 0.21.0

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    if norm:
        data = data.copy()  # operate on a copy
        norms = _compute_row_norms(data)
        data /= norms[:, np.newaxis]
    s = linalg.svdvals(data)
    rank = _estimate_rank_from_s(s, tol, tol_kind)
    if return_singular is True:
        return rank, s
    else:
        return rank


def _estimate_rank_from_s(s, tol="auto", tol_kind="absolute"):
    """Estimate the rank of a matrix from its singular values.

    Parameters
    ----------
    s : ndarray, shape (..., ndim)
        The singular values of the matrix.
    tol : float | ``'auto'``
        Tolerance for singular values to consider non-zero in calculating the
        rank. Can be 'auto' to use the same thresholding as
        ``scipy.linalg.orth`` (assuming np.float64 datatype) adjusted
        by a factor of 2.
    tol_kind : str
        Can be ``"absolute"`` or ``"relative"``.

    Returns
    -------
    rank : ndarray, shape (...)
        The estimated rank.
    """
    s = np.array(s, float)
    max_s = np.amax(s, axis=-1)
    if isinstance(tol, str):
        if tol not in ("auto", "float32"):
            raise ValueError(f'tol must be "auto" or float, got {repr(tol)}')
        # XXX this should be float32 probably due to how we save and
        # load data, but it breaks test_make_inverse_operator (!)
        # The factor of 2 gets test_compute_covariance_auto_reg[None]
        # to pass without breaking minimum norm tests. :(
        # Passing 'float32' is a hack workaround for test_maxfilter_get_rank :(
        if tol == "float32":
            eps = np.finfo(np.float32).eps
        else:
            eps = np.finfo(np.float64).eps
        tol = s.shape[-1] * max_s * eps
        if s.ndim == 1:  # typical
            logger.info(
                "    Using tolerance %0.2g (%0.2g eps * %d dim * %0.2g"
                "  max singular value)",
                tol,
                eps,
                len(s),
                max_s,
            )
    elif not (isinstance(tol, np.ndarray) and tol.dtype.kind == "f"):
        tol = float(tol)
        if tol_kind == "relative":
            tol = tol * max_s

    rank = np.sum(s > tol, axis=-1)
    return rank


def _estimate_rank_raw(
    raw,
    picks=None,
    tol=1e-4,
    scalings="norm",
    with_ref_meg=False,
    tol_kind="absolute",
    on_few_samples="warn",
):
    """Aid the transition away from raw.estimate_rank."""
    if picks is None:
        picks = _picks_to_idx(raw.info, picks, with_ref_meg=with_ref_meg)
    # conveniency wrapper to expose the expert "tol" option + scalings options
    return _estimate_rank_meeg_signals(
        raw[picks][0],
        pick_info(raw.info, picks),
        scalings,
        tol,
        False,
        tol_kind,
        log_ch_type=None,
        on_few_samples=on_few_samples,
    )


@fill_doc_static("info_not_none")
def _estimate_rank_meeg_signals(
    data,
    info,
    scalings,
    tol="auto",
    return_singular=False,
    tol_kind="absolute",
    log_ch_type=None,
    on_few_samples="warn",
):
    """Estimate rank for M/EEG data.

    Parameters
    ----------
    data : np.ndarray of float, shape(n_channels, n_samples)
        The M/EEG signals.
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    scalings : dict | ``'norm'`` | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e15, grad=1e13, eeg=1e6)

        If ``'norm'`` data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    tol : float | str
        Tolerance. See ``estimate_rank``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    tol_kind : str
        Tolerance kind. See ``estimate_rank``.
    on_few_samples : str
        Can be 'warn' (default), 'ignore', or 'raise' to control behavior when
        there are fewer samples than channels, which can lead to inaccurate rank
        estimates.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    picks_list = _picks_by_type(info)
    assert data.ndim == 2, data.shape
    n_channels, n_samples = data.shape
    if n_samples < n_channels:
        msg = (
            f"Too few samples ({n_samples=} is less than {n_channels=}), "
            "rank estimate may be unreliable"
        )
        _on_missing(on_few_samples, msg, "on_few_samples")
    with _scaled_array(data, picks_list, scalings):
        out = estimate_rank(
            data,
            tol=tol,
            norm=False,
            return_singular=return_singular,
            tol_kind=tol_kind,
        )
    rank = out[0] if isinstance(out, tuple) else out
    if log_ch_type is None:
        ch_type = " + ".join(list(zip(*picks_list))[0])
    else:
        ch_type = log_ch_type
    logger.info("    Estimated rank (%s): %d", ch_type, rank)
    return out


@_verbose_control
@fill_doc_static("info_not_none")
def _estimate_rank_meeg_cov(
    data,
    info,
    scalings,
    tol="auto",
    return_singular=False,
    *,
    log_ch_type=None,
    on_few_samples="warn",
    verbose=None,
):
    """Estimate rank of M/EEG covariance data, given the covariance.

    Parameters
    ----------
    data : np.ndarray of float, shape (n_channels, n_channels)
        The M/EEG covariance.
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    scalings : dict | 'norm' | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e12, grad=1e11, eeg=1e5)

        If 'norm' data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    tol : float | str
        Tolerance. See ``estimate_rank``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    on_few_samples : str
        Can be 'warn' (default), 'ignore', or 'raise' to control behavior when
        there are fewer samples than channels, which can lead to inaccurate rank
        estimates.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    picks_list = _picks_by_type(info, exclude=[])
    scalings = _handle_default("scalings_cov_rank", scalings)
    _apply_scaling_cov(data, picks_list, scalings)
    if data.shape[1] < data.shape[0]:
        msg = (
            "You've got fewer samples than channels, your "
            "rank estimate might be inaccurate."
        )
        _on_missing(on_few_samples, msg, "on_few_samples")
    out = estimate_rank(data, tol=tol, norm=False, return_singular=return_singular)
    rank = out[0] if isinstance(out, tuple) else out
    if log_ch_type is None:
        ch_type_ = " + ".join(list(zip(*picks_list))[0])
    else:
        ch_type_ = log_ch_type
    logger.info(f"    Estimated rank ({ch_type_}): {rank}")
    _undo_scaling_cov(data, picks_list, scalings)
    return out


@_verbose_control
def _get_rank_sss(
    inst, msg="You should use data-based rank estimate instead", verbose=None
):
    """Look up rank from SSS data.

    .. note::
        Throws an error if SSS has not been applied.

    Parameters
    ----------
    inst : instance of Raw, Epochs or Evoked, or Info
        Any MNE object with an .info attribute

    Returns
    -------
    rank : int
        The numerical rank as predicted by the number of SSS
        components.
    """
    # XXX this is too basic for movement compensated data
    # https://github.com/mne-tools/mne-python/issues/4676
    info = inst if isinstance(inst, Info) else inst.info
    del inst

    # Take the first record that actually holds an SSS expansion. MaxFilter 2.2
    # writes it first, but 3.0 emits an empty sss_info ahead of the populated
    # one, so we cannot just use proc_history[0].
    proc_info = [
        pp
        for pp in info.get("proc_history", [])
        if "in_order" in pp.get("max_info", {}).get("sss_info", {})
    ]
    if len(proc_info) == 0:
        raise ValueError(
            f'Could not find Maxfilter information in info["proc_history"]. {msg}'
        )
    if len(proc_info) > 1:
        logger.info("Found multiple SSS records. Using the first.")
    max_info = proc_info[0]["max_info"]
    inside = max_info["sss_info"]["in_order"]
    nfree = (inside + 1) ** 2 - 1
    nfree -= (
        len(max_info["sss_info"]["components"][:nfree])
        - max_info["sss_info"]["components"][:nfree].sum()
    )
    return nfree


def _info_rank(info, ch_type, picks, rank):
    if ch_type in ["meg", "mag", "grad"] and rank != "full":
        try:
            return _get_rank_sss(info)
        except ValueError:
            pass
    return len(picks)


def _compute_rank_int(inst, *args, **kwargs):
    """Wrap compute_rank but yield an int."""
    # XXX eventually we should unify how channel types are handled
    # so that we don't need to do this, or we do it everywhere.
    # Using pca=True in compute_whitener might help.
    return sum(compute_rank(inst, *args, on_few_samples="ignore", **kwargs).values())


@verbose_static("rank_none", "info", "tol_rank", "tol_kind_rank", "on_rank_mismatch")
def compute_rank(
    inst,
    rank=None,
    scalings=None,
    info=None,
    tol="auto",
    *,
    proj=True,
    tol_kind="absolute",
    on_rank_mismatch="ignore",
    on_few_samples=None,
    verbose=None,
):
    """Compute the rank of data or noise covariance.

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one. It operates on :term:`data channels` only.

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Covariance
        Raw measurements to compute the rank from or the covariance.
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

        The default is ``None``.
    scalings : dict | None
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale different channel types
        to comparable values.
    info : mne.Info | None
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
        Only necessary if ``inst`` is a :class:`mne.Covariance`
        object (since this does not provide ``inst.info``).
    tol : float | 'auto'
        Tolerance for singular values to consider non-zero in
        calculating the rank. The singular values are calculated
        in this method such that independent data are expected to
        have singular value around one. Can be 'auto' to use the
        same thresholding as :func:`scipy.linalg.orth`.
    proj : bool
        If True, all projs in ``inst`` and ``info`` will be applied or
        considered when ``rank=None`` or ``rank='info'``.
    tol_kind : str
        Can be: "absolute" (default) or "relative". Only used if ``tol`` is a
        float, because when ``tol`` is a string the mode is implicitly relative.
        After applying the chosen scale factors / normalization to the data,
        the singular values are computed, and the rank is then taken as:

        - ``'absolute'``
            The number of singular values ``s`` greater than ``tol``.
            This mode can fail if your data do not adhere to typical
            data scalings.
        - ``'relative'``
            The number of singular values ``s`` greater than ``tol * s.max()``.
            This mode can fail if you have one or more large components in the
            data (e.g., artifacts).

        .. versionadded:: 0.21.0
    on_rank_mismatch : str
        If an explicit MEG value is passed, what to do when it does not match
        an empirically computed rank (only used for covariances).
        Can be 'raise' to raise an error, 'warn' (default) to emit a warning, or
        'ignore' to ignore.

        .. versionadded:: 0.23
    on_few_samples : str | None
        Can be 'warn', 'ignore', or 'raise' to control behavior when
        there are fewer samples than channels, which can lead to inaccurate rank
        estimates. None (default) means "ignore" if ``inst`` is a
        :class:`mne.Covariance` or ``rank in ("info", "full")``, and "warn" otherwise.

        .. versionadded:: 1.11
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    rank : dict
        Estimated rank of the data for each channel type.
        To get the total rank, you can use ``sum(rank.values())``.

    Notes
    -----
    .. versionadded:: 0.18
    """
    return _compute_rank(
        inst=inst,
        rank=rank,
        scalings=scalings,
        info=info,
        tol=tol,
        proj=proj,
        tol_kind=tol_kind,
        on_rank_mismatch=on_rank_mismatch,
        on_few_samples=on_few_samples,
    )


@_verbose_control
def _compute_rank(
    inst,
    rank=None,
    scalings=None,
    info=None,
    *,
    tol="auto",
    proj=True,
    tol_kind="absolute",
    on_rank_mismatch="ignore",
    on_few_samples=None,
    log_ch_type=None,
    verbose=None,
):
    from .cov import Covariance
    from .epochs import BaseEpochs
    from .io import BaseRaw

    rank = _check_rank(rank)
    scalings = _handle_default("scalings_cov_rank", scalings)
    _check_on_missing(on_rank_mismatch, "on_rank_mismatch")

    if isinstance(inst, Covariance):
        inst_type = "covariance"
        if info is None:
            raise ValueError("info cannot be None if inst is a Covariance.")
        # Reset bads as it's already taken into account in inst['names']
        info = info.copy()
        info["bads"] = []
        inst = pick_channels_cov(
            inst,
            set(inst["names"]) & set(info["ch_names"]),
            exclude=info["bads"] + inst["bads"],
            ordered=False,
        )
        if info["ch_names"] != inst["names"]:
            info = pick_info(
                info, [info["ch_names"].index(name) for name in inst["names"]]
            )
    else:
        info = inst.info
        inst_type = "data"
    logger.info(f"Computing rank from {inst_type} with rank={repr(rank)}")

    _validate_type(rank, (str, dict, None), "rank")
    if isinstance(rank, str):  # string, either 'info' or 'full'
        rank_type = "info"
        info_type = rank
        rank = dict()
    else:  # None or dict
        rank_type = "estimated"
        if rank is None:
            rank = dict()

    if on_few_samples is None:
        if inst_type != "covariance" and rank_type == "estimated":
            on_few_samples = "warn"
        else:
            on_few_samples = "ignore"

    simple_info = _simplify_info(info)
    picks_list = _picks_by_type(info, meg_combined=True, ref_meg=False, exclude="bads")
    for ch_type, picks in picks_list:
        est_verbose = None
        if ch_type in rank:
            # raise an error of user-supplied rank exceeds number of channels
            if rank[ch_type] > len(picks):
                raise ValueError(
                    f"rank[{repr(ch_type)}]={rank[ch_type]} exceeds the number"
                    f" of channels ({len(picks)})"
                )
            # special case: if whitening a covariance, check the passed rank
            # against the estimated one
            est_verbose = False
            if not (
                on_rank_mismatch != "ignore"
                and rank_type == "estimated"
                and ch_type == "meg"
                and isinstance(inst, Covariance)
                and not inst["diag"]
            ):
                continue
        ch_names = [info["ch_names"][pick] for pick in picks]
        n_chan = len(ch_names)
        if proj:
            proj_op, n_proj, _ = make_projector(info["projs"], ch_names)
        else:
            proj_op, n_proj = None, 0
        if log_ch_type is None:
            ch_type_ = ch_type.upper()
        else:
            ch_type_ = log_ch_type
        if rank_type == "info":
            # use info
            this_rank = _info_rank(info, ch_type, picks, info_type)
            if info_type != "full":
                this_rank -= n_proj
                logger.info(
                    f"    {ch_type_}: rank {this_rank} after "
                    f"{n_proj} projector{_pl(n_proj)} applied to "
                    f"{n_chan} channel{_pl(n_chan)}"
                )
            else:
                logger.info(f"    {ch_type_}: rank {this_rank} from info")
        else:
            # Use empirical estimation
            assert rank_type == "estimated"
            if isinstance(inst, BaseRaw | BaseEpochs):
                if isinstance(inst, BaseRaw):
                    data = inst.get_data(picks, reject_by_annotation="omit")
                else:  # isinstance(inst, BaseEpochs):
                    data = np.concatenate(inst.get_data(picks), axis=1)
                if proj:
                    data = np.dot(proj_op, data)
                this_rank = _estimate_rank_meeg_signals(
                    data,
                    pick_info(simple_info, picks),
                    scalings,
                    tol,
                    False,
                    tol_kind,
                    log_ch_type=log_ch_type,
                    on_few_samples=on_few_samples,
                )
            else:
                assert isinstance(inst, Covariance)
                if inst["diag"]:
                    this_rank = (inst["data"][picks] > 0).sum() - n_proj
                else:
                    data = inst["data"][picks][:, picks]
                    if proj:
                        data = np.dot(np.dot(proj_op, data), proj_op.T)

                    this_rank, sing = _estimate_rank_meeg_cov(
                        data,
                        pick_info(simple_info, picks),
                        scalings,
                        tol,
                        return_singular=True,
                        log_ch_type=log_ch_type,
                        on_few_samples=on_few_samples,
                        verbose=est_verbose,
                    )
                    if ch_type in rank:
                        ratio = sing[this_rank - 1] / sing[rank[ch_type] - 1]
                        if ratio > 100:
                            msg = (
                                f"The passed rank[{repr(ch_type)}]="
                                f"{rank[ch_type]} exceeds the estimated rank "
                                f"of the noise covariance ({this_rank}) "
                                f"leading to a potential increase in "
                                f"noise during whitening by a factor "
                                f"of {np.sqrt(ratio):0.1g}. Ensure that the "
                                f"rank correctly corresponds to that of the "
                                f"given noise covariance matrix."
                            )
                            _on_missing(on_rank_mismatch, msg, "on_rank_mismatch")
                        continue
            this_info_rank = _info_rank(info, ch_type, picks, "info")
            logger.info(
                f"    {ch_type_}: rank {this_rank} computed from "
                f"{n_chan} data channel{_pl(n_chan)} with "
                f"{n_proj} projector{_pl(n_proj)}"
            )
            if this_rank > this_info_rank:
                warn(
                    "Something went wrong in the data-driven estimation of the data "
                    "rank as it exceeds the theoretical rank from the info "
                    f"({this_rank} > {this_info_rank}). Consider setting rank "
                    'to "auto" or setting it explicitly as an integer.'
                )
        if ch_type not in rank:
            rank[ch_type] = int(this_rank)

    return rank
