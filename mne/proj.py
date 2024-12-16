# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ._fiff.constants import FIFF
from ._fiff.open import fiff_open
from ._fiff.pick import _picks_to_idx, pick_types, pick_types_forward
from ._fiff.proj import (
    Projection,
    _has_eeg_average_ref_proj,
    _read_proj,
    _write_proj,
    make_eeg_average_ref_proj,
    make_projector,
)
from ._fiff.write import start_and_end_file
from .cov import _check_n_samples
from .epochs import Epochs
from .event import make_fixed_length_events
from .fixes import _safe_svd
from .forward import _subject_from_forward, convert_forward_solution, is_fixed_orient
from .parallel import parallel_func
from .source_estimate import _make_stc
from .utils import (
    _check_fname,
    _check_option,
    _validate_type,
    check_fname,
    logger,
    verbose,
)


@verbose
def read_proj(fname, *, verbose=None):
    """Read projections from a FIF file.

    Parameters
    ----------
    fname : path-like
        The name of file containing the projections vectors. It should end with
        ``-proj.fif`` or ``-proj.fif.gz``.
    %(verbose)s

    Returns
    -------
    projs : list of Projection
        The list of projection vectors.

    See Also
    --------
    write_proj
    """
    check_fname(
        fname, "projection", ("-proj.fif", "-proj.fif.gz", "_proj.fif", "_proj.fif.gz")
    )
    fname = _check_fname(fname, overwrite="read", must_exist=True)

    ff, tree, _ = fiff_open(fname)
    with ff as fid:
        projs = _read_proj(fid, tree)
    return projs


@verbose
def write_proj(fname, projs, *, overwrite=False, verbose=None):
    """Write projections to a FIF file.

    Parameters
    ----------
    fname : path-like
        The name of file containing the projections vectors. It should end with
        ``-proj.fif`` or ``-proj.fif.gz``.
    projs : list of Projection
        The list of projection vectors.
    %(overwrite)s

        .. versionadded:: 1.0
    %(verbose)s

        .. versionadded:: 1.0

    See Also
    --------
    read_proj
    """
    fname = _check_fname(fname, overwrite=overwrite)
    check_fname(
        fname, "projection", ("-proj.fif", "-proj.fif.gz", "_proj.fif", "_proj.fif.gz")
    )
    with start_and_end_file(fname) as fid:
        _write_proj(fid, projs)


@verbose
def _compute_proj(
    data, info, n_grad, n_mag, n_eeg, desc_prefix, meg="separate", verbose=None
):
    _validate_type(n_grad, "numeric", "n_grad", "float or int")
    _validate_type(n_mag, "numeric", "n_grad", "float or int")
    _validate_type(n_eeg, "numeric", "n_eeg", "float or int")
    for n_, n_name in ((n_grad, "n_grad"), (n_mag, "n_mag"), (n_eeg, "n_eeg")):
        if n_ < 0:
            raise ValueError(
                f"Argument '{n_name}' must be either a positive integer or a float "
                f"between 0 and 1. '{n_}' is invalid."
            )
    _check_option("meg", meg, ("separate", "combined"))
    if meg == "combined":
        if n_grad != n_mag:
            raise ValueError(
                f"n_grad ({n_grad}) must be equal to n_mag ({n_mag}) when using "
                "meg='combined'."
            )
        ch_types = ("meg", "eeg")
        n_vectors = (n_grad, n_eeg)
        kinds = ("meg", "eeg")
    else:
        ch_types = ("grad", "mag", "eeg")
        n_vectors = (n_grad, n_mag, n_eeg)
        kinds = ("planar", "axial", "eeg")

    projs = []
    for ch_type, n_vector, kind in zip(ch_types, n_vectors, kinds):
        # select channels to use
        try:
            idx = _picks_to_idx(info, ch_type, with_ref_meg=False, exclude="bads")
        except ValueError:
            logger.info("No channels '%s' found. Skipping.", ch_type)
            continue
        names = [info["ch_names"][k] for k in idx]

        data_ = data[idx][:, idx]  # data is the covariance matrix: U * S**2 * Ut
        U, Sexp2, _ = _safe_svd(data_, full_matrices=False)
        exp_var = Sexp2 / Sexp2.sum()

        # select vectors to use
        if 0 < n_vector < 1:
            n_vector = np.searchsorted(np.cumsum(exp_var), n_vector, "left") + 1
        U = U[:, :n_vector]
        exp_var = exp_var[:n_vector]

        # create projectors
        for k, (u, var) in enumerate(zip(U.T, exp_var)):
            proj_data = dict(
                col_names=names,
                row_names=None,
                data=u[np.newaxis, :],
                nrow=1,
                ncol=u.size,
            )
            desc = f"{kind}-{desc_prefix}-PCA-{k + 1:02d}"
            logger.info(f"Adding projection: {desc} (exp var={100 * float(var):0.1f}%)")
            proj = Projection(
                active=False,
                data=proj_data,
                desc=desc,
                kind=FIFF.FIFFV_PROJ_ITEM_FIELD,
                explained_var=var,
            )
            projs.append(proj)
    return projs


@verbose
def compute_proj_epochs(
    epochs,
    n_grad=2,
    n_mag=2,
    n_eeg=2,
    n_jobs=None,
    desc_prefix=None,
    meg="separate",
    verbose=None,
):
    """Compute SSP (signal-space projection) vectors on epoched data.

    %(compute_ssp)s

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs containing the artifact.
    %(n_proj_vectors)s
    %(n_jobs)s
        Number of jobs to use to compute covariance.
    desc_prefix : str | None
        The description prefix to use. If None, one will be created based on
        the event_id, tmin, and tmax.
    meg : str
        Can be ``'separate'`` (default) or ``'combined'`` to compute projectors
        for magnetometers and gradiometers separately or jointly.
        If ``'combined'``, ``n_mag == n_grad`` is required and the number of
        projectors computed for MEG will be ``n_mag``.

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    projs: list of Projection
        List of projection vectors.

    See Also
    --------
    compute_proj_raw, compute_proj_evoked
    """
    # compute data covariance
    data = _compute_cov_epochs(epochs, n_jobs)
    event_id = epochs.event_id
    if event_id is None or len(list(event_id.keys())) == 0:
        event_id = "0"
    elif len(event_id.keys()) == 1:
        event_id = str(list(event_id.values())[0])
    else:
        event_id = "Multiple-events"
    if desc_prefix is None:
        desc_prefix = f"{event_id}-{epochs.tmin:<.3f}-{epochs.tmax:<.3f}"
    return _compute_proj(data, epochs.info, n_grad, n_mag, n_eeg, desc_prefix, meg=meg)


def _compute_cov_epochs(epochs, n_jobs, *, log_drops=False):
    """Compute epochs covariance."""
    parallel, p_fun, n_jobs = parallel_func(np.dot, n_jobs)
    n_start = len(epochs.events)
    data = parallel(p_fun(e, e.T) for e in epochs)
    n_epochs = len(data)
    if n_epochs == 0:
        raise RuntimeError("No good epochs found")
    if log_drops:
        logger.info(f"Dropped {n_start - n_epochs}/{n_start} epochs")

    n_chan, n_samples = epochs.info["nchan"], len(epochs.times)
    _check_n_samples(n_samples * n_epochs, n_chan)
    data = sum(data)
    return data


@verbose
def compute_proj_evoked(
    evoked, n_grad=2, n_mag=2, n_eeg=2, desc_prefix=None, meg="separate", verbose=None
):
    """Compute SSP (signal-space projection) vectors on evoked data.

    %(compute_ssp)s

    Parameters
    ----------
    evoked : instance of Evoked
        The Evoked obtained by averaging the artifact.
    %(n_proj_vectors)s
    desc_prefix : str | None
        The description prefix to use. If None, one will be created based on
        tmin and tmax.

        .. versionadded:: 0.17
    meg : str
        Can be ``'separate'`` (default) or ``'combined'`` to compute projectors
        for magnetometers and gradiometers separately or jointly.
        If ``'combined'``, ``n_mag == n_grad`` is required and the number of
        projectors computed for MEG will be ``n_mag``.

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    projs : list of Projection
        List of projection vectors.

    See Also
    --------
    compute_proj_raw, compute_proj_epochs
    """
    data = np.dot(evoked.data, evoked.data.T)  # compute data covariance
    if desc_prefix is None:
        desc_prefix = f"{evoked.times[0]:<.3f}-{evoked.times[-1]:<.3f}"
    return _compute_proj(data, evoked.info, n_grad, n_mag, n_eeg, desc_prefix, meg=meg)


@verbose
def compute_proj_raw(
    raw,
    start=0,
    stop=None,
    duration=1,
    n_grad=2,
    n_mag=2,
    n_eeg=0,
    reject=None,
    flat=None,
    n_jobs=None,
    meg="separate",
    verbose=None,
):
    """Compute SSP (signal-space projection) vectors on continuous data.

    %(compute_ssp)s

    Parameters
    ----------
    raw : instance of Raw
        A raw object to use the data from.
    start : float
        Time (in seconds) to start computing SSP.
    stop : float | None
        Time (in seconds) to stop computing SSP. None will go to the end of the file.
    duration : float | None
        Duration (in seconds) to chunk data into for SSP
        If duration is ``None``, data will not be chunked.
    %(n_proj_vectors)s
    reject : dict | None
        Epoch PTP rejection threshold used if ``duration != None``. See `~mne.Epochs`.
    flat : dict | None
        Epoch flatness rejection threshold used if ``duration != None``. See
        `~mne.Epochs`.
    %(n_jobs)s
        Number of jobs to use to compute covariance.
    meg : str
        Can be ``'separate'`` (default) or ``'combined'`` to compute projectors
        for magnetometers and gradiometers separately or jointly.
        If ``'combined'``, ``n_mag == n_grad`` is required and the number of
        projectors computed for MEG will be ``n_mag``.

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    projs: list of Projection
        List of projection vectors.

    See Also
    --------
    compute_proj_epochs, compute_proj_evoked
    """
    if duration is not None:
        duration = np.round(duration * raw.info["sfreq"]) / raw.info["sfreq"]
        events = make_fixed_length_events(raw, 999, start, stop, duration)
        picks = pick_types(
            raw.info, meg=True, eeg=True, eog=True, ecg=True, emg=True, exclude="bads"
        )
        epochs = Epochs(
            raw,
            events,
            None,
            tmin=0.0,
            tmax=duration - 1.0 / raw.info["sfreq"],
            picks=picks,
            reject=reject,
            flat=flat,
            baseline=None,
            proj=False,
        )
        data = _compute_cov_epochs(epochs, n_jobs, log_drops=True)
        info = epochs.info
        if not stop:
            stop = raw.n_times / raw.info["sfreq"]
    else:
        # convert to sample indices
        start = max(raw.time_as_index(start)[0], 0)
        stop = raw.time_as_index(stop)[0] if stop else raw.n_times
        stop = min(stop, raw.n_times)
        data, times = raw[:, start:stop]
        _check_n_samples(stop - start, data.shape[0])
        data = np.dot(data, data.T)  # compute data covariance
        info = raw.info
        # convert back to times
        start = start / raw.info["sfreq"]
        stop = stop / raw.info["sfreq"]

    desc_prefix = f"Raw-{start:<.3f}-{stop:<.3f}"
    projs = _compute_proj(data, info, n_grad, n_mag, n_eeg, desc_prefix, meg=meg)
    return projs


@verbose
def sensitivity_map(
    fwd, projs=None, ch_type="grad", mode="fixed", exclude=(), *, verbose=None
):
    """Compute sensitivity map.

    Such maps are used to know how much sources are visible by a type
    of sensor, and how much projections shadow some sources.

    Parameters
    ----------
    fwd : Forward
        The forward operator.
    projs : list
        List of projection vectors.
    ch_type : ``'grad'`` | ``'mag'`` | ``'eeg'``
        The type of sensors to use.
    mode : str
        The type of sensitivity map computed. See manual. Should be ``'free'``,
        ``'fixed'``, ``'ratio'``, ``'radiality'``, ``'angle'``,
        ``'remaining'``, or ``'dampening'`` corresponding to the argument
        ``--map 1, 2, 3, 4, 5, 6, 7`` of the command ``mne_sensitivity_map``.
    exclude : list of str | str
        List of channels to exclude. If empty do not exclude any (default).
        If ``'bads'``, exclude channels in ``fwd['info']['bads']``.
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        The sensitivity map as a SourceEstimate or VolSourceEstimate instance
        for visualization.

    Notes
    -----
    When mode is ``'fixed'`` or ``'free'``, the sensitivity map is normalized
    by its maximum value.
    """
    # check strings
    _check_option("ch_type", ch_type, ["eeg", "grad", "mag"])
    _check_option(
        "mode",
        mode,
        ["free", "fixed", "ratio", "radiality", "angle", "remaining", "dampening"],
    )

    # check forward
    if is_fixed_orient(fwd, orig=True):
        raise ValueError("fwd should must be computed with free orientation")

    # limit forward (this will make a copy of the data for us)
    if ch_type == "eeg":
        fwd = pick_types_forward(fwd, meg=False, eeg=True, exclude=exclude)
    else:
        fwd = pick_types_forward(fwd, meg=ch_type, eeg=False, exclude=exclude)

    convert_forward_solution(
        fwd, surf_ori=True, force_fixed=False, copy=False, verbose=False
    )
    assert fwd["surf_ori"] and not is_fixed_orient(fwd)

    gain = fwd["sol"]["data"]

    # Make sure EEG has average
    if ch_type == "eeg":
        if projs is None or not _has_eeg_average_ref_proj(fwd["info"], projs=projs):
            eeg_ave = [make_eeg_average_ref_proj(fwd["info"])]
        else:
            eeg_ave = []
        projs = eeg_ave if projs is None else projs + eeg_ave

    # Construct the projector
    residual_types = ["angle", "remaining", "dampening"]
    if projs is not None:
        proj, ncomp, U = make_projector(
            projs, fwd["sol"]["row_names"], include_active=True
        )
        # do projection for most types
        if mode not in residual_types:
            gain = np.dot(proj, gain)
        elif ncomp == 0:
            raise RuntimeError(
                "No valid projectors found for channel type "
                f"{ch_type}, cannot compute {mode}"
            )
    # can only run the last couple methods if there are projectors
    elif mode in residual_types:
        raise ValueError(f"No projectors used, cannot compute {mode}")

    _, n_dipoles = gain.shape
    n_locations = n_dipoles // 3
    del n_dipoles
    sensitivity_map = np.empty(n_locations)

    for k in range(n_locations):
        gg = gain[:, 3 * k : 3 * (k + 1)]  # noqa: E203
        if mode != "fixed":
            s = _safe_svd(gg, full_matrices=False, compute_uv=False)
        if mode == "free":
            sensitivity_map[k] = s[0]
        else:
            gz = np.linalg.norm(gg[:, 2])  # the normal component
            if mode == "fixed":
                sensitivity_map[k] = gz
            elif mode == "ratio":
                sensitivity_map[k] = gz / s[0]
            elif mode == "radiality":
                sensitivity_map[k] = 1.0 - (gz / s[0])
            else:
                if mode == "angle":
                    co = np.linalg.norm(np.dot(gg[:, 2], U))
                    sensitivity_map[k] = co / gz
                else:
                    p = np.linalg.norm(np.dot(proj, gg[:, 2]))
                    if mode == "remaining":
                        sensitivity_map[k] = p / gz
                    elif mode == "dampening":
                        sensitivity_map[k] = 1.0 - p / gz
                    else:
                        raise ValueError(f"Unknown mode type (got {mode})")

    # only normalize fixed and free methods
    if mode in ["fixed", "free"]:
        sensitivity_map /= np.max(sensitivity_map)

    subject = _subject_from_forward(fwd)
    vertices = [s["vertno"] for s in fwd["src"]]
    return _make_stc(
        sensitivity_map[:, np.newaxis],
        vertices,
        fwd["src"].kind,
        tmin=0.0,
        tstep=1.0,
        subject=subject,
    )
