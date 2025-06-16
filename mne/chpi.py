"""Functions for fitting head positions with (c)HPI coils.

``compute_head_pos`` can be used to:

1. Drop coils whose GOF are below ``gof_limit``. If fewer than 3 coils
   remain, abandon fitting for the chunk.
2. Fit dev_head_t quaternion (using ``_fit_chpi_quat_subset``),
   iteratively dropping coils (as long as 3 remain) to find the best GOF
   (using ``_fit_chpi_quat``).
3. If fewer than 3 coils meet the ``dist_limit`` criteria following
   projection of the fitted device coil locations into the head frame,
   abandon fitting for the chunk.

The function ``filter_chpi`` uses the same linear model to filter cHPI
and (optionally) line frequencies from the data.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy
import itertools
from functools import partial

import numpy as np
from scipy.linalg import orth
from scipy.optimize import fmin_cobyla
from scipy.spatial.distance import cdist

from ._fiff.constants import FIFF
from ._fiff.meas_info import Info, _simplify_info
from ._fiff.pick import (
    _picks_to_idx,
    pick_channels,
    pick_channels_regexp,
    pick_info,
    pick_types,
)
from ._fiff.proj import Projection, setup_proj
from .channels.channels import _get_meg_system
from .cov import compute_whitener, make_ad_hoc_cov
from .dipole import _make_guesses
from .event import find_events
from .fixes import jit
from .forward import _concatenate_coils, _create_meg_coils, _magnetic_dipole_field_vec
from .io import BaseRaw
from .io.ctf.trans import _make_ctf_coord_trans_set
from .io.kit.constants import KIT
from .io.kit.kit import RawKIT as _RawKIT
from .preprocessing.maxwell import (
    _get_mf_picks_fix_mags,
    _prep_mf_coils,
    _regularize_out,
    _sss_basis,
)
from .transforms import (
    _angle_between_quats,
    _fit_matched_points,
    _quat_to_affine,
    als_ras_trans,
    apply_trans,
    invert_transform,
    quat_to_rot,
    rot_to_quat,
)
from .utils import (
    ProgressBar,
    _check_fname,
    _check_option,
    _on_missing,
    _pl,
    _validate_type,
    _verbose_safe_false,
    logger,
    use_log_level,
    verbose,
    warn,
)

# Eventually we should add:
#   hpicons
#   high-passing of data during fits
#   parsing cHPI coil information from acq pars, then to PSD if necessary


# ############################################################################
# Reading from text or FIF file


def read_head_pos(fname):
    """Read MaxFilter-formatted head position parameters.

    Parameters
    ----------
    fname : path-like
        The filename to read. This can be produced by e.g.,
        ``maxfilter -headpos <name>.pos``.

    Returns
    -------
    quats : array, shape (n_pos, 10)
        The position and quaternion parameters from cHPI fitting.
        See :func:`mne.chpi.compute_head_pos` for details on the columns.

    See Also
    --------
    write_head_pos
    head_pos_to_trans_rot_t

    Notes
    -----
    .. versionadded:: 0.12
    """
    _check_fname(fname, must_exist=True, overwrite="read")
    data = np.loadtxt(fname, skiprows=1)  # first line is header, skip it
    data.shape = (-1, 10)  # ensure it's the right size even if empty
    if np.isnan(data).any():  # make sure we didn't do something dumb
        raise RuntimeError(f"positions could not be read properly from {fname}")
    return data


def write_head_pos(fname, pos):
    """Write MaxFilter-formatted head position parameters.

    Parameters
    ----------
    fname : path-like
        The filename to write.
    pos : array, shape (n_pos, 10)
        The position and quaternion parameters from cHPI fitting.
        See :func:`mne.chpi.compute_head_pos` for details on the columns.

    See Also
    --------
    read_head_pos
    head_pos_to_trans_rot_t

    Notes
    -----
    .. versionadded:: 0.12
    """
    _check_fname(fname, overwrite=True)
    pos = np.array(pos, np.float64)
    if pos.ndim != 2 or pos.shape[1] != 10:
        raise ValueError(
            f"pos must be a 2D array of shape (N, 10), got shape {pos.shape}"
        )
    with open(fname, "wb") as fid:
        fid.write(
            " Time       q1       q2       q3       q4       q5       "
            "q6       g-value  error    velocity\n".encode("ASCII")
        )
        for p in pos:
            fmts = ["% 9.3f"] + ["% 8.5f"] * 9
            fid.write(((" " + " ".join(fmts) + "\n") % tuple(p)).encode("ASCII"))


def head_pos_to_trans_rot_t(quats):
    """Convert Maxfilter-formatted head position quaternions.

    Parameters
    ----------
    quats : ndarray, shape (n_pos, 10)
        MaxFilter-formatted position and quaternion parameters.
        See :func:`mne.chpi.read_head_pos` for details on the columns.

    Returns
    -------
    translation : ndarray, shape (n_pos, 3)
        Translations at each time point.
    rotation : ndarray, shape (n_pos, 3, 3)
        Rotations at each time point.
    t : ndarray, shape (n_pos,)
        The time points.

    See Also
    --------
    read_head_pos
    write_head_pos
    """
    t = quats[..., 0].copy()
    rotation = quat_to_rot(quats[..., 1:4])
    translation = quats[..., 4:7].copy()
    return translation, rotation, t


@verbose
def extract_chpi_locs_ctf(raw, verbose=None):
    r"""Extract cHPI locations from CTF data.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with CTF cHPI information.
    %(verbose)s

    Returns
    -------
    %(chpi_locs)s

    Notes
    -----
    CTF continuous head monitoring stores the x,y,z location (m) of each chpi
    coil as separate channels in the dataset:

    - ``HLC001[123]\\*`` - nasion
    - ``HLC002[123]\\*`` - lpa
    - ``HLC003[123]\\*`` - rpa

    This extracts these positions for use with
    :func:`~mne.chpi.compute_head_pos`.

    .. versionadded:: 0.20
    """
    # Pick channels corresponding to the cHPI positions
    hpi_picks = pick_channels_regexp(raw.info["ch_names"], "HLC00[123][123].*")

    # make sure we get 9 channels
    if len(hpi_picks) != 9:
        raise RuntimeError("Could not find all 9 cHPI channels")

    # get indices in alphabetical order
    sorted_picks = np.array(sorted(hpi_picks, key=lambda k: raw.info["ch_names"][k]))

    # make picks to match order of dig cardinial ident codes.
    # LPA (HPIC002[123]-*), NAS(HPIC001[123]-*), RPA(HPIC003[123]-*)
    hpi_picks = sorted_picks[[3, 4, 5, 0, 1, 2, 6, 7, 8]]
    del sorted_picks

    # process the entire run
    time_sl = slice(0, len(raw.times))
    chpi_data = raw[hpi_picks, time_sl][0]

    # transforms
    tmp_trans = _make_ctf_coord_trans_set(None, None)
    ctf_dev_dev_t = tmp_trans["t_ctf_dev_dev"]
    del tmp_trans

    # find indices where chpi locations change
    indices = [0]
    indices.extend(np.where(np.any(np.diff(chpi_data, axis=1), axis=0))[0] + 1)
    # data in channels are in ctf device coordinates (cm)
    rrs = chpi_data[:, indices].T.reshape(len(indices), 3, 3)  # m
    # map to mne device coords
    rrs = apply_trans(ctf_dev_dev_t, rrs)
    gofs = np.ones(rrs.shape[:2])  # not encoded, set all good
    moments = np.zeros(rrs.shape)  # not encoded, set all zero
    times = raw.times[indices] + raw._first_time
    return dict(rrs=rrs, gofs=gofs, times=times, moments=moments)


@verbose
def extract_chpi_locs_kit(raw, stim_channel="MISC 064", *, verbose=None):
    """Extract cHPI locations from KIT data.

    Parameters
    ----------
    raw : instance of RawKIT
        Raw data with KIT cHPI information.
    stim_channel : str
        The stimulus channel that encodes HPI measurement intervals.
    %(verbose)s

    Returns
    -------
    %(chpi_locs)s

    Notes
    -----
    .. versionadded:: 0.23
    """
    _validate_type(raw, (_RawKIT,), "raw")
    stim_chs = [
        raw.info["ch_names"][pick]
        for pick in pick_types(raw.info, stim=True, misc=True, ref_meg=False)
    ]
    _validate_type(stim_channel, str, "stim_channel")
    _check_option("stim_channel", stim_channel, stim_chs)
    idx = raw.ch_names.index(stim_channel)
    safe_false = _verbose_safe_false()
    events_on = find_events(
        raw, stim_channel=raw.ch_names[idx], output="onset", verbose=safe_false
    )[:, 0]
    events_off = find_events(
        raw, stim_channel=raw.ch_names[idx], output="offset", verbose=safe_false
    )[:, 0]
    bad = False
    if len(events_on) == 0 or len(events_off) == 0:
        bad = True
    else:
        if events_on[-1] > events_off[-1]:
            events_on = events_on[:-1]
        if events_on.size != events_off.size or not (events_on < events_off).all():
            bad = True
    if bad:
        raise RuntimeError(
            f"Could not find appropriate cHPI intervals from {stim_channel}"
        )
    # use the midpoint for times
    times = (events_on + events_off) / (2 * raw.info["sfreq"])
    del events_on, events_off
    # XXX remove first two rows. It is unknown currently if there is a way to
    # determine from the con file the number of initial pulses that
    # indicate the start of reading. The number is shown by opening the con
    # file in MEG160, but I couldn't find the value in the .con file, so it
    # may just always be 2...
    times = times[2:]
    n_coils = 5  # KIT always has 5 (hard-coded in reader)
    header = raw._raw_extras[0]["dirs"][KIT.DIR_INDEX_CHPI_DATA]
    dtype = np.dtype([("good", "<u4"), ("data", "<f8", (4,))])
    assert dtype.itemsize == header["size"], (dtype.itemsize, header["size"])
    all_data = list()
    for fname in raw.filenames:
        with open(fname) as fid:
            fid.seek(header["offset"])
            all_data.append(
                np.fromfile(fid, dtype, count=header["count"]).reshape(-1, n_coils)
            )
    data = np.concatenate(all_data)
    extra = ""
    if len(times) < len(data):
        extra = f", truncating to {len(times)} based on events"
    logger.info(f"Found {len(data)} cHPI measurement{_pl(len(data))}{extra}")
    data = data[: len(times)]
    # good is not currently used, but keep this in case we want it later
    # good = data['good'] == 1
    data = data["data"]
    rrs, gofs = data[:, :, :3], data[:, :, 3]
    rrs = apply_trans(als_ras_trans, rrs)
    moments = np.zeros(rrs.shape)  # not encoded, set all zero
    return dict(rrs=rrs, gofs=gofs, times=times, moments=moments)


# ############################################################################
# Estimate positions from data


@verbose
def get_chpi_info(info, on_missing="raise", verbose=None):
    """Retrieve cHPI information from the data.

    Parameters
    ----------
    %(info_not_none)s
    %(on_missing_chpi)s
    %(verbose)s

    Returns
    -------
    hpi_freqs : array, shape (n_coils,)
        The frequency used for each individual cHPI coil.
    hpi_pick : int | None
        The index of the ``STIM`` channel containing information about when
        which cHPI coils were switched on.
    hpi_on : array, shape (n_coils,)
        The values coding for the "on" state of each individual cHPI coil.

    Notes
    -----
    .. versionadded:: 0.24
    """
    _validate_type(item=info, item_name="info", types=Info)
    _check_option(
        parameter="on_missing",
        value=on_missing,
        allowed_values=["ignore", "raise", "warn"],
    )

    if len(info["hpi_meas"]) == 0 or (
        "coil_freq" not in info["hpi_meas"][0]["hpi_coils"][0]
    ):
        _on_missing(
            on_missing,
            msg="No appropriate cHPI information found in "
            'info["hpi_meas"] and info["hpi_subsystem"]',
        )
        return np.empty(0), None, np.empty(0)

    hpi_coils = sorted(
        info["hpi_meas"][-1]["hpi_coils"], key=lambda x: x["number"]
    )  # ascending (info) order

    # get frequencies
    hpi_freqs = np.array([float(x["coil_freq"]) for x in hpi_coils])
    logger.info(
        f"Using {len(hpi_freqs)} HPI coils: {' '.join(str(int(s)) for s in hpi_freqs)} "
        "Hz"
    )

    # how cHPI active is indicated in the FIF file
    hpi_sub = info["hpi_subsystem"]
    hpi_pick = None  # there is no pick!
    if hpi_sub is not None:
        if "event_channel" in hpi_sub:
            hpi_pick = pick_channels(
                info["ch_names"], [hpi_sub["event_channel"]], ordered=False
            )
            hpi_pick = hpi_pick[0] if len(hpi_pick) > 0 else None
        # grab codes indicating a coil is active
        hpi_on = [coil["event_bits"][0] for coil in hpi_sub["hpi_coils"]]
        # not all HPI coils will actually be used
        hpi_on = np.array([hpi_on[hc["number"] - 1] for hc in hpi_coils])
        # mask for coils that may be active
        hpi_mask = np.array([event_bit != 0 for event_bit in hpi_on])
        hpi_on = hpi_on[hpi_mask]
        hpi_freqs = hpi_freqs[hpi_mask]
    else:
        hpi_on = np.zeros(len(hpi_freqs))

    return hpi_freqs, hpi_pick, hpi_on


@verbose
def _get_hpi_initial_fit(info, adjust=False, verbose=None):
    """Get HPI fit locations from raw."""
    if info["hpi_results"] is None or len(info["hpi_results"]) == 0:
        raise RuntimeError("no initial cHPI head localization performed")

    hpi_result = info["hpi_results"][-1]
    hpi_dig = sorted(
        [d for d in info["dig"] if d["kind"] == FIFF.FIFFV_POINT_HPI],
        key=lambda x: x["ident"],
    )  # ascending (dig) order
    if len(hpi_dig) == 0:  # CTF data, probably
        msg = "HPIFIT: No HPI dig points, using hpifit result"
        hpi_dig = sorted(hpi_result["dig_points"], key=lambda x: x["ident"])
        if all(
            d["coord_frame"] in (FIFF.FIFFV_COORD_DEVICE, FIFF.FIFFV_COORD_UNKNOWN)
            for d in hpi_dig
        ):
            # Do not modify in place!
            hpi_dig = copy.deepcopy(hpi_dig)
            msg += " transformed to head coords"
            for dig in hpi_dig:
                dig.update(
                    r=apply_trans(info["dev_head_t"], dig["r"]),
                    coord_frame=FIFF.FIFFV_COORD_HEAD,
                )
        logger.debug(msg)

    # zero-based indexing, dig->info
    # CTF does not populate some entries so we use .get here
    pos_order = hpi_result.get("order", np.arange(1, len(hpi_dig) + 1)) - 1
    used = hpi_result.get("used", np.arange(len(hpi_dig)))
    dist_limit = hpi_result.get("dist_limit", 0.005)
    good_limit = hpi_result.get("good_limit", 0.98)
    goodness = hpi_result.get("goodness", np.ones(len(hpi_dig)))

    # this shouldn't happen, eventually we could add the transforms
    # necessary to put it in head coords
    if not all(d["coord_frame"] == FIFF.FIFFV_COORD_HEAD for d in hpi_dig):
        raise RuntimeError("cHPI coordinate frame incorrect")
    # Give the user some info
    logger.info(
        f"HPIFIT: {len(pos_order)} coils digitized in order "
        f"{' '.join(str(o + 1) for o in pos_order)}"
    )
    logger.debug(
        f"HPIFIT: {len(used)} coils accepted: {' '.join(str(h) for h in used)}"
    )
    hpi_rrs = np.array([d["r"] for d in hpi_dig])[pos_order]
    assert len(hpi_rrs) >= 3

    # Fitting errors
    hpi_rrs_fit = sorted(
        [d for d in info["hpi_results"][-1]["dig_points"]], key=lambda x: x["ident"]
    )
    hpi_rrs_fit = np.array([d["r"] for d in hpi_rrs_fit])
    # hpi_result['dig_points'] are in FIFFV_COORD_UNKNOWN coords, but this
    # is probably a misnomer because it should be FIFFV_COORD_DEVICE for this
    # to work
    assert hpi_result["coord_trans"]["to"] == FIFF.FIFFV_COORD_HEAD
    hpi_rrs_fit = apply_trans(hpi_result["coord_trans"]["trans"], hpi_rrs_fit)
    if "moments" in hpi_result:
        logger.debug(f"Hpi coil moments {hpi_result['moments'].shape[::-1]}:")
        for moment in hpi_result["moments"]:
            logger.debug(f"{moment[0]:g} {moment[1]:g} {moment[2]:g}")
    errors = np.linalg.norm(hpi_rrs - hpi_rrs_fit, axis=1)
    logger.debug(f"HPIFIT errors:  {', '.join(f'{1000 * e:0.1f}' for e in errors)} mm.")
    if errors.sum() < len(errors) * dist_limit:
        logger.info("HPI consistency of isotrak and hpifit is OK.")
    elif not adjust and (len(used) == len(hpi_dig)):
        warn("HPI consistency of isotrak and hpifit is poor.")
    else:
        # adjust HPI coil locations using the hpifit transformation
        for hi, (err, r_fit) in enumerate(zip(errors, hpi_rrs_fit)):
            # transform to head frame
            d = 1000 * err
            if not adjust:
                if err >= dist_limit:
                    warn(
                        f"Discrepancy of HPI coil {hi + 1} isotrak and hpifit is "
                        f"{d:.1f} mm!"
                    )
            elif hi + 1 not in used:
                if goodness[hi] >= good_limit:
                    logger.info(
                        f"Note: HPI coil {hi + 1} isotrak is adjusted by {d:.1f} mm!"
                    )
                    hpi_rrs[hi] = r_fit
                else:
                    warn(
                        f"Discrepancy of HPI coil {hi + 1} isotrak and hpifit of "
                        f"{d:.1f} mm was not adjusted!"
                    )
    logger.debug(
        f"HP fitting limits: err = {1000 * dist_limit:.1f} mm, gval = {good_limit:.3f}."
    )

    return hpi_rrs.astype(float)


def _magnetic_dipole_objective(
    x, B, B2, coils, whitener, too_close, return_moment=False
):
    """Project data onto right eigenvectors of whitened forward."""
    fwd = _magnetic_dipole_field_vec(x[np.newaxis], coils, too_close)
    out, u, s, one = _magnetic_dipole_delta(fwd, whitener, B, B2)
    if return_moment:
        one /= s
        Q = np.dot(one, u.T)
        out = (out, Q)
    return out


@jit()
def _magnetic_dipole_delta(fwd, whitener, B, B2):
    # Here we use .T to get whitener to Fortran order, which speeds things up
    fwd = np.dot(fwd, whitener.T)
    u, s, v = np.linalg.svd(fwd, full_matrices=False)
    one = np.dot(v, B)
    Bm2 = np.dot(one, one)
    return B2 - Bm2, u, s, one


def _magnetic_dipole_delta_multi(whitened_fwd_svd, B, B2):
    # Here we use .T to get whitener to Fortran order, which speeds things up
    one = np.matmul(whitened_fwd_svd, B)
    Bm2 = np.sum(one * one, axis=1)
    return B2 - Bm2


def _fit_magnetic_dipole(B_orig, x0, too_close, whitener, coils, guesses):
    """Fit a single bit of data (x0 = pos)."""
    B = np.dot(whitener, B_orig)
    B2 = np.dot(B, B)
    objective = partial(
        _magnetic_dipole_objective,
        B=B,
        B2=B2,
        coils=coils,
        whitener=whitener,
        too_close=too_close,
    )
    if guesses is not None:
        res0 = objective(x0)
        res = _magnetic_dipole_delta_multi(guesses["whitened_fwd_svd"], B, B2)
        assert res.shape == (guesses["rr"].shape[0],)
        idx = np.argmin(res)
        if res[idx] < res0:
            x0 = guesses["rr"][idx]
    x = fmin_cobyla(objective, x0, (), rhobeg=1e-3, rhoend=1e-5, disp=False)
    gof, moment = objective(x, return_moment=True)
    gof = 1.0 - gof / B2
    return x, gof, moment


@jit()
def _chpi_objective(x, coil_dev_rrs, coil_head_rrs):
    """Compute objective function."""
    d = np.dot(coil_dev_rrs, quat_to_rot(x[:3]).T)
    d += x[3:]
    d -= coil_head_rrs
    d *= d
    return d.sum()


def _fit_chpi_quat(coil_dev_rrs, coil_head_rrs):
    """Fit rotation and translation (quaternion) parameters for cHPI coils."""
    denom = np.linalg.norm(coil_head_rrs - np.mean(coil_head_rrs, axis=0))
    denom *= denom
    # We could try to solve it the analytic way:
    # XXX someday we could choose to weight these points by their goodness
    # of fit somehow.
    quat = _fit_matched_points(coil_dev_rrs, coil_head_rrs)[0]
    gof = 1.0 - _chpi_objective(quat, coil_dev_rrs, coil_head_rrs) / denom
    return quat, gof


def _fit_coil_order_dev_head_trans(dev_pnts, head_pnts, bias=True):
    """Compute Device to Head transform allowing for permutiatons of points."""
    id_quat = np.zeros(6)
    best_order = None
    best_g = -999
    best_quat = id_quat
    for this_order in itertools.permutations(np.arange(len(head_pnts))):
        head_pnts_tmp = head_pnts[np.array(this_order)]
        this_quat, g = _fit_chpi_quat(dev_pnts, head_pnts_tmp)
        assert np.linalg.det(quat_to_rot(this_quat[:3])) > 0.9999
        if bias:
            # For symmetrical arrangements, flips can produce roughly
            # equivalent g values. To avoid this, heavily penalize
            # large rotations.
            rotation = _angle_between_quats(this_quat[:3], np.zeros(3))
            check_g = g * max(1.0 - rotation / np.pi, 0) ** 0.25
        else:
            check_g = g
        if check_g > best_g:
            out_g = g
            best_g = check_g
            best_order = np.array(this_order)
            best_quat = this_quat

    # Convert Quaterion to transform
    dev_head_t = _quat_to_affine(best_quat)
    return dev_head_t, best_order, out_g


@verbose
def _setup_hpi_amplitude_fitting(
    info, t_window, remove_aliased=False, ext_order=1, allow_empty=False, verbose=None
):
    """Generate HPI structure for HPI localization."""
    # grab basic info.
    on_missing = "raise" if not allow_empty else "ignore"
    hpi_freqs, hpi_pick, hpi_ons = get_chpi_info(info, on_missing=on_missing)

    # check for maxwell filtering
    for ent in info["proc_history"]:
        for key in ("sss_info", "max_st"):
            if len(ent["max_info"]["sss_info"]) > 0:
                warn(
                    "Fitting cHPI amplitudes after Maxwell filtering may not work, "
                    "consider fitting on the original data."
                )
                break

    _validate_type(t_window, (str, "numeric"), "t_window")
    if info["line_freq"] is not None:
        line_freqs = np.arange(
            info["line_freq"], info["sfreq"] / 3.0, info["line_freq"]
        )
    else:
        line_freqs = np.zeros([0])
    lfs = " ".join(f"{lf}" for lf in line_freqs)
    logger.info(f"Line interference frequencies: {lfs} Hz")
    # worry about resampled/filtered data.
    # What to do e.g. if Raw has been resampled and some of our
    # HPI freqs would now be aliased
    highest = info.get("lowpass")
    highest = info["sfreq"] / 2.0 if highest is None else highest
    keepers = hpi_freqs <= highest
    if remove_aliased:
        hpi_freqs = hpi_freqs[keepers]
        hpi_ons = hpi_ons[keepers]
    elif not keepers.all():
        raise RuntimeError(
            f"Found HPI frequencies {hpi_freqs[~keepers].tolist()} above the lowpass ("
            f"or Nyquist) frequency {highest:0.1f}"
        )
    # calculate optimal window length.
    if isinstance(t_window, str):
        _check_option("t_window", t_window, ("auto",), extra="if a string")
        if len(hpi_freqs):
            all_freqs = np.concatenate((hpi_freqs, line_freqs))
            delta_freqs = np.diff(np.unique(all_freqs))
            t_window = max(5.0 / all_freqs.min(), 1.0 / delta_freqs.min())
        else:
            t_window = 0.2
    t_window = float(t_window)
    if t_window <= 0:
        raise ValueError(f"t_window ({t_window}) must be > 0")
    logger.info(f"Using time window: {1000 * t_window:0.1f} ms")
    window_nsamp = np.rint(t_window * info["sfreq"]).astype(int)
    model = _setup_hpi_glm(hpi_freqs, line_freqs, info["sfreq"], window_nsamp)
    inv_model = np.linalg.pinv(model)
    inv_model_reord = _reorder_inv_model(inv_model, len(hpi_freqs))
    proj, proj_op, meg_picks = _setup_ext_proj(info, ext_order)
    # include mag and grad picks separately, for SNR computations
    mag_subpicks = _picks_to_idx(info, "mag", allow_empty=True)
    mag_subpicks = np.searchsorted(meg_picks, mag_subpicks)
    grad_subpicks = _picks_to_idx(info, "grad", allow_empty=True)
    grad_subpicks = np.searchsorted(meg_picks, grad_subpicks)
    # Set up magnetic dipole fits
    hpi = dict(
        meg_picks=meg_picks,
        mag_subpicks=mag_subpicks,
        grad_subpicks=grad_subpicks,
        hpi_pick=hpi_pick,
        model=model,
        inv_model=inv_model,
        t_window=t_window,
        inv_model_reord=inv_model_reord,
        on=hpi_ons,
        n_window=window_nsamp,
        proj=proj,
        proj_op=proj_op,
        freqs=hpi_freqs,
        line_freqs=line_freqs,
    )
    return hpi


def _setup_hpi_glm(hpi_freqs, line_freqs, sfreq, window_nsamp):
    """Initialize a general linear model for HPI amplitude estimation."""
    slope = np.linspace(-0.5, 0.5, window_nsamp)[:, np.newaxis]
    radians_per_sec = 2 * np.pi * np.arange(window_nsamp, dtype=float) / sfreq
    f_t = hpi_freqs[np.newaxis, :] * radians_per_sec[:, np.newaxis]
    l_t = line_freqs[np.newaxis, :] * radians_per_sec[:, np.newaxis]
    model = [
        np.sin(f_t),
        np.cos(f_t),  # hpi freqs
        np.sin(l_t),
        np.cos(l_t),  # line freqs
        slope,
        np.ones_like(slope),
    ]  # drift, DC
    return np.hstack(model)


@jit()
def _reorder_inv_model(inv_model, n_freqs):
    # Reorder for faster computation
    idx = np.arange(2 * n_freqs).reshape(2, n_freqs).T.ravel()
    return inv_model[idx]


def _setup_ext_proj(info, ext_order):
    meg_picks = pick_types(info, meg=True, eeg=False, exclude="bads")
    info = pick_info(_simplify_info(info), meg_picks)  # makes a copy
    _, _, _, _, mag_or_fine = _get_mf_picks_fix_mags(
        info, int_order=0, ext_order=ext_order, ignore_ref=True, verbose="error"
    )
    mf_coils = _prep_mf_coils(info, verbose="error")
    ext = _sss_basis(
        dict(origin=(0.0, 0.0, 0.0), int_order=0, ext_order=ext_order), mf_coils
    ).T
    out_removes = _regularize_out(0, 1, mag_or_fine, [])
    ext = ext[~np.isin(np.arange(len(ext)), out_removes)]
    ext = orth(ext.T).T
    assert ext.shape[1] == len(meg_picks)
    proj = Projection(
        kind=FIFF.FIFFV_PROJ_ITEM_HOMOG_FIELD,
        desc="SSS",
        active=False,
        data=dict(
            data=ext, ncol=info["nchan"], col_names=info["ch_names"], nrow=len(ext)
        ),
    )
    with info._unlock():
        info["projs"] = [proj]
    proj_op, _ = setup_proj(
        info, add_eeg_ref=False, activate=False, verbose=_verbose_safe_false()
    )
    assert proj_op.shape == (len(meg_picks),) * 2
    return proj, proj_op, meg_picks


def _time_prefix(fit_time):
    """Format log messages."""
    return (f"    t={fit_time:0.3f}:").ljust(17)


def _fit_chpi_amplitudes(raw, time_sl, hpi, snr=False):
    """Fit amplitudes for each channel from each of the N cHPI sinusoids.

    Returns
    -------
    sin_fit : ndarray, shape (n_freqs, n_channels)
        The sin amplitudes matching each cHPI frequency.
        Will be all nan if this time window should be skipped.
    snr : ndarray, shape (n_freqs, 2)
        Estimated SNR for this window, separately for mag and grad channels.
    """
    # No need to detrend the data because our model has a DC term
    with use_log_level(False):
        # loads good channels
        this_data = raw[hpi["meg_picks"], time_sl][0]

    # which HPI coils to use
    if hpi["hpi_pick"] is not None:
        with use_log_level(False):
            # loads hpi_stim channel
            chpi_data = raw[hpi["hpi_pick"], time_sl][0]

        ons = (np.round(chpi_data).astype(np.int64) & hpi["on"][:, np.newaxis]).astype(
            bool
        )
        n_on = ons.all(axis=-1).sum(axis=0)
        if not (n_on >= 3).all():
            return None
    if snr:
        return _fast_fit_snr(
            this_data,
            len(hpi["freqs"]),
            hpi["model"],
            hpi["inv_model"],
            hpi["mag_subpicks"],
            hpi["grad_subpicks"],
        )
    return _fast_fit(
        this_data,
        hpi["proj_op"],
        len(hpi["freqs"]),
        hpi["model"],
        hpi["inv_model_reord"],
    )


@jit()
def _fast_fit(this_data, proj, n_freqs, model, inv_model_reord):
    # first or last window
    if this_data.shape[1] != model.shape[0]:
        model = model[: this_data.shape[1]]
        inv_model_reord = _reorder_inv_model(np.linalg.pinv(model), n_freqs)
    proj_data = proj @ this_data
    X = inv_model_reord @ proj_data.T

    sin_fit = np.zeros((n_freqs, X.shape[1]))
    for fi in range(n_freqs):
        # use SVD across all sensors to estimate the sinusoid phase
        u, s, vt = np.linalg.svd(X[2 * fi : 2 * fi + 2], full_matrices=False)
        # the first component holds the predominant phase direction
        # (so ignore the second, effectively doing s[1] = 0):
        sin_fit[fi] = vt[0] * s[0]
    return sin_fit


@jit()
def _fast_fit_snr(this_data, n_freqs, model, inv_model, mag_picks, grad_picks):
    # first or last window
    if this_data.shape[1] != model.shape[0]:
        model = model[: this_data.shape[1]]
        inv_model = np.linalg.pinv(model)
    coefs = np.ascontiguousarray(inv_model) @ np.ascontiguousarray(this_data.T)
    # average sin & cos terms (special property of sinusoids: power=A²/2)
    hpi_power = (coefs[:n_freqs] ** 2 + coefs[n_freqs : (2 * n_freqs)] ** 2) / 2
    resid = this_data - np.ascontiguousarray((model @ coefs).T)
    # can't use np.var(..., axis=1) with Numba, so do it manually:
    resid_mean = np.atleast_2d(resid.sum(axis=1) / resid.shape[1]).T
    squared_devs = np.abs(resid - resid_mean) ** 2
    resid_var = squared_devs.sum(axis=1) / squared_devs.shape[1]
    # output array will be (n_freqs, 3 * n_ch_types). The 3 columns for each
    # channel type are the SNR, the mean cHPI power and the residual variance
    # (which gets tiled to shape (n_freqs,) because it's a scalar).
    snrs = np.empty((n_freqs, 0))
    # average power & compute residual variance separately for each ch type
    for _picks in (mag_picks, grad_picks):
        if len(_picks):
            avg_power = hpi_power[:, _picks].sum(axis=1) / len(_picks)
            avg_resid = resid_var[_picks].mean() * np.ones(n_freqs)
            snr = 10 * np.log10(avg_power / avg_resid)
            snrs = np.hstack((snrs, np.stack((snr, avg_power, avg_resid), 1)))
    return snrs


def _check_chpi_param(chpi_, name):
    if name == "chpi_locs":
        want_ndims = dict(times=1, rrs=3, moments=3, gofs=2)
        extra_keys = list()
    else:
        assert name == "chpi_amplitudes"
        want_ndims = dict(times=1, slopes=3)
        extra_keys = ["proj"]

    _validate_type(chpi_, dict, name)
    want_keys = list(want_ndims.keys()) + extra_keys
    if set(want_keys).symmetric_difference(chpi_):
        raise ValueError(
            f"{name} must be a dict with entries {want_keys}, got "
            f"{sorted(chpi_.keys())}"
        )
    n_times = None
    for key, want_ndim in want_ndims.items():
        key_str = f"{name}[{key}]"
        val = chpi_[key]
        _validate_type(val, np.ndarray, key_str)
        shape = val.shape
        if val.ndim != want_ndim:
            raise ValueError(f"{key_str} must have ndim={want_ndim}, got {val.ndim}")
        if n_times is None and key != "proj":
            n_times = shape[0]
        if n_times != shape[0] and key != "proj":
            raise ValueError(
                f"{name} have inconsistent number of time points in {want_keys}"
            )
    if name == "chpi_locs":
        n_coils = chpi_["rrs"].shape[1]
        for key in ("gofs", "moments"):
            val = chpi_[key]
            if val.shape[1] != n_coils:
                raise ValueError(
                    f'chpi_locs["rrs"] had values for {n_coils} coils but '
                    f'chpi_locs["{key}"] had values for {val.shape[1]} coils'
                )
        for key in ("rrs", "moments"):
            val = chpi_[key]
            if val.shape[2] != 3:
                raise ValueError(
                    f'chpi_locs["{key}"].shape[2] must be 3, got shape {shape}'
                )
    else:
        assert name == "chpi_amplitudes"
        slopes, proj = chpi_["slopes"], chpi_["proj"]
        _validate_type(proj, Projection, 'chpi_amplitudes["proj"]')
        n_ch = len(proj["data"]["col_names"])
        if slopes.shape[0] != n_times or slopes.shape[2] != n_ch:
            raise ValueError(
                f"slopes must have shape[0]=={n_times} and shape[2]=={n_ch}, got shape "
                f"{slopes.shape}"
            )


@verbose
def compute_head_pos(
    info, chpi_locs, dist_limit=0.005, gof_limit=0.98, adjust_dig=False, verbose=None
):
    """Compute time-varying head positions.

    Parameters
    ----------
    %(info_not_none)s
    %(chpi_locs)s
        Typically obtained by :func:`~mne.chpi.compute_chpi_locs` or
        :func:`~mne.chpi.extract_chpi_locs_ctf`.
    dist_limit : float
        Minimum distance (m) to accept for coil position fitting.
    gof_limit : float
        Minimum goodness of fit to accept for each coil.
    %(adjust_dig_chpi)s
    %(verbose)s

    Returns
    -------
    quats : ndarray, shape (n_pos, 10)
        MaxFilter-formatted head position parameters. The columns correspond to
        ``[t, q1, q2, q3, x, y, z, gof, err, v]`` for each time point.

    See Also
    --------
    compute_chpi_locs
    extract_chpi_locs_ctf
    read_head_pos
    write_head_pos

    Notes
    -----
    .. versionadded:: 0.20
    """
    _check_chpi_param(chpi_locs, "chpi_locs")
    _validate_type(info, Info, "info")
    hpi_dig_head_rrs = _get_hpi_initial_fit(info, adjust=adjust_dig, verbose="error")
    n_coils = len(hpi_dig_head_rrs)
    coil_dev_rrs = apply_trans(invert_transform(info["dev_head_t"]), hpi_dig_head_rrs)
    dev_head_t = info["dev_head_t"]["trans"]
    pos_0 = dev_head_t[:3, 3]
    last = dict(
        quat_fit_time=-0.1,
        coil_dev_rrs=coil_dev_rrs,
        quat=np.concatenate([rot_to_quat(dev_head_t[:3, :3]), dev_head_t[:3, 3]]),
    )
    del coil_dev_rrs
    quats = []
    for fit_time, this_coil_dev_rrs, g_coils in zip(
        *(chpi_locs[key] for key in ("times", "rrs", "gofs"))
    ):
        use_idx = np.where(g_coils >= gof_limit)[0]

        #
        # 1. Check number of good ones
        #
        if len(use_idx) < 3:
            gofs = ", ".join(f"{g:0.2f}" for g in g_coils)
            warn(
                f"{_time_prefix(fit_time)}{len(use_idx)}/{n_coils} "
                "good HPI fits, cannot determine the transformation "
                f"({gofs} GOF)!"
            )
            continue

        #
        # 2. Fit the head translation and rotation params (minimize error
        #    between coil positions and the head coil digitization
        #    positions) iteratively using different sets of coils.
        #
        this_quat, g, use_idx = _fit_chpi_quat_subset(
            this_coil_dev_rrs, hpi_dig_head_rrs, use_idx
        )

        #
        # 3. Stop if < 3 good
        #

        # Convert quaterion to transform
        this_dev_head_t = _quat_to_affine(this_quat)
        est_coil_head_rrs = apply_trans(this_dev_head_t, this_coil_dev_rrs)
        errs = np.linalg.norm(hpi_dig_head_rrs - est_coil_head_rrs, axis=1)
        n_good = ((g_coils >= gof_limit) & (errs < dist_limit)).sum()
        if n_good < 3:
            warn_str = ", ".join(
                f"{1000 * e:0.1f}::{g:0.2f}" for e, g in zip(errs, g_coils)
            )
            warn(
                f"{_time_prefix(fit_time)}{n_good}/{n_coils} good HPI fits, cannot "
                f"determine the transformation ({warn_str} mm/GOF)!"
            )
            continue

        # velocities, in device coords, of HPI coils
        dt = fit_time - last["quat_fit_time"]
        vs = tuple(
            1000.0
            * np.linalg.norm(last["coil_dev_rrs"] - this_coil_dev_rrs, axis=1)
            / dt
        )
        logger.info(
            _time_prefix(fit_time)
            + (
                "%s/%s good HPI fits, movements [mm/s] = "
                + " / ".join(["% 8.1f"] * n_coils)
            )
            % ((n_good, n_coils) + vs)
        )

        # Log results
        # MaxFilter averages over a 200 ms window for display, but we don't
        for ii in range(n_coils):
            if ii in use_idx:
                start, end = " ", "/"
            else:
                start, end = "(", ")"
            log_str = (
                "    "
                + start
                + "{0:6.1f} {1:6.1f} {2:6.1f} / "
                + "{3:6.1f} {4:6.1f} {5:6.1f} / "
                + "g = {6:0.3f} err = {7:4.1f} "
                + end
            )
            vals = np.concatenate(
                (
                    1000 * hpi_dig_head_rrs[ii],
                    1000 * est_coil_head_rrs[ii],
                    [g_coils[ii], 1000 * errs[ii]],
                )
            )
            if len(use_idx) >= 3:
                if ii <= 2:
                    log_str += "{8:6.3f} {9:6.3f} {10:6.3f}"
                    vals = np.concatenate((vals, this_dev_head_t[ii, :3]))
                elif ii == 3:
                    log_str += "{8:6.1f} {9:6.1f} {10:6.1f}"
                    vals = np.concatenate((vals, this_dev_head_t[:3, 3] * 1000.0))
            logger.debug(log_str.format(*vals))

        # resulting errors in head coil positions
        d = np.linalg.norm(last["quat"][3:] - this_quat[3:])  # m
        r = _angle_between_quats(last["quat"][:3], this_quat[:3]) / dt
        v = d / dt  # m/s
        d = 100 * np.linalg.norm(this_quat[3:] - pos_0)  # dis from 1st
        logger.debug(
            f"    #t = {fit_time:0.3f}, #e = {100 * errs.mean():0.2f} cm, #g = {g:0.3f}"
            f", #v = {100 * v:0.2f} cm/s, #r = {r:0.2f} rad/s, #d = {d:0.2f} cm"
        )
        q_rep = " ".join(f"{qq:8.5f}" for qq in this_quat)
        logger.debug(f"    #t = {fit_time:0.3f}, #q = {q_rep}")

        quats.append(
            np.concatenate(([fit_time], this_quat, [g], [errs[use_idx].mean()], [v]))
        )
        last["quat_fit_time"] = fit_time
        last["quat"] = this_quat
        last["coil_dev_rrs"] = this_coil_dev_rrs
    quats = np.array(quats, np.float64)
    quats = np.zeros((0, 10)) if quats.size == 0 else quats
    return quats


def _fit_chpi_quat_subset(coil_dev_rrs, coil_head_rrs, use_idx):
    quat, g = _fit_chpi_quat(coil_dev_rrs[use_idx], coil_head_rrs[use_idx])
    out_idx = use_idx.copy()
    if len(use_idx) > 3:  # try dropping one (recursively)
        for di in range(len(use_idx)):
            this_use_idx = list(use_idx[:di]) + list(use_idx[di + 1 :])
            this_quat, this_g, this_use_idx = _fit_chpi_quat_subset(
                coil_dev_rrs, coil_head_rrs, this_use_idx
            )
            if this_g > g:
                quat, g, out_idx = this_quat, this_g, this_use_idx
    return quat, g, np.array(out_idx, int)


@verbose
def compute_chpi_snr(
    raw, t_step_min=0.01, t_window="auto", ext_order=1, tmin=0, tmax=None, verbose=None
):
    """Compute time-varying estimates of cHPI SNR.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information.
    t_step_min : float
        Minimum time step to use.
    %(t_window_chpi_t)s
    %(ext_order_chpi)s
    %(tmin_raw)s
    %(tmax_raw)s
    %(verbose)s

    Returns
    -------
    chpi_snrs : dict
        The time-varying cHPI SNR estimates, with entries "times", "freqs",
        "snr_mag", "power_mag", and "resid_mag" (and/or "snr_grad",
        "power_grad", and "resid_grad", depending on which channel types are
        present in ``raw``).

    See Also
    --------
    mne.chpi.compute_chpi_locs, mne.chpi.compute_chpi_amplitudes

    Notes
    -----
    .. versionadded:: 0.24
    """
    return _compute_chpi_amp_or_snr(
        raw, t_step_min, t_window, ext_order, tmin, tmax, verbose, snr=True
    )


@verbose
def compute_chpi_amplitudes(
    raw, t_step_min=0.01, t_window="auto", ext_order=1, tmin=0, tmax=None, verbose=None
):
    """Compute time-varying cHPI amplitudes.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information.
    t_step_min : float
        Minimum time step to use.
    %(t_window_chpi_t)s
    %(ext_order_chpi)s
    %(tmin_raw)s
    %(tmax_raw)s
    %(verbose)s

    Returns
    -------
    %(chpi_amplitudes)s

    See Also
    --------
    mne.chpi.compute_chpi_locs, mne.chpi.compute_chpi_snr

    Notes
    -----
    This function will:

    1. Get HPI frequencies,  HPI status channel, HPI status bits,
       and digitization order using ``_setup_hpi_amplitude_fitting``.
    2. Window data using ``t_window`` (half before and half after ``t``) and
       ``t_step_min``.
    3. Use a linear model (DC + linear slope + sin + cos terms) to fit
       sinusoidal amplitudes to MEG channels.
       It uses SVD to determine the phase/amplitude of the sinusoids.

    In "auto" mode, ``t_window`` will be set to the longer of:

    1. Five cycles of the lowest HPI or line frequency.
          Ensures that the frequency estimate is stable.
    2. The reciprocal of the smallest difference between HPI and line freqs.
          Ensures that neighboring frequencies can be disambiguated.

    The output is meant to be used with :func:`~mne.chpi.compute_chpi_locs`.

    .. versionadded:: 0.20
    """
    return _compute_chpi_amp_or_snr(
        raw, t_step_min, t_window, ext_order, tmin, tmax, verbose
    )


def _compute_chpi_amp_or_snr(
    raw,
    t_step_min=0.01,
    t_window="auto",
    ext_order=1,
    tmin=0,
    tmax=None,
    verbose=None,
    snr=False,
):
    """Compute cHPI amplitude or SNR.

    See compute_chpi_amplitudes for parameter descriptions. One additional
    boolean parameter ``snr`` signals whether to return SNR instead of
    amplitude.
    """
    hpi = _setup_hpi_amplitude_fitting(raw.info, t_window, ext_order=ext_order)
    tmin, tmax = raw._tmin_tmax_to_start_stop(tmin, tmax)
    tmin = tmin / raw.info["sfreq"]
    tmax = tmax / raw.info["sfreq"]
    need_win = hpi["t_window"] / 2.0
    fit_idxs = raw.time_as_index(
        np.arange(tmin + need_win, tmax, t_step_min), use_rounding=True
    )
    logger.info(
        f"Fitting {len(hpi['freqs'])} HPI coil locations at up to "
        f"{len(fit_idxs)} time points ({tmax - tmin:.1f} s duration)"
    )
    del tmin, tmax
    sin_fits = dict()
    sin_fits["proj"] = hpi["proj"]
    sin_fits["times"] = (
        np.round(fit_idxs + raw.first_samp - hpi["n_window"] / 2.0) / raw.info["sfreq"]
    )
    n_times = len(sin_fits["times"])
    n_freqs = len(hpi["freqs"])
    n_chans = len(sin_fits["proj"]["data"]["col_names"])
    if snr:
        del sin_fits["proj"]
        sin_fits["freqs"] = hpi["freqs"]
        ch_types = raw.get_channel_types()
        grad_offset = 3 if "mag" in ch_types else 0
        for ch_type in ("mag", "grad"):
            if ch_type in ch_types:
                for key in ("snr", "power", "resid"):
                    cols = 1 if key == "resid" else n_freqs
                    sin_fits[f"{ch_type}_{key}"] = np.empty((n_times, cols))
    else:
        sin_fits["slopes"] = np.empty((n_times, n_freqs, n_chans))
    message = f"cHPI {'SNRs' if snr else 'amplitudes'}"
    for mi, midpt in enumerate(ProgressBar(fit_idxs, mesg=message)):
        #
        # 0. determine samples to fit.
        #
        time_sl = midpt - hpi["n_window"] // 2
        time_sl = slice(max(time_sl, 0), min(time_sl + hpi["n_window"], len(raw.times)))

        #
        # 1. Fit amplitudes for each channel from each of the N sinusoids
        #
        amps_or_snrs = _fit_chpi_amplitudes(raw, time_sl, hpi, snr)
        if snr:
            if amps_or_snrs is None:
                amps_or_snrs = np.full((n_freqs, grad_offset + 3), np.nan)
            # unpack the SNR estimates. mag & grad are returned in one array
            # (because of Numba) so take care with which column is which.
            # note that mean residual is a scalar (same for all HPI freqs) but
            # is returned as a (tiled) vector (again, because Numba) so that's
            # why below we take amps_or_snrs[0, 2] instead of [:, 2]
            ch_types = raw.get_channel_types()
            if "mag" in ch_types:
                sin_fits["mag_snr"][mi] = amps_or_snrs[:, 0]  # SNR
                sin_fits["mag_power"][mi] = amps_or_snrs[:, 1]  # mean power
                sin_fits["mag_resid"][mi] = amps_or_snrs[0, 2]  # mean resid
            if "grad" in ch_types:
                sin_fits["grad_snr"][mi] = amps_or_snrs[:, grad_offset]
                sin_fits["grad_power"][mi] = amps_or_snrs[:, grad_offset + 1]
                sin_fits["grad_resid"][mi] = amps_or_snrs[0, grad_offset + 2]
        else:
            sin_fits["slopes"][mi] = amps_or_snrs
    return sin_fits


@verbose
def compute_chpi_locs(
    info,
    chpi_amplitudes,
    t_step_max=1.0,
    too_close="raise",
    adjust_dig=False,
    verbose=None,
):
    """Compute locations of each cHPI coils over time.

    Parameters
    ----------
    %(info_not_none)s
    %(chpi_amplitudes)s
        Typically obtained by :func:`mne.chpi.compute_chpi_amplitudes`.
    t_step_max : float
        Maximum time step to use.
    too_close : str
        How to handle HPI positions too close to the sensors,
        can be ``'raise'`` (default), ``'warning'``, or ``'info'``.
    %(adjust_dig_chpi)s
    %(verbose)s

    Returns
    -------
    %(chpi_locs)s

    See Also
    --------
    compute_chpi_amplitudes
    compute_head_pos
    read_head_pos
    write_head_pos
    extract_chpi_locs_ctf

    Notes
    -----
    This function is designed to take the output of
    :func:`mne.chpi.compute_chpi_amplitudes` and:

    1. Get HPI coil locations (as digitized in ``info['dig']``) in head coords.
    2. If the amplitudes are 98%% correlated with last position
       (and Δt < t_step_max), skip fitting.
    3. Fit magnetic dipoles using the amplitudes for each coil frequency.

    The number of fitted points ``n_pos`` will depend on the velocity of head
    movements as well as ``t_step_max`` (and ``t_step_min`` from
    :func:`mne.chpi.compute_chpi_amplitudes`).

    .. versionadded:: 0.20
    """
    # Set up magnetic dipole fits
    _check_option("too_close", too_close, ["raise", "warning", "info"])
    _check_chpi_param(chpi_amplitudes, "chpi_amplitudes")
    _validate_type(info, Info, "info")
    sin_fits = chpi_amplitudes  # use the old name below
    del chpi_amplitudes
    proj = sin_fits["proj"]
    meg_picks = pick_channels(info["ch_names"], proj["data"]["col_names"], ordered=True)
    info = pick_info(info, meg_picks)  # makes a copy
    with info._unlock():
        info["projs"] = [proj]
    del meg_picks, proj
    meg_coils = _concatenate_coils(_create_meg_coils(info["chs"], "accurate"))

    # Set up external model for interference suppression
    safe_false = _verbose_safe_false()
    cov = make_ad_hoc_cov(info, verbose=safe_false)
    whitener, _ = compute_whitener(cov, info, verbose=safe_false)

    # Make some location guesses (1 cm grid)
    R = np.linalg.norm(meg_coils[0], axis=1).min()
    guesses = _make_guesses(
        dict(R=R, r0=np.zeros(3)), 0.01, 0.0, 0.005, verbose=safe_false
    )[0]["rr"]
    logger.info(
        f"Computing {len(guesses)} HPI location guesses "
        f"(1 cm grid in a {R * 100:.1f} cm sphere)"
    )
    fwd = _magnetic_dipole_field_vec(guesses, meg_coils, too_close)
    fwd = np.dot(fwd, whitener.T)
    fwd.shape = (guesses.shape[0], 3, -1)
    fwd = np.linalg.svd(fwd, full_matrices=False)[2]
    guesses = dict(rr=guesses, whitened_fwd_svd=fwd)
    del fwd, R

    iter_ = list(zip(sin_fits["times"], sin_fits["slopes"]))
    chpi_locs = dict(times=[], rrs=[], gofs=[], moments=[])
    # setup last iteration structure
    hpi_dig_dev_rrs = apply_trans(
        invert_transform(info["dev_head_t"])["trans"],
        _get_hpi_initial_fit(info, adjust=adjust_dig),
    )
    last = dict(
        sin_fit=None,
        coil_fit_time=sin_fits["times"][0] - 1,
        coil_dev_rrs=hpi_dig_dev_rrs,
    )
    n_hpi = len(hpi_dig_dev_rrs)
    del hpi_dig_dev_rrs
    for fit_time, sin_fit in ProgressBar(iter_, mesg="cHPI locations "):
        # skip this window if bad
        if not np.isfinite(sin_fit).all():
            continue

        # check if data has sufficiently changed
        if last["sin_fit"] is not None:  # first iteration
            corrs = np.array(
                [np.corrcoef(s, lst)[0, 1] for s, lst in zip(sin_fit, last["sin_fit"])]
            )
            corrs *= corrs
            # check to see if we need to continue
            if (
                fit_time - last["coil_fit_time"] <= t_step_max - 1e-7
                and (corrs > 0.98).sum() >= 3
            ):
                # don't need to refit data
                continue

        # update 'last' sin_fit *before* inplace sign mult
        last["sin_fit"] = sin_fit.copy()

        #
        # 2. Fit magnetic dipole for each coil to obtain coil positions
        #    in device coordinates
        #
        coil_fits = [
            _fit_magnetic_dipole(f, x0, too_close, whitener, meg_coils, guesses)
            for f, x0 in zip(sin_fit, last["coil_dev_rrs"])
        ]
        rrs, gofs, moments = zip(*coil_fits)
        chpi_locs["times"].append(fit_time)
        chpi_locs["rrs"].append(rrs)
        chpi_locs["gofs"].append(gofs)
        chpi_locs["moments"].append(moments)
        last["coil_fit_time"] = fit_time
        last["coil_dev_rrs"] = rrs
    n_times = len(chpi_locs["times"])
    shapes = dict(
        times=(n_times,),
        rrs=(n_times, n_hpi, 3),
        gofs=(n_times, n_hpi),
        moments=(n_times, n_hpi, 3),
    )
    for key, val in chpi_locs.items():
        chpi_locs[key] = np.array(val, float).reshape(shapes[key])
    return chpi_locs


def _chpi_locs_to_times_dig(chpi_locs):
    """Reformat chpi_locs as list of dig (dict)."""
    dig = list()
    for rrs, gofs in zip(*(chpi_locs[key] for key in ("rrs", "gofs"))):
        dig.append(
            [
                {
                    "r": rr,
                    "ident": idx,
                    "gof": gof,
                    "kind": FIFF.FIFFV_POINT_HPI,
                    "coord_frame": FIFF.FIFFV_COORD_DEVICE,
                }
                for idx, (rr, gof) in enumerate(zip(rrs, gofs), 1)
            ]
        )
    return chpi_locs["times"], dig


@verbose
def filter_chpi(
    raw,
    include_line=True,
    t_step=0.01,
    t_window="auto",
    ext_order=1,
    allow_line_only=False,
    verbose=None,
):
    """Remove cHPI and line noise from data.

    .. note:: This function will only work properly if cHPI was on
              during the recording.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information. Must be preloaded. Operates in-place.
    include_line : bool
        If True, also filter line noise.
    t_step : float
        Time step to use for estimation, default is 0.01 (10 ms).
    %(t_window_chpi_t)s
    %(ext_order_chpi)s
    allow_line_only : bool
        If True, allow filtering line noise only. The default is False,
        which only allows the function to run when cHPI information is present.

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The raw data.

    Notes
    -----
    cHPI signals are in general not stationary, because head movements act
    like amplitude modulators on cHPI signals. Thus it is recommended to
    use this procedure, which uses an iterative fitting method, to
    remove cHPI signals, as opposed to notch filtering.

    .. versionadded:: 0.12
    """
    _validate_type(raw, BaseRaw, "raw")
    if not raw.preload:
        raise RuntimeError("raw data must be preloaded")
    t_step = float(t_step)
    if t_step <= 0:
        raise ValueError(f"t_step ({t_step}) must be > 0")
    n_step = int(np.ceil(t_step * raw.info["sfreq"]))
    if include_line and raw.info["line_freq"] is None:
        raise RuntimeError(
            'include_line=True but raw.info["line_freq"] is '
            "None, consider setting it to the line frequency"
        )
    hpi = _setup_hpi_amplitude_fitting(
        raw.info,
        t_window,
        remove_aliased=True,
        ext_order=ext_order,
        allow_empty=allow_line_only,
        verbose=_verbose_safe_false(),
    )

    fit_idxs = np.arange(0, len(raw.times) + hpi["n_window"] // 2, n_step)
    n_freqs = len(hpi["freqs"])
    n_remove = 2 * n_freqs
    meg_picks = pick_types(raw.info, meg=True, exclude=())  # filter all chs
    n_times = len(raw.times)

    msg = f"Removing {n_freqs} cHPI"
    if include_line:
        n_remove += 2 * len(hpi["line_freqs"])
        msg += f" and {len(hpi['line_freqs'])} line harmonic"
    msg += f" frequencies from {len(meg_picks)} MEG channels"

    recon = np.dot(hpi["model"][:, :n_remove], hpi["inv_model"][:n_remove]).T
    logger.info(msg)
    chunks = list()  # the chunks to subtract
    last_endpt = 0
    pb = ProgressBar(fit_idxs, mesg="Filtering")
    for ii, midpt in enumerate(pb):
        left_edge = midpt - hpi["n_window"] // 2
        time_sl = slice(
            max(left_edge, 0), min(left_edge + hpi["n_window"], len(raw.times))
        )
        this_len = time_sl.stop - time_sl.start
        if this_len == hpi["n_window"]:
            this_recon = recon
        else:  # first or last window
            model = hpi["model"][:this_len]
            inv_model = np.linalg.pinv(model)
            this_recon = np.dot(model[:, :n_remove], inv_model[:n_remove]).T
        this_data = raw._data[meg_picks, time_sl]
        subt_pt = min(midpt + n_step, n_times)
        if last_endpt != subt_pt:
            fit_left_edge = left_edge - time_sl.start + hpi["n_window"] // 2
            fit_sl = slice(fit_left_edge, fit_left_edge + (subt_pt - last_endpt))
            chunks.append((subt_pt, np.dot(this_data, this_recon[:, fit_sl])))
        last_endpt = subt_pt

        # Consume (trailing) chunks that are now safe to remove because
        # our windows will no longer touch them
        if ii < len(fit_idxs) - 1:
            next_left_edge = fit_idxs[ii + 1] - hpi["n_window"] // 2
        else:
            next_left_edge = np.inf
        while len(chunks) > 0 and chunks[0][0] <= next_left_edge:
            right_edge, chunk = chunks.pop(0)
            raw._data[meg_picks, right_edge - chunk.shape[1] : right_edge] -= chunk
    return raw


def _compute_good_distances(hpi_coil_dists, new_pos, dist_limit=0.005):
    """Compute good coils based on distances."""
    these_dists = cdist(new_pos, new_pos)
    these_dists = np.abs(hpi_coil_dists - these_dists)
    # there is probably a better algorithm for finding the bad ones...
    good = False
    use_mask = np.ones(len(hpi_coil_dists), bool)
    while not good:
        d = these_dists[use_mask][:, use_mask]
        d_bad = d > dist_limit
        good = not d_bad.any()
        if not good:
            if use_mask.sum() == 2:
                use_mask[:] = False
                break  # failure
            # exclude next worst point
            badness = (d * d_bad).sum(axis=0)
            exclude_coils = np.where(use_mask)[0][np.argmax(badness)]
            use_mask[exclude_coils] = False
    return use_mask, these_dists


@verbose
def get_active_chpi(raw, *, on_missing="raise", verbose=None):
    """Determine how many HPI coils were active for a time point.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information.
    %(on_missing_chpi)s
    %(verbose)s

    Returns
    -------
    n_active : array, shape (n_times)
        The number of active cHPIs for every timepoint in raw.

    Notes
    -----
    .. versionadded:: 1.2
    """
    # get meg system
    system, _ = _get_meg_system(raw.info)

    # check whether we have a neuromag system
    if system not in ["122m", "306m"]:
        raise NotImplementedError(
            "Identifying active HPI channels is not implemented for other systems than "
            "neuromag."
        )
    # extract hpi info
    chpi_info = get_chpi_info(raw.info, on_missing=on_missing)
    if (len(chpi_info[2]) == 0) or (chpi_info[1] is None):
        return np.zeros_like(raw.times)

    # extract hpi time series and infer which one was on
    chpi_ts = raw[chpi_info[1]][0].astype(int)
    chpi_active = (chpi_ts & chpi_info[2][:, np.newaxis]).astype(bool)
    return chpi_active.sum(axis=0)
