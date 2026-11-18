# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
import warnings
from contextlib import contextmanager

import numpy as np

from ..._fiff.pick import _picks_to_idx, pick_types
from ...utils import _check_option, _validate_type, fill_doc_static, warn

# Standardized fNIRS channel name, e.g. "S1_D2 760" or "S1_D2 hbo"
_S_D_RE = re.compile(r"S(\d+)_D(\d+) (\d+\.?\d*|\w+)")


def _parse_nirs_ch_name(ch_name):
    """Split an fNIRS channel name ``S<source>_D<detector> <value>``.

    Parameters
    ----------
    ch_name : str
        The channel name, e.g. ``"S1_D2 760"`` or ``"S1_D2 hbo"``.

    Returns
    -------
    source : int
        The source number.
    detector : int
        The detector number.
    value : str
        The wavelength or chromophore.
    """
    match = _S_D_RE.match(ch_name)
    if match is None:
        raise ValueError(
            f"Cannot parse fNIRS channel name {repr(ch_name)}, expected "
            '"S<source>_D<detector> <value>"'
        )
    source, detector, value = match.groups()
    return int(source), int(detector), value


@fill_doc_static("info_not_none", "picks_all_data")
def source_detector_distances(info, picks=None):
    r"""Determine the distance between NIRS source and detectors.

    Parameters
    ----------
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick all data channels. Note
        that channels in ``info['bads']`` *will be included* if their names or
        indices are explicitly provided.

    Returns
    -------
    dists : array of float
        Array containing distances in meters.
        Of shape equal to number of channels, or shape of picks if supplied.
    """
    return np.array(
        [
            np.linalg.norm(
                np.diff(info["chs"][pick]["loc"][3:9].reshape(2, 3), axis=0)[0]
            )
            for pick in _picks_to_idx(info, picks, exclude=[])
        ],
        float,
    )


@fill_doc_static("info_not_none")
def short_channels(info, threshold=0.01):
    r"""Determine which NIRS channels are short.

    Channels with a source to detector distance of less than
    ``threshold`` are reported as short. The default threshold is 0.01 m.

    Parameters
    ----------
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    threshold : float
        The threshold distance for what is considered short in meters.

    Returns
    -------
    short : array of bool
        Array indicating which channels are short.
        Of shape equal to number of channels.
    """
    return source_detector_distances(info) < threshold


def _has_source_detector_distances(info, picks=None):
    """Return True if source-detector distances can be computed."""
    distances = source_detector_distances(info, picks=picks)
    return len(distances) > 0 and np.all(distances > 0)


def _channel_frequencies(info):
    """Return the light frequency for each channel."""
    # Only valid for fNIRS data before conversion to haemoglobin
    picks = _picks_to_idx(
        info, ["fnirs_cw_amplitude", "fnirs_od"], exclude=[], allow_empty=True
    )
    freqs = [round(float(_parse_nirs_ch_name(info.ch_names[p])[2])) for p in picks]
    return np.array(freqs, int)


def _channel_chromophore(info):
    """Return the chromophore of each channel."""
    # Only valid for fNIRS data after conversion to haemoglobin
    picks = _picks_to_idx(info, ["hbo", "hbr"], exclude=[], allow_empty=True)
    return [_parse_nirs_ch_name(info.ch_names[p])[2] for p in picks]


def _check_channels_ordered(info, pair_vals, *, throw_errors=True, check_bads=True):
    """Check channels follow expected fNIRS format.

    If the channels are correctly ordered then an array of valid picks
    will be returned.

    If throw_errors is True then any errors in fNIRS formatting will be
    thrown to inform the user. If throw_errors is False then an empty array
    will be returned if the channels are not sufficiently formatted.
    """
    # Every second channel should be same SD pair
    # and have the specified light frequencies.

    # All wavelength based fNIRS data.
    picks_wave = _picks_to_idx(
        info, ["fnirs_cw_amplitude", "fnirs_od"], exclude=[], allow_empty=True
    )
    # All chromophore fNIRS data
    picks_chroma = _picks_to_idx(info, ["hbo", "hbr"], exclude=[], allow_empty=True)

    # All TD moments
    td_moments = [
        "fnirs_td_moments_intensity",
        "fnirs_td_moments_mean",
        "fnirs_td_moments_variance",
    ]
    picks_moments = _picks_to_idx(info, td_moments, exclude=[], allow_empty=True)

    # All TD gated
    picks_gated = _picks_to_idx(
        info, ["fnirs_td_gated_amplitude"], exclude=[], allow_empty=True
    )

    n_found = sum(
        len(x) > 0
        for x in (
            picks_wave,
            picks_chroma,
            picks_moments,
            picks_gated,
        )
    )
    if n_found != 1:
        picks = _throw_or_return_empty(
            "MNE supports exactly one of amplitude, optical density, "
            "TD moments, TD gated, and haemoglobin data in a given raw "
            f"structure, found {n_found}",
            throw_errors,
        )

    # All continuous wave fNIRS data
    if len(picks_wave):
        error_word = "frequencies"
        picks = picks_wave
    elif len(picks_chroma):
        error_word = "chromophore"
        picks = picks_chroma
    else:
        assert len(picks_moments) or len(picks_gated)
        return  # nothing to check

    pair_vals = np.array(pair_vals)
    if pair_vals.shape[0] < 2:
        raise ValueError(
            f"At least two {error_word} must exist in info, got {list(pair_vals)}"
        )
    # In principle we do not need to require that these be sorted --
    # all we need to do is change our sorted() below to make use of a
    # pair_vals.index(...) in a sort key -- but in practice we always want
    # (hbo, hbr) or (lowest_freq, higher_freq, ...) pairings, both of which will
    # work with a naive string sort, so let's just enforce sorted-ness here
    is_str = pair_vals.dtype.kind == "U"
    if is_str:
        if pair_vals.tolist() != ["hbo", "hbr"]:
            raise ValueError(
                f'The {error_word} in info must be ["hbo", "hbr"], but got '
                f"{pair_vals} instead"
            )
    elif not np.array_equal(np.unique(pair_vals), pair_vals):
        raise ValueError(
            f"The {error_word} in info must be unique and sorted, but got "
            f"got {pair_vals} instead"
        )

    # Check that the total number of channels is divisible by the number of pair values
    # (e.g., for 2 wavelengths, we need an even number of channels)
    if len(picks) % len(pair_vals) != 0:
        picks = _throw_or_return_empty(
            "NIRS channels not ordered correctly. The number of channels "
            f"must be a multiple of {len(pair_vals)} values, but "
            f"{len(picks)} channels were provided.",
            throw_errors,
        )

    # Ensure wavelength info exists for waveform data
    # Note: currently, the only requirement for the wavelength field in info is
    # that it cannot be NaN. It depends on the data readers what is stored in it.
    if len(picks_wave) > 0:
        all_freqs = [info["chs"][ii]["loc"][9] for ii in picks_wave]
        # test for nan values first as those mess up the output of set()
        if np.any(np.isnan(all_freqs)):
            picks = _throw_or_return_empty(
                f"NIRS channels is missing wavelength information in the "
                f'info["chs"] structure. The encoded wavelengths are {all_freqs}.',
                throw_errors,
            )

    # Validate the channel naming scheme
    for pick in picks:
        try:
            _, _, value = _parse_nirs_ch_name(info.ch_names[pick])
        except ValueError:
            picks = _throw_or_return_empty(
                "NIRS channels have specified naming conventions. "
                "The provided channel name can not be parsed: "
                f"{repr(info.ch_names[pick])}",
                throw_errors,
            )
            break
        if len(picks_wave) > 0 and not value[0].isdigit():
            picks = _throw_or_return_empty(
                "NIRS channels have specified naming conventions. "
                "Wavelength data must be labeled with a number. "
                f"The failing channel is {info['chs'][pick]['ch_name']}",
                throw_errors,
            )
            break
        if len(picks_chroma) > 0:
            if value not in ["hbo", "hbr"]:
                picks = _throw_or_return_empty(
                    "NIRS channels have specified naming conventions."
                    "Chromophore data must be labeled either hbo or hbr. "
                    f"The failing channel is {info['chs'][pick]['ch_name']}",
                    throw_errors,
                )
                break

    # Reorder to be paired (naive sort okay here given validation above)
    picks = picks[np.argsort([info["ch_names"][pick] for pick in picks])]

    # Validate channel grouping (same source-detector pairs, all pair_vals match)
    for ii in range(0, len(picks), len(pair_vals)):
        # Extract a group of channels (e.g., all wavelengths for one S-D pair)
        group_picks = picks[ii : ii + len(pair_vals)]

        group_info = [_parse_nirs_ch_name(info.ch_names[pick]) for pick in group_picks]

        # Separate the parsed components:
        # source IDs, detector IDs, and values (freq/chromophore)
        s_group, d_group, val_group = zip(*group_info)

        # For wavelength data, convert string frequencies to float for comparison
        if len(picks_wave) > 0:
            val_group = np.array([float(v) for v in val_group])

        # Verify that all channels in this group have the same source-detector pair
        # and that the values match the expected pair_vals sequence
        if not (
            len(set(s_group)) == 1
            and len(set(d_group)) == 1
            and np.array_equal(val_group, pair_vals)
        ):
            picks = _throw_or_return_empty(
                "NIRS channels not ordered correctly. Channels must be "
                f"grouped by source-detector pairs with alternating {error_word} "
                f"values {pair_vals}, but got mismatching names "
                f"{[info['ch_names'][pick] for pick in group_picks]}.",
                throw_errors,
            )
            break

    if check_bads:
        for ii in range(0, len(picks), len(pair_vals)):
            group_picks = picks[ii : ii + len(pair_vals)]

            want = [info.ch_names[pick] for pick in group_picks]
            got = list(set(info["bads"]).intersection(want))
            if 0 < len(got) < len(want):
                raise RuntimeError(
                    "NIRS bad labelling is not consistent. "
                    f"Found {got} but needed {want}. "
                )
    return picks


@contextmanager
def _warn_channel(ch_name):
    """Re-emit warnings raised while processing one channel, naming it.

    NumPy emits numerical warnings (e.g. "divide by zero encountered in log")
    without any channel context.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        warn(f"{w.message} in channel {ch_name}", w.category)


def _throw_or_return_empty(msg, throw_errors):
    if throw_errors:
        raise ValueError(msg)
    else:
        return []


def _validate_nirs_info(
    info,
    *,
    throw_errors=True,
    fnirs=None,
    which=None,
    check_bads=True,
    allow_empty=True,
):
    """Apply all checks to fNIRS info. Works on all continuous wave types."""
    _validate_type(fnirs, (None, str), "fnirs")
    kinds = dict(
        od="optical density",
        cw_amplitude="continuous wave",
        hb="chromophore",
    )
    _check_option("fnirs", fnirs, (None,) + tuple(kinds))
    if fnirs is not None:
        kind = kinds[fnirs]
        fnirs = ["hbo", "hbr"] if fnirs == "hb" else f"fnirs_{fnirs}"
        if not len(pick_types(info, fnirs=fnirs)):
            raise RuntimeError(
                f"{which} must operate on {kind} data, but none was found."
            )
    freqs = np.unique(_channel_frequencies(info))
    if freqs.size > 0:
        pair_vals = freqs
    else:
        pair_vals = np.unique(_channel_chromophore(info))
    out = _check_channels_ordered(
        info, pair_vals, throw_errors=throw_errors, check_bads=check_bads
    )
    return out


def _fnirs_spread_bads(info):
    """Spread bad labeling across fnirs channels."""
    # For an optode pair if any component (light frequency or chroma) is marked
    # as bad, then they all should be. This function will find any pairs marked
    # as bad and spread the bad marking to all components of the optode pair.
    picks = _validate_nirs_info(info, check_bads=False)
    new_bads = set(info["bads"])

    # Extract SD pair groups from channel names
    # E.g. all channels belonging to S1D1, S1D2, etc.
    # Assumes valid channels (naming convention and number)
    ch_names = [info.ch_names[i] for i in picks]

    # Create dict with keys corresponding to SD pairs
    # Defaultdict would require another import
    sd_groups = {}
    for ch_name in ch_names:
        sd_pair = _parse_nirs_ch_name(ch_name)[:2]
        if sd_pair not in sd_groups:
            sd_groups[sd_pair] = [ch_name]
        else:
            sd_groups[sd_pair].append(ch_name)

    # Spread bad labeling across SD pairs
    for channels in sd_groups.values():
        if any(channel in new_bads for channel in channels):
            new_bads.update(channels)

    info["bads"] = sorted(new_bads)
    return info


def _fnirs_optode_names(info):
    """Return list of unique optode names."""
    picks_wave = _picks_to_idx(
        info, ["fnirs_cw_amplitude", "fnirs_od"], exclude=[], allow_empty=True
    )
    picks_chroma = _picks_to_idx(info, ["hbo", "hbr"], exclude=[], allow_empty=True)

    if len(picks_wave) == 0 and len(picks_chroma) == 0:
        return [], []

    parsed = [_parse_nirs_ch_name(ch) for ch in info.ch_names]
    sources = np.unique([source for source, _, _ in parsed])
    detectors = np.unique([detector for _, detector, _ in parsed])

    src_names = [f"S{s}" for s in sources]
    det_names = [f"D{d}" for d in detectors]

    return src_names, det_names


def _optode_position(info, optode):
    """Find the position of an optode."""
    if "S" in optode:
        group, loc_idx = 0, range(3, 6)
    elif "D" in optode:
        group, loc_idx = 1, range(6, 9)
    # Compare optode numbers, so that e.g. D1 does not match D13
    number = int(optode[1:])
    idx = [_parse_nirs_ch_name(ch)[group] == number for ch in info.ch_names].index(True)

    return info["chs"][idx]["loc"][loc_idx]


def _reorder_nirx(raw):
    # Maybe someday we should make this public like
    # mne.preprocessing.nirs.reorder_standard(raw, order='nirx')
    info = raw.info
    picks = pick_types(info, fnirs=True, exclude=[])
    prefixes = [info["ch_names"][pick].split()[0] for pick in picks]
    nirs_names = [info["ch_names"][pick] for pick in picks]
    nirs_sorted = sorted(
        nirs_names,
        key=lambda name: (prefixes.index(name.split()[0]), name.split(maxsplit=1)[1]),
    )
    raw.reorder_channels(nirs_sorted)
