# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mstats

from ..._fiff.constants import FIFF
from ...annotations import Annotations, _annotations_starts_stops
from ...io import BaseRaw
from ...utils import _check_preload, _validate_type, logger, warn


def interpolate_blinks(raw, buffer=0.05, match="BAD_blink", interpolate_gaze=False):
    """Interpolate eyetracking signals during blinks.

    This function uses the timing of blink annotations to estimate missing
    data. Missing values are then interpolated linearly. Operates in place.

    Parameters
    ----------
    raw : instance of Raw
        The raw data with at least one ``'pupil'`` or ``'eyegaze'`` channel.
    buffer : float | array-like of float, shape ``(2,))``
        The time in seconds before and after a blink to consider invalid and
        include in the segment to be interpolated over. Default is ``0.05`` seconds
        (50 ms). If array-like, the first element is the time before the blink and the
        second element is the time after the blink to consider invalid, for example,
        ``(0.025, .1)``.
    match : str | list of str
        The description of annotations to interpolate over. If a list, the data within
        all annotations that match any of the strings in the list will be interpolated
        over. If a ``match`` starts with ``'BAD_'``, that part will be removed from the
        annotation description after interpolation. Defaults to ``'BAD_blink'``.
    interpolate_gaze : bool
        If False, only apply interpolation to ``'pupil channels'``. If True, interpolate
        over ``'eyegaze'`` channels as well. Defaults to False, because eye position can
        change in unpredictable ways during blinks.

    Returns
    -------
    self : instance of Raw
        Returns the modified instance.

    Notes
    -----
    .. versionadded:: 1.5
    """
    _check_preload(raw, "interpolate_blinks")
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(buffer, (float, tuple, list, np.ndarray), "buffer")
    _validate_type(match, (str, tuple, list, np.ndarray), "match")

    # determine the buffer around blinks to include in the interpolation
    buffer = np.array(buffer, dtype=float)
    if buffer.size == 1:
        buffer = np.array([buffer, buffer])

    if isinstance(match, str):
        match = [match]

    # get the blink annotations
    blink_annots = [annot for annot in raw.annotations if annot["description"] in match]
    if not blink_annots:
        warn(f"No annotations matching {match} found. Aborting.")
        return raw
    _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze=interpolate_gaze)

    # remove bad from the annotation description
    for desc in match:
        if desc.startswith("BAD_"):
            logger.info(f"Removing 'BAD_' from {desc}.")
            raw.annotations.rename({desc: desc.replace("BAD_", "")})
    return raw


def _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze):
    """Interpolate eyetracking signals during blinks in-place."""
    logger.info("Interpolating missing data during blinks...")
    pre_buffer, post_buffer = buffer
    # iterate over each eyetrack channel and interpolate the blinks
    interpolated_chs = []
    for ci, ch_info in enumerate(raw.info["chs"]):
        if interpolate_gaze:  # interpolate over all eyetrack channels
            if ch_info["kind"] != FIFF.FIFFV_EYETRACK_CH:
                continue
        else:  # interpolate over pupil channels only
            if ch_info["coil_type"] != FIFF.FIFFV_COIL_EYETRACK_PUPIL:
                continue
        # Create an empty boolean mask
        mask = np.zeros_like(raw.times, dtype=bool)
        starts, ends = _annotations_starts_stops(raw, "BAD_blink")
        starts = np.divide(starts, raw.info["sfreq"])
        ends = np.divide(ends, raw.info["sfreq"])
        for annot, start, end in zip(blink_annots, starts, ends):
            if "ch_names" not in annot or not annot["ch_names"]:
                msg = f"Blink annotation missing values for 'ch_names' key: {annot}"
                raise ValueError(msg)
            start -= pre_buffer
            end += post_buffer
            if ch_info["ch_name"] not in annot["ch_names"]:
                continue  # skip if the channel is not in the blink annotation
            # Update the mask for times within the current blink period
            mask |= (raw.times >= start) & (raw.times <= end)
        blink_indices = np.where(mask)[0]
        non_blink_indices = np.where(~mask)[0]

        # Linear interpolation
        interpolated_samples = np.interp(
            raw.times[blink_indices],
            raw.times[non_blink_indices],
            raw._data[ci, non_blink_indices],
        )
        # Replace the samples at the blink_indices with the interpolated values
        raw._data[ci, blink_indices] = interpolated_samples
        interpolated_chs.append(ch_info["ch_name"])
    if interpolated_chs:
        logger.info(
            f"Interpolated {len(interpolated_chs)} channels: {interpolated_chs}"
        )
    else:
        warn("No channels were interpolated.")


def annotate_velocity_blinks(
    raw,
    channel_names=("pupil_left", "pupil_right"),
    margin_before=0.01,
    margin_after=0.01,
    max_duration=0.5,
    description="BAD_velo_blink",
    sigma=5,
    vt_start_multiplyer=40,
    vt_end_multiplyer=40,
    low_v_thresh_multiplyer=1,
):
    """
    Annotate blinks based on pupil velocity profile.

    fast constriction -> fast dilation -> low velocity.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with pupil data.
    channel_names : tuple of str
        Names of pupil channels (e.g., ('pupil_left', 'pupil_right')).
    margin_before : float
        Time (in seconds) to extend before blink onset.
    margin_after : float
        Time (in seconds) to extend after blink offset.
    max_duration : float
        Maximum allowable blink duration (in seconds).
    description : str
        Annotation label.
    sigma : float
        Sigma for Gaussian smoothing.
    vt_start_multiplyer : float
        Multiplier for median absolute deviation (MAD) to set blink onset threshold.
    vt_end_multiplyer : float
        Multiplier for median absolute deviation (MAD) to set blink dilation threshold.
    low_v_thresh_multiplyer : float
        Multiplier for median velocity (MED) to define stabilization (low velocity)
        threshold.

    Returns
    -------
    raw : mne.io.Raw
        Modified Raw object with blink annotations.

    """
    annotations = raw.annotations

    sfreq = raw.info["sfreq"]
    max_search_duration = max_duration

    onsets, durations, descriptions, ch_names_list = [], [], [], []

    for ch_name in channel_names:
        if ch_name not in raw.ch_names:
            raise ValueError(f"Channel '{ch_name}' not found in raw.")

        data = raw.get_data(picks=ch_name)[0]

        if sigma > 0:
            smoothed = gaussian_filter1d(data, sigma=sigma)
            velocity = np.gradient(smoothed) * sfreq
        else:
            velocity = np.gradient(data) * sfreq

        # clipping the top 1 percent of data to not have outliers
        # influence the threshold (winsorizing)
        velocity_clipped = mstats.winsorize(
            velocity, limits=[0.01, 0.01]
        )  # clip top and bottom 1%
        abs_velocity = np.abs(velocity_clipped.data)

        # median velocity
        med = np.nanmedian(abs_velocity)
        # median absolute deviation
        mad = np.nanmedian(np.abs(abs_velocity - med))

        vt_start = med + (vt_start_multiplyer * mad)
        vt_end = med + (vt_end_multiplyer * mad)
        low_v_thresh = med * low_v_thresh_multiplyer

        # Optionally print thresholds per participant and channel
        # print(
        #     f"{ch_name}: med={med:.2f}, "
        #     "mad={mad:.2f}, "
        #     "vt_start={vt_start:.2f}, "
        #     "vt_end={vt_end:.2f}, "
        #     "low_v_thresh={low_v_thresh:.2f}"
        # )

        i = 0
        while i < len(velocity) - 1:
            imid = None

            # Blink onset: rapid constriction
            if velocity[i] < -vt_start:
                istart = i

                # Limit search range
                max_search = int(sfreq * max_search_duration)

                # Dilation: search after constriction
                dilation_zone = velocity[istart : istart + max_search]
                dilation_indices = np.where(dilation_zone > vt_end)[0]
                if dilation_indices.size > 0:
                    imid = dilation_indices[0] + istart

                    # Look for stabilization
                    post_dilate_zone = velocity[imid : imid + max_search]
                    low_v_indices = np.where(np.abs(post_dilate_zone) < low_v_thresh)[0]
                    if low_v_indices.size > 0:
                        iend = low_v_indices[0] + imid

                        # Validate and annotate
                        duration_sec = (iend - istart) / sfreq
                        if duration_sec <= max_duration:
                            onset = (istart / sfreq) - margin_before
                            onset = max(0, onset)
                            total_duration = duration_sec + margin_before + margin_after

                            onsets.append(onset)
                            durations.append(total_duration)
                            descriptions.append(description)
                            ch_names_list.append([ch_name])
                            i = iend - 1
                            continue  # go to next blink candidate
                # Adaptive skip logic
            if imid is not None:
                i = imid + 1
            else:
                i += int(sfreq * 0.01)  # skip 10 ms

    if onsets:
        new_annotations = Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.annotations.orig_time,
            ch_names=ch_names_list,
        )
        raw.set_annotations(annotations + new_annotations)
    else:
        print("No blinks detected.")

    return raw
