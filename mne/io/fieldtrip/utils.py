# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..._fiff._digitization import DigPoint, _ensure_fiducials_head
from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ..._fiff.pick import pick_info
from ...transforms import rotation3d_align_z_axis
from ...utils import _check_pandas_installed, warn

_supported_megs = ["neuromag306"]

_unit_dict = {
    "m": 1,
    "cm": 1e-2,
    "mm": 1e-3,
    "V": 1,
    "mV": 1e-3,
    "uV": 1e-6,
    "T": 1,
    "T/m": 1,
    "T/cm": 1e2,
}

NOINFO_WARNING = (
    "Importing FieldTrip data without an info dict from the "
    "original file. Channel locations, orientations and types "
    "will be incorrect. The imported data cannot be used for "
    "source analysis, channel interpolation etc."
)


def _validate_ft_struct(ft_struct):
    """Run validation checks on the ft_structure."""
    if isinstance(ft_struct, list):
        raise RuntimeError("Loading of data in cell arrays is not supported")


def _create_info(ft_struct, raw_info):
    """Create MNE info structure from a FieldTrip structure."""
    if raw_info is None:
        warn(NOINFO_WARNING)

    sfreq = _set_sfreq(ft_struct)
    ch_names = ft_struct["label"]
    if raw_info:
        info = raw_info.copy()
        missing_channels = set(ch_names) - set(info["ch_names"])
        if missing_channels:
            warn(
                "The following channels are present in the FieldTrip data "
                f"but cannot be found in the provided info: {missing_channels}.\n"
                "These channels will be removed from the resulting data!"
            )

            missing_chan_idx = [ch_names.index(ch) for ch in missing_channels]
            new_chs = [ch for ch in ch_names if ch not in missing_channels]
            ch_names = new_chs
            ft_struct["label"] = ch_names

            if "trial" in ft_struct:
                ft_struct["trial"] = _remove_missing_channels_from_trial(
                    ft_struct["trial"], missing_chan_idx
                )

            if "avg" in ft_struct:
                if ft_struct["avg"].ndim == 2:
                    ft_struct["avg"] = np.delete(
                        ft_struct["avg"], missing_chan_idx, axis=0
                    )

        with info._unlock():
            info["sfreq"] = sfreq
        ch_idx = [info["ch_names"].index(ch) for ch in ch_names]
        pick_info(info, ch_idx, copy=False)
    else:
        info = create_info(ch_names, sfreq)
        chs, dig = _create_info_chs_dig(ft_struct)
        with info._unlock(update_redundant=True):
            info.update(chs=chs, dig=dig)

    return info


def _remove_missing_channels_from_trial(trial, missing_chan_idx):
    if isinstance(trial, list):
        for idx_trial in range(len(trial)):
            trial[idx_trial] = _remove_missing_channels_from_trial(
                trial[idx_trial], missing_chan_idx
            )
    elif isinstance(trial, np.ndarray):
        if trial.ndim == 2:
            trial = np.delete(trial, missing_chan_idx, axis=0)
    else:
        raise ValueError(
            '"trial" field of the FieldTrip structure has an unknown format.'
        )

    return trial


def _create_info_chs_dig(ft_struct):
    """Create the chs info field from the FieldTrip structure."""
    all_channels = ft_struct["label"]
    ch_defaults = dict(
        coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
        cal=1.0,
        range=1.0,
        unit_mul=FIFF.FIFF_UNITM_NONE,
        loc=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
        unit=FIFF.FIFF_UNIT_V,
    )
    try:
        elec = ft_struct["elec"]
    except KeyError:
        elec = None

    try:
        grad = ft_struct["grad"]
    except KeyError:
        grad = None

    if elec is None and grad is None:
        warn(
            "The supplied FieldTrip structure does not have an elec or grad "
            "field. No channel locations will extracted and the kind of "
            "channel might be inaccurate."
        )
    if "chanpos" not in (elec or grad or {"chanpos": None}):
        raise RuntimeError(
            "This file was created with an old version of FieldTrip. You can "
            "convert the data to the new version by loading it into FieldTrip "
            "and applying ft_selectdata with an empty cfg structure on it. "
            "Otherwise you can supply the Info field."
        )

    chs = list()
    dig = list()
    counter = 0
    for idx_chan, cur_channel_label in enumerate(all_channels):
        cur_ch = ch_defaults.copy()
        cur_ch["ch_name"] = cur_channel_label
        cur_ch["logno"] = idx_chan + 1
        cur_ch["scanno"] = idx_chan + 1
        if elec and cur_channel_label in elec["label"]:
            cur_ch = _process_channel_eeg(cur_ch, elec)
            assert cur_ch["coord_frame"] == FIFF.FIFFV_COORD_HEAD
            # Ref gets ident=0 and we don't have it, so start at 1
            counter += 1
            d = DigPoint(
                r=cur_ch["loc"][:3],
                coord_frame=FIFF.FIFFV_COORD_HEAD,
                kind=FIFF.FIFFV_POINT_EEG,
                ident=counter,
            )
            dig.append(d)
        elif grad and cur_channel_label in grad["label"]:
            cur_ch = _process_channel_meg(cur_ch, grad)
        else:
            if cur_channel_label.startswith("EOG"):
                cur_ch["kind"] = FIFF.FIFFV_EOG_CH
                cur_ch["coil_type"] = FIFF.FIFFV_COIL_EEG
            elif cur_channel_label.startswith("ECG"):
                cur_ch["kind"] = FIFF.FIFFV_ECG_CH
                cur_ch["coil_type"] = FIFF.FIFFV_COIL_EEG_BIPOLAR
            elif cur_channel_label.startswith("STI"):
                cur_ch["kind"] = FIFF.FIFFV_STIM_CH
                cur_ch["coil_type"] = FIFF.FIFFV_COIL_NONE
            else:
                warn(
                    f"Cannot guess the correct type of channel {cur_channel_label}. "
                    "Making it a MISC channel."
                )
                cur_ch["kind"] = FIFF.FIFFV_MISC_CH
                cur_ch["coil_type"] = FIFF.FIFFV_COIL_NONE

        chs.append(cur_ch)
    _ensure_fiducials_head(dig)

    return chs, dig


def _set_sfreq(ft_struct):
    """Set the sample frequency."""
    try:
        sfreq = ft_struct["fsample"]
    except KeyError:
        try:
            time = ft_struct["time"]
        except KeyError:
            raise ValueError("No Source for sfreq found")
        else:
            t1, t2 = float(time[0]), float(time[1])
            sfreq = 1 / (t2 - t1)
    try:
        sfreq = float(sfreq)
    except TypeError:
        warn(
            "FieldTrip structure contained multiple sample rates, trying the "
            f"first of:\n{sfreq} Hz"
        )
        sfreq = float(sfreq.ravel()[0])
    return sfreq


def _set_tmin(ft_struct):
    """Set the start time before the event in evoked data if possible."""
    times = ft_struct["time"]
    time_check = all(times[i][0] == times[i - 1][0] for i, x in enumerate(times))
    if time_check:
        tmin = times[0][0]
    else:
        raise RuntimeError(
            "Loading data with non-uniform times per epoch is not supported"
        )
    return tmin


def _create_events(ft_struct, trialinfo_column):
    """Create an event matrix from the FieldTrip structure."""
    if "trialinfo" not in ft_struct:
        return None

    event_type = ft_struct["trialinfo"]
    event_number = range(len(event_type))

    if trialinfo_column < 0:
        raise ValueError("trialinfo_column must be positive")

    available_ti_cols = 1
    if event_type.ndim == 2:
        available_ti_cols = event_type.shape[1]

    if trialinfo_column > (available_ti_cols - 1):
        raise ValueError(
            "trialinfo_column is higher than the amount of columns in trialinfo."
        )

    event_trans_val = np.zeros(len(event_type))

    if event_type.ndim == 2:
        event_type = event_type[:, trialinfo_column]

    events = (
        np.vstack([np.array(event_number), event_trans_val, event_type]).astype("int").T
    )

    return events


def _create_event_metadata(ft_struct):
    """Create event metadata from trialinfo."""
    pandas = _check_pandas_installed(strict=False)
    if not pandas:
        warn(
            "The Pandas library is not installed. Not returning the original "
            "trialinfo matrix as metadata."
        )
        return None

    metadata = pandas.DataFrame(ft_struct["trialinfo"])

    return metadata


def _process_channel_eeg(cur_ch, elec):
    """Convert EEG channel from FieldTrip to MNE.

    Parameters
    ----------
    cur_ch: dict
        Channel specific dictionary to populate.

    elec: dict
        elec dict as loaded from the FieldTrip structure

    Returns
    -------
    cur_ch: dict
        The original dict (cur_ch) with the added information
    """
    all_labels = np.asanyarray(elec["label"])
    chan_idx_in_elec = np.where(all_labels == cur_ch["ch_name"])[0][0]
    position = np.squeeze(elec["chanpos"][chan_idx_in_elec, :])
    # chanunit = elec['chanunit'][chan_idx_in_elec]  # not used/needed yet
    position_unit = elec["unit"]

    position = position * _unit_dict[position_unit]
    cur_ch["loc"] = np.hstack((position, np.zeros((9,))))
    cur_ch["unit"] = FIFF.FIFF_UNIT_V
    cur_ch["kind"] = FIFF.FIFFV_EEG_CH
    cur_ch["coil_type"] = FIFF.FIFFV_COIL_EEG
    cur_ch["coord_frame"] = FIFF.FIFFV_COORD_HEAD

    return cur_ch


def _process_channel_meg(cur_ch, grad):
    """Convert MEG channel from FieldTrip to MNE.

    Parameters
    ----------
    cur_ch: dict
        Channel specific dictionary to populate.

    grad: dict
        grad dict as loaded from the FieldTrip structure

    Returns
    -------
    dict: The original dict (cur_ch) with the added information
    """
    all_labels = np.asanyarray(grad["label"])
    chan_idx_in_grad = np.where(all_labels == cur_ch["ch_name"])[0][0]
    gradtype = grad["type"]
    chantype = grad["chantype"][chan_idx_in_grad]
    position_unit = grad["unit"]
    position = np.squeeze(grad["chanpos"][chan_idx_in_grad, :])
    position = position * _unit_dict[position_unit]

    if gradtype == "neuromag306" and "tra" in grad and "coilpos" in grad:
        # Try to regenerate original channel pos.
        idx_in_coilpos = np.where(grad["tra"][chan_idx_in_grad, :] != 0)[0]
        cur_coilpos = grad["coilpos"][idx_in_coilpos, :]
        cur_coilpos = cur_coilpos * _unit_dict[position_unit]
        cur_coilori = grad["coilori"][idx_in_coilpos, :]
        if chantype == "megmag":
            position = cur_coilpos[0] - 0.0003 * cur_coilori[0]
        if chantype == "megplanar":
            tmp_pos = cur_coilpos - 0.0003 * cur_coilori
            position = np.average(tmp_pos, axis=0)

    original_orientation = np.squeeze(grad["chanori"][chan_idx_in_grad, :])
    try:
        orientation = rotation3d_align_z_axis(original_orientation).T
    except AssertionError:
        orientation = np.eye(3)
    assert orientation.shape == (3, 3)
    orientation = orientation.flatten()
    # chanunit = grad['chanunit'][chan_idx_in_grad]  # not used/needed yet

    cur_ch["loc"] = np.hstack((position, orientation))
    cur_ch["kind"] = FIFF.FIFFV_MEG_CH
    if chantype == "megmag":
        cur_ch["coil_type"] = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
        cur_ch["unit"] = FIFF.FIFF_UNIT_T
    elif chantype == "megplanar":
        cur_ch["coil_type"] = FIFF.FIFFV_COIL_VV_PLANAR_T1
        cur_ch["unit"] = FIFF.FIFF_UNIT_T_M
    elif chantype == "refmag":
        cur_ch["coil_type"] = FIFF.FIFFV_COIL_MAGNES_REF_MAG
        cur_ch["unit"] = FIFF.FIFF_UNIT_T
    elif chantype == "refgrad":
        cur_ch["coil_type"] = FIFF.FIFFV_COIL_MAGNES_REF_GRAD
        cur_ch["unit"] = FIFF.FIFF_UNIT_T
    elif chantype == "meggrad":
        cur_ch["coil_type"] = FIFF.FIFFV_COIL_AXIAL_GRAD_5CM
        cur_ch["unit"] = FIFF.FIFF_UNIT_T
    else:
        raise RuntimeError(f"Unexpected coil type: {chantype}.")

    cur_ch["coord_frame"] = FIFF.FIFFV_COORD_HEAD

    return cur_ch
