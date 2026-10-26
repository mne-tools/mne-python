# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from inspect import getfullargspec

import numpy as np

from ..annotations import _sync_onset
from ..utils import _check_eeglabio_installed, check_version

_check_eeglabio_installed()
import eeglabio.epochs  # noqa: E402
import eeglabio.raw  # noqa: E402

# MATLAB v5 .mat files cannot hold variables of 2 GB or more
_V5_MAX_BYTES = 2**31


def _export_raw(fname, raw):
    # load data first
    raw.load_data()

    # remove extra epoc and STI channels
    drop_chs = ["epoc"]
    # filenames attribute of RawArray is filled with None
    if raw.filenames[0] and raw.filenames[0].suffix != ".fif":
        drop_chs.append("STI 014")

    ch_names = [ch for ch in raw.ch_names if ch not in drop_chs]
    cart_coords = _get_als_coords_from_chs(raw.info["chs"], drop_chs)

    if raw.annotations:
        annotations = [
            # eeglabio builds a structured array, which does not support StringDType
            raw.annotations.description.tolist(),
            # subtract raw.first_time because EEGLAB marks events starting from
            # the first available data point and ignores raw.first_time
            _sync_onset(raw, raw.annotations.onset, inverse=False),
            raw.annotations.duration,
        ]
    else:
        annotations = None
    data = raw.get_data(picks=ch_names)
    eeglabio.raw.export_set(
        fname,
        data=data,
        sfreq=raw.info["sfreq"],
        ch_names=ch_names,
        ch_locs=cart_coords,
        annotations=annotations,
        **_fmt_kwargs(data),
    )


def _export_epochs(fname, epochs):
    _check_eeglabio_installed()
    # load data first
    epochs.load_data()

    # remove extra epoc and STI channels
    drop_chs = ["epoc", "STI 014"]
    ch_names = [ch for ch in epochs.ch_names if ch not in drop_chs]
    cart_coords = _get_als_coords_from_chs(epochs.info["chs"], drop_chs)

    if epochs.annotations:
        annot = [
            # eeglabio builds a structured array, which does not support StringDType
            epochs.annotations.description.tolist(),
            epochs.annotations.onset,
            epochs.annotations.duration,
        ]
    else:
        annot = None

    events = epochs.events
    kwargs = dict()
    # TODO VERSION: remove once eeglabio > 0.1.3 is required (jackz314/eeglabio#26)
    if not check_version("eeglabio", "0.1.4"):
        # older eeglabio uses events[:, 0] as positions in the concatenated data
        zero = np.clip(epochs.time_as_index(0)[0], 0, len(epochs.times) - 1)
        events = events.copy()
        events[:, 0] = np.arange(len(epochs)) * len(epochs.times) + zero
        if "epoch_indices" in getfullargspec(eeglabio.epochs.export_set).kwonlyargs:
            # eeglabio 0.1.2-0.1.3 use these as 1-based EEGLAB epoch numbers
            kwargs["epoch_indices"] = np.arange(1, len(epochs) + 1)

    data = epochs.get_data(picks=ch_names)
    kwargs.update(_fmt_kwargs(data))
    eeglabio.epochs.export_set(
        fname,
        data=data,
        sfreq=epochs.info["sfreq"],
        events=events,
        tmin=epochs.tmin,
        tmax=epochs.tmax,
        ch_names=ch_names,
        event_id=epochs.event_id,
        ch_locs=cart_coords,
        annotations=annot,
        **kwargs,
    )


def _fmt_kwargs(data):
    # eeglabio exports single precision by default
    if data.size * 4 >= _V5_MAX_BYTES and check_version("eeglabio", "0.1.4"):
        return dict(fmt="v7.3")
    return dict()


def _get_als_coords_from_chs(chs, drop_chs=None):
    """Extract channel locations in ALS format (x, y, z) from a chs instance.

    Returns
    -------
    None if no valid coordinates are found (all zeros)
    """
    if drop_chs is None:
        drop_chs = []
    cart_coords = np.array([d["loc"][:3] for d in chs if d["ch_name"] not in drop_chs])
    if cart_coords.any():  # has coordinates
        # (-y x z) to (x y z)
        cart_coords[:, 0] = -cart_coords[:, 0]  # -y to y
        # swap x (1) and y (0)
        cart_coords[:, [0, 1]] = cart_coords[:, [1, 0]]
    else:
        cart_coords = None
    return cart_coords
