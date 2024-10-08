# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..utils import _check_eeglabio_installed

_check_eeglabio_installed()
import eeglabio.epochs  # noqa: E402
import eeglabio.raw  # noqa: E402


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

    annotations = [
        raw.annotations.description,
        raw.annotations.onset,
        raw.annotations.duration,
    ]
    eeglabio.raw.export_set(
        fname,
        data=raw.get_data(picks=ch_names),
        sfreq=raw.info["sfreq"],
        ch_names=ch_names,
        ch_locs=cart_coords,
        annotations=annotations,
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
            epochs.annotations.description,
            epochs.annotations.onset,
            epochs.annotations.duration,
        ]
    else:
        annot = None

    eeglabio.epochs.export_set(
        fname,
        data=epochs.get_data(picks=ch_names),
        sfreq=epochs.info["sfreq"],
        events=epochs.events,
        tmin=epochs.tmin,
        tmax=epochs.tmax,
        ch_names=ch_names,
        event_id=epochs.event_id,
        ch_locs=cart_coords,
        annotations=annot,
    )


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
