"""Optional edfio-backed reader engine for EDF files.

This module implements an alternative ``engine="edfio"`` for
:func:`mne.io.read_raw_edf` that parses the file with the
`edfio <https://github.com/the-siesta-group/edfio>`_ package instead of the
native reader. It is faster on uniform-sampling-rate recordings and always
returns preloaded data.

Scope (kept deliberately minimal):

- uniform sampling rates only (the native engine handles mixed rates);
- all channels are typed ``eeg``;
- ``meas_date`` is not set;
- data is returned in volts, scaled from the header's physical dimension
  using the same unit mapping as the native reader.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors

import numpy as np

from ..._fiff.meas_info import _unique_channel_names
from ...annotations import Annotations
from ...utils import _check_fname, fill_doc, verbose
from ..base import BaseRaw

_UNIT_MULT = {
    "\u03bcV": 1e-6,  # greek mu
    "\u00b5V": 1e-6,  # micro symbol
    "uV": 1e-6,
    "mV": 1e-3,
}


class _RawEdfio(BaseRaw):
    """Raw from edfio-parsed EDF (always preloaded)."""

    _extra_attributes = ()

    def __init__(self, info, data, annotations, *, verbose=None):
        super().__init__(
            info,
            preload=data,
            last_samps=[data.shape[1] - 1],
            filenames=None,
            orig_format="double",
            verbose=verbose,
        )
        if len(annotations):
            self.set_annotations(annotations)


@fill_doc
@verbose
def read_raw_edf_edfio(
    input_fname,
    *,
    preload=True,
    exclude=(),
    include=None,
    verbose=None,
) -> _RawEdfio:
    """Read an EDF file using the edfio parser.

    Parameters
    ----------
    input_fname : path-like
        Path to the EDF/EDF+ file.
    %(preload)s
        The edfio engine currently supports only preloaded reads; ``True``
        (or a truthy string) is required.
    exclude : list of str
        Channel names to exclude.
    include : list of str | None
        Restrict channels to these names (after ``exclude``).
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        Preloaded raw data in volts.

    Notes
    -----
    Uniform sampling rates only; all channels are typed ``eeg``;
    ``info['meas_date']`` is not populated.
    """
    from edfio import read_edf as _read_edf

    input_fname = str(_check_fname(input_fname, "read", True, "input_fname"))
    if not preload:
        raise NotImplementedError(
            'The "edfio" engine currently always loads data into memory; '
            'use preload=True.'
        )
    edf = _read_edf(input_fname)

    signals = edf.signals
    ch_names = [sig.label for sig in signals]
    sfreqs = {float(sig.sampling_frequency) for sig in signals}
    if len(sfreqs) != 1:
        raise NotImplementedError(
            "The edfio engine requires a uniform sampling rate; this file has "
            f"{len(sfreqs)} distinct rates. Use the default engine instead."
        )
    sfreq = sfreqs.pop()

    keep = np.arange(len(signals))
    if include is not None:
        keep = [i for i in keep if ch_names[i] in set(include)]
    if len(exclude):
        excluded = set(exclude)
        keep = [i for i in keep if ch_names[i] not in excluded]
    keep = np.asarray(keep, dtype=int)
    if keep.size == 0:
        raise ValueError("No channels selected")

    ch_names = list(np.array(ch_names)[keep])
    ch_names = _unique_channel_names(ch_names)
    unit_mults = np.array(
        [
            _UNIT_MULT.get(str(signals[i].physical_dimension).strip(), 1.0)
            for i in keep
        ],
        dtype=float,
    )
    # Stack digital samples once, then decode all channels in two fused
    # passes: physical = (digital + offset) * (gain * unit_mult), matching
    # edfio's calibration op order.
    n_times = min(len(signals[i].digital) for i in keep)
    dig = np.empty((len(keep), n_times), dtype=np.int16)
    gains = np.empty(len(keep))
    offsets = np.empty(len(keep))
    for row_i, sig_i in enumerate(keep):
        digital = signals[sig_i].digital
        dig[row_i] = digital[:n_times]
        sig = signals[sig_i]
        gains[row_i] = (sig.physical_max - sig.physical_min) / (
            sig.digital_max - sig.digital_min
        )
        offsets[row_i] = sig.physical_max / gains[row_i] - sig.digital_max

    info = _make_info_edfio(ch_names, sfreq)
    data = np.empty((len(keep), n_times), dtype=np.float64)
    np.add(dig, offsets[:, np.newaxis], out=data, casting="unsafe")
    data *= (gains * unit_mults)[:, np.newaxis]

    annots = edf.annotations
    mne_annots = Annotations(
        onset=[a.onset for a in annots],
        duration=[a.duration for a in annots],
        description=[str(a.text) for a in annots],
    )
    return _RawEdfio(info, data, mne_annots, verbose=verbose)


def _make_info_edfio(ch_names, sfreq):
    import mne

    return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
