# Authors: The MNE-Python contributors.
# License: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray


@pytest.fixture
def multi_wavelength_raw(request: pytest.FixtureRequest) -> RawArray:
    """Create a raw CW fNIRS object with 3 wavelengths per source-detector pair."""
    n_pairs = getattr(request, "param", None)
    if n_pairs is None:
        raise RuntimeError(
            "parametrize multi_wavelength_raw with the desired number of optode pairs"
        )
    sampling_freq = 10.0
    n_times = 128
    freqs = [700, 730, 850]

    ch_names = [f"S{ii}_D{ii} {wl}" for ii in range(1, n_pairs + 1) for wl in freqs]
    rng = np.random.default_rng()
    data = rng.random((len(ch_names), n_times)) + 0.01

    info = create_info(
        ch_names=ch_names, ch_types="fnirs_cw_amplitude", sfreq=sampling_freq
    )
    raw = RawArray(data, info, verbose=True)
    for ii, (ch, freq) in enumerate(zip(raw.info["chs"], freqs * n_pairs)):
        ch["loc"][9] = freq
        ch["loc"][3:6] = (ii // 3 * 0.01, 0.0, 0.0)
        ch["loc"][6:9] = (ii // 3 * 0.01, 0.03, 0.0)

    return raw
