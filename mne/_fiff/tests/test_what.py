# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD

import glob
from pathlib import Path

import numpy as np
import pytest

from mne import what, create_info
from mne.datasets import testing
from mne.io import RawArray
from mne.preprocessing import ICA
from mne.utils import _record_warnings

data_path = testing.data_path(download=False)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_what(tmp_path, verbose_debug):
    """Test mne.what."""
    pytest.importorskip("sklearn")
    # ICA
    ica = ICA(max_iter=1)
    raw = RawArray(np.random.RandomState(0).randn(3, 10), create_info(3, 1000.0, "eeg"))
    with _record_warnings():  # convergence sometimes
        ica.fit(raw)
    fname = tmp_path / "x-ica.fif"
    ica.save(fname)
    assert what(fname) == "ica"
    # test files
    fnames = glob.glob(str(data_path / "MEG" / "sample" / "*.fif"))
    fnames += glob.glob(str(data_path / "subjects" / "sample" / "bem" / "*.fif"))
    fnames = sorted(fnames)
    want_dict = dict(
        eve="events",
        ave="evoked",
        cov="cov",
        inv="inverse",
        fwd="forward",
        trans="transform",
        proj="proj",
        raw="raw",
        meg="raw",
        sol="bem solution",
        bem="bem surfaces",
        src="src",
        dense="bem surfaces",
        sparse="bem surfaces",
        head="bem surfaces",
        fiducials="fiducials",
    )
    for fname in fnames:
        kind = Path(fname).stem.split("-")[-1]
        if len(kind) > 5:
            kind = kind.split("_")[-1]
        this = what(fname)
        assert this == want_dict[kind]
    fname = data_path / "MEG" / "sample" / "sample_audvis-ave_xfit.dip"
    assert what(fname) == "unknown"
