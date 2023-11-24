import numpy as np
import pytest

import mne
from mne._fiff.constants import FIFF

# read and preprocess raw data
fpath = mne.datasets.testing.data_path(download=False)
fname = fpath / "eyetrack" / "test_eyelink.asc"
raw = mne.io.read_raw_eyelink(fname)
cal = mne.preprocessing.eyetracking.read_eyelink_calibration(fname)[0]
cal["screen_size"] = ((569 / 1000), (340 / 1000))
cal["screen_resolution"] = (1920, 1080)
cal["screen_distance"] = 0.65
mne.preprocessing.eyetracking.interpolate_blinks(raw, interpolate_gaze=True)
epochs = mne.make_fixed_length_epochs(raw, preload=True)
evoked = epochs.average()


@pytest.mark.parametrize("inst", [raw, epochs, evoked])
def test_convert_units(inst):
    """Test unit conversion."""
    # roundtrip conversion should be identical to original data
    data_orig = inst.get_data(picks=[0])  # take the first x-coord channel
    mne.preprocessing.eyetracking.convert_units(inst, calibration=cal, to="radians")
    assert inst.info["chs"][0]["unit"] == FIFF.FIFF_UNIT_RAD
    mne.preprocessing.eyetracking.convert_units(inst, calibration=cal, to="pixels")
    assert inst.info["chs"][1]["unit"] == FIFF.FIFF_UNIT_PX
    data_new = inst.get_data(picks=[0])
    # mask nan values. only necessary for raw but still works for epochs/evoked
    orig_masked = np.ma.masked_where(np.isnan(data_orig.squeeze()), data_orig.squeeze())
    new_masked = np.ma.masked_where(np.isnan(data_new.squeeze()), data_new.squeeze())
    np.testing.assert_allclose(orig_masked, new_masked)
