# Authors: George O'Neill <g.o'neill@ucl.ac.uk>
#
# License: BSD-3-Clause

from numpy import isnan, empty
from numpy.testing import assert_array_equal, assert_array_almost_equal

import pytest

from mne.datasets import testing
from mne.io import read_raw_fil
from mne.io.fil.sensors import _get_pos_units
from mne.io.pick import pick_types

import scipy.io


fil_path = testing.data_path(download=False) / "FIL"


# TODO: Ignore this warning in all these tests until we deal with this properly
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*problems later!:RuntimeWarning",
)


def unpack_mat(matin):
    """Extract relevant entries from unstructred readmat."""
    data = matin["data"]
    grad = data[0][0]["grad"]
    label = list()
    coil_label = list()
    for ii in range(len(data[0][0]["label"])):
        label.append(str(data[0][0]["label"][ii][0][0]))
    for ii in range(len(grad[0][0]["label"])):
        coil_label.append(str(grad[0][0]["label"][ii][0][0]))

    matout = {
        "label": label,
        "trial": data["trial"][0][0][0][0],
        "coil_label": coil_label,
        "coil_pos": grad[0][0]["coilpos"],
        "coil_ori": grad[0][0]["coilori"],
    }
    return matout


def _match_str(A_list, B_list):
    """Locate where in a list matches another."""
    B_inds = list()
    for ii in A_list:
        if ii in B_list:
            B_inds.append(B_list.index(ii))
    return B_inds


def _get_channels_with_positions(info):
    """Parse channel orientation/position."""
    ch_list = list()
    ch_inds = list()
    for ii, ch in enumerate(info["chs"]):
        if not (any(isnan(ch["loc"]))):
            ch_inds.append(ii)
            ch_list.append(ch["ch_name"])
    return ch_list, ch_inds


def _fil_megmag(raw_test, raw_mat):
    """Test the magnetometer channels."""
    test_inds = pick_types(raw_test.info, meg="mag", ref_meg=False, exclude="bads")
    test_list = list(raw_test.info["ch_names"][i] for i in test_inds)
    mat_list = raw_mat["label"]
    mat_inds = _match_str(test_list, mat_list)

    assert len(mat_inds) == len(
        test_inds
    ), "Number of magnetometer channels in RAW does not match .mat file!"

    a = raw_test._data[test_inds, :]
    b = raw_mat["trial"][mat_inds, :] * 1e-15  # fT to T

    assert_array_equal(a, b)


def _fil_stim(raw_test, raw_mat):
    """Test the trigger channels."""
    test_inds = pick_types(
        raw_test.info, meg=False, ref_meg=False, stim=True, exclude="bads"
    )
    test_list = list(raw_test.info["ch_names"][i] for i in test_inds)
    mat_list = raw_mat["label"]
    mat_inds = _match_str(test_list, mat_list)

    assert len(mat_inds) == len(
        test_inds
    ), "Number of stim channels in RAW does not match .mat file!"

    a = raw_test._data[test_inds, :]
    b = raw_mat["trial"][mat_inds, :]  # fT to T

    assert_array_equal(a, b)


def _fil_sensorpos(raw_test, raw_mat):
    """Test the sensor positions/orientations."""
    test_list, test_inds = _get_channels_with_positions(raw_test.info)
    grad_list = raw_mat["coil_label"]
    grad_inds = _match_str(test_list, grad_list)

    assert len(grad_inds) == len(
        test_inds
    ), "Number of channels with position data in RAW does not match .mat file!"

    mat_pos = raw_mat["coil_pos"][grad_inds, :]
    mat_ori = raw_mat["coil_ori"][grad_inds, :]
    _, sf1 = _get_pos_units(mat_pos)

    test_pos = empty((len(test_inds), 3))
    test_ori = empty((len(test_inds), 3))
    for i, ind in enumerate(test_inds):
        test_pos[i, :] = raw_test.info["chs"][ind]["loc"][0:3]
        test_ori[i, :] = raw_test.info["chs"][ind]["loc"][-3:]
    _, sf2 = _get_pos_units(test_pos)

    assert_array_almost_equal(test_pos / sf2, mat_pos / sf1)
    assert_array_almost_equal(test_ori, mat_ori)


@testing.requires_testing_data
def test_fil_all():
    """Test FIL reader, match to known answers from .mat file."""
    binname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
    matname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_fieldtrip.mat"

    raw = read_raw_fil(binname)
    raw.load_data(verbose=False)
    tmp = scipy.io.loadmat(matname)
    mat = unpack_mat(tmp)

    _fil_megmag(raw, mat)
    _fil_stim(raw, mat)
    _fil_sensorpos(raw, mat)
