# Authors: George O'Neill <g.o'neill@ucl.ac.uk>
#
# License: BSD-3-Clause


import numpy as np
import pytest

from mne.datasets import testing
from mne.io import read_raw_fil
from mne.preprocessing.hfc import (compute_proj_hfc,
                                   _filter_channels_with_positions)
from mne.io.pick import pick_types, pick_info

import scipy.io

fil_path = testing.data_path(download=False) / 'FIL'
fname_root = "sub-noise_ses-001_task-noise220622_run-001"

# TODO: Ignore this warning in all these tests until we deal with this properly
pytestmark = pytest.mark.filterwarnings(
    'ignore:.*problems later!:RuntimeWarning',
    'ignore:.*Projection vector.*'
)


def _unpack_mat(matin):
    """Extract relevant entries from unstructred readmat."""
    data = matin['data']
    grad = data[0][0]['grad']
    label = list()
    coil_label = list()
    for ii in range(len(data[0][0]['label'])):
        label.append(str(data[0][0]['label'][ii][0][0]))
    for ii in range(len(grad[0][0]['label'])):
        coil_label.append(str(grad[0][0]['label'][ii][0][0]))

    matout = {'label': label,
              'trial': data['trial'][0][0][0][0],
              'coil_label': coil_label,
              'coil_pos': grad[0][0]['coilpos'],
              'coil_ori': grad[0][0]['coilori']}
    return matout


def _angle_between(A, B):
    """Measure the angle between two vectors."""
    A = A[:, np.newaxis]
    nA = np.linalg.norm(A, axis=0)
    nB = np.linalg.norm(B, axis=1)
    A = A / nA
    B = B / nB[:, np.newaxis]
    d = np.dot(B, A)
    idh = [i for i in range(len(d)) if d[i] > 1.]
    d[idh] = 1
    idl = [i for i in range(len(d)) if d[i] < -1.]
    d[idl] = -1
    ang = np.arccos(d)
    ang = np.minimum(ang, (2 * np.pi) - ang)
    return ang


def _match_str(A_list, B_list):
    """Locate where in a list matches another."""
    B_inds = list()
    for ii in A_list:
        if ii in B_list:
            B_inds.append(B_list.index(ii))
    return B_inds


def _compare_hfc_results(order, rtol=1e-7):
    """Apply HFC and compare to previous computed solutions."""
    binname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
    raw = read_raw_fil(binname, verbose=False)
    raw.load_data(verbose=False)
    projs = compute_proj_hfc(raw.info, order=order, accuracy='point')
    raw.add_proj(projs).apply_proj()

    fname = fname_root + "_hfc_l{0}.mat".format(order)
    matname = fil_path / fname
    tmp = scipy.io.loadmat(matname)
    mat = _unpack_mat(tmp)

    test_list = projs[0]['data']['col_names']
    test_inds = _match_str(test_list, raw.ch_names)
    mat_list = mat["coil_label"]
    mat_inds = _match_str(test_list, mat_list)

    a = mat['trial'][mat_inds]
    b = raw._data[test_inds, 0:300] * 1e15

    np.testing.assert_allclose(a, b, verbose=False)


@testing.requires_testing_data
def test_l1_basis_orientations():
    """Test that angles between the basis components matches orientations."""
    binname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
    raw = read_raw_fil(binname)
    projs = compute_proj_hfc(raw.info, accuracy='point')
    basis = np.hstack([p['data']['data'].T for p in projs])
    ang_model = np.concatenate([_angle_between(b, basis)
                                for b in basis])

    idx = pick_types(raw.info, meg='mag')
    idx_loc = _filter_channels_with_positions(raw.info, idx)
    chs = pick_info(raw.info, idx_loc)['chs']
    ori_sens = np.concatenate([ch['loc'][-3:] for ch in chs])
    ori_sens = np.reshape(ori_sens, (len(idx_loc), 3))
    ang_sens = np.concatenate([_angle_between(o, ori_sens)
                               for o in ori_sens])

    np.testing.assert_allclose(ang_sens, ang_model, atol=1e-7)


@testing.requires_testing_data
def test_l1_correction():
    """Compare HFC (l=1) to previous computed results in another language."""
    _compare_hfc_results(1)


@testing.requires_testing_data
def test_l2_correction():
    """Compare HFC (l=2) to previous computed results in another language."""
    _compare_hfc_results(2)


@testing.requires_testing_data
def test_l3_correction():
    """Compare HFC (l=3) to previous computed results in another language."""
    _compare_hfc_results(3)
