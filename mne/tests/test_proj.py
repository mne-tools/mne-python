import os.path as op
from nose.tools import assert_true

import numpy as np
from numpy.testing import assert_array_almost_equal

from mne.fiff import Raw, pick_types
from mne import compute_proj_epochs, compute_proj_evoked
from mne.fiff.proj import make_projector, activate_proj
from mne.proj import read_proj
from mne import read_events, Epochs

base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')
proj_fname = op.join(base_dir, 'test_proj.fif')


def test_compute_proj():
    """Test SSP computation"""
    event_id, tmin, tmax = 1, -0.2, 0.3

    raw = Raw(raw_fname)
    events = read_events(event_fname)
    exclude = []
    bad_ch = 'MEG 2443'
    picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                            exclude=exclude)
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=None, proj=False)

    evoked = epochs.average()
    projs = compute_proj_epochs(epochs, n_grad=1, n_mag=1, n_eeg=0)

    projs2 = read_proj(proj_fname)

    assert_true(len(projs) == len(projs2))

    for p1, p2 in zip(projs, projs2):
        assert_true(p1['desc'] == p2['desc'])
        assert_true(p1['data']['col_names'] == p2['data']['col_names'])
        assert_true(p1['active'] == p2['active'])
        # compare with sign invariance
        p1_data = p1['data']['data'] * np.sign(p1['data']['data'][0, 0])
        p2_data = p2['data']['data'] * np.sign(p2['data']['data'][0, 0])
        if bad_ch in p1['data']['col_names']:
            bad = p1['data']['col_names'].index('MEG 2443')
            mask = np.ones(p1_data.size, dtype=np.bool)
            mask[bad] = False
            p1_data = p1_data[:, mask]
            p2_data = p2_data[:, mask]
        corr = np.corrcoef(p1_data, p2_data)[0, 1]
        assert_array_almost_equal(corr, 1.0, 7)

    # test that you can compute the projection matrix
    projs = activate_proj(projs)
    proj, nproj, U = make_projector(projs, epochs.ch_names, bads=[])

    assert_true(nproj == 2)
    assert_true(U.shape[1] == 2)

    # test that you can save them
    epochs.info['projs'] += projs
    evoked = epochs.average()
    evoked.save('foo.fif')

    projs = read_proj(proj_fname)

    projs_evoked = compute_proj_evoked(evoked, n_grad=1, n_mag=1, n_eeg=0)
    # XXX : test something
