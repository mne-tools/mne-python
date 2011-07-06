import os.path as op

from numpy.testing import assert_array_almost_equal

from .. import Raw, pick_types, compute_spatial_vectors
from ..proj import make_projector
from ..open import fiff_open
from ... import read_events, Epochs

raw_fname = op.join(op.dirname(__file__), 'data', 'test_raw.fif')
event_fname = op.join(op.dirname(__file__), 'data', 'test-eve.fif')
proj_fname = op.join(op.dirname(__file__), 'data', 'test_proj.fif')

# XXX
# def test_compute_spatial_vectors():
#     """Test SSP computation
#     """
#     event_id, tmin, tmax = 1, -0.2, 0.3
# 
#     raw = Raw(raw_fname)
#     events = read_events(event_fname)
#     exclude = ['MEG 2443', 'EEG 053']
#     picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
#                             exclude=exclude)
#     epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
#                         baseline=(None, 0), proj=False,
#                         reject=dict(mag=5000e-15, grad=16e-10))
# 
#     projs = compute_spatial_vectors(epochs, n_grad=1, n_mag=2, n_eeg=2)
# 
#     proj, nproj, U = make_projector(projs, epochs.ch_names, bads=[])
#     assert nproj == 3
#     assert U.shape[1] == 3
# 
#     epochs.info['projs'] += projs
#     evoked = epochs.average()
#     evoked.save('foo.fif')
# 
#     fid, tree, _ = fiff_open(proj_fname)
#     projs = read_proj(fid, tree)
