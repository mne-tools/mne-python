import numpy as np
from os import path as op
from numpy.polynomial import legendre
from numpy.testing.utils import assert_allclose, assert_array_equal

from mne import (read_trans, make_surface_mapping, get_meg_helmet_surf,
                 get_head_surface)
from mne.datasets import sample
from mne.forward._lead_dots import _comp_sum
from mne.fiff import read_info
from mne.transforms import _get_mri_head_t_from_trans_file


base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
short_raw_fname = op.join(base_dir, 'test_raw.fif')
trans_txt_fname = op.join(base_dir, 'sample-audvis-raw-trans.txt')

data_path = sample.data_path(download=False)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
subjects_dir = op.join(data_path, 'subjects')


def test_legendre_val():
    """Test legendre equivalence
    """
    # pick some values for beta and ctheta, not too close to 0 or 1
    beta = np.random.rand(10, 11) * 0.8 + 0.1
    ctheta = np.random.rand(10, 11) * 0.8 + 0.1

    # use our "optimized" code
    c1 = np.array([[_comp_sum(bb, cc) for bb, cc in zip(b, c)]
                   for b, c in zip(beta, ctheta)])

    # compare to numpy
    n_terms = 100
    n = np.arange(1, n_terms, dtype=float)[:, np.newaxis, np.newaxis]
    coeffs = np.empty((n_terms,) + beta.shape)
    coeffs[1:] = (np.cumprod([beta] * (n_terms - 1), axis=0)
                  * (2.0 * n + 1.0) * (2.0 * n + 1.0) / n)
    c2 = legendre.legval(ctheta, coeffs, tensor=False)
    assert_allclose(c1, c2, 1e-1, 1e-3)


@sample.requires_sample_data
def _test_eeg_field_interpolation():
    """Test interpolation of EEG field onto head
    """
    trans = read_trans(trans_fname)
    info = read_info(raw_fname)
    surf = get_head_surface('sample', subjects_dir=subjects_dir)
    data = make_surface_mapping(info, surf, trans, 'meg')
    assert_array_equal(data.shape, (2562, 59))  # maps data onto surf


def _test_meg_field_interpolation_helmet():
    """Test interpolation of MEG field onto helmet
    """
    info = read_info(short_raw_fname)
    surf = get_meg_helmet_surf(info)
    trans = _get_mri_head_t_from_trans_file(trans_txt_fname)
    data = make_surface_mapping(info, surf, trans, 'meg')
    assert_array_equal(data.shape, (304, 305))  # data onto surf
