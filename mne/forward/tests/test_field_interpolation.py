import numpy as np
from os import path as op
from numpy.polynomial import legendre
from numpy.testing.utils import assert_allclose, assert_array_equal

from mne import (read_trans, make_surface_mapping, get_meg_helmet_surf,
                 get_head_surface)
from mne.datasets import sample
from mne.forward._lead_dots import _comp_sum, _get_legen_table, _get_legen_lut
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
    # check table equiv
    xs = np.linspace(-1., 1., 4000)

    # True, numpy
    vals_np = legendre.legval(xs, np.ones(100), tensor=False)

    # Table approximation
    lut, n_fact = _get_legen_table()
    vals_i = np.sum(_get_legen_lut(xs, lut), axis=1)
    assert_allclose(vals_np, vals_i, rtol=1e-4, atol=1e-5)

    beta = np.random.RandomState(0).rand(20, 30) * 0.8
    ctheta = np.random.RandomState(0).rand(20, 30) * 2.0 - 1.0

    lut, n_fact = _get_legen_table()
    c1 = _comp_sum(beta.flatten(), ctheta.flatten(), lut, n_fact)
    c1.shape = beta.shape

    # compare to numpy
    n_terms = 100
    n = np.arange(1, n_terms, dtype=float)[:, np.newaxis, np.newaxis]
    coeffs = np.zeros((n_terms,) + beta.shape)
    coeffs[1:] = (np.cumprod([beta] * (n_terms - 1), axis=0)
                  * (2.0 * n + 1.0) * (2.0 * n + 1.0) / n)
    c2 = legendre.legval(ctheta, coeffs, tensor=False)
    assert_allclose(c1, c2, 1e-3, 1e-5)


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
