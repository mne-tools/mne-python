import numpy as np
from os import path as op
from numpy.polynomial import legendre
from numpy.testing.utils import assert_allclose, assert_array_equal

from mne import (read_trans, make_surface_mapping, get_meg_helmet_surf,
                 get_head_surface)
from mne.datasets import sample
from mne.forward._lead_dots import (_comp_sum_eeg, _comp_sums_meg,
                                    _get_legen_table,
                                    _get_legen_lut_fast,
                                    _get_legen_lut_accurate)
from mne.fiff import read_info
from mne.transforms import _get_mri_head_t_from_trans_file
from mne.fixes import partial
from mne.externals.six.moves import zip


base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
short_raw_fname = op.join(base_dir, 'test_raw.fif')
trans_txt_fname = op.join(base_dir, 'sample-audvis-raw-trans.txt')

data_path = sample.data_path(download=False)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
subjects_dir = op.join(data_path, 'subjects')


def test_legendre_val():
    """Test Legendre polynomial (derivative) equivalence
    """
    # check table equiv
    xs = np.linspace(-1., 1., 4000)
    n_terms = 100

    # True, numpy
    vals_np = legendre.legvander(xs, n_terms - 1)

    # Table approximation
    for fun, nc in zip([_get_legen_lut_fast, _get_legen_lut_accurate],
                       [100, 50]):
        lut, n_fact = _get_legen_table('eeg', n_coeff=nc)
        vals_i = fun(xs, lut)
        # Need a "1:" here because we omit the first coefficient in our table!
        assert_allclose(vals_np[:, 1:vals_i.shape[1] + 1], vals_i,
                        rtol=1e-2, atol=1e-3)

        # Now let's look at our sums
        ctheta = np.random.rand(20, 30) * 2.0 - 1.0
        beta = np.random.rand(20, 30) * 0.8
        lut_fun = partial(fun, lut=lut)
        c1 = _comp_sum_eeg(beta.flatten(), ctheta.flatten(), lut_fun, n_fact)
        c1.shape = beta.shape

        # compare to numpy
        n = np.arange(1, n_terms, dtype=float)[:, np.newaxis, np.newaxis]
        coeffs = np.zeros((n_terms,) + beta.shape)
        coeffs[1:] = (np.cumprod([beta] * (n_terms - 1), axis=0)
                      * (2.0 * n + 1.0) * (2.0 * n + 1.0) / n)
        # can't use tensor=False here b/c it isn't in old numpy
        c2 = np.empty((20, 30))
        for ci1 in range(20):
            for ci2 in range(30):
                c2[ci1, ci2] = legendre.legval(ctheta[ci1, ci2],
                                               coeffs[:, ci1, ci2])
        assert_allclose(c1, c2, 1e-2, 1e-3)  # close enough...

    # compare fast and slow for MEG
    ctheta = np.random.rand(20 * 30) * 2.0 - 1.0
    beta = np.random.rand(20 * 30) * 0.8
    lut, n_fact = _get_legen_table('meg', n_coeff=50)
    fun = partial(_get_legen_lut_fast, lut=lut)
    coeffs = _comp_sums_meg(beta, ctheta, fun, n_fact, False)
    lut, n_fact = _get_legen_table('meg', n_coeff=100)
    fun = partial(_get_legen_lut_accurate, lut=lut)
    coeffs = _comp_sums_meg(beta, ctheta, fun, n_fact, False)


@sample.requires_sample_data
def _test_eeg_field_interpolation():
    """Test interpolation of EEG field onto head
    """
    trans = read_trans(trans_fname)
    info = read_info(raw_fname)
    surf = get_head_surface('sample', subjects_dir=subjects_dir)
    data = make_surface_mapping(info, surf, trans, 'eeg')
    assert_array_equal(data.shape, (2562, 59))  # maps data onto surf


def _test_meg_field_interpolation_helmet():
    """Test interpolation of MEG field onto helmet
    """
    info = read_info(short_raw_fname)
    surf = get_meg_helmet_surf(info)
    trans = _get_mri_head_t_from_trans_file(trans_txt_fname)
    data = make_surface_mapping(info, surf, trans, 'meg')
    assert_array_equal(data.shape, (304, 305))  # data onto surf
