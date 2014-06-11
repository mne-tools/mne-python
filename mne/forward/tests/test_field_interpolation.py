import numpy as np
from os import path as op
from numpy.polynomial import legendre
from numpy.testing.utils import assert_allclose, assert_array_equal
from nose.tools import assert_raises, assert_true

from mne.forward import _make_surface_mapping, make_field_map
from mne.surface import get_meg_helmet_surf, get_head_surf
from mne.datasets import sample
from mne.forward._lead_dots import (_comp_sum_eeg, _comp_sums_meg,
                                    _get_legen_table,
                                    _get_legen_lut_fast,
                                    _get_legen_lut_accurate)
from mne import pick_types_evoked, read_evokeds
from mne.fixes import partial
from mne.externals.six.moves import zip


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')

data_path = sample.data_path(download=False)
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
subjects_dir = op.join(data_path, 'subjects')


def test_legendre_val():
    """Test Legendre polynomial (derivative) equivalence
    """
    # check table equiv
    xs = np.linspace(-1., 1., 1000)
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
                        rtol=1e-2, atol=5e-3)

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


def test_legendre_table():
    """Test Legendre table calculation
    """
    # double-check our table generation
    n_do = 10
    for ch_type in ['eeg', 'meg']:
        lut1, n_fact1 = _get_legen_table(ch_type, n_coeff=50)
        lut1 = lut1[:, :n_do - 1].copy()
        n_fact1 = n_fact1[:n_do - 1].copy()
        lut2, n_fact2 = _get_legen_table(ch_type, n_coeff=n_do,
                                         force_calc=True)
        assert_allclose(lut1, lut2)
        assert_allclose(n_fact1, n_fact2)


@sample.requires_sample_data
def test_make_field_map_eeg():
    """Test interpolation of EEG field onto head
    """
    evoked = read_evokeds(evoked_fname, condition='Left Auditory')
    evoked.info['bads'] = ['MEG 2443', 'EEG 053']  # add some bads
    surf = get_head_surf('sample', subjects_dir=subjects_dir)
    # we must have trans if surface is in MRI coords
    assert_raises(ValueError, _make_surface_mapping, evoked.info, surf, 'eeg')

    evoked = pick_types_evoked(evoked, meg=False, eeg=True)
    fmd = make_field_map(evoked, trans_fname=trans_fname,
                         subject='sample', subjects_dir=subjects_dir)

    # trans is necessary for EEG only
    assert_raises(RuntimeError, make_field_map, evoked, trans_fname=None,
                  subject='sample', subjects_dir=subjects_dir)

    fmd = make_field_map(evoked, trans_fname=trans_fname,
                         subject='sample', subjects_dir=subjects_dir)
    assert_true(len(fmd) == 1)
    assert_array_equal(fmd[0]['data'].shape, (2562, 59))  # maps data onto surf
    assert_true(len(fmd[0]['ch_names']), 59)


def test_make_field_map_meg():
    """Test interpolation of MEG field onto helmet
    """
    evoked = read_evokeds(evoked_fname, condition='Left Auditory')
    info = evoked.info
    surf = get_meg_helmet_surf(info)
    # let's reduce the number of channels by a bunch to speed it up
    info['bads'] = info['ch_names'][:200]
    # bad ch_type
    assert_raises(ValueError, _make_surface_mapping, info, surf, 'foo')
    # bad mode
    assert_raises(ValueError, _make_surface_mapping, info, surf, 'meg',
                  mode='foo')
    # no picks
    evoked_eeg = pick_types_evoked(evoked, meg=False, eeg=True)
    assert_raises(RuntimeError, _make_surface_mapping, evoked_eeg.info,
                  surf, 'meg')
    # bad surface def
    nn = surf['nn']
    del surf['nn']
    assert_raises(KeyError, _make_surface_mapping, info, surf, 'meg')
    surf['nn'] = nn
    cf = surf['coord_frame']
    del surf['coord_frame']
    assert_raises(KeyError, _make_surface_mapping, info, surf, 'meg')
    surf['coord_frame'] = cf

    # now do it with make_field_map
    evoked = pick_types_evoked(evoked, meg=True, eeg=False)
    fmd = make_field_map(evoked, trans_fname=None,
                         subject='sample', subjects_dir=subjects_dir)
    assert_true(len(fmd) == 1)
    assert_array_equal(fmd[0]['data'].shape, (304, 106))  # maps data onto surf
    assert_true(len(fmd[0]['ch_names']), 106)

    assert_raises(ValueError, make_field_map, evoked, ch_type='foobar')
