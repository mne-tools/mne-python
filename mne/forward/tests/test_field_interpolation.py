import numpy as np
from os import path as op
from numpy.polynomial import legendre
from numpy.testing.utils import (assert_allclose, assert_array_equal,
                                 assert_array_almost_equal)
from nose.tools import assert_raises, assert_true

from mne.forward import _make_surface_mapping, make_field_map
from mne.forward._lead_dots import (_comp_sum_eeg, _comp_sums_meg,
                                    _get_legen_table,
                                    _get_legen_lut_fast,
                                    _get_legen_lut_accurate,
                                    _do_cross_dots)
from mne.forward._make_forward import _create_coils
from mne.forward._field_interpolation import _setup_dots
from mne.surface import get_meg_helmet_surf, get_head_surf
from mne.datasets import testing
from mne import read_evokeds
from mne.io.constants import FIFF
from mne.fixes import partial
from mne.externals.six.moves import zip
from mne.utils import run_tests_if_main, slow_test


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')

data_path = testing.data_path(download=False)
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
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
        lut, n_fact = _get_legen_table('eeg', n_coeff=nc, force_calc=True)
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
        coeffs[1:] = (np.cumprod([beta] * (n_terms - 1), axis=0) *
                      (2.0 * n + 1.0) * (2.0 * n + 1.0) / n)
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
    lut, n_fact = _get_legen_table('meg', n_coeff=10, force_calc=True)
    fun = partial(_get_legen_lut_fast, lut=lut)
    coeffs = _comp_sums_meg(beta, ctheta, fun, n_fact, False)
    lut, n_fact = _get_legen_table('meg', n_coeff=20, force_calc=True)
    fun = partial(_get_legen_lut_accurate, lut=lut)
    coeffs = _comp_sums_meg(beta, ctheta, fun, n_fact, False)


def test_legendre_table():
    """Test Legendre table calculation
    """
    # double-check our table generation
    n = 10
    for ch_type in ['eeg', 'meg']:
        lut1, n_fact1 = _get_legen_table(ch_type, n_coeff=25, force_calc=True)
        lut1 = lut1[:, :n - 1].copy()
        n_fact1 = n_fact1[:n - 1].copy()
        lut2, n_fact2 = _get_legen_table(ch_type, n_coeff=n, force_calc=True)
        assert_allclose(lut1, lut2)
        assert_allclose(n_fact1, n_fact2)


@testing.requires_testing_data
def test_make_field_map_eeg():
    """Test interpolation of EEG field onto head
    """
    evoked = read_evokeds(evoked_fname, condition='Left Auditory')
    evoked.info['bads'] = ['MEG 2443', 'EEG 053']  # add some bads
    surf = get_head_surf('sample', subjects_dir=subjects_dir)
    # we must have trans if surface is in MRI coords
    assert_raises(ValueError, _make_surface_mapping, evoked.info, surf, 'eeg')

    evoked.pick_types(meg=False, eeg=True)
    fmd = make_field_map(evoked, trans_fname,
                         subject='sample', subjects_dir=subjects_dir)

    # trans is necessary for EEG only
    assert_raises(RuntimeError, make_field_map, evoked, None,
                  subject='sample', subjects_dir=subjects_dir)

    fmd = make_field_map(evoked, trans_fname,
                         subject='sample', subjects_dir=subjects_dir)
    assert_true(len(fmd) == 1)
    assert_array_equal(fmd[0]['data'].shape, (642, 59))  # maps data onto surf
    assert_true(len(fmd[0]['ch_names']), 59)


@testing.requires_testing_data
@slow_test
def test_make_field_map_meg():
    """Test interpolation of MEG field onto helmet | head
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
    evoked_eeg = evoked.pick_types(meg=False, eeg=True, copy=True)
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
    evoked.pick_types(meg=True, eeg=False)
    fmd = make_field_map(evoked, None,
                         subject='sample', subjects_dir=subjects_dir)
    assert_true(len(fmd) == 1)
    assert_array_equal(fmd[0]['data'].shape, (304, 106))  # maps data onto surf
    assert_true(len(fmd[0]['ch_names']), 106)

    assert_raises(ValueError, make_field_map, evoked, ch_type='foobar')

    # now test the make_field_map on head surf for MEG
    evoked.pick_types(meg=True, eeg=False)
    fmd = make_field_map(evoked, trans_fname, meg_surf='head',
                         subject='sample', subjects_dir=subjects_dir)
    assert_true(len(fmd) == 1)
    assert_array_equal(fmd[0]['data'].shape, (642, 106))  # maps data onto surf
    assert_true(len(fmd[0]['ch_names']), 106)

    assert_raises(ValueError, make_field_map, evoked, meg_surf='foobar',
                  subjects_dir=subjects_dir, trans=trans_fname)


def _setup_args(info):
    """Helper to test_as_meg_type_evoked."""
    coils = _create_coils(info['chs'], FIFF.FWD_COIL_ACCURACY_NORMAL,
                          info['dev_head_t'], 'meg')
    my_origin, int_rad, noise, lut_fun, n_fact = _setup_dots('fast',
                                                             coils,
                                                             'meg')
    args_dict = dict(intrad=int_rad, volume=False, coils1=coils, r0=my_origin,
                     ch_type='meg', lut=lut_fun, n_fact=n_fact)
    return args_dict


@testing.requires_testing_data
def test_as_meg_type_evoked():
    """Test interpolation of data on to virtual channels."""

    # validation tests
    evoked = read_evokeds(evoked_fname, condition='Left Auditory')
    assert_raises(ValueError, evoked.as_type, 'meg')
    assert_raises(ValueError, evoked.copy().pick_types(meg='grad').as_type,
                  'meg')

    # channel names
    ch_names = evoked.info['ch_names']
    virt_evoked = evoked.pick_channels(ch_names=ch_names[:10:1],
                                       copy=True).as_type('mag')
    assert_true(all('_virtual' in ch for ch in virt_evoked.info['ch_names']))

    # pick from and to channels
    evoked_from = evoked.pick_channels(ch_names=ch_names[2:10:3], copy=True)
    evoked_to = evoked.pick_channels(ch_names=ch_names[0:10:3], copy=True)

    info_from, info_to = evoked_from.info, evoked_to.info

    # set up things
    args1, args2 = _setup_args(info_from), _setup_args(info_to)
    args1.update(coils2=args2['coils1']), args2.update(coils2=args1['coils1'])

    # test cross dots
    cross_dots1 = _do_cross_dots(**args1)
    cross_dots2 = _do_cross_dots(**args2)

    assert_array_almost_equal(cross_dots1, cross_dots2.T)

    # correlation test
    evoked = evoked.pick_channels(ch_names=ch_names[:10:]).copy()
    data1 = evoked.pick_types(meg='grad').data.ravel()
    data2 = evoked.as_type('grad').data.ravel()
    assert_true(np.corrcoef(data1, data2)[0, 1] > 0.95)

run_tests_if_main()
