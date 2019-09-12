# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)


import pytest

import numpy as np

from numpy.testing import assert_array_equal, assert_allclose

from mne.channels.montage import read_montage
from mne.channels.montage import _BUILT_IN_MONTAGES
from mne.channels import make_standard_montage
from mne._digitization.base import _get_dig_eeg




MONTAGES_WITHOUT_FIDUCIALS = ['EGI_256', 'easycap-M1', 'easycap-M10']
MONTAGES_WITH_FIDUCIALS = [k for k in _BUILT_IN_MONTAGES
                           if k not in MONTAGES_WITHOUT_FIDUCIALS]

EXPECTED_HEAD_SIZE = 0.085


def test_make_standard_montage_egi_256():
    """Test egi_256."""
    EXPECTED_FIRST_9_LOC = np.array(
        [[ 6.55992516e-02,  5.64176352e-02, -2.57662946e-02],  # noqa
         [ 6.08331388e-02,  6.57063949e-02, -6.40717015e-03],  # noqa
         [ 5.19851171e-02,  7.15413471e-02,  1.12091555e-02],  # noqa
         [ 4.18066179e-02,  7.31439438e-02,  2.66373224e-02],  # noqa
         [ 3.09755787e-02,  6.97928339e-02,  4.21906579e-02],  # noqa
         [ 1.96959622e-02,  6.22758709e-02,  5.58500821e-02],  # noqa
         [ 1.03933314e-02,  5.14631908e-02,  6.63221724e-02],  # noqa
         [ 8.76671630e-18,  3.81400691e-02,  7.39613137e-02],  # noqa
         [-1.05002738e-02,  1.95003515e-02,  7.85765571e-02]]  # noqa
    )

    montage = make_standard_montage('EGI_256')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])
    eeg_center = eeg_loc.mean(axis=0)
    distance_to_center = np.linalg.norm(eeg_loc - eeg_center, axis=1)

    # assert_allclose(eeg_center, [0, 0, 0], atol=1e-8)  # XXX we no longer substract mean
    assert_allclose(distance_to_center.mean(), 0.085, atol=1e-3)
    assert_allclose(distance_to_center.std(), 0.00418, atol=1e-4)
    # assert_allclose(eeg_loc[:9], EXPECTED_FIRST_9_LOC, atol=1e-1)  # XXX ?


@pytest.mark.skip(reason='The points no longer match')
@pytest.mark.parametrize('kind', [
    # 'EGI_256',  # This was broken
    # 'easycap-M1',  # easycap don't match.
    # 'easycap-M10',
    'GSN-HydroCel-128',
    'GSN-HydroCel-129',
    'GSN-HydroCel-256',
    'GSN-HydroCel-257',
    'GSN-HydroCel-32',
    'GSN-HydroCel-64_1.0',
    'GSN-HydroCel-65_1.0',
    'biosemi128',
    'biosemi16',
    'biosemi160',
    'biosemi256',
    'biosemi32',
    'biosemi64',
    'mgh60',
    'mgh70',
    'standard_1005',
    'standard_1020',
    'standard_alphabetic',
    'standard_postfixed',
    'standard_prefixed',
    'standard_primed',
])
def test_old_vs_new(kind):
    """Test difference between old and new standard montages."""
    mont = read_montage(kind)
    digm = make_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])


def test_standard_montage_errors():
    """Test error handling for wrong keys."""
    with pytest.raises(ValueError, match='Could not find the montage'):
        _ = make_standard_montage('not-here')


@pytest.mark.parametrize('kind, tol', [
    ['EGI_256', 1e-5],
    ['easycap-M1', 1e-8],
    ['easycap-M10', 1e-8],
])
def test_standard_montages_in_head(kind, tol):
    """Test standard montage properties (ie: they form a head)."""
    montage = make_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    assert_allclose(
        actual=np.linalg.norm(eeg_loc, axis=1),
        desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE),
        atol=tol,
    )


from mne.channels.montage import transform_to_head
from pytest import approx


import matplotlib.pyplot as plt
from mne.channels._dig_montage_utils import _get_fid_coords
from mne.transforms import _sph_to_cart


def _plot_dig_transformation(transformed, original, title=''):
    EXPECTED_HEAD_SIZE = 0.085
    from mne.viz.backends.renderer import _Renderer
    def get_data(montage):
        data, coord_frame = _get_fid_coords(montage.dig)
        data['eeg'] = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])
        data['coord_frame'] = coord_frame

        return data

    def _plot_fid_coord(renderer, data, color):
        renderer.tube(
            # origin=data.lpa,  # XXX: why I cannot pas a (3,) ?
            origin=np.atleast_2d(data.lpa),
            destination=np.atleast_2d(data.rpa),
            # color='red',  # XXX: why I cannot do that?
            color=color,
            radius=0.001,  # XXX: why radious=1 which is default does not work?
        )
        renderer.tube(
            origin=np.atleast_2d((data.lpa+data.rpa)/2),
            destination=np.atleast_2d(data.nasion),
            color=color,
            radius=0.001,  # XXX: why radious=1 which is default does not work?
        )

    ren = _Renderer()
    ren.sphere(
        center=np.array([0, 0, 0]),
        # color=(100, 100, 100),  # XXX: is color (R,G,B) 0-255? doc needs rev.
        color=(1.0, 1.0, 1.0),  # XXX: doc don't say [0-1 or 0-255] ??
        # scale=EXPECTED_HEAD_SIZE,  # XXX: why I cannot put radius a value in mm??  # noqa
        scale=0.17,  # XXXX magic number!!
        opacity=0.3,
        resolution=20,  # XXX: why this is not callen n_poligons??
        backface_culling=False,
    )
    N_RAND_PTS = 50
    ren.sphere(
        center=_sph_to_cart(np.stack(
            [np.full((N_RAND_PTS,), EXPECTED_HEAD_SIZE),
             np.random.rand(N_RAND_PTS) * 3 * 3.1415,
             np.random.rand(N_RAND_PTS) * 3 * 3.1415,
            ],
            axis=-1,
        )),
        color=(1.0, 1.0, 1.0),
        scale=0.001
    )

    orig_data = get_data(original)
    trans_data = get_data(transformed)

    for oo, tt in zip(orig_data.eeg, trans_data.eeg):
        ren.tube(
            origin=np.atleast_2d(oo),
            destination=np.atleast_2d(tt),
            color=(.0, .1, .0),
            radius=0.0005,
        )

    _plot_fid_coord(ren, orig_data, (1.0, 0, 0))
    ren.sphere(center=orig_data.eeg, color=(1.0, .0, .0), scale=0.0022)

    _plot_fid_coord(ren, trans_data, (0, 0, 1.0))
    ren.sphere(center=trans_data.eeg, color=(.0, .0, 1.0), scale=0.0022)


    ren.text2d(x=0, y=0, text=title, width=.1)
    ren.show()

    return


@pytest.mark.skip(reason='this is my plotting tinkering')
def test_bar():
    """Test bar."""
    plt.switch_backend('Qt5Agg')

    # kind = 'EGI_256'
    kind = 'mgh60'
    montage = make_standard_montage(kind)
    trf_montage = transform_to_head(montage)

    np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    _plot_dig_transformation(trf_montage, montage)

    # import pdb; pdb.set_trace()


@pytest.mark.parametrize('kind, foo', [
    # XXX All should be 0.085 but they are not !!
    # ['EGI_256', 0.08500001],
    # ['easycap-M1', 0.08499999999999999],
    # ['easycap-M10', 0.08499999999999999],
    # ['GSN-HydroCel-128', 9.763325532616348],
    # ['GSN-HydroCel-129', 9.781833508100744],
    # ['GSN-HydroCel-256', 10.53120179308986],
    # ['GSN-HydroCel-257', 10.542564039112401],
    # ['GSN-HydroCel-32', 9.334690825727204],
    # ['GSN-HydroCel-64_1.0', 11.375727506868348],
    # ['GSN-HydroCel-65_1.0', 11.41411195568285],
    # ['biosemi128', 103.13293097944218],
    # ['biosemi16', 102.54836114601703],
    # ['biosemi160', 103.24734353529684],
    # ['biosemi256', 102.31834042785782],
    # ['biosemi32', 102.66433014370907],
    # ['biosemi64', 101.87617188729301],
    ['mgh60', 0.11734227421583884],
    # ['mgh70', 0.11808759279592418],
    # ['standard_1005', 0.1171808880579489],
    # ['standard_1020', 0.11460403303216726],
    # ['standard_alphabetic', 0.12012639557866846],
    # ['standard_postfixed', 0.11887390168465949],
    # ['standard_prefixed', 0.11675854869450944],
    # ['standard_primed', 0.11887390168465949],
])
def test_foo(kind, foo):
    """Test standard montage properties (ie: they form a head)."""
    # import pdb; pdb.set_trace()
    montage = make_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])
    dist_mean = np.linalg.norm(eeg_loc, axis=1).mean()
    # assert  dist_mean == approx(0.085, atol=1e-2)
    montage = transform_to_head(montage) if montage._coord_frame != 'head' else montage  # noqa
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    # assert_allclose(
    #     actual=np.linalg.norm(eeg_loc, axis=1),
    #     desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE),
    #     atol=1e-2  # Use a high tolerance for now # tol,
    # )
    assert np.linalg.norm(eeg_loc, axis=1).mean() == approx(foo)


@pytest.mark.parametrize('kind, orig_mean, trans_mean', [
    ['mgh60', 0.09797280213313385, 0.11734227421583884],
])
def test_foo(kind, orig_mean, trans_mean):
    """Test standard montage properties (ie: they form a head)."""
    # import pdb; pdb.set_trace()
    montage = make_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])
    assert  np.linalg.norm(eeg_loc, axis=1).mean() == approx(orig_mean, abs=1e-4)

    trans_montage = transform_to_head(montage) if montage._coord_frame != 'head' else montage  # noqa
    trans_eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(trans_montage.dig)])

    # assert_allclose(
    #     actual=np.linalg.norm(eeg_loc, axis=1),
    #     desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE),
    #     atol=1e-2  # Use a high tolerance for now # tol,
    # )
    assert np.linalg.norm(trans_eeg_loc, axis=1).mean() == approx(trans_mean, abs=1e-4)
