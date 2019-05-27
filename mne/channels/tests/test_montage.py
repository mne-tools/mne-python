# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op

import pytest

import numpy as np
from scipy.io import savemat

from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_allclose, assert_array_almost_equal,
                           assert_array_less, assert_equal)
from mne.channels.montage import (read_montage, _set_montage, read_dig_montage,
                                  get_builtin_montages)
from mne.utils import _TempDir, run_tests_if_main, assert_dig_allclose
from mne import create_info, EvokedArray, read_evokeds, __file__ as _mne_file
from mne.bem import _fit_sphere
from mne.coreg import fit_matched_points
from mne.transforms import apply_trans, get_ras_to_neuromag_trans
from mne.io.constants import FIFF
from mne.io.meas_info import _read_dig_points
from mne.viz._3d import _fiducial_coords

from mne.io.kit import read_mrk
from mne.io import (read_raw_brainvision, read_raw_egi, read_raw_fif,
                    read_fiducials)

from mne.datasets import testing

data_path = testing.data_path(download=False)
fif_dig_montage_fname = op.join(data_path, 'montage', 'eeganes07.fif')
egi_dig_montage_fname = op.join(data_path, 'montage', 'coordinates.xml')
egi_raw_fname = op.join(data_path, 'montage', 'egi_dig_test.raw')
egi_fif_fname = op.join(data_path, 'montage', 'egi_dig_raw.fif')
locs_montage_fname = op.join(data_path, 'EEGLAB', 'test_chans.locs')
evoked_fname = op.join(data_path, 'montage', 'level2_raw-ave.fif')

io_dir = op.join(op.dirname(__file__), '..', '..', 'io')
kit_dir = op.join(io_dir, 'kit', 'tests', 'data')
elp = op.join(kit_dir, 'test_elp.txt')
hsp = op.join(kit_dir, 'test_hsp.txt')
hpi = op.join(kit_dir, 'test_mrk.sqd')
bv_fname = op.join(io_dir, 'brainvision', 'tests', 'data', 'test.vhdr')
fif_fname = op.join(io_dir, 'tests', 'data', 'test_raw.fif')
ctf_fif_fname = op.join(io_dir, 'tests', 'data', 'test_ctf_comp_raw.fif')


def test_fiducials():
    """Test handling of fiducials."""
    # Eventually the code used here should be unified with montage.py, but for
    # now it uses code in odd places
    for fname in (fif_fname, ctf_fif_fname):
        fids, coord_frame = read_fiducials(fname)
        points = _fiducial_coords(fids, coord_frame)
        assert points.shape == (3, 3)
        # Fids
        assert_allclose(points[:, 2], 0., atol=1e-6)
        assert_allclose(points[::2, 1], 0., atol=1e-6)
        assert points[2, 0] > 0  # RPA
        assert points[0, 0] < 0  # LPA
        # Nasion
        assert_allclose(points[1, 0], 0., atol=1e-6)
        assert points[1, 1] > 0


def test_documented():
    """Test that montages are documented."""
    docs = read_montage.__doc__
    lines = [line[4:] for line in docs.splitlines()]
    start = stop = None
    for li, line in enumerate(lines):
        if line.startswith('====') and li < len(lines) - 2 and \
                lines[li + 1].startswith('Kind') and\
                lines[li + 2].startswith('===='):
            start = li + 3
        elif start is not None and li > start and line.startswith('===='):
            stop = li
            break
    assert (start is not None)
    assert (stop is not None)
    kinds = [line.split(' ')[0] for line in lines[start:stop]]
    kinds = [kind for kind in kinds if kind != '']
    montages = os.listdir(op.join(op.dirname(_mne_file), 'channels', 'data',
                                  'montages'))
    montages = sorted(op.splitext(m)[0] for m in montages)
    assert_equal(len(set(montages)), len(montages))
    assert_equal(len(set(kinds)), len(kinds), err_msg=str(sorted(kinds)))
    assert_equal(set(montages), set(kinds))


def test_montage():
    """Test making montages."""
    tempdir = _TempDir()
    inputs = dict(
        sfp='FidNz 0       9.071585155     -2.359754454\n'
            'FidT9 -6.711765       0.040402876     -3.251600355\n'
            'very_very_very_long_name -5.831241498 -4.494821698  4.955347697\n'
            'Cz 0       0       8.899186843',
        csd='// MatLab   Sphere coordinates [degrees]         Cartesian coordinates\n'  # noqa: E501
            '// Label       Theta       Phi    Radius         X         Y         Z       off sphere surface\n'  # noqa: E501
            'E1      37.700     -14.000       1.000    0.7677    0.5934   -0.2419  -0.00000000000000011\n'  # noqa: E501
            'E3      51.700      11.000       1.000    0.6084    0.7704    0.1908   0.00000000000000000\n'  # noqa: E501
            'E31      90.000     -11.000       1.000    0.0000    0.9816   -0.1908   0.00000000000000000\n'  # noqa: E501
            'E61     158.000     -17.200       1.000   -0.8857    0.3579   -0.2957  -0.00000000000000022',  # noqa: E501
        mm_elc='# ASA electrode file\nReferenceLabel  avg\nUnitPosition    mm\n'  # noqa:E501
               'NumberPositions=    68\n'
               'Positions\n'
               '-86.0761 -19.9897 -47.9860\n'
               '85.7939 -20.0093 -48.0310\n'
               '0.0083 86.8110 -39.9830\n'
               '-86.0761 -24.9897 -67.9860\n'
               'Labels\nLPA\nRPA\nNz\nDummy\n',
        m_elc='# ASA electrode file\nReferenceLabel  avg\nUnitPosition    m\n'
              'NumberPositions=    68\nPositions\n-.0860761 -.0199897 -.0479860\n'  # noqa:E501
              '.0857939 -.0200093 -.0480310\n.0000083 .00868110 -.0399830\n'
              '.08 -.02 -.04\n'
              'Labels\nLPA\nRPA\nNz\nDummy\n',
        txt='Site  Theta  Phi\n'
            'Fp1  -92    -72\n'
            'Fp2   92     72\n'
            'very_very_very_long_name       -92     72\n'
            'O2        92    -90\n',
        elp='346\n'
            'EEG\t      F3\t -62.027\t -50.053\t      85\n'
            'EEG\t      Fz\t  45.608\t      90\t      85\n'
            'EEG\t      F4\t   62.01\t  50.103\t      85\n'
            'EEG\t      FCz\t   68.01\t  58.103\t      85\n',
        hpts='eeg Fp1 -95.0 -3. -3.\n'
             'eeg AF7 -1 -1 -3\n'
             'eeg A3 -2 -2 2\n'
             'eeg A 0 0 0',
        bvef='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
             '<!-- Generated by EasyCap Configurator 19.05.2014 -->\n'
             '<Electrodes defaults="false">\n'
             '  <Electrode>\n'
             '    <Name>Fp1</Name>\n'
             '    <Theta>-90</Theta>\n'
             '    <Phi>-72</Phi>\n'
             '    <Radius>1</Radius>\n'
             '    <Number>1</Number>\n'
             '  </Electrode>\n'
             '  <Electrode>\n'
             '    <Name>Fz</Name>\n'
             '    <Theta>45</Theta>\n'
             '    <Phi>90</Phi>\n'
             '    <Radius>1</Radius>\n'
             '    <Number>2</Number>\n'
             '  </Electrode>\n'
             '  <Electrode>\n'
             '    <Name>F3</Name>\n'
             '    <Theta>-60</Theta>\n'
             '    <Phi>-51</Phi>\n'
             '    <Radius>1</Radius>\n'
             '    <Number>3</Number>\n'
             '  </Electrode>\n'
             '  <Electrode>\n'
             '    <Name>F7</Name>\n'
             '    <Theta>-90</Theta>\n'
             '    <Phi>-36</Phi>\n'
             '    <Radius>1</Radius>\n'
             '    <Number>4</Number>\n'
             '  </Electrode>\n'
             '</Electrodes>',
    )
    # Get actual positions and save them for checking
    # csd comes from the string above, all others come from commit 2fa35d4
    poss = dict(
        sfp=[[0.0, 9.07159, -2.35975], [-6.71176, 0.0404, -3.2516],
             [-5.83124, -4.49482, 4.95535], [0.0, 0.0, 8.89919]],
        mm_elc=[[-0.08608, -0.01999, -0.04799], [0.08579, -0.02001, -0.04803],
                [1e-05, 0.08681, -0.03998], [-0.08608, -0.02499, -0.06799]],
        m_elc=[[-0.08608, -0.01999, -0.04799], [0.08579, -0.02001, -0.04803],
               [1e-05, 0.00868, -0.03998], [0.08, -0.02, -0.04]],
        txt=[[-26.25044, 80.79056, -2.96646], [26.25044, 80.79056, -2.96646],
             [-26.25044, -80.79056, -2.96646], [0.0, -84.94822, -2.96646]],
        elp=[[-48.20043, 57.55106, 39.86971], [0.0, 60.73848, 59.4629],
             [48.1426, 57.58403, 39.89198], [41.64599, 66.91489, 31.8278]],
        hpts=[[-95, -3, -3], [-1, -1., -3.], [-2, -2, 2.], [0, 0, 0]],
        bvef=[[-26.266444, 80.839803, 5.204748e-15],
              [3.680313e-15, 60.104076, 60.104076],
              [-46.325632, 57.207392, 42.500000],
              [-68.766444, 49.961746, 5.204748e-15]],
    )
    for key, text in inputs.items():
        kind = key.split('_')[-1]
        fname = op.join(tempdir, 'test.' + kind)
        with open(fname, 'w') as fid:
            fid.write(text)
        montage = read_montage(fname)
        if kind in ('sfp', 'txt'):
            assert ('very_very_very_long_name' in montage.ch_names)
        assert_equal(len(montage.ch_names), 4)
        assert_equal(len(montage.ch_names), len(montage.pos))
        assert_equal(montage.pos.shape, (4, 3))
        assert_equal(montage.kind, 'test')
        if kind == 'csd':
            dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                     ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                     ('off_sph', 'f8')]
            try:
                table = np.loadtxt(fname, skip_header=2, dtype=dtype)
            except TypeError:
                table = np.loadtxt(fname, skiprows=2, dtype=dtype)
            poss['csd'] = np.c_[table['x'], table['y'], table['z']]
        if kind == 'elc':
            # Make sure points are reasonable distance from geometric centroid
            centroid = np.sum(montage.pos, axis=0) / montage.pos.shape[0]
            distance_from_centroid = np.apply_along_axis(
                np.linalg.norm, 1,
                montage.pos - centroid)
            assert_array_less(distance_from_centroid, 0.2)
            assert_array_less(0.01, distance_from_centroid)
        assert_array_almost_equal(poss[key], montage.pos, 4, err_msg=key)

    # Test reading in different letter case.
    ch_names = ["F3", "FZ", "F4", "FC3", "FCz", "FC4", "C3", "CZ", "C4", "CP3",
                "CPZ", "CP4", "P3", "PZ", "P4", "O1", "OZ", "O2"]
    montage = read_montage('standard_1020', ch_names=ch_names)
    assert_array_equal(ch_names, montage.ch_names)

    # test transform
    input_strs = ["""
    eeg Fp1 -95.0 -31.0 -3.0
    eeg AF7 -81 -59 -3
    eeg AF3 -87 -41 28
    cardinal 2 -91 0 -42
    cardinal 1 0 -91 -42
    cardinal 3 0 91 -42
    """, """
    Fp1 -95.0 -31.0 -3.0
    AF7 -81 -59 -3
    AF3 -87 -41 28
    FidNz -91 0 -42
    FidT9 0 -91 -42
    FidT10 0 91 -42
    """]
    # sfp files seem to have Nz, T9, and T10 as fiducials:
    # https://github.com/mne-tools/mne-python/pull/4482#issuecomment-321980611

    kinds = ['test_fid.hpts',  'test_fid.sfp']

    for kind, input_str in zip(kinds, input_strs):
        fname = op.join(tempdir, kind)
        with open(fname, 'w') as fid:
            fid.write(input_str)
        montage = read_montage(op.join(tempdir, kind), transform=True)

        # check coordinate transformation
        pos = np.array([-95.0, -31.0, -3.0])
        nasion = np.array([-91, 0, -42])
        lpa = np.array([0, -91, -42])
        rpa = np.array([0, 91, -42])
        fids = np.vstack((nasion, lpa, rpa))
        trans = get_ras_to_neuromag_trans(fids[0], fids[1], fids[2])
        pos = apply_trans(trans, pos)
        assert_array_equal(montage.pos[0], pos)
        assert_array_equal(montage.nasion[[0, 2]], [0, 0])
        assert_array_equal(montage.lpa[[1, 2]], [0, 0])
        assert_array_equal(montage.rpa[[1, 2]], [0, 0])
        pos = np.array([-95.0, -31.0, -3.0])
        montage_fname = op.join(tempdir, kind)
        montage = read_montage(montage_fname, unit='mm')
        assert_array_equal(montage.pos[0], pos * 1e-3)

        # test with last
        info = create_info(montage.ch_names, 1e3,
                           ['eeg'] * len(montage.ch_names))
        _set_montage(info, montage)
        pos2 = np.array([c['loc'][:3] for c in info['chs']])
        assert_array_equal(pos2, montage.pos)
        assert_equal(montage.ch_names, info['ch_names'])

        info = create_info(
            montage.ch_names, 1e3, ['eeg'] * len(montage.ch_names))

        evoked = EvokedArray(
            data=np.zeros((len(montage.ch_names), 1)), info=info, tmin=0)

        # test return type as well as set montage
        assert (isinstance(evoked.set_montage(montage), type(evoked)))

        pos3 = np.array([c['loc'][:3] for c in evoked.info['chs']])
        assert_array_equal(pos3, montage.pos)
        assert_equal(montage.ch_names, evoked.info['ch_names'])

        # Warning should be raised when some EEG are not specified in montage
        info = create_info(montage.ch_names + ['foo', 'bar'], 1e3,
                           ['eeg'] * (len(montage.ch_names) + 2))
        with pytest.warns(RuntimeWarning, match='position specified'):
            _set_montage(info, montage)

    # Channel names can be treated case insensitive
    info = create_info(['FP1', 'af7', 'AF3'], 1e3, ['eeg'] * 3)
    _set_montage(info, montage)

    # Unless there is a collision in names
    info = create_info(['FP1', 'Fp1', 'AF3'], 1e3, ['eeg'] * 3)
    assert (info['dig'] is None)
    with pytest.warns(RuntimeWarning, match='position specified'):
        _set_montage(info, montage)
    assert len(info['dig']) == 5  # 2 EEG w/pos, 3 fiducials
    montage.ch_names = ['FP1', 'Fp1', 'AF3']
    info = create_info(['fp1', 'AF3'], 1e3, ['eeg', 'eeg'])
    assert (info['dig'] is None)
    with pytest.warns(RuntimeWarning, match='position specified'):
        _set_montage(info, montage, set_dig=False)
    assert (info['dig'] is None)

    # test get_pos2d method
    montage = read_montage("standard_1020")
    c3 = montage.get_pos2d()[montage.ch_names.index("C3")]
    c4 = montage.get_pos2d()[montage.ch_names.index("C4")]
    fz = montage.get_pos2d()[montage.ch_names.index("Fz")]
    oz = montage.get_pos2d()[montage.ch_names.index("Oz")]
    f1 = montage.get_pos2d()[montage.ch_names.index("F1")]
    assert (c3[0] < 0)  # left hemisphere
    assert (c4[0] > 0)  # right hemisphere
    assert (fz[1] > 0)  # frontal
    assert (oz[1] < 0)  # occipital
    assert_allclose(fz[0], 0, atol=1e-2)  # midline
    assert_allclose(oz[0], 0, atol=1e-2)  # midline
    assert (f1[0] < 0 and f1[1] > 0)  # left frontal

    # test get_builtin_montages function
    montages = get_builtin_montages()
    assert (len(montages) > 0)  # MNE should always ship with montages
    assert ("standard_1020" in montages)  # 10/20 montage
    assert ("standard_1005" in montages)  # 10/05 montage


@testing.requires_testing_data
def test_read_locs():
    """Test reading EEGLAB locs."""
    pos = read_montage(locs_montage_fname).pos
    expected = [[0., 9.99779165e-01, -2.10157875e-02],
                [3.08738197e-01, 7.27341573e-01, -6.12907052e-01],
                [-5.67059636e-01, 6.77066318e-01, 4.69067752e-01],
                [0., 7.14575231e-01, 6.99558616e-01]]
    assert_allclose(pos[:4], expected, atol=1e-7)


def test_read_dig_montage():
    """Test read_dig_montage."""
    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    montage = read_dig_montage(hsp, hpi, elp, names, transform=False)
    elp_points = _read_dig_points(elp)
    hsp_points = _read_dig_points(hsp)
    hpi_points = read_mrk(hpi)
    assert_equal(montage.point_names, names)
    assert_array_equal(montage.elp, elp_points)
    assert_array_equal(montage.hsp, hsp_points)
    assert_array_equal(montage.hpi, hpi_points)
    assert (montage.dev_head_t is None)
    montage = read_dig_montage(hsp, hpi, elp, names,
                               transform=True, dev_head_t=True)
    # check coordinate transformation
    # nasion
    assert_almost_equal(montage.nasion[0], 0)
    assert_almost_equal(montage.nasion[2], 0)
    # lpa and rpa
    assert_allclose(montage.lpa[1:], 0, atol=1e-16)
    assert_allclose(montage.rpa[1:], 0, atol=1e-16)
    # device head transform
    dev_head_t = fit_matched_points(tgt_pts=montage.elp,
                                    src_pts=montage.hpi, out='trans')
    assert_array_equal(montage.dev_head_t, dev_head_t)

    # Digitizer as array
    m2 = read_dig_montage(hsp_points, hpi_points, elp_points, names, unit='m')
    assert_array_equal(m2.hsp, montage.hsp)
    m3 = read_dig_montage(hsp_points * 1000, hpi_points, elp_points * 1000,
                          names)
    assert_allclose(m3.hsp, montage.hsp)

    # test unit parameter and .mat support
    tempdir = _TempDir()
    mat_hsp = op.join(tempdir, 'test.mat')
    savemat(mat_hsp, dict(Points=(1000 * hsp_points).T), oned_as='row')
    montage_cm = read_dig_montage(mat_hsp, hpi, elp, names, unit='cm')
    assert_allclose(montage_cm.hsp, montage.hsp * 10.)
    assert_allclose(montage_cm.elp, montage.elp * 10.)
    assert_array_equal(montage_cm.hpi, montage.hpi)
    pytest.raises(ValueError, read_dig_montage, hsp, hpi, elp, names,
                  unit='km')
    # extra columns
    extra_hsp = op.join(tempdir, 'test.txt')
    with open(hsp, 'rb') as fin:
        with open(extra_hsp, 'wb') as fout:
            for line in fin:
                if line.startswith(b'%'):
                    fout.write(line)
                else:
                    # extra column
                    fout.write(line.rstrip() + b' 0.0 0.0 0.0\n')
    with pytest.warns(RuntimeWarning, match='Found .* columns instead of 3'):
        montage_extra = read_dig_montage(extra_hsp, hpi, elp, names)
    assert_allclose(montage_extra.hsp, montage.hsp)
    assert_allclose(montage_extra.elp, montage.elp)


def test_set_dig_montage():
    """Test applying DigMontage to inst."""
    # Extensive testing of applying `dig` to info is done in test_meas_info
    # with `test_make_dig_points`.
    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    hsp_points = _read_dig_points(hsp)
    elp_points = _read_dig_points(elp)
    nasion, lpa, rpa = elp_points[:3]
    nm_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    elp_points = apply_trans(nm_trans, elp_points)
    nasion, lpa, rpa = elp_points[:3]
    hsp_points = apply_trans(nm_trans, hsp_points)

    montage = read_dig_montage(hsp, hpi, elp, names, transform=True,
                               dev_head_t=True)
    temp_dir = _TempDir()
    fname_temp = op.join(temp_dir, 'test.fif')
    montage.save(fname_temp)
    montage_read = read_dig_montage(fif=fname_temp)
    for use_mon in (montage, montage_read):
        info = create_info(['Test Ch'], 1e3, ['eeg'])
        with pytest.warns(None):  # warns on one run about not all positions
            _set_montage(info, use_mon)
        hs = np.array([p['r'] for i, p in enumerate(info['dig'])
                       if p['kind'] == FIFF.FIFFV_POINT_EXTRA])
        nasion_dig = np.array([p['r'] for p in info['dig']
                               if all([p['ident'] == FIFF.FIFFV_POINT_NASION,
                                       p['kind'] == FIFF.FIFFV_POINT_CARDINAL])
                               ])
        lpa_dig = np.array([p['r'] for p in info['dig']
                            if all([p['ident'] == FIFF.FIFFV_POINT_LPA,
                                    p['kind'] == FIFF.FIFFV_POINT_CARDINAL])])
        rpa_dig = np.array([p['r'] for p in info['dig']
                            if all([p['ident'] == FIFF.FIFFV_POINT_RPA,
                                    p['kind'] == FIFF.FIFFV_POINT_CARDINAL])])
        hpi_dig = np.array([p['r'] for p in info['dig']
                            if p['kind'] == FIFF.FIFFV_POINT_HPI])
        assert_allclose(hs, hsp_points, atol=1e-7)
        assert_allclose(nasion_dig.ravel(), nasion, atol=1e-7)
        assert_allclose(lpa_dig.ravel(), lpa, atol=1e-7)
        assert_allclose(rpa_dig.ravel(), rpa, atol=1e-7)
        assert_allclose(hpi_dig, elp_points[3:], atol=1e-7)


@testing.requires_testing_data
def test_fif_dig_montage():
    """Test FIF dig montage support."""
    dig_montage = read_dig_montage(fif=fif_dig_montage_fname)

    # test round-trip IO
    temp_dir = _TempDir()
    fname_temp = op.join(temp_dir, 'test.fif')
    _check_roundtrip(dig_montage, fname_temp)

    # Make a BrainVision file like the one the user would have had
    raw_bv = read_raw_brainvision(bv_fname, preload=True)
    raw_bv_2 = raw_bv.copy()
    mapping = dict()
    for ii, ch_name in enumerate(raw_bv.ch_names):
        mapping[ch_name] = 'EEG%03d' % (ii + 1,)
    raw_bv.rename_channels(mapping)
    for ii, ch_name in enumerate(raw_bv_2.ch_names):
        mapping[ch_name] = 'EEG%03d' % (ii + 33,)
    raw_bv_2.rename_channels(mapping)
    raw_bv.add_channels([raw_bv_2])

    for ii in range(2):
        if ii == 1:
            dig_montage.transform_to_head()  # should have no meaningful effect

        # Set the montage
        raw_bv.set_montage(dig_montage)

        # Check the result
        evoked = read_evokeds(evoked_fname)[0]

        assert_equal(len(raw_bv.ch_names), len(evoked.ch_names) - 1)
        for ch_py, ch_c in zip(raw_bv.info['chs'], evoked.info['chs'][:-1]):
            assert_equal(ch_py['ch_name'],
                         ch_c['ch_name'].replace('EEG ', 'EEG'))
            # C actually says it's unknown, but it's not (?):
            # assert_equal(ch_py['coord_frame'], ch_c['coord_frame'])
            assert_equal(ch_py['coord_frame'], FIFF.FIFFV_COORD_HEAD)
            c_loc = ch_c['loc'].copy()
            c_loc[c_loc == 0] = np.nan
            assert_allclose(ch_py['loc'], c_loc, atol=1e-7)
        assert_dig_allclose(raw_bv.info, evoked.info)

    # Roundtrip of non-FIF start
    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    montage = read_dig_montage(hsp, hpi, elp, names, transform=False)
    pytest.raises(RuntimeError, montage.save, fname_temp)  # must be head coord
    montage = read_dig_montage(hsp, hpi, elp, names)
    _check_roundtrip(montage, fname_temp)


@testing.requires_testing_data
def test_egi_dig_montage():
    """Test EGI MFF XML dig montage support."""
    dig_montage = read_dig_montage(egi=egi_dig_montage_fname, unit='m')

    # # test round-trip IO
    temp_dir = _TempDir()
    fname_temp = op.join(temp_dir, 'egi_test.fif')
    _check_roundtrip(dig_montage, fname_temp)

    # Test coordinate transform
    dig_montage.transform_to_head()
    # nasion
    assert_almost_equal(dig_montage.nasion[0], 0)
    assert_almost_equal(dig_montage.nasion[2], 0)
    # lpa and rpa
    assert_allclose(dig_montage.lpa[1:], 0, atol=1e-16)
    assert_allclose(dig_montage.rpa[1:], 0, atol=1e-16)

    # Test accuracy and embedding within raw object
    raw_egi = read_raw_egi(egi_raw_fname, channel_naming='EEG %03d')
    raw_egi.set_montage(dig_montage)
    test_raw_egi = read_raw_fif(egi_fif_fname)

    assert_equal(len(raw_egi.ch_names), len(test_raw_egi.ch_names))
    for ch_raw, ch_test_raw in zip(raw_egi.info['chs'],
                                   test_raw_egi.info['chs']):
        assert_equal(ch_raw['ch_name'], ch_test_raw['ch_name'])
        assert_equal(ch_raw['coord_frame'], FIFF.FIFFV_COORD_HEAD)
        assert_allclose(ch_raw['loc'], ch_test_raw['loc'], atol=1e-7)
    assert_dig_allclose(raw_egi.info, test_raw_egi.info)


def test_set_montage():
    """Test setting a montage."""
    raw = read_raw_fif(fif_fname)
    orig_pos = np.array([ch['loc'][:3] for ch in raw.info['chs']
                         if ch['ch_name'].startswith('EEG')])
    raw.set_montage('mgh60')  # test loading with string argument
    new_pos = np.array([ch['loc'][:3] for ch in raw.info['chs']
                        if ch['ch_name'].startswith('EEG')])
    assert ((orig_pos != new_pos).all())
    r0 = _fit_sphere(new_pos)[1]
    assert_allclose(r0, [0., -0.016, 0.], atol=1e-3)
    # mgh70 has no 61/62/63/64 (these are EOG/ECG)
    mon = read_montage('mgh70')
    assert 'EEG061' not in mon.ch_names
    assert 'EEG074' in mon.ch_names


def _check_roundtrip(montage, fname):
    """Check roundtrip writing."""
    assert_equal(montage.coord_frame, 'head')
    montage.save(fname)
    montage_read = read_dig_montage(fif=fname)
    assert_equal(str(montage), str(montage_read))
    for kind in ('elp', 'hsp', 'nasion', 'lpa', 'rpa'):
        if getattr(montage, kind) is not None:
            assert_allclose(getattr(montage, kind),
                            getattr(montage_read, kind), err_msg=kind)
    assert_equal(montage_read.coord_frame, 'head')


run_tests_if_main()
