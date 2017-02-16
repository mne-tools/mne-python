# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_raises, assert_equal

from mne import (make_field_map, pick_channels_evoked, read_evokeds,
                 read_trans, read_dipole, SourceEstimate)
from mne.io import read_raw_ctf, read_raw_bti, read_raw_kit, read_info
from mne.io.meas_info import write_dig
from mne.viz import (plot_sparse_source_estimates, plot_source_estimates,
                     plot_trans, snapshot_brain_montage)
from mne.utils import (requires_mayavi, requires_pysurfer, run_tests_if_main,
                       _import_mlab, _TempDir)
from mne.datasets import testing
from mne.source_space import read_source_spaces


# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
trans_fname = op.join(data_dir, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
src_fname = op.join(data_dir, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')
dip_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')
ctf_fname = op.join(data_dir, 'CTF', 'testdata_ctf.ds')

io_dir = op.join(op.abspath(op.dirname(__file__)), '..', '..', 'io')
base_dir = op.join(io_dir, 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')

base_dir = op.join(io_dir, 'bti', 'tests', 'data')
pdf_fname = op.join(base_dir, 'test_pdf_linux')
config_fname = op.join(base_dir, 'test_config_linux')
hs_fname = op.join(base_dir, 'test_hs_linux')
sqd_fname = op.join(io_dir, 'kit', 'tests', 'data', 'test.sqd')
warnings.simplefilter('always')  # enable b/c these tests throw warnings


@testing.requires_testing_data
@requires_pysurfer
@requires_mayavi
def test_plot_sparse_source_estimates():
    """Test plotting of (sparse) source estimates."""
    sample_src = read_source_spaces(src_fname)

    # dense version
    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_size = stc_data.size
    stc_data[(np.random.rand(stc_size / 20) * stc_size).astype(int)] = \
        np.random.RandomState(0).rand(stc_data.size / 20)
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    colormap = 'mne_analyze'
    plot_source_estimates(stc, 'sample', colormap=colormap,
                          background=(1, 1, 0),
                          subjects_dir=subjects_dir, colorbar=True,
                          clim='auto')
    assert_raises(TypeError, plot_source_estimates, stc, 'sample',
                  figure='foo', hemi='both', clim='auto',
                  subjects_dir=subjects_dir)

    # now do sparse version
    vertices = sample_src[0]['vertno']
    inds = [111, 333]
    stc_data = np.zeros((len(inds), n_time))
    stc_data[0, 1] = 1.
    stc_data[1, 4] = 2.
    vertices = [vertices[inds], np.empty(0, dtype=np.int)]
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    plot_sparse_source_estimates(sample_src, stc, bgcolor=(1, 1, 1),
                                 opacity=0.5, high_resolution=False)


@testing.requires_testing_data
@requires_mayavi
def test_plot_evoked_field():
    """Test plotting evoked field."""
    evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(-0.2, 0.0))
    evoked = pick_channels_evoked(evoked, evoked.ch_names[::10])  # speed
    for t in ['meg', None]:
        with warnings.catch_warnings(record=True):  # bad proj
            maps = make_field_map(evoked, trans_fname, subject='sample',
                                  subjects_dir=subjects_dir, n_jobs=1,
                                  ch_type=t)
        evoked.plot_field(maps, time=0.1)


@testing.requires_testing_data
@requires_mayavi
def test_plot_trans():
    """Test plotting of -trans.fif files and MEG sensor layouts."""
    # generate fiducials file for testing
    tempdir = _TempDir()
    fiducials_path = op.join(tempdir, 'fiducials.fif')
    fid = [{'coord_frame': 5, 'ident': 1, 'kind': 1,
            'r': [-0.08061612, -0.02908875, -0.04131077]},
           {'coord_frame': 5, 'ident': 2, 'kind': 1,
            'r': [0.00146763, 0.08506715, -0.03483611]},
           {'coord_frame': 5, 'ident': 3, 'kind': 1,
            'r': [0.08436285, -0.02850276, -0.04127743]}]
    write_dig(fiducials_path, fid, 5)

    mlab = _import_mlab()
    evoked = read_evokeds(evoked_fname)[0]
    sample_src = read_source_spaces(src_fname)
    with warnings.catch_warnings(record=True):  # 4D weight tables
        bti = read_raw_bti(pdf_fname, config_fname, hs_fname, convert=True,
                           preload=False).info
    infos = dict(
        Neuromag=evoked.info,
        CTF=read_raw_ctf(ctf_fname).info,
        BTi=bti,
        KIT=read_raw_kit(sqd_fname).info,
    )
    for system, info in infos.items():
        ref_meg = False if system == 'KIT' else True
        plot_trans(info, trans_fname, subject='sample', meg_sensors=True,
                   subjects_dir=subjects_dir, ref_meg=ref_meg)
        mlab.close(all=True)
    # KIT ref sensor coil def is defined
    plot_trans(infos['KIT'], None, meg_sensors=True, ref_meg=True)
    mlab.close(all=True)
    info = infos['Neuromag']
    assert_raises(ValueError, plot_trans, info, trans_fname,
                  subject='sample', subjects_dir=subjects_dir,
                  ch_type='bad-chtype')
    assert_raises(TypeError, plot_trans, 'foo', trans_fname,
                  subject='sample', subjects_dir=subjects_dir)
    assert_raises(TypeError, plot_trans, info, trans_fname,
                  subject='sample', subjects_dir=subjects_dir, src='foo')
    assert_raises(ValueError, plot_trans, info, trans_fname,
                  subject='fsaverage', subjects_dir=subjects_dir,
                  src=sample_src)
    sample_src.plot(subjects_dir=subjects_dir)
    mlab.close(all=True)
    # no-head version
    plot_trans(info, None, meg_sensors=True, dig=True, coord_frame='head')
    mlab.close(all=True)
    # all coord frames
    for coord_frame in ('meg', 'head', 'mri'):
        plot_trans(info, meg_sensors=True, dig=True, coord_frame=coord_frame,
                   trans=trans_fname, subject='sample',
                   mri_fiducials=fiducials_path, subjects_dir=subjects_dir)
        mlab.close(all=True)
    # EEG only with strange options
    evoked_eeg_ecog = evoked.copy().pick_types(meg=False, eeg=True)
    evoked_eeg_ecog.info['projs'] = []  # "remove" avg proj
    evoked_eeg_ecog.set_channel_types({'EEG 001': 'ecog'})
    with warnings.catch_warnings(record=True) as w:
        plot_trans(evoked_eeg_ecog.info, subject='sample', trans=trans_fname,
                   source='outer_skin', meg_sensors=True, skull=True,
                   eeg_sensors=['original', 'projected'], ecog_sensors=True,
                   brain='white', head=True, subjects_dir=subjects_dir)
    mlab.close(all=True)
    assert_true(['Cannot plot MEG' in str(ww.message) for ww in w])


@testing.requires_testing_data
@requires_pysurfer
@requires_mayavi
def test_limits_to_control_points():
    """Test functionality for determing control points."""
    sample_src = read_source_spaces(src_fname)

    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.random.RandomState(0).rand((n_verts * n_time))
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')

    # Test for simple use cases
    mlab = _import_mlab()
    stc.plot(subjects_dir=subjects_dir)
    stc.plot(clim=dict(pos_lims=(10, 50, 90)), subjects_dir=subjects_dir)
    stc.plot(clim=dict(kind='value', lims=(10, 50, 90)), figure=99,
             subjects_dir=subjects_dir)
    stc.plot(colormap='hot', clim='auto', subjects_dir=subjects_dir)
    stc.plot(colormap='mne', clim='auto', subjects_dir=subjects_dir)
    figs = [mlab.figure(), mlab.figure()]
    assert_raises(ValueError, stc.plot, clim='auto', figure=figs,
                  subjects_dir=subjects_dir)

    # Test both types of incorrect limits key (lims/pos_lims)
    assert_raises(KeyError, plot_source_estimates, stc, colormap='mne',
                  clim=dict(kind='value', lims=(5, 10, 15)),
                  subjects_dir=subjects_dir)
    assert_raises(KeyError, plot_source_estimates, stc, colormap='hot',
                  clim=dict(kind='value', pos_lims=(5, 10, 15)),
                  subjects_dir=subjects_dir)

    # Test for correct clim values
    assert_raises(ValueError, stc.plot,
                  clim=dict(kind='value', pos_lims=[0, 1, 0]),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, colormap='mne',
                  clim=dict(pos_lims=(5, 10, 15, 20)),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot,
                  clim=dict(pos_lims=(5, 10, 15), kind='foo'),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, colormap='mne', clim='foo',
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, clim=(5, 10, 15),
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, plot_source_estimates, 'foo', clim='auto',
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, stc.plot, hemi='foo', clim='auto',
                  subjects_dir=subjects_dir)

    # Test handling of degenerate data
    stc.plot(clim=dict(kind='value', lims=[0, 0, 1]),
             subjects_dir=subjects_dir)  # ok
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # thresholded maps
        stc._data.fill(1.)
        plot_source_estimates(stc, subjects_dir=subjects_dir, time_unit='s')
        assert_equal(len(w), 0)
        stc._data[0].fill(0.)
        plot_source_estimates(stc, subjects_dir=subjects_dir, time_unit='s')
        assert_equal(len(w), 0)
        stc._data.fill(0.)
        plot_source_estimates(stc, subjects_dir=subjects_dir, time_unit='s')
        assert_equal(len(w), 1)
    mlab.close(all=True)


@testing.requires_testing_data
@requires_mayavi
def test_plot_dipole_locations():
    """Test plotting dipole locations."""
    dipoles = read_dipole(dip_fname)
    trans = read_trans(trans_fname)
    dipoles.plot_locations(trans, 'sample', subjects_dir, fig_name='foo')
    assert_raises(ValueError, dipoles.plot_locations, trans, 'sample',
                  subjects_dir, mode='foo')


@requires_mayavi
def test_snapshot_brain_montage():
    info = read_info(evoked_fname)
    fig = plot_trans(info, trans=None, subject='sample',
                     subjects_dir=subjects_dir)

    xyz = np.vstack([ich['loc'][:3] for ich in info['chs']])
    ch_names = [ich['ch_name'] for ich in info['chs']]
    xyz_dict = dict(zip(ch_names, xyz))
    xyz_dict[info['chs'][0]['ch_name']] = [1, 2]  # Set one ch to only 2 vals

    # Make sure wrong types are checked
    assert_raises(ValueError, snapshot_brain_montage, fig, xyz)

    # All chs must have 3 position values
    assert_raises(ValueError, snapshot_brain_montage, fig, xyz_dict)

    # Make sure we raise error if the figure has no scene
    assert_raises(TypeError, snapshot_brain_montage, fig, info)

run_tests_if_main()
