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
                 read_trans, read_dipole, SourceEstimate, VectorSourceEstimate,
                 make_sphere_model)
from mne.io import read_raw_ctf, read_raw_bti, read_raw_kit, read_info
from mne.io.meas_info import write_dig
from mne.viz import (plot_sparse_source_estimates, plot_source_estimates,
                     plot_trans, snapshot_brain_montage, plot_head_positions,
                     plot_alignment)
from mne.viz.utils import _fake_click
from mne.utils import (requires_mayavi, requires_pysurfer, run_tests_if_main,
                       _import_mlab, _TempDir, requires_nibabel, check_version,
                       requires_version)
from mne.datasets import testing
from mne.source_space import read_source_spaces
from mne.bem import read_bem_solution, read_bem_surfaces


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


def test_plot_head_positions():
    """Test plotting of head positions."""
    import matplotlib.pyplot as plt
    pos = np.random.RandomState(0).randn(4, 10)
    pos[:, 0] = np.arange(len(pos))
    with warnings.catch_warnings(record=True):  # old MPL will cause a warning
        plot_head_positions(pos)
        if check_version('matplotlib', '1.4'):
            plot_head_positions(pos, mode='field')
        else:
            assert_raises(RuntimeError, plot_head_positions, pos, mode='field')
    assert_raises(ValueError, plot_head_positions, pos, 'foo')
    plt.close('all')


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
    stc_data[(np.random.rand(stc_size // 20) * stc_size).astype(int)] = \
        np.random.RandomState(0).rand(stc_data.size // 20)
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
def test_plot_alignment():
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
        meg = ['helmet', 'sensors']
        if system == 'KIT':
            meg.append('ref')
        plot_alignment(info, trans_fname, subject='sample',
                       subjects_dir=subjects_dir, meg=meg)
        mlab.close(all=True)
    # KIT ref sensor coil def is defined
    with warnings.catch_warnings(record=True):  # deprecated
        plot_trans(infos['KIT'], None, meg_sensors=True, ref_meg=True)
    mlab.close(all=True)
    info = infos['Neuromag']
    assert_raises(TypeError, plot_alignment, 'foo', trans_fname,
                  subject='sample', subjects_dir=subjects_dir)
    assert_raises(TypeError, plot_alignment, info, trans_fname,
                  subject='sample', subjects_dir=subjects_dir, src='foo')
    assert_raises(ValueError, plot_alignment, info, trans_fname,
                  subject='fsaverage', subjects_dir=subjects_dir,
                  src=sample_src)
    sample_src.plot(subjects_dir=subjects_dir, head=True, skull=True,
                    brain='white')
    mlab.close(all=True)
    # no-head version
    with warnings.catch_warnings(record=True):  # deprecated
        plot_trans(info, None, meg_sensors=True, dig=True, coord_frame='head')
    mlab.close(all=True)
    # all coord frames
    for coord_frame in ('meg', 'head', 'mri'):
        plot_alignment(info, meg=['helmet', 'sensors'], dig=True,
                       coord_frame=coord_frame, trans=trans_fname,
                       subject='sample', mri_fiducials=fiducials_path,
                       subjects_dir=subjects_dir, src=sample_src)
        mlab.close(all=True)
    # EEG only with strange options
    evoked_eeg_ecog = evoked.copy().pick_types(meg=False, eeg=True)
    evoked_eeg_ecog.info['projs'] = []  # "remove" avg proj
    evoked_eeg_ecog.set_channel_types({'EEG 001': 'ecog'})
    with warnings.catch_warnings(record=True) as w:
        plot_alignment(evoked_eeg_ecog.info, subject='sample',
                       trans=trans_fname, subjects_dir=subjects_dir,
                       surfaces=['white', 'outer_skin', 'outer_skull'],
                       meg=['helmet', 'sensors'],
                       eeg=['original', 'projected'], ecog=True)
    mlab.close(all=True)
    assert_true(['Cannot plot MEG' in str(ww.message) for ww in w])

    sphere = make_sphere_model(info=evoked.info, r0='auto', head_radius='auto')
    bem_sol = read_bem_solution(op.join(subjects_dir, 'sample', 'bem',
                                        'sample-1280-1280-1280-bem-sol.fif'))
    bem_surfs = read_bem_surfaces(op.join(subjects_dir, 'sample', 'bem',
                                          'sample-1280-1280-1280-bem.fif'))
    sample_src[0]['coord_frame'] = 4  # hack for coverage
    plot_alignment(info, trans_fname, subject='sample', meg='helmet',
                   subjects_dir=subjects_dir, eeg='projected', bem=sphere,
                   surfaces=['head', 'brain', 'inner_skull', 'outer_skull'],
                   src=sample_src)
    plot_alignment(info, trans_fname, subject='sample', meg=[],
                   subjects_dir=subjects_dir, bem=bem_sol, eeg=True,
                   surfaces=['head', 'inflated', 'outer_skull', 'inner_skull'])
    plot_alignment(info, trans_fname, subject='sample',
                   meg=True, subjects_dir=subjects_dir,
                   surfaces=['head', 'inner_skull'], bem=bem_surfs)
    sphere = make_sphere_model('auto', None, evoked.info)  # one layer
    plot_alignment(info, trans_fname, subject='sample', meg=False,
                   coord_frame='mri', subjects_dir=subjects_dir,
                   surfaces=['brain'], bem=sphere)
    # one layer bem with skull surfaces:
    assert_raises(ValueError, plot_alignment, info=info, trans=trans_fname,
                  subject='sample', subjects_dir=subjects_dir,
                  surfaces=['brain', 'head', 'inner_skull'], bem=sphere)
    # wrong eeg value:
    assert_raises(ValueError, plot_alignment, info=info, trans=trans_fname,
                  subject='sample', subjects_dir=subjects_dir, eeg='foo')
    # wrong meg value:
    assert_raises(ValueError, plot_alignment, info=info, trans=trans_fname,
                  subject='sample', subjects_dir=subjects_dir, meg='bar')
    # multiple brain surfaces:
    assert_raises(ValueError, plot_alignment, info=info, trans=trans_fname,
                  subject='sample', subjects_dir=subjects_dir,
                  surfaces=['white', 'pial'])


@testing.requires_testing_data
@requires_pysurfer
@requires_mayavi
def test_limits_to_control_points():
    """Test functionality for determing control points."""
    sample_src = read_source_spaces(src_fname)
    kwargs = dict(subjects_dir=subjects_dir, smoothing_steps=1)

    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.random.RandomState(0).rand((n_verts * n_time))
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')

    # Test for simple use cases
    mlab = _import_mlab()
    stc.plot(**kwargs)
    stc.plot(clim=dict(pos_lims=(10, 50, 90)), **kwargs)
    stc.plot(colormap='hot', clim='auto', **kwargs)
    stc.plot(colormap='mne', clim='auto', **kwargs)
    figs = [mlab.figure(), mlab.figure()]
    stc.plot(clim=dict(kind='value', lims=(10, 50, 90)), figure=99, **kwargs)
    assert_raises(ValueError, stc.plot, clim='auto', figure=figs, **kwargs)

    # Test both types of incorrect limits key (lims/pos_lims)
    assert_raises(KeyError, plot_source_estimates, stc, colormap='mne',
                  clim=dict(kind='value', lims=(5, 10, 15)), **kwargs)
    assert_raises(KeyError, plot_source_estimates, stc, colormap='hot',
                  clim=dict(kind='value', pos_lims=(5, 10, 15)), **kwargs)

    # Test for correct clim values
    assert_raises(ValueError, stc.plot,
                  clim=dict(kind='value', pos_lims=[0, 1, 0]), **kwargs)
    assert_raises(ValueError, stc.plot, colormap='mne',
                  clim=dict(pos_lims=(5, 10, 15, 20)), **kwargs)
    assert_raises(ValueError, stc.plot,
                  clim=dict(pos_lims=(5, 10, 15), kind='foo'), **kwargs)
    assert_raises(ValueError, stc.plot, colormap='mne', clim='foo', **kwargs)
    assert_raises(ValueError, stc.plot, clim=(5, 10, 15), **kwargs)
    assert_raises(ValueError, plot_source_estimates, 'foo', clim='auto',
                  **kwargs)
    assert_raises(ValueError, stc.plot, hemi='foo', clim='auto', **kwargs)

    # Test handling of degenerate data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # thresholded maps
        stc._data.fill(0.)
        plot_source_estimates(stc, **kwargs)
        assert_equal(len(w), 1)
    mlab.close(all=True)


@testing.requires_testing_data
@requires_nibabel()
def test_stc_mpl():
    """Test plotting source estimates with matplotlib."""
    import matplotlib.pyplot as plt
    sample_src = read_source_spaces(src_fname)

    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.ones((n_verts * n_time))
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1, 'sample')
    with warnings.catch_warnings(record=True):  # vertices not included
        stc.plot(subjects_dir=subjects_dir, time_unit='s', views='ven',
                 hemi='rh', smoothing_steps=2, subject='sample',
                 backend='matplotlib', spacing='oct1', initial_time=0.001,
                 colormap='Reds')
        fig = stc.plot(subjects_dir=subjects_dir, time_unit='ms', views='dor',
                       hemi='lh', smoothing_steps=2, subject='sample',
                       backend='matplotlib', spacing='ico2', time_viewer=True)
        time_viewer = fig.time_viewer
        _fake_click(time_viewer, time_viewer.axes[0], (0.5, 0.5))  # change t
        time_viewer.canvas.key_press_event('ctrl+right')
        time_viewer.canvas.key_press_event('left')
    assert_raises(ValueError, stc.plot, subjects_dir=subjects_dir,
                  hemi='both', subject='sample', backend='matplotlib')
    assert_raises(ValueError, stc.plot, subjects_dir=subjects_dir,
                  time_unit='ss', subject='sample', backend='matplotlib')
    plt.close('all')


@testing.requires_testing_data
@requires_nibabel()
def test_plot_dipole_mri_orthoview():
    """Test mpl dipole plotting."""
    import matplotlib.pyplot as plt
    dipoles = read_dipole(dip_fname)
    trans = read_trans(trans_fname)
    for coord_frame, idx, show_all in zip(['head', 'mri'],
                                          ['gof', 'amplitude'], [True, False]):
        fig = dipoles.plot_locations(trans, 'sample', subjects_dir,
                                     coord_frame=coord_frame, idx=idx,
                                     show_all=show_all, mode='orthoview')
        fig.canvas.scroll_event(0.5, 0.5, 1)  # scroll up
        fig.canvas.scroll_event(0.5, 0.5, -1)  # scroll down
        fig.canvas.key_press_event('up')
        fig.canvas.key_press_event('down')
        fig.canvas.key_press_event('a')  # some other key
    ax = plt.subplot(111)
    assert_raises(ValueError, dipoles.plot_locations, trans, 'sample',
                  subjects_dir, ax=ax)
    plt.close('all')


@requires_mayavi
def test_snapshot_brain_montage():
    """Test snapshot brain montage."""
    info = read_info(evoked_fname)
    with warnings.catch_warnings(record=True):  # deprecated
        fig = plot_trans(
            info, trans=None, subject='sample', subjects_dir=subjects_dir,
            skull=['outer_skull', 'inner_skull'])

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


@testing.requires_testing_data
@requires_version('surfer', '0.8')
@requires_mayavi
def test_plot_vec_source_estimates():
    """Test plotting of vector source estimates."""
    sample_src = read_source_spaces(src_fname)

    vertices = [s['vertno'] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 5
    data = np.random.RandomState(0).rand(n_verts, 3, n_time)
    stc = VectorSourceEstimate(data, vertices, 1, 1)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        stc.plot('sample', subjects_dir=subjects_dir)


run_tests_if_main()
