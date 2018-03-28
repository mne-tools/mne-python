import os.path as op
import warnings

import numpy as np
from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from mne import (read_dipole, read_forward_solution,
                 convert_forward_solution, read_evokeds, read_cov,
                 SourceEstimate, write_evokeds, fit_dipole,
                 transform_surface_to, make_sphere_model, pick_types,
                 pick_info, EvokedArray, read_source_spaces, make_ad_hoc_cov,
                 make_forward_solution, Dipole, DipoleFixed, Epochs,
                 make_fixed_length_events)
from mne.dipole import get_phantom_dipoles
from mne.simulation import simulate_evoked
from mne.datasets import testing
from mne.utils import run_tests_if_main, _TempDir, requires_mne, run_subprocess
from mne.proj import make_eeg_average_ref_proj

from mne.io import read_raw_fif, read_raw_ctf
from mne.io.constants import FIFF

from mne.surface import _compute_nearest
from mne.bem import _bem_find_surface, read_bem_solution
from mne.transforms import apply_trans, _get_trans

import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')
data_path = testing.data_path(download=False)
meg_path = op.join(data_path, 'MEG', 'sample')
fname_dip_xfit = op.join(meg_path, 'sample_audvis-ave_xfit.dip')
fname_raw = op.join(meg_path, 'sample_audvis_trunc_raw.fif')
fname_dip = op.join(meg_path, 'sample_audvis_trunc_set1.dip')
fname_evo = op.join(meg_path, 'sample_audvis_trunc-ave.fif')
fname_evo_full = op.join(meg_path, 'sample_audvis-ave.fif')
fname_cov = op.join(meg_path, 'sample_audvis_trunc-cov.fif')
fname_trans = op.join(meg_path, 'sample_audvis_trunc-trans.fif')
fname_fwd = op.join(meg_path, 'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
fname_bem = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
fname_src = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-oct-2-src.fif')
fname_xfit_dip = op.join(data_path, 'dip', 'fixed_auto.fif')
fname_xfit_dip_txt = op.join(data_path, 'dip', 'fixed_auto.dip')
fname_xfit_seq_txt = op.join(data_path, 'dip', 'sequential.dip')
fname_ctf = op.join(data_path, 'CTF', 'testdata_ctf_short.ds')
subjects_dir = op.join(data_path, 'subjects')


def _compare_dipoles(orig, new):
    """Compare dipole results for equivalence."""
    assert_allclose(orig.times, new.times, atol=1e-3, err_msg='times')
    assert_allclose(orig.pos, new.pos, err_msg='pos')
    assert_allclose(orig.amplitude, new.amplitude, err_msg='amplitude')
    assert_allclose(orig.gof, new.gof, err_msg='gof')
    assert_allclose(orig.ori, new.ori, rtol=1e-4, atol=1e-4, err_msg='ori')
    assert_equal(orig.name, new.name)


def _check_dipole(dip, n_dipoles):
    """Check dipole sizes."""
    assert_equal(len(dip), n_dipoles)
    assert_equal(dip.pos.shape, (n_dipoles, 3))
    assert_equal(dip.ori.shape, (n_dipoles, 3))
    assert_equal(dip.gof.shape, (n_dipoles,))
    assert_equal(dip.amplitude.shape, (n_dipoles,))


@testing.requires_testing_data
def test_io_dipoles():
    """Test IO for .dip files."""
    tempdir = _TempDir()
    dipole = read_dipole(fname_dip)
    print(dipole)  # test repr
    out_fname = op.join(tempdir, 'temp.dip')
    dipole.save(out_fname)
    dipole_new = read_dipole(out_fname)
    _compare_dipoles(dipole, dipole_new)


@testing.requires_testing_data
def test_dipole_fitting_ctf():
    """Test dipole fitting with CTF data."""
    raw_ctf = read_raw_ctf(fname_ctf).set_eeg_reference(projection=True)
    events = make_fixed_length_events(raw_ctf, 1)
    evoked = Epochs(raw_ctf, events, 1, 0, 0, baseline=None).average()
    cov = make_ad_hoc_cov(evoked.info)
    sphere = make_sphere_model((0., 0., 0.))
    # XXX Eventually we should do some better checks about accuracy, but
    # for now our CTF phantom fitting tutorials will have to do
    # (otherwise we need to add that to the testing dataset, which is
    # a bit too big)
    fit_dipole(evoked, cov, sphere)


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_mne
def test_dipole_fitting():
    """Test dipole fitting."""
    amp = 100e-9
    tempdir = _TempDir()
    rng = np.random.RandomState(0)
    fname_dtemp = op.join(tempdir, 'test.dip')
    fname_sim = op.join(tempdir, 'test-ave.fif')
    fwd = convert_forward_solution(read_forward_solution(fname_fwd),
                                   surf_ori=False, force_fixed=True,
                                   use_cps=True)
    evoked = read_evokeds(fname_evo)[0]
    cov = read_cov(fname_cov)
    n_per_hemi = 5
    vertices = [np.sort(rng.permutation(s['vertno'])[:n_per_hemi])
                for s in fwd['src']]
    nv = sum(len(v) for v in vertices)
    stc = SourceEstimate(amp * np.eye(nv), vertices, 0, 0.001)
    evoked = simulate_evoked(fwd, stc, evoked.info, cov, nave=evoked.nave,
                             random_state=rng)
    # For speed, let's use a subset of channels (strange but works)
    picks = np.sort(np.concatenate([
        pick_types(evoked.info, meg=True, eeg=False)[::2],
        pick_types(evoked.info, meg=False, eeg=True)[::2]]))
    evoked.pick_channels([evoked.ch_names[p] for p in picks])
    evoked.add_proj(make_eeg_average_ref_proj(evoked.info))
    write_evokeds(fname_sim, evoked)

    # Run MNE-C version
    run_subprocess([
        'mne_dipole_fit', '--meas', fname_sim, '--meg', '--eeg',
        '--noise', fname_cov, '--dip', fname_dtemp,
        '--mri', fname_fwd, '--reg', '0', '--tmin', '0',
    ])
    dip_c = read_dipole(fname_dtemp)

    # Run mne-python version
    sphere = make_sphere_model(head_radius=0.1)
    with warnings.catch_warnings(record=True):
        dip, residuals = fit_dipole(evoked, cov, sphere, fname_fwd)

    # Sanity check: do our residuals have less power than orig data?
    data_rms = np.sqrt(np.sum(evoked.data ** 2, axis=0))
    resi_rms = np.sqrt(np.sum(residuals ** 2, axis=0))
    assert_true((data_rms > resi_rms * 0.95).all(),
                msg='%s (factor: %s)' % ((data_rms / resi_rms).min(), 0.95))

    # Compare to original points
    transform_surface_to(fwd['src'][0], 'head', fwd['mri_head_t'])
    transform_surface_to(fwd['src'][1], 'head', fwd['mri_head_t'])
    assert_equal(fwd['src'][0]['coord_frame'], FIFF.FIFFV_COORD_HEAD)
    src_rr = np.concatenate([s['rr'][v] for s, v in zip(fwd['src'], vertices)],
                            axis=0)
    src_nn = np.concatenate([s['nn'][v] for s, v in zip(fwd['src'], vertices)],
                            axis=0)

    # MNE-C skips the last "time" point :(
    out = dip.crop(dip_c.times[0], dip_c.times[-1])
    assert_true(dip is out)
    src_rr, src_nn = src_rr[:-1], src_nn[:-1]

    # check that we did about as well
    corrs, dists, gc_dists, amp_errs, gofs = [], [], [], [], []
    for d in (dip_c, dip):
        new = d.pos
        diffs = new - src_rr
        corrs += [np.corrcoef(src_rr.ravel(), new.ravel())[0, 1]]
        dists += [np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))]
        gc_dists += [180 / np.pi * np.mean(np.arccos(np.sum(src_nn * d.ori,
                                                     axis=1)))]
        amp_errs += [np.sqrt(np.mean((amp - d.amplitude) ** 2))]
        gofs += [np.mean(d.gof)]
    factor = 0.8
    assert_true(dists[0] / factor >= dists[1], 'dists: %s' % dists)
    assert_true(corrs[0] * factor <= corrs[1], 'corrs: %s' % corrs)
    assert_true(gc_dists[0] / factor >= gc_dists[1] * 0.8,
                'gc-dists (ori): %s' % gc_dists)
    assert_true(amp_errs[0] / factor >= amp_errs[1],
                'amplitude errors: %s' % amp_errs)
    # This one is weird because our cov/sim/picking is weird
    assert_true(gofs[0] * factor <= gofs[1] * 2, 'gof: %s' % gofs)


@testing.requires_testing_data
def test_dipole_fitting_fixed():
    """Test dipole fitting with a fixed position."""
    import matplotlib.pyplot as plt
    tpeak = 0.073
    sphere = make_sphere_model(head_radius=0.1)
    evoked = read_evokeds(fname_evo, baseline=(None, 0))[0]
    evoked.pick_types(meg=True)
    t_idx = np.argmin(np.abs(tpeak - evoked.times))
    evoked_crop = evoked.copy().crop(tpeak, tpeak)
    assert_equal(len(evoked_crop.times), 1)
    cov = read_cov(fname_cov)
    dip_seq, resid = fit_dipole(evoked_crop, cov, sphere)
    assert_true(isinstance(dip_seq, Dipole))
    assert_equal(len(dip_seq.times), 1)
    pos, ori, gof = dip_seq.pos[0], dip_seq.ori[0], dip_seq.gof[0]
    amp = dip_seq.amplitude[0]
    # Fix position, allow orientation to change
    dip_free, resid_free = fit_dipole(evoked, cov, sphere, pos=pos)
    assert_true(isinstance(dip_free, Dipole))
    assert_allclose(dip_free.times, evoked.times)
    assert_allclose(np.tile(pos[np.newaxis], (len(evoked.times), 1)),
                    dip_free.pos)
    assert_allclose(ori, dip_free.ori[t_idx])  # should find same ori
    assert_true(np.dot(dip_free.ori, ori).mean() < 0.9)  # but few the same
    assert_allclose(gof, dip_free.gof[t_idx])  # ... same gof
    assert_allclose(amp, dip_free.amplitude[t_idx])  # and same amp
    assert_allclose(resid, resid_free[:, [t_idx]])
    # Fix position and orientation
    dip_fixed, resid_fixed = fit_dipole(evoked, cov, sphere, pos=pos, ori=ori)
    assert_true(isinstance(dip_fixed, DipoleFixed))
    assert_allclose(dip_fixed.times, evoked.times)
    assert_allclose(dip_fixed.info['chs'][0]['loc'][:3], pos)
    assert_allclose(dip_fixed.info['chs'][0]['loc'][3:6], ori)
    assert_allclose(dip_fixed.data[1, t_idx], gof)
    assert_allclose(resid, resid_fixed[:, [t_idx]])
    _check_roundtrip_fixed(dip_fixed)
    # bad resetting
    evoked.info['bads'] = [evoked.ch_names[3]]
    dip_fixed, resid_fixed = fit_dipole(evoked, cov, sphere, pos=pos, ori=ori)
    # Degenerate conditions
    evoked_nan = evoked.copy().crop(0, 0)
    evoked_nan.data[0, 0] = None
    assert_raises(ValueError, fit_dipole, evoked_nan, cov, sphere)
    assert_raises(ValueError, fit_dipole, evoked, cov, sphere, ori=[1, 0, 0])
    assert_raises(ValueError, fit_dipole, evoked, cov, sphere, pos=[0, 0, 0],
                  ori=[2, 0, 0])
    assert_raises(ValueError, fit_dipole, evoked, cov, sphere, pos=[0.1, 0, 0])
    # copying
    dip_fixed_2 = dip_fixed.copy()
    dip_fixed_2.data[:] = 0.
    assert not np.isclose(dip_fixed.data, 0., atol=1e-20).any()
    # plotting
    plt.close('all')
    dip_fixed.plot()
    plt.close('all')


@testing.requires_testing_data
def test_len_index_dipoles():
    """Test len and indexing of Dipole objects."""
    dipole = read_dipole(fname_dip)
    d0 = dipole[0]
    d1 = dipole[:1]
    _check_dipole(d0, 1)
    _check_dipole(d1, 1)
    _compare_dipoles(d0, d1)
    mask = dipole.gof > 15
    idx = np.where(mask)[0]
    d_mask = dipole[mask]
    _check_dipole(d_mask, 4)
    _compare_dipoles(d_mask, dipole[idx])


@testing.requires_testing_data
def test_min_distance_fit_dipole():
    """Test dipole min_dist to inner_skull."""
    subject = 'sample'
    raw = read_raw_fif(fname_raw, preload=True)

    # select eeg data
    picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    info = pick_info(raw.info, picks)

    # Let's use cov = Identity
    cov = read_cov(fname_cov)
    cov['data'] = np.eye(cov['data'].shape[0])

    # Simulated scal map
    simulated_scalp_map = np.zeros(picks.shape[0])
    simulated_scalp_map[27:34] = 1

    simulated_scalp_map = simulated_scalp_map[:, None]

    evoked = EvokedArray(simulated_scalp_map, info, tmin=0)

    min_dist = 5.  # distance in mm

    bem = read_bem_solution(fname_bem)
    dip, residual = fit_dipole(evoked, cov, bem, fname_trans,
                               min_dist=min_dist)

    dist = _compute_depth(dip, fname_bem, fname_trans, subject, subjects_dir)

    # Constraints are not exact, so bump the minimum slightly
    assert_true(min_dist - 0.1 < (dist[0] * 1000.) < (min_dist + 1.))

    assert_raises(ValueError, fit_dipole, evoked, cov, fname_bem, fname_trans,
                  -1.)


def _compute_depth(dip, fname_bem, fname_trans, subject, subjects_dir):
    """Compute dipole depth."""
    trans = _get_trans(fname_trans)[0]
    bem = read_bem_solution(fname_bem)
    surf = _bem_find_surface(bem, 'inner_skull')
    points = surf['rr']
    points = apply_trans(trans['trans'], points)
    depth = _compute_nearest(points, dip.pos, return_dists=True)[1][0]
    return np.ravel(depth)


@testing.requires_testing_data
def test_accuracy():
    """Test dipole fitting to sub-mm accuracy."""
    evoked = read_evokeds(fname_evo)[0].crop(0., 0.,)
    evoked.pick_types(meg=True, eeg=False)
    evoked.pick_channels([c for c in evoked.ch_names[::4]])
    for rad, perc_90 in zip((0.09, None), (0.002, 0.004)):
        bem = make_sphere_model('auto', rad, evoked.info,
                                relative_radii=(0.999, 0.998, 0.997, 0.995))
        src = read_source_spaces(fname_src)

        fwd = make_forward_solution(evoked.info, None, src, bem)
        fwd = convert_forward_solution(fwd, force_fixed=True, use_cps=True)
        vertices = [src[0]['vertno'], src[1]['vertno']]
        n_vertices = sum(len(v) for v in vertices)
        amp = 10e-9
        data = np.eye(n_vertices + 1)[:n_vertices]
        data[-1, -1] = 1.
        data *= amp
        stc = SourceEstimate(data, vertices, 0., 1e-3, 'sample')
        evoked.info.normalize_proj()
        sim = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf)

        cov = make_ad_hoc_cov(evoked.info)
        dip = fit_dipole(sim, cov, bem, min_dist=0.001)[0]

        ds = []
        for vi in range(n_vertices):
            if vi < len(vertices[0]):
                hi = 0
                vertno = vi
            else:
                hi = 1
                vertno = vi - len(vertices[0])
            vertno = src[hi]['vertno'][vertno]
            rr = src[hi]['rr'][vertno]
            d = np.sqrt(np.sum((rr - dip.pos[vi]) ** 2))
            ds.append(d)
        # make sure that our median is sub-mm and the large majority are very
        # close (we expect some to be off by a bit e.g. because they are
        # radial)
        assert_true((np.percentile(ds, [50, 90]) < [0.0005, perc_90]).all())


@testing.requires_testing_data
def test_dipole_fixed():
    """Test reading a fixed-position dipole (from Xfit)."""
    dip = read_dipole(fname_xfit_dip)
    # print the representation of the objet DipoleFixed
    print(dip)

    _check_roundtrip_fixed(dip)
    with warnings.catch_warnings(record=True) as w:  # unused fields
        dip_txt = read_dipole(fname_xfit_dip_txt)
    assert_true(any('extra fields' in str(ww.message) for ww in w))
    assert_allclose(dip.info['chs'][0]['loc'][:3], dip_txt.pos[0])
    assert_allclose(dip_txt.amplitude[0], 12.1e-9)
    with warnings.catch_warnings(record=True):  # unused fields
        dip_txt_seq = read_dipole(fname_xfit_seq_txt)
    assert_allclose(dip_txt_seq.gof, [27.3, 46.4, 43.7, 41., 37.3, 32.5])


def _check_roundtrip_fixed(dip):
    """Helper to test roundtrip IO for fixed dipoles."""
    tempdir = _TempDir()
    dip.save(op.join(tempdir, 'test-dip.fif.gz'))
    dip_read = read_dipole(op.join(tempdir, 'test-dip.fif.gz'))
    assert_allclose(dip_read.data, dip_read.data)
    assert_allclose(dip_read.times, dip.times)
    assert_equal(dip_read.info['xplotter_layout'], dip.info['xplotter_layout'])
    assert_equal(dip_read.ch_names, dip.ch_names)
    for ch_1, ch_2 in zip(dip_read.info['chs'], dip.info['chs']):
        assert_equal(ch_1['ch_name'], ch_2['ch_name'])
        for key in ('loc', 'kind', 'unit_mul', 'range', 'coord_frame', 'unit',
                    'cal', 'coil_type', 'scanno', 'logno'):
            assert_allclose(ch_1[key], ch_2[key], err_msg=key)


def test_get_phantom_dipoles():
    """Test getting phantom dipole locations."""
    assert_raises(ValueError, get_phantom_dipoles, 0)
    assert_raises(ValueError, get_phantom_dipoles, 'foo')
    for kind in ('vectorview', 'otaniemi'):
        pos, ori = get_phantom_dipoles(kind)
        assert_equal(pos.shape, (32, 3))
        assert_equal(ori.shape, (32, 3))


@testing.requires_testing_data
def test_confidence():
    """Test confidence limits."""
    tempdir = _TempDir()
    evoked = read_evokeds(fname_evo_full, 'Left Auditory', baseline=(None, 0))
    evoked.crop(0.08, 0.08).pick_types()  # MEG-only
    cov = make_ad_hoc_cov(evoked.info)
    sphere = make_sphere_model((0., 0., 0.04), 0.08)
    dip_py = fit_dipole(evoked, cov, sphere)[0]
    fname_test = op.join(tempdir, 'temp-dip.txt')
    dip_py.save(fname_test)
    dip_read = read_dipole(fname_test)
    with warnings.catch_warnings(record=True) as w:
        dip_xfit = read_dipole(fname_dip_xfit)
    assert_equal(len(w), 1)
    assert_true("['noise/ft/cm', 'prob']" in str(w[0].message))
    for dip_check in (dip_py, dip_read):
        assert_allclose(dip_check.pos, dip_xfit.pos, atol=5e-4)  # < 0.5 mm
        assert_allclose(dip_check.gof, dip_xfit.gof, atol=5e-1)  # < 0.5%
        assert_array_equal(dip_check.nfree, dip_xfit.nfree)  # exact match
        assert_allclose(dip_check.khi2, dip_xfit.khi2, rtol=2e-2)  # 2% miss
        assert_equal(set(dip_check.conf.keys()), set(dip_xfit.conf.keys()))
        for key in sorted(dip_check.conf.keys()):
            assert_allclose(dip_check.conf[key], dip_xfit.conf[key],
                            rtol=1.5e-1, err_msg=key)

run_tests_if_main(False)
