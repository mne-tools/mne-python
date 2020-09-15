import os
import os.path as op

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import matplotlib.pyplot as plt
import pytest

from mne import (read_dipole, read_forward_solution,
                 convert_forward_solution, read_evokeds, read_cov,
                 SourceEstimate, write_evokeds, fit_dipole,
                 transform_surface_to, make_sphere_model, pick_types,
                 pick_info, EvokedArray, read_source_spaces, make_ad_hoc_cov,
                 make_forward_solution, Dipole, DipoleFixed, Epochs,
                 make_fixed_length_events, Evoked)
from mne.dipole import get_phantom_dipoles, _BDIP_ERROR_KEYS
from mne.simulation import simulate_evoked
from mne.datasets import testing
from mne.utils import run_tests_if_main, requires_mne, run_subprocess
from mne.proj import make_eeg_average_ref_proj

from mne.io import read_raw_fif, read_raw_ctf
from mne.io.constants import FIFF

from mne.surface import _compute_nearest
from mne.bem import _bem_find_surface, read_bem_solution
from mne.transforms import apply_trans, _get_trans

data_path = testing.data_path(download=False)
meg_path = op.join(data_path, 'MEG', 'sample')
fname_dip_xfit_80 = op.join(meg_path, 'sample_audvis-ave_xfit.dip')
fname_raw = op.join(meg_path, 'sample_audvis_trunc_raw.fif')
fname_dip = op.join(meg_path, 'sample_audvis_trunc_set1.dip')
fname_bdip = op.join(meg_path, 'sample_audvis_trunc_set1.bdip')
fname_dip_xfit = op.join(meg_path, 'sample_audvis_trunc_xfit.dip')
fname_bdip_xfit = op.join(meg_path, 'sample_audvis_trunc_xfit.bdip')
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
    assert orig.name == new.name


def _check_dipole(dip, n_dipoles):
    """Check dipole sizes."""
    assert len(dip) == n_dipoles
    assert dip.pos.shape == (n_dipoles, 3)
    assert dip.ori.shape == (n_dipoles, 3)
    assert dip.gof.shape == (n_dipoles,)
    assert dip.amplitude.shape == (n_dipoles,)


@testing.requires_testing_data
def test_io_dipoles(tmpdir):
    """Test IO for .dip files."""
    dipole = read_dipole(fname_dip)
    assert 'Dipole ' in repr(dipole)  # test repr
    out_fname = op.join(str(tmpdir), 'temp.dip')
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
    fit_dipole(evoked, cov, sphere, rank=dict(meg=len(evoked.data)))


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_mne
def test_dipole_fitting(tmpdir):
    """Test dipole fitting."""
    amp = 100e-9
    tempdir = str(tmpdir)
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
    with pytest.warns(RuntimeWarning, match='projection'):
        dip, residual = fit_dipole(evoked, cov, sphere, fname_fwd,
                                   rank='info')  # just to test rank support
    assert isinstance(residual, Evoked)

    # Sanity check: do our residuals have less power than orig data?
    data_rms = np.sqrt(np.sum(evoked.data ** 2, axis=0))
    resi_rms = np.sqrt(np.sum(residual.data ** 2, axis=0))
    assert (data_rms > resi_rms * 0.95).all(), \
        '%s (factor: %s)' % ((data_rms / resi_rms).min(), 0.95)

    # Compare to original points
    transform_surface_to(fwd['src'][0], 'head', fwd['mri_head_t'])
    transform_surface_to(fwd['src'][1], 'head', fwd['mri_head_t'])
    assert fwd['src'][0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD
    src_rr = np.concatenate([s['rr'][v] for s, v in zip(fwd['src'], vertices)],
                            axis=0)
    src_nn = np.concatenate([s['nn'][v] for s, v in zip(fwd['src'], vertices)],
                            axis=0)

    # MNE-C skips the last "time" point :(
    out = dip.crop(dip_c.times[0], dip_c.times[-1])
    assert (dip is out)
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
    # XXX possibly some OpenBLAS numerical differences make
    # things slightly worse for us
    factor = 0.7
    assert dists[0] / factor >= dists[1], 'dists: %s' % dists
    assert corrs[0] * factor <= corrs[1], 'corrs: %s' % corrs
    assert gc_dists[0] / factor >= gc_dists[1] * 0.8, \
        'gc-dists (ori): %s' % gc_dists
    assert amp_errs[0] / factor >= amp_errs[1],\
        'amplitude errors: %s' % amp_errs
    # This one is weird because our cov/sim/picking is weird
    assert gofs[0] * factor <= gofs[1] * 2, 'gof: %s' % gofs


@testing.requires_testing_data
def test_dipole_fitting_fixed(tmpdir):
    """Test dipole fitting with a fixed position."""
    tpeak = 0.073
    sphere = make_sphere_model(head_radius=0.1)
    evoked = read_evokeds(fname_evo, baseline=(None, 0))[0]
    evoked.pick_types(meg=True)
    t_idx = np.argmin(np.abs(tpeak - evoked.times))
    evoked_crop = evoked.copy().crop(tpeak, tpeak)
    assert len(evoked_crop.times) == 1
    cov = read_cov(fname_cov)
    dip_seq, resid = fit_dipole(evoked_crop, cov, sphere)
    assert isinstance(dip_seq, Dipole)
    assert isinstance(resid, Evoked)
    assert len(dip_seq.times) == 1
    pos, ori, gof = dip_seq.pos[0], dip_seq.ori[0], dip_seq.gof[0]
    amp = dip_seq.amplitude[0]
    # Fix position, allow orientation to change
    dip_free, resid_free = fit_dipole(evoked, cov, sphere, pos=pos)
    assert isinstance(dip_free, Dipole)
    assert isinstance(resid_free, Evoked)
    assert_allclose(dip_free.times, evoked.times)
    assert_allclose(np.tile(pos[np.newaxis], (len(evoked.times), 1)),
                    dip_free.pos)
    assert_allclose(ori, dip_free.ori[t_idx])  # should find same ori
    assert (np.dot(dip_free.ori, ori).mean() < 0.9)  # but few the same
    assert_allclose(gof, dip_free.gof[t_idx])  # ... same gof
    assert_allclose(amp, dip_free.amplitude[t_idx])  # and same amp
    assert_allclose(resid.data, resid_free.data[:, [t_idx]])
    # Fix position and orientation
    dip_fixed, resid_fixed = fit_dipole(evoked, cov, sphere, pos=pos, ori=ori)
    assert (isinstance(dip_fixed, DipoleFixed))
    assert_allclose(dip_fixed.times, evoked.times)
    assert_allclose(dip_fixed.info['chs'][0]['loc'][:3], pos)
    assert_allclose(dip_fixed.info['chs'][0]['loc'][3:6], ori)
    assert_allclose(dip_fixed.data[1, t_idx], gof)
    assert_allclose(resid.data, resid_fixed.data[:, [t_idx]])
    _check_roundtrip_fixed(dip_fixed, tmpdir)
    # bad resetting
    evoked.info['bads'] = [evoked.ch_names[3]]
    dip_fixed, resid_fixed = fit_dipole(evoked, cov, sphere, pos=pos, ori=ori)
    # Degenerate conditions
    evoked_nan = evoked.copy().crop(0, 0)
    evoked_nan.data[0, 0] = None
    pytest.raises(ValueError, fit_dipole, evoked_nan, cov, sphere)
    pytest.raises(ValueError, fit_dipole, evoked, cov, sphere, ori=[1, 0, 0])
    pytest.raises(ValueError, fit_dipole, evoked, cov, sphere, pos=[0, 0, 0],
                  ori=[2, 0, 0])
    pytest.raises(ValueError, fit_dipole, evoked, cov, sphere, pos=[0.1, 0, 0])
    # copying
    dip_fixed_2 = dip_fixed.copy()
    dip_fixed_2.data[:] = 0.
    assert not np.isclose(dip_fixed.data, 0., atol=1e-20).any()
    # plotting
    plt.close('all')
    dip_fixed.plot()
    plt.close('all')
    orig_times = np.array(dip_fixed.times)
    shift_times = dip_fixed.shift_time(1.).times
    assert_allclose(shift_times, orig_times + 1)


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


@pytest.mark.slowtest  # slow-ish on Travis OSX
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
    assert isinstance(residual, Evoked)

    dist = _compute_depth(dip, fname_bem, fname_trans, subject, subjects_dir)

    # Constraints are not exact, so bump the minimum slightly
    assert (min_dist - 0.1 < (dist[0] * 1000.) < (min_dist + 1.))

    pytest.raises(ValueError, fit_dipole, evoked, cov, fname_bem, fname_trans,
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
        assert ((np.percentile(ds, [50, 90]) < [0.0005, perc_90]).all())


@testing.requires_testing_data
def test_dipole_fixed(tmpdir):
    """Test reading a fixed-position dipole (from Xfit)."""
    dip = read_dipole(fname_xfit_dip)
    # print the representation of the object DipoleFixed
    assert 'DipoleFixed ' in repr(dip)

    _check_roundtrip_fixed(dip, tmpdir)
    with pytest.warns(RuntimeWarning, match='extra fields'):
        dip_txt = read_dipole(fname_xfit_dip_txt)
    assert_allclose(dip.info['chs'][0]['loc'][:3], dip_txt.pos[0])
    assert_allclose(dip_txt.amplitude[0], 12.1e-9)
    with pytest.warns(RuntimeWarning, match='extra fields'):
        dip_txt_seq = read_dipole(fname_xfit_seq_txt)
    assert_allclose(dip_txt_seq.gof, [27.3, 46.4, 43.7, 41., 37.3, 32.5])


def _check_roundtrip_fixed(dip, tmpdir):
    """Check roundtrip IO for fixed dipoles."""
    tempdir = str(tmpdir)
    dip.save(op.join(tempdir, 'test-dip.fif.gz'))
    dip_read = read_dipole(op.join(tempdir, 'test-dip.fif.gz'))
    assert_allclose(dip_read.data, dip_read.data)
    assert_allclose(dip_read.times, dip.times, atol=1e-8)
    assert dip_read.info['xplotter_layout'] == dip.info['xplotter_layout']
    assert dip_read.ch_names == dip.ch_names
    for ch_1, ch_2 in zip(dip_read.info['chs'], dip.info['chs']):
        assert ch_1['ch_name'] == ch_2['ch_name']
        for key in ('loc', 'kind', 'unit_mul', 'range', 'coord_frame', 'unit',
                    'cal', 'coil_type', 'scanno', 'logno'):
            assert_allclose(ch_1[key], ch_2[key], err_msg=key)


def test_get_phantom_dipoles():
    """Test getting phantom dipole locations."""
    pytest.raises(ValueError, get_phantom_dipoles, 0)
    pytest.raises(ValueError, get_phantom_dipoles, 'foo')
    for kind in ('vectorview', 'otaniemi'):
        pos, ori = get_phantom_dipoles(kind)
        assert pos.shape == (32, 3)
        assert ori.shape == (32, 3)


@testing.requires_testing_data
def test_confidence(tmpdir):
    """Test confidence limits."""
    evoked = read_evokeds(fname_evo_full, 'Left Auditory', baseline=(None, 0))
    evoked.crop(0.08, 0.08).pick_types(meg=True)  # MEG-only
    cov = make_ad_hoc_cov(evoked.info)
    sphere = make_sphere_model((0., 0., 0.04), 0.08)
    dip_py = fit_dipole(evoked, cov, sphere)[0]
    fname_test = op.join(str(tmpdir), 'temp-dip.txt')
    dip_py.save(fname_test)
    dip_read = read_dipole(fname_test)
    with pytest.warns(RuntimeWarning, match="'noise/ft/cm', 'prob'"):
        dip_xfit = read_dipole(fname_dip_xfit_80)
    for dip_check in (dip_py, dip_read):
        assert_allclose(dip_check.pos, dip_xfit.pos, atol=5e-4)  # < 0.5 mm
        assert_allclose(dip_check.gof, dip_xfit.gof, atol=5e-1)  # < 0.5%
        assert_array_equal(dip_check.nfree, dip_xfit.nfree)  # exact match
        assert_allclose(dip_check.khi2, dip_xfit.khi2, rtol=2e-2)  # 2% miss
        assert set(dip_check.conf.keys()) == set(dip_xfit.conf.keys())
        for key in sorted(dip_check.conf.keys()):
            assert_allclose(dip_check.conf[key], dip_xfit.conf[key],
                            rtol=1.5e-1, err_msg=key)


# bdip created with:
# mne_dipole_fit --meas sample_audvis_trunc-ave.fif --set 1 --meg --tmin 40 --tmax 95 --bmin -200 --bmax 0 --noise sample_audvis_trunc-cov.fif --bem ../../subjects/sample/bem/sample-1280-1280-1280-bem-sol.fif --origin 0\:0\:40 --mri sample_audvis_trunc-trans.fif --bdip sample_audvis_trunc_set1.bdip  # noqa: E501
# It gives equivalent results to .dip in non-dipole mode.
# xfit bdip created by taking sample_audvis_trunc-ave.fif, picking MEG
# channels, writitng to disk (with MNE), then running xfit on 40-95 ms
# with a 3.3 ms step
@testing.requires_testing_data
@pytest.mark.parametrize('fname_dip_, fname_bdip_', [
    (fname_dip, fname_bdip),
    (fname_dip_xfit, fname_bdip_xfit),
])
def test_bdip(fname_dip_, fname_bdip_, tmpdir):
    """Test bdip I/O."""
    # use text as veridical
    with pytest.warns(None):  # ignored fields
        dip = read_dipole(fname_dip_)
    # read binary
    orig_size = os.stat(fname_bdip_).st_size
    bdip = read_dipole(fname_bdip_)
    # test round-trip by writing and reading, too
    fname = tmpdir.join('test.bdip')
    bdip.save(fname)
    bdip_read = read_dipole(fname)
    write_size = os.stat(str(fname)).st_size
    assert orig_size == write_size
    assert len(dip) == len(bdip) == len(bdip_read) == 17
    dip_has_conf = fname_dip_ == fname_dip_xfit
    for kind, this_bdip in (('orig', bdip), ('read', bdip_read)):
        for key, atol in (
                ('pos', 5e-5),
                ('ori', 5e-3),
                ('gof', 0.5e-1),
                ('times', 5e-5),
                ('khi2', 1e-2)):
            d = getattr(dip, key)
            b = getattr(this_bdip, key)
            if key == 'khi2' and dip_has_conf:
                if d is not None:
                    assert_allclose(d, b, atol=atol,
                                    err_msg='%s: %s' % (kind, key))
                else:
                    assert b is None
        if dip_has_conf:
            # conf
            conf_keys = _BDIP_ERROR_KEYS + ('vol',)
            assert (set(this_bdip.conf.keys()) ==
                    set(dip.conf.keys()) ==
                    set(conf_keys))
            for key in conf_keys:
                d = dip.conf[key]
                b = this_bdip.conf[key]
                assert_allclose(d, b, rtol=0.12,  # no so great, text I/O
                                err_msg='%s: %s' % (kind, key))
        # Not stored
        assert this_bdip.name is None
        assert_allclose(this_bdip.nfree, 0.)


run_tests_if_main()
