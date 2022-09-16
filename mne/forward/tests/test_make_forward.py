from itertools import product
import os
import os.path as op
from pathlib import Path

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numpy.testing import assert_array_less

from mne.bem import read_bem_surfaces, make_bem_solution
from mne.channels import make_standard_montage
from mne.datasets import testing
from mne.io import read_raw_fif, read_raw_kit, read_raw_bti, read_info
from mne.io.constants import FIFF
from mne import (read_forward_solution, write_forward_solution,
                 make_forward_solution, convert_forward_solution,
                 setup_volume_source_space, read_source_spaces, create_info,
                 make_sphere_model, pick_types_forward, pick_info, pick_types,
                 read_evokeds, read_cov, read_dipole,
                 get_volume_labels_from_aseg)
from mne.surface import _get_ico_surface
from mne.transforms import Transform
from mne.utils import (requires_mne, requires_nibabel, run_subprocess,
                       catch_logging, requires_mne_mark,
                       requires_openmeeg_mark)
from mne.forward._make_forward import _create_meg_coils, make_forward_dipole
from mne.forward._compute_forward import _magnetic_dipole_field_vec
from mne.forward import Forward, _do_forward_solution, use_coil_def
from mne.dipole import Dipole, fit_dipole
from mne.simulation import simulate_evoked
from mne.source_estimate import VolSourceEstimate
from mne.source_space import (write_source_spaces, _compare_source_spaces,
                              setup_source_space)

from mne.forward.tests.test_forward import assert_forward_allclose

data_path = testing.data_path(download=False)
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_raw = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')
fname_evo = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_dip = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
subjects_dir = os.path.join(data_path, 'subjects')
fname_src = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-4-src.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
fname_aseg = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
fname_bem_meg = op.join(subjects_dir, 'sample', 'bem',
                        'sample-1280-bem-sol.fif')

io_path = Path(__file__).parent.parent.parent / 'io'
bti_dir = io_path / 'bti' / 'tests' / 'data'
kit_dir = io_path / 'kit' / 'tests' / 'data'
trans_path = op.join(kit_dir, 'trans-sample.fif')
fname_ctf_raw = io_path / 'tests' / 'data' / 'test_ctf_comp_raw.fif'


def _col_corrs(a, b):
    """Compute correlation between paired columns, being careful about 0."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    num = (a * b).mean(0)
    a_std = np.sqrt((a * a).mean(0))
    b_std = np.sqrt((b * b).mean(0))
    all_zero = (a_std == 0) & (b_std == 0)
    num[all_zero] = 1.
    a_std[all_zero] = 1.
    b_std[all_zero] = 1.
    return num / (a_std * b_std)


def _rdm(a, b):
    """Compute the ratio of norms, being careful about 0."""
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    all_zero = (a_norm == 0) & (b_norm == 0)
    a_norm[all_zero] = 1.
    b_norm[all_zero] = 1.
    return a_norm / b_norm


def _compare_forwards(fwd, fwd_py, n_sensors, n_src,
                      meg_rtol=1e-4, meg_atol=1e-9,
                      meg_corr_tol=0.99, meg_rdm_tol=0.01,
                      eeg_rtol=1e-3, eeg_atol=1e-3,
                      eeg_corr_tol=0.99, eeg_rdm_tol=0.01):
    """Test forwards."""
    # check source spaces
    assert len(fwd['src']) == len(fwd_py['src'])
    _compare_source_spaces(fwd['src'], fwd_py['src'], mode='approx')
    for surf_ori, force_fixed in product([False, True], [False, True]):
        # use copy here to leave our originals unmodified
        fwd = convert_forward_solution(fwd, surf_ori, force_fixed, copy=True,
                                       use_cps=True)
        fwd_py = convert_forward_solution(fwd_py, surf_ori, force_fixed,
                                          copy=True, use_cps=True)
        check_src = n_src // 3 if force_fixed else n_src

        for key in ('nchan', 'source_rr', 'source_ori',
                    'surf_ori', 'coord_frame', 'nsource'):
            assert_allclose(fwd_py[key], fwd[key], rtol=1e-4, atol=1e-7,
                            err_msg=key)
        # In surf_ori=True only Z matters for source_nn
        if surf_ori and not force_fixed:
            ori_sl = slice(2, None, 3)
        else:
            ori_sl = slice(None)
        assert_allclose(fwd_py['source_nn'][ori_sl], fwd['source_nn'][ori_sl],
                        rtol=1e-4, atol=1e-6)
        assert_allclose(fwd_py['mri_head_t']['trans'],
                        fwd['mri_head_t']['trans'], rtol=1e-5, atol=1e-8)

        assert fwd_py['sol']['data'].shape == (n_sensors, check_src)
        assert len(fwd['sol']['row_names']) == n_sensors
        assert len(fwd_py['sol']['row_names']) == n_sensors

        # check MEG
        fwd_meg = fwd['sol']['data'][:306, ori_sl]
        fwd_meg_py = fwd_py['sol']['data'][:306, ori_sl]
        assert_allclose(fwd_meg, fwd_meg_py, rtol=meg_rtol, atol=meg_atol,
                        err_msg='MEG mismatch')
        meg_corrs = _col_corrs(fwd_meg, fwd_meg_py)
        assert_array_less(meg_corr_tol, meg_corrs, err_msg='MEG corr/MAG')
        meg_rdm = _rdm(fwd_meg, fwd_meg_py)
        assert_allclose(meg_rdm, 1, atol=meg_rdm_tol, err_msg='MEG RDM')
        # check EEG
        if fwd['sol']['data'].shape[0] > 306:
            fwd_eeg = fwd['sol']['data'][306:, ori_sl]
            fwd_eeg_py = fwd['sol']['data'][306:, ori_sl]
            assert_allclose(fwd_eeg, fwd_eeg_py, rtol=eeg_rtol, atol=eeg_atol,
                            err_msg='EEG mismatch')
            # To test so-called MAG we use correlation (related to cosine
            # similarity) and also RDM to test the amplitude mismatch
            eeg_corrs = _col_corrs(fwd_eeg, fwd_eeg_py)
            assert_array_less(eeg_corr_tol, eeg_corrs, err_msg='EEG corr/MAG')
            eeg_rdm = _rdm(fwd_eeg, fwd_eeg_py)
            assert_allclose(eeg_rdm, 1, atol=eeg_rdm_tol, err_msg='EEG RDM')


def test_magnetic_dipole():
    """Test basic magnetic dipole forward calculation."""
    info = read_info(fname_raw)
    picks = pick_types(info, meg=True, eeg=False, exclude=[])
    info = pick_info(info, picks[:12])
    coils = _create_meg_coils(info['chs'], 'normal', None)
    # magnetic dipole far (meters!) from device origin
    r0 = np.array([0., 13., -6.])
    for ch, coil in zip(info['chs'], coils):
        rr = (ch['loc'][:3] + r0) / 2.  # get halfway closer
        far_fwd = _magnetic_dipole_field_vec(r0[np.newaxis, :], [coil])
        near_fwd = _magnetic_dipole_field_vec(rr[np.newaxis, :], [coil])
        ratio = 8. if ch['ch_name'][-1] == '1' else 16.  # grad vs mag
        assert_allclose(np.median(near_fwd / far_fwd), ratio, atol=1e-1)
    # degenerate case
    r0 = coils[0]['rmag'][[0]]
    with pytest.raises(RuntimeError, match='Coil too close'):
        _magnetic_dipole_field_vec(r0, coils[:1])
    with pytest.warns(RuntimeWarning, match='Coil too close'):
        fwd = _magnetic_dipole_field_vec(r0, coils[:1], too_close='warning')
    assert not np.isfinite(fwd).any()
    with np.errstate(invalid='ignore'):
        fwd = _magnetic_dipole_field_vec(r0, coils[:1], too_close='info')
    assert not np.isfinite(fwd).any()


@pytest.mark.slowtest  # slow-ish on Travis OSX
@requires_mne
def test_make_forward_solution_kit(tmp_path, fname_src_small):
    """Test making fwd using KIT (compensated) files."""
    sqd_path = op.join(kit_dir, 'test.sqd')
    mrk_path = op.join(kit_dir, 'test_mrk.sqd')
    elp_path = op.join(kit_dir, 'test_elp.txt')
    hsp_path = op.join(kit_dir, 'test_hsp.txt')
    fname_kit_raw = op.join(kit_dir, 'test_bin_raw.fif')

    # first use mne-C: convert file, make forward solution
    fwd = _do_forward_solution('sample', fname_kit_raw, src=fname_src_small,
                               bem=fname_bem_meg, mri=trans_path,
                               eeg=False, meg=True, subjects_dir=subjects_dir)
    assert (isinstance(fwd, Forward))

    # now let's use python with the same raw file
    src = read_source_spaces(fname_src_small)
    fwd_py = make_forward_solution(fname_kit_raw, trans_path, src,
                                   fname_bem_meg, eeg=False, meg=True)
    _compare_forwards(fwd, fwd_py, 157, n_src_small)
    assert (isinstance(fwd_py, Forward))

    # now let's use mne-python all the way
    raw_py = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)
    # without ignore_ref=True, this should throw an error:
    with pytest.raises(NotImplementedError, match='Cannot.*KIT reference'):
        make_forward_solution(raw_py.info, src=src, eeg=False, meg=True,
                              bem=fname_bem_meg, trans=trans_path)

    # check that asking for eeg channels (even if they don't exist) is handled
    meg_only_info = pick_info(raw_py.info, pick_types(raw_py.info, meg=True,
                                                      eeg=False))
    fwd_py = make_forward_solution(meg_only_info, src=src, meg=True, eeg=True,
                                   bem=fname_bem_meg, trans=trans_path,
                                   ignore_ref=True)
    _compare_forwards(fwd, fwd_py, 157, n_src_small,
                      meg_rtol=1e-3, meg_atol=1e-7)


@requires_mne
def test_make_forward_solution_bti(fname_src_small):
    """Test BTI end-to-end versus C."""
    bti_pdf = bti_dir / 'test_pdf_linux'
    bti_config = bti_dir / 'test_config_linux'
    bti_hs = bti_dir / 'test_hs_linux'
    fname_bti_raw = bti_dir / 'exported4D_linux_raw.fif'
    raw_py = read_raw_bti(bti_pdf, bti_config, bti_hs, preload=False)
    src = read_source_spaces(fname_src_small)
    fwd_py = make_forward_solution(raw_py.info, src=src, eeg=False, meg=True,
                                   bem=fname_bem_meg, trans=trans_path)
    fwd = _do_forward_solution('sample', fname_bti_raw, src=fname_src_small,
                               bem=fname_bem_meg, mri=trans_path,
                               eeg=False, meg=True, subjects_dir=subjects_dir)
    _compare_forwards(fwd, fwd_py, 248, n_src_small)


@pytest.mark.parametrize('other', [
    pytest.param('MNE-C', marks=requires_mne_mark()),
    pytest.param('openmeeg', marks=requires_openmeeg_mark()),
])
def test_make_forward_solution_ctf(tmp_path, fname_src_small, other):
    """Test CTF w/compensation against MNE-C or OpenMEEG."""
    src = read_source_spaces(fname_src_small)
    raw = read_raw_fif(fname_ctf_raw)
    assert raw.compensation_grade == 3
    if other == 'openmeeg':
        mindist = 20.
        n_src_want = 51
    else:
        assert other == 'MNE-C'
        mindist = 0.
        n_src_want = n_src_small
        assert n_src_want == 108
    mindist = 20. if other == 'openmeeg' else 0.
    fwd_py = make_forward_solution(
        fname_ctf_raw, fname_trans, src, fname_bem_meg, eeg=False,
        mindist=mindist, verbose=True)

    if other == 'openmeeg':
        # TODO: This should be a 1-layer, but it's broken
        # (some correlations become negative!)...
        bem_surfaces = read_bem_surfaces(fname_bem)  # fname_bem_meg
        bem = make_bem_solution(bem_surfaces, solver='openmeeg')
        # TODO: These tolerances are bad
        tol_kwargs = dict(meg_atol=1, meg_corr_tol=0.65, meg_rdm_tol=0.6)
        fwd = make_forward_solution(
            fname_ctf_raw, fname_trans, src, bem, eeg=False, mindist=mindist,
            verbose=True)
    else:
        assert other == 'MNE-C'
        bem = None
        tol_kwargs = dict()
        fwd = _do_forward_solution(
            'sample', fname_ctf_raw, mri=fname_trans, src=fname_src_small,
            bem=fname_bem_meg, eeg=False, meg=True, subjects_dir=subjects_dir,
            mindist=mindist)
    _compare_forwards(fwd, fwd_py, 274, n_src_want, **tol_kwargs)

    # CTF with compensation changed in python
    ctf_raw = read_raw_fif(fname_ctf_raw)
    ctf_raw.info['bads'] = ['MRO24-2908']  # test that it works with some bads
    ctf_raw.apply_gradient_compensation(2)

    fwd_py = make_forward_solution(
        ctf_raw.info, fname_trans, src, fname_bem_meg, eeg=False, meg=True,
        mindist=mindist)
    if other == 'openmeeg':
        assert bem is not None
        fwd = make_forward_solution(
            ctf_raw.info, fname_trans, src, bem, eeg=False, mindist=mindist,
            verbose=True)
    else:
        fwd = _do_forward_solution(
            'sample', ctf_raw, mri=fname_trans, src=fname_src_small,
            bem=fname_bem_meg, eeg=False, meg=True, subjects_dir=subjects_dir,
            mindist=mindist)
    _compare_forwards(fwd, fwd_py, 274, n_src_want, **tol_kwargs)

    fname_temp = tmp_path / 'test-ctf-fwd.fif'
    write_forward_solution(fname_temp, fwd_py)
    fwd_py2 = read_forward_solution(fname_temp)
    _compare_forwards(fwd_py, fwd_py2, 274, n_src_want, **tol_kwargs)
    repr(fwd_py)


@testing.requires_testing_data
def test_make_forward_solution_basic():
    """Test making M-EEG forward solution from python."""
    with catch_logging() as log:
        # make sure everything can be path-like (gh #10872)
        fwd_py = make_forward_solution(
            Path(fname_raw), Path(fname_trans), Path(fname_src),
            Path(fname_bem), mindist=5., verbose=True)
    log = log.getvalue()
    assert 'Total 258/258 points inside the surface' in log
    assert (isinstance(fwd_py, Forward))
    fwd = read_forward_solution(fname_meeg)
    assert (isinstance(fwd, Forward))
    _compare_forwards(fwd, fwd_py, 366, 1494, meg_rtol=1e-3)
    # Homogeneous model
    with pytest.raises(RuntimeError, match='homogeneous.*1-layer.*EEG'):
        make_forward_solution(fname_raw, fname_trans, fname_src,
                              fname_bem_meg)


@requires_openmeeg_mark()
@pytest.mark.parametrize("n_layers", [
    3,
    pytest.param(1, marks=pytest.mark.xfail(raises=RuntimeError)),
])
@testing.requires_testing_data
def test_make_forward_solution_openmeeg(n_layers):
    """Test making M-EEG forward solution from OpenMEEG."""
    solver = "openmeeg"
    bem_surfaces = read_bem_surfaces(fname_bem)
    raw = read_raw_fif(fname_raw)
    n_sensors = 366
    ch_types = ['eeg', 'meg']
    if n_layers == 1:
        ch_types = ['meg']
        bem_surfaces = bem_surfaces[-1:]
        assert bem_surfaces[0]['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN
        n_sensors = 306
    raw.pick(ch_types)
    n_sources_kept = 501 // 3
    fwds = dict()
    for solver in ["openmeeg", "mne"]:
        bem = make_bem_solution(bem_surfaces, solver=solver)
        assert bem['solver'] == solver
        with catch_logging() as log:
            # make sure everything can be path-like (gh #10872)
            fwd = make_forward_solution(
                raw.info, Path(fname_trans), Path(fname_src),
                bem, mindist=20., verbose=True)
        log = log.getvalue()
        assert 'Total 258/258 points inside the surface' in log
        assert (isinstance(fwd, Forward))
        fwds[solver] = fwd
        del fwd
    _compare_forwards(fwds["openmeeg"],
                      fwds["mne"], n_sensors, n_sources_kept * 3,
                      meg_atol=1, eeg_atol=100,
                      meg_corr_tol=0.98, eeg_corr_tol=0.98,
                      meg_rdm_tol=0.1, eeg_rdm_tol=0.2)


def test_make_forward_solution_discrete(tmp_path, small_surf_src):
    """Test making and converting a forward solution with discrete src."""
    # smoke test for depth weighting and discrete source spaces
    src = small_surf_src
    src = src + setup_volume_source_space(
        pos=dict(rr=src[0]['rr'][src[0]['vertno'][:3]].copy(),
                 nn=src[0]['nn'][src[0]['vertno'][:3]].copy()))
    sphere = make_sphere_model()
    fwd = make_forward_solution(fname_raw, fname_trans, src, sphere,
                                meg=True, eeg=False)
    convert_forward_solution(fwd, surf_ori=True)


n_src_small = 108  # this is the resulting # of verts in fwd


@pytest.fixture(scope='module', params=[testing._pytest_param()])
def small_surf_src():
    """Create a small surface source space."""
    src = setup_source_space('sample', 'oct2', subjects_dir=subjects_dir,
                             add_dist=False)
    assert sum(s['nuse'] for s in src) * 3 == n_src_small
    return src


@pytest.fixture()
def fname_src_small(tmp_path, small_surf_src):
    """Create a small source space."""
    fname_src_small = tmp_path / 'sample-oct-2-src.fif'
    write_source_spaces(fname_src_small, small_surf_src)
    return fname_src_small


@requires_mne
@pytest.mark.timeout(90)  # can take longer than 60 sec on Travis
def test_make_forward_solution_sphere(tmp_path, fname_src_small):
    """Test making a forward solution with a sphere model."""
    out_name = tmp_path / 'tmp-fwd.fif'
    run_subprocess(['mne_forward_solution', '--meg', '--eeg',
                    '--meas', fname_raw, '--src', fname_src_small,
                    '--mri', fname_trans, '--fwd', out_name])
    fwd = read_forward_solution(out_name)
    sphere = make_sphere_model(verbose=True)
    src = read_source_spaces(fname_src_small)
    fwd_py = make_forward_solution(fname_raw, fname_trans, src, sphere,
                                   meg=True, eeg=True, verbose=True)
    _compare_forwards(fwd, fwd_py, 366, 108,
                      meg_rtol=5e-1, meg_atol=1e-6,
                      eeg_rtol=5e-1, eeg_atol=5e-1)
    # Since the above is pretty lax, let's check a different way
    for meg, eeg in zip([True, False], [False, True]):
        fwd_ = pick_types_forward(fwd, meg=meg, eeg=eeg)
        fwd_py_ = pick_types_forward(fwd, meg=meg, eeg=eeg)
        assert_allclose(np.corrcoef(fwd_['sol']['data'].ravel(),
                                    fwd_py_['sol']['data'].ravel())[0, 1],
                        1.0, rtol=1e-3)
    # Number of layers in the sphere model doesn't matter for MEG
    # (as long as no sources are omitted due to distance)
    assert len(sphere['layers']) == 4
    fwd = make_forward_solution(fname_raw, fname_trans, src, sphere,
                                meg=True, eeg=False)
    sphere_1 = make_sphere_model(head_radius=None)
    assert len(sphere_1['layers']) == 0
    assert_array_equal(sphere['r0'], sphere_1['r0'])
    fwd_1 = make_forward_solution(fname_raw, fname_trans, src, sphere,
                                  meg=True, eeg=False)
    _compare_forwards(fwd, fwd_1, 306, 108, meg_rtol=1e-12, meg_atol=1e-12)
    # Homogeneous model
    sphere = make_sphere_model(head_radius=None)
    with pytest.raises(RuntimeError, match='zero shells.*EEG'):
        make_forward_solution(fname_raw, fname_trans, src, sphere)


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_nibabel()
def test_forward_mixed_source_space(tmp_path):
    """Test making the forward solution for a mixed source space."""
    # get the surface source space
    rng = np.random.RandomState(0)
    surf = read_source_spaces(fname_src)

    # setup two volume source spaces
    label_names = get_volume_labels_from_aseg(fname_aseg)
    vol_labels = rng.choice(label_names, 2)
    with pytest.warns(RuntimeWarning, match='Found no usable.*CC_Mid_Ant.*'):
        vol1 = setup_volume_source_space('sample', pos=20., mri=fname_aseg,
                                         volume_label=vol_labels[0],
                                         add_interpolator=False)
    vol2 = setup_volume_source_space('sample', pos=20., mri=fname_aseg,
                                     volume_label=vol_labels[1],
                                     add_interpolator=False)

    # merge surfaces and volume
    src = surf + vol1 + vol2

    # calculate forward solution
    fwd = make_forward_solution(fname_raw, fname_trans, src, fname_bem)
    assert (repr(fwd))

    # extract source spaces
    src_from_fwd = fwd['src']

    # get the coordinate frame of each source space
    coord_frames = np.array([s['coord_frame'] for s in src_from_fwd])

    # assert that all source spaces are in head coordinates
    assert ((coord_frames == FIFF.FIFFV_COORD_HEAD).all())

    # run tests for SourceSpaces.export_volume
    fname_img = tmp_path / 'temp-image.mgz'

    # head coordinates and mri_resolution, but trans file
    with pytest.raises(ValueError, match='trans containing mri to head'):
        src_from_fwd.export_volume(fname_img, mri_resolution=True, trans=None)

    # head coordinates and mri_resolution, but wrong trans file
    vox_mri_t = vol1[0]['vox_mri_t']
    with pytest.raises(ValueError, match='head<->mri, got mri_voxel->mri'):
        src_from_fwd.export_volume(fname_img, mri_resolution=True,
                                   trans=vox_mri_t)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_make_forward_dipole(tmp_path):
    """Test forward-projecting dipoles."""
    rng = np.random.RandomState(0)

    evoked = read_evokeds(fname_evo)[0]
    cov = read_cov(fname_cov)
    cov['projs'] = []  # avoid proj warning
    dip_c = read_dipole(fname_dip)

    # Only use magnetometers for speed!
    picks = pick_types(evoked.info, meg='mag', eeg=False)[::8]
    evoked.pick_channels([evoked.ch_names[p] for p in picks])
    evoked.info.normalize_proj()
    info = evoked.info

    # Make new Dipole object with n_test_dipoles picked from the dipoles
    # in the test dataset.
    n_test_dipoles = 3  # minimum 3 needed to get uneven sampling in time
    dipsel = np.sort(rng.permutation(np.arange(len(dip_c)))[:n_test_dipoles])
    dip_test = Dipole(times=dip_c.times[dipsel],
                      pos=dip_c.pos[dipsel],
                      amplitude=dip_c.amplitude[dipsel],
                      ori=dip_c.ori[dipsel],
                      gof=dip_c.gof[dipsel])

    sphere = make_sphere_model(head_radius=0.1)

    # Warning emitted due to uneven sampling in time
    with pytest.warns(RuntimeWarning, match='unevenly spaced'):
        fwd, stc = make_forward_dipole(dip_test, sphere, info,
                                       trans=fname_trans)

    # stc is list of VolSourceEstimate's
    assert isinstance(stc, list)
    for n_dip in range(n_test_dipoles):
        assert isinstance(stc[n_dip], VolSourceEstimate)

    # Now simulate evoked responses for each of the test dipoles,
    # and fit dipoles to them (sphere model, MEG and EEG)
    times, pos, amplitude, ori, gof = [], [], [], [], []
    nave = 400  # add a tiny amount of noise to the simulated evokeds
    for s in stc:
        evo_test = simulate_evoked(fwd, s, info, cov,
                                   nave=nave, random_state=rng)
        # evo_test.add_proj(make_eeg_average_ref_proj(evo_test.info))
        dfit, resid = fit_dipole(evo_test, cov, sphere, None)
        times += dfit.times.tolist()
        pos += dfit.pos.tolist()
        amplitude += dfit.amplitude.tolist()
        ori += dfit.ori.tolist()
        gof += dfit.gof.tolist()

    # Create a new Dipole object with the dipole fits
    dip_fit = Dipole(times, pos, amplitude, ori, gof)

    # check that true (test) dipoles and fits are "close"
    # cf. mne/tests/test_dipole.py
    diff = dip_test.pos - dip_fit.pos
    corr = np.corrcoef(dip_test.pos.ravel(), dip_fit.pos.ravel())[0, 1]
    dist = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    gc_dist = 180 / np.pi * \
        np.mean(np.arccos(np.sum(dip_test.ori * dip_fit.ori, axis=1)))
    amp_err = np.sqrt(np.mean((dip_test.amplitude - dip_fit.amplitude) ** 2))

    # Make sure each coordinate is close to reference
    # NB tolerance should be set relative to snr of simulated evoked!
    assert_allclose(dip_fit.pos, dip_test.pos, rtol=0, atol=1e-2,
                    err_msg='position mismatch')
    assert dist < 1e-2  # within 1 cm
    assert corr > 0.985
    assert gc_dist < 20  # less than 20 degrees
    assert amp_err < 10e-9  # within 10 nAm

    # Make sure rejection works with BEM: one dipole at z=1m
    # NB _make_forward.py:_prepare_for_forward will raise a RuntimeError
    # if no points are left after min_dist exclusions, hence 2 dips here!
    dip_outside = Dipole(times=[0., 0.001],
                         pos=[[0., 0., 1.0], [0., 0., 0.040]],
                         amplitude=[100e-9, 100e-9],
                         ori=[[1., 0., 0.], [1., 0., 0.]], gof=1)
    with pytest.raises(ValueError, match='outside the inner skull'):
        make_forward_dipole(dip_outside, fname_bem, info, fname_trans)
    # if we get this far, can safely assume the code works with BEMs too
    # -> use sphere again below for speed

    # Now make an evenly sampled set of dipoles, some simultaneous,
    # should return a VolSourceEstimate regardless
    times = [0., 0., 0., 0.001, 0.001, 0.002]
    pos = np.random.rand(6, 3) * 0.020 + \
        np.array([0., 0., 0.040])[np.newaxis, :]
    amplitude = np.random.rand(6) * 100e-9
    ori = np.eye(6, 3) + np.eye(6, 3, -3)
    gof = np.arange(len(times)) / len(times)  # arbitrary

    dip_even_samp = Dipole(times, pos, amplitude, ori, gof)

    # I/O round-trip
    fname = str(tmp_path / 'test-fwd.fif')
    with pytest.warns(RuntimeWarning, match='free orientation'):
        write_forward_solution(fname, fwd)
    fwd_read = convert_forward_solution(
        read_forward_solution(fname), force_fixed=True)
    assert_forward_allclose(fwd, fwd_read, rtol=1e-6)

    fwd, stc = make_forward_dipole(dip_even_samp, sphere, info,
                                   trans=fname_trans)
    assert isinstance(stc, VolSourceEstimate)
    assert_allclose(stc.times, np.arange(0., 0.003, 0.001))

    # Test passing a list of Dipoles instead of a single Dipole object
    fwd2, stc2 = make_forward_dipole([dip_even_samp[0], dip_even_samp[1:]],
                                     sphere, info, trans=fname_trans)
    assert_array_equal(fwd['sol']['data'], fwd2['sol']['data'])
    assert_array_equal(stc.data, stc2.data)


@testing.requires_testing_data
def test_make_forward_no_meg(tmp_path):
    """Test that we can make and I/O forward solution with no MEG channels."""
    pos = dict(rr=[[0.05, 0, 0]], nn=[[0, 0, 1.]])
    src = setup_volume_source_space(pos=pos)
    bem = make_sphere_model()
    trans = None
    montage = make_standard_montage('standard_1020')
    info = create_info(['Cz'], 1000., 'eeg').set_montage(montage)
    fwd = make_forward_solution(info, trans, src, bem)
    fname = tmp_path / 'test-fwd.fif'
    write_forward_solution(fname, fwd)
    fwd_read = read_forward_solution(fname)
    assert_allclose(fwd['sol']['data'], fwd_read['sol']['data'])


def test_use_coil_def(tmp_path):
    """Test use_coil_def."""
    info = create_info(1, 1000., 'mag')
    info['chs'][0]['coil_type'] = 9999
    info['chs'][0]['loc'][:] = [0, 0, 0.02, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    sphere = make_sphere_model((0., 0., 0.), 0.01)
    src = setup_volume_source_space(pos=5, sphere=sphere)
    trans = Transform('head', 'mri', None)
    with pytest.raises(RuntimeError, match='coil definition not found'):
        make_forward_solution(info, trans, src, sphere)
    coil_fname = tmp_path / 'coil_def.dat'
    with open(coil_fname, 'w') as fid:
        fid.write("""# custom cube coil def
1   9999    2   8  3e-03  0.000e+00     "Test"
  0.1250 -0.750e-03 -0.750e-03 -0.750e-03  0.000  0.000""")
    with pytest.raises(RuntimeError, match='Could not interpret'):
        with use_coil_def(coil_fname):
            make_forward_solution(info, trans, src, sphere)
    with open(coil_fname, 'w') as fid:
        fid.write("""# custom cube coil def
1   9999    2   8  3e-03  0.000e+00     "Test"
  0.1250 -0.750e-03 -0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250 -0.750e-03  0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03 -0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03  0.750e-03 -0.750e-03  0.000  0.000  1.000
  0.1250 -0.750e-03 -0.750e-03  0.750e-03  0.000  0.000  1.000
  0.1250 -0.750e-03  0.750e-03  0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03 -0.750e-03  0.750e-03  0.000  0.000  1.000
  0.1250  0.750e-03  0.750e-03  0.750e-03  0.000  0.000  1.000""")
    with use_coil_def(coil_fname):
        make_forward_solution(info, trans, src, sphere)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_sensors_inside_bem():
    """Test that sensors inside the BEM are problematic."""
    rr = _get_ico_surface(1)['rr']
    rr /= np.linalg.norm(rr, axis=1, keepdims=True)
    rr *= 0.1
    assert len(rr) == 42
    info = create_info(len(rr), 1000., 'mag')
    info['dev_head_t'] = Transform('meg', 'head', np.eye(4))
    for ii, ch in enumerate(info['chs']):
        ch['loc'][:] = np.concatenate((rr[ii], np.eye(3).ravel()))
    trans = Transform('head', 'mri', np.eye(4))
    trans['trans'][2, 3] = 0.03
    sphere_noshell = make_sphere_model((0., 0., 0.), None)
    sphere = make_sphere_model((0., 0., 0.), 1.01)
    with pytest.raises(RuntimeError, match='.* 15 MEG.*inside the scalp.*'):
        make_forward_solution(info, trans, fname_src, fname_bem)
    make_forward_solution(info, trans, fname_src, fname_bem_meg)  # okay
    make_forward_solution(info, trans, fname_src, sphere_noshell)  # okay
    with pytest.raises(RuntimeError, match='.* 42 MEG.*outermost sphere sh.*'):
        make_forward_solution(info, trans, fname_src, sphere)
    sphere = make_sphere_model((0., 0., 2.0), 1.01)  # weird, but okay
    make_forward_solution(info, trans, fname_src, sphere)
    for ch in info['chs']:
        ch['loc'][:3] *= 0.1
    with pytest.raises(RuntimeError, match='.* 42 MEG.*the inner skull.*'):
        make_forward_solution(info, trans, fname_src, fname_bem_meg)
