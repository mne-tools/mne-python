# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_true
import warnings

from mne.chpi import (_calculate_chpi_positions,
                      head_pos_to_trans_rot_t)
from mne.transforms import rot_to_quat, _angle_between_quats
from mne.utils import run_tests_if_main, slow_test
from mne.datasets import testing

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
test_fif_fname = op.join(base_dir, 'test_raw.fif')

warnings.simplefilter('always')


def _assert_quats(actual, desired, dist_tol=0.003, angle_tol=5.):
    """Compare estimated cHPI positions."""
    from scipy.interpolate import interp1d
    trans_est, rot_est, t_est = head_pos_to_trans_rot_t(actual)
    trans, rot, t = head_pos_to_trans_rot_t(desired)
    quats_est = rot_to_quat(rot_est)

    use_mask = (t >= t_est[0]) & (t <= t_est[-1])
    t = t[use_mask]
    trans = trans[use_mask]
    quats = rot_to_quat(rot)
    quats = quats[use_mask]

    # double-check our angle function
    for q in (quats, quats_est):
        angles = _angle_between_quats(q, q)
        assert_allclose(angles, 0., atol=1e-5)

    # limit translation difference between MF and our estimation
    trans_est_interp = interp1d(t_est, trans_est, axis=0)(t)
    worst = np.sqrt(np.sum((trans - trans_est_interp) ** 2, axis=1)).max()
    assert_true(worst <= dist_tol, '%0.3f > %0.3f mm'
                % (1000 * worst, 1000 * dist_tol))

    # limit rotation difference between MF and our estimation
    # (note that the interpolation will make this slightly worse)
    quats_est_interp = interp1d(t_est, quats_est, axis=0)(t)
    worst = 180 * _angle_between_quats(quats_est_interp, quats).max() / np.pi
    assert_true(worst <= angle_tol, '%0.3f > %0.3f deg' % (worst, angle_tol,))


@slow_test
@testing.requires_testing_data
def test_simulate_calculate_chpi_positions():
    """Test calculation of cHPI positions with simulated data."""

    raw_fname = op.join(base_dir, 'test_raw.fif')

    from mne.io import read_info, RawArray
    from mne.forward import make_forward_dipole
    from mne.simulation import simulate_raw
    from mne import pick_types, Dipole, make_sphere_model, read_cov

    # Read info dict from raw FIF file
    info = read_info(raw_fname)
    pick = pick_types(info, meg=True, stim=True, eeg=False, exclude=[])

    # Tune the info structure
    chpi_channel = u'STI201'
    ncoil = len(info['hpi_results'][0]['order'])

    coil_freq = np.linspace(info['lowpass'] - 100.0,
                            info['lowpass'] - 10.0, ncoil)

    hpi_subsystem = {'event_channel': chpi_channel,
                     'hpi_coils': [{'event_bits': np.array([256, 0, 256, 256],
                                                           dtype=np.int32)},
                                   {'event_bits': np.array([512, 0, 512, 512],
                                                           dtype=np.int32)},
                                   {'event_bits':
                                       np.array([1024, 0, 1024, 1024],
                                                dtype=np.int32)},
                                   {'event_bits':
                                       np.array([2048, 0, 2048, 2048],
                                                dtype=np.int32)}],
                     'ncoil': ncoil}

    info['hpi_subsystem'] = hpi_subsystem
    for l, freq in enumerate(coil_freq):
            info['hpi_meas'][0]['hpi_coils'][l]['coil_freq'] = freq

    info['line_freq'] = 50.0
    info['nchan'] = len(pick)
    info['bads'] = []
    info['chs'] = [ch for ci, ch in enumerate(info['chs']) if ci in pick]
    info['ch_names'] = [ch for ci, ch in enumerate(info['ch_names'])
                        if ci in pick]
    info['ch_names'][313] = chpi_channel
    info['chs'][313]['ch_name'] = chpi_channel
    info['projs'] = []

    info_trans = info['dev_head_t']['trans'].copy()

    dev_head_pos_ini = np.concatenate([rot_to_quat(info_trans[:3, :3]),
                                      info_trans[:3, 3]])
    ez = np.array([0, 0, 1])  # Unit vector in z-direction of head coordinates

    # Define some constants
    T = 30  # Time / s

    # Quotient of head position sampling frequency
    # and raw sampling frequency
    head_pos_sfreq_quotient = 0.001

    # Round number of head positions to the next integer
    S = int(T / (info['sfreq'] * head_pos_sfreq_quotient) + 0.5)
    dz = 0.0001  # Shift in z-direction is 0.1mm for each step

    dev_head_pos = np.zeros((S, 10))
    dev_head_pos[:, 0] = np.arange(S) * info['sfreq'] * head_pos_sfreq_quotient
    dev_head_pos[:, 1:4] = dev_head_pos_ini[:3]
    dev_head_pos[:, 4:7] = dev_head_pos_ini[3:] + \
        np.outer(np.arange(S) * dz, ez)
    dev_head_pos[:, 7] = 1.0

    # cm/s
    dev_head_pos[:, 9] = 100 * dz / (info['sfreq'] * head_pos_sfreq_quotient)

    # Round number of samples to the next integer
    raw_data = np.zeros((len(pick), int(T * info['sfreq'] + 0.5)))
    raw = RawArray(raw_data, info)

    dip = Dipole(np.array([0.0, 0.1, 0.2]),
                 np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                 np.array([1e-9, 1e-9, 1e-9]),
                 np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                 np.array([1.0, 1.0, 1.0]), 'dip')
    sphere = make_sphere_model('auto', 'auto', info=info,
                               relative_radii=(1.0, 0.9), sigmas=(0.33, 0.3))
    fwd, stc = make_forward_dipole(dip, sphere, info)
    stc.resample(info['sfreq'])

    cov = read_cov(op.join(base_dir, 'test_erm-cov.fif'))

    fwd['src'][0]['pinfo'] = []
    fwd['src'][0]['nuse_tri'] = []
    fwd['src'][0]['use_tris'] = []
    fwd['src'][0]['patch_inds'] = []
    raw = simulate_raw(raw, stc, None, fwd['src'], sphere, cov=cov,
                       blink=False, ecg=False, chpi=True,
                       head_pos=dev_head_pos, mindist=1.0, interp='zero',
                       iir_filter=[0.2, -0.2, 0.04], n_jobs=-1,
                       random_state=1, verbose=None)

    quats = _calculate_chpi_positions(raw, t_step_min=raw.info['sfreq'] *
                                      head_pos_sfreq_quotient,
                                      t_step_max=raw.info['sfreq'] *
                                      head_pos_sfreq_quotient,
                                      t_window=1.0)

    _assert_quats(quats, dev_head_pos, 0.002, 3.0)


run_tests_if_main()
