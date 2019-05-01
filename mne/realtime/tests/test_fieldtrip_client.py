# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import contextlib
import os
import os.path as op
import socket
import time

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

from mne import Epochs, find_events, pick_types
from mne.io import read_raw_fif
from mne.utils import requires_neuromag2ft, running_subprocess
from mne.utils import run_tests_if_main
from mne.realtime import FieldTripClient, RtEpochs

from mne.realtime.tests.test_mockclient import _call_base_epochs_public_api

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.realpath(op.join(base_dir, 'test_raw.fif'))


@pytest.fixture
def free_tcp_port():
    """Get a free TCP port."""
    with contextlib.closing(socket.socket()) as free_socket:
        free_socket.bind(('127.0.0.1', 0))
        return free_socket.getsockname()[1]


@pytest.mark.slowtest
@requires_neuromag2ft
def test_fieldtrip_rtepochs(free_tcp_port, tmpdir):
    """Test FieldTrip RtEpochs."""
    raw_tmax = 7
    raw = read_raw_fif(raw_fname, preload=True)
    raw.crop(tmin=0, tmax=raw_tmax)
    events_offline = find_events(raw, stim_channel='STI 014')
    event_id = list(np.unique(events_offline[:, 2]))
    tmin, tmax = -0.2, 0.5
    epochs_offline = Epochs(raw, events_offline, event_id=event_id,
                            tmin=tmin, tmax=tmax)
    epochs_offline.drop_bad()
    isi_max = (np.max(np.diff(epochs_offline.events[:, 0])) /
               raw.info['sfreq']) + 1.0

    neuromag2ft_fname = op.realpath(op.join(os.environ['NEUROMAG2FT_ROOT'],
                                            'neuromag2ft'))
    # Works with neuromag2ft-3.0.2
    cmd = (neuromag2ft_fname, '--file', raw_fname, '--speed', '8.0',
           '--bufport', str(free_tcp_port))

    with running_subprocess(cmd, after='terminate', verbose=False):
        data_rt = None
        events_ids_rt = None
        with pytest.warns(RuntimeWarning, match='Trying to guess it'):
            with FieldTripClient(host='localhost', port=free_tcp_port,
                                 tmax=raw_tmax, wait_max=2) as rt_client:
                # get measurement info guessed by MNE-Python
                raw_info = rt_client.get_measurement_info()
                assert ([ch['ch_name'] for ch in raw_info['chs']] ==
                        [ch['ch_name'] for ch in raw.info['chs']])

                # create the real-time epochs object
                epochs_rt = RtEpochs(rt_client, event_id, tmin, tmax,
                                     stim_channel='STI 014', isi_max=isi_max)
                epochs_rt.start()

                time.sleep(0.5)
                for ev_num, ev in enumerate(epochs_rt.iter_evoked()):
                    if ev_num == 0:
                        data_rt = ev.data[None, :, :]
                        events_ids_rt = int(
                            ev.comment)  # comment attribute contains event_id
                    else:
                        data_rt = np.concatenate(
                            (data_rt, ev.data[None, :, :]), axis=0)
                        events_ids_rt = np.append(events_ids_rt,
                                                  int(ev.comment))

                _call_base_epochs_public_api(epochs_rt, tmpdir)
                epochs_rt.stop(stop_receive_thread=True)

        assert_array_equal(events_ids_rt, epochs_rt.events[:, 2])
        assert_array_equal(data_rt, epochs_rt.get_data())
        assert len(epochs_rt) == len(epochs_offline)
        assert_array_equal(events_ids_rt, epochs_offline.events[:, 2])
        assert_allclose(epochs_rt.get_data(), epochs_offline.get_data(),
                        rtol=1.e-5, atol=1.e-8)  # defaults of np.isclose


@requires_neuromag2ft
def test_fieldtrip_client(free_tcp_port):
    """Test fieldtrip_client."""
    neuromag2ft_fname = op.realpath(op.join(os.environ['NEUROMAG2FT_ROOT'],
                                            'neuromag2ft'))
    # Works with neuromag2ft-3.0.2
    cmd = (neuromag2ft_fname, '--file', raw_fname, '--speed', '8.0',
           '--bufport', str(free_tcp_port))

    time.sleep(0.5)

    with running_subprocess(cmd, after='terminate', verbose=False):
        # Start the FieldTrip buffer
        with pytest.warns(RuntimeWarning):
            with FieldTripClient(host='localhost', port=free_tcp_port,
                                 tmax=5, wait_max=2) as rt_client:
                tmin_samp1 = rt_client.tmin_samp

        time.sleep(1)  # Pause measurement

        # Start the FieldTrip buffer again
        with pytest.warns(RuntimeWarning):
            with FieldTripClient(host='localhost', port=free_tcp_port,
                                 tmax=5, wait_max=2) as rt_client:
                raw_info = rt_client.get_measurement_info()

                tmin_samp2 = rt_client.tmin_samp
                picks = pick_types(raw_info, meg='grad', eeg=False,
                                   stim=False, eog=False)
                epoch = rt_client.get_data_as_epoch(n_samples=5, picks=picks)
                n_channels, n_samples = epoch.get_data().shape[1:]

                epoch2 = rt_client.get_data_as_epoch(n_samples=5, picks=picks)
                n_channels2, n_samples2 = epoch2.get_data().shape[1:]

                # case of picks=None
                epoch = rt_client.get_data_as_epoch(n_samples=5)

        assert tmin_samp2 > tmin_samp1
        assert n_samples == 5
        assert n_samples2 == 5
        assert n_channels == len(picks)
        assert n_channels2 == len(picks)


run_tests_if_main()
