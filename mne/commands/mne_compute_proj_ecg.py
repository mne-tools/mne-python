#!/usr/bin/env python
r"""Compute SSP/PCA projections for ECG artifacts.

Examples
--------
.. code-block:: console

    $ mne compute_proj_ecg -i sample_audvis_raw.fif -c "MEG 1531" -a \
                           --l-freq 1 --h-freq 100 \
                           --rej-grad 3000 --rej-mag 4000 --rej-eeg 100

"""
# Authors : Alexandre Gramfort, Ph.D.
#           Martin Luessi, Ph.D.

import os
import sys
import mne


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-i", "--in", dest="raw_in",
                      help="Input raw FIF file", metavar="FILE")
    parser.add_option("--tmin", dest="tmin", type="float",
                      help="Time before event in seconds",
                      default=-0.2)
    parser.add_option("--tmax", dest="tmax", type="float",
                      help="Time after event in seconds",
                      default=0.4)
    parser.add_option("-g", "--n-grad", dest="n_grad", type="int",
                      help="Number of SSP vectors for gradiometers",
                      default=2)
    parser.add_option("-m", "--n-mag", dest="n_mag", type="int",
                      help="Number of SSP vectors for magnetometers",
                      default=2)
    parser.add_option("-e", "--n-eeg", dest="n_eeg", type="int",
                      help="Number of SSP vectors for EEG",
                      default=2)
    parser.add_option("--l-freq", dest="l_freq", type="float",
                      help="Filter low cut-off frequency in Hz",
                      default=1)
    parser.add_option("--h-freq", dest="h_freq", type="float",
                      help="Filter high cut-off frequency in Hz",
                      default=100)
    parser.add_option("--ecg-l-freq", dest="ecg_l_freq", type="float",
                      help="Filter low cut-off frequency in Hz used "
                      "for ECG event detection",
                      default=5)
    parser.add_option("--ecg-h-freq", dest="ecg_h_freq", type="float",
                      help="Filter high cut-off frequency in Hz used "
                      "for ECG event detection",
                      default=35)
    parser.add_option("-p", "--preload", dest="preload",
                      help="Temporary file used during computation "
                      "(to save memory)",
                      default=True)
    parser.add_option("-a", "--average", dest="average", action="store_true",
                      help="Compute SSP after averaging",
                      default=False)
    parser.add_option("--proj", dest="proj",
                      help="Use SSP projections from a fif file.",
                      default=None)
    parser.add_option("--filtersize", dest="filter_length", type="int",
                      help="Number of taps to use for filtering",
                      default=2048)
    parser.add_option("-j", "--n-jobs", dest="n_jobs", type="int",
                      help="Number of jobs to run in parallel",
                      default=1)
    parser.add_option("-c", "--channel", dest="ch_name",
                      help="Channel to use for ECG detection "
                      "(Required if no ECG found)",
                      default=None)
    parser.add_option("--rej-grad", dest="rej_grad", type="float",
                      help="Gradiometers rejection parameter "
                      "in fT/cm (peak to peak amplitude)",
                      default=2000)
    parser.add_option("--rej-mag", dest="rej_mag", type="float",
                      help="Magnetometers rejection parameter "
                      "in fT (peak to peak amplitude)",
                      default=3000)
    parser.add_option("--rej-eeg", dest="rej_eeg", type="float",
                      help="EEG rejection parameter in µV "
                      "(peak to peak amplitude)",
                      default=50)
    parser.add_option("--rej-eog", dest="rej_eog", type="float",
                      help="EOG rejection parameter in µV "
                      "(peak to peak amplitude)",
                      default=250)
    parser.add_option("--avg-ref", dest="avg_ref", action="store_true",
                      help="Add EEG average reference proj",
                      default=False)
    parser.add_option("--no-proj", dest="no_proj", action="store_true",
                      help="Exclude the SSP projectors currently "
                      "in the fiff file",
                      default=False)
    parser.add_option("--bad", dest="bad_fname",
                      help="Text file containing bad channels list "
                      "(one per line)",
                      default=None)
    parser.add_option("--event-id", dest="event_id", type="int",
                      help="ID to use for events",
                      default=999)
    parser.add_option("--event-raw", dest="raw_event_fname",
                      help="raw file to use for event detection",
                      default=None)
    parser.add_option("--tstart", dest="tstart", type="float",
                      help="Start artifact detection after tstart seconds",
                      default=0.)
    parser.add_option("--qrsthr", dest="qrs_threshold", type="string",
                      help="QRS detection threshold. Between 0 and 1. Can "
                      "also be 'auto' for automatic selection",
                      default='auto')

    options, args = parser.parse_args()

    raw_in = options.raw_in

    if raw_in is None:
        parser.print_help()
        sys.exit(1)

    tmin = options.tmin
    tmax = options.tmax
    n_grad = options.n_grad
    n_mag = options.n_mag
    n_eeg = options.n_eeg
    l_freq = options.l_freq
    h_freq = options.h_freq
    ecg_l_freq = options.ecg_l_freq
    ecg_h_freq = options.ecg_h_freq
    average = options.average
    preload = options.preload
    filter_length = options.filter_length
    n_jobs = options.n_jobs
    ch_name = options.ch_name
    reject = dict(grad=1e-13 * float(options.rej_grad),
                  mag=1e-15 * float(options.rej_mag),
                  eeg=1e-6 * float(options.rej_eeg),
                  eog=1e-6 * float(options.rej_eog))
    avg_ref = options.avg_ref
    no_proj = options.no_proj
    bad_fname = options.bad_fname
    event_id = options.event_id
    proj_fname = options.proj
    raw_event_fname = options.raw_event_fname
    tstart = options.tstart
    qrs_threshold = options.qrs_threshold
    if qrs_threshold != 'auto':
        try:
            qrs_threshold = float(qrs_threshold)
        except ValueError:
            raise ValueError('qrsthr must be "auto" or a float')

    if bad_fname is not None:
        with open(bad_fname, 'r') as fid:
            bads = [w.rstrip() for w in fid.readlines()]
        print('Bad channels read : %s' % bads)
    else:
        bads = []

    if raw_in.endswith('_raw.fif') or raw_in.endswith('-raw.fif'):
        prefix = raw_in[:-8]
    else:
        prefix = raw_in[:-4]

    ecg_event_fname = prefix + '_ecg-eve.fif'

    if average:
        ecg_proj_fname = prefix + '_ecg_avg-proj.fif'
    else:
        ecg_proj_fname = prefix + '_ecg-proj.fif'

    raw = mne.io.read_raw_fif(raw_in, preload=preload)

    if raw_event_fname is not None:
        raw_event = mne.io.read_raw_fif(raw_event_fname)
    else:
        raw_event = raw

    flat = None
    projs, events = mne.preprocessing.compute_proj_ecg(
        raw, raw_event, tmin, tmax, n_grad, n_mag, n_eeg, l_freq, h_freq,
        average, filter_length, n_jobs, ch_name, reject, flat, bads, avg_ref,
        no_proj, event_id, ecg_l_freq, ecg_h_freq, tstart, qrs_threshold,
        copy=False)

    raw.close()

    if raw_event_fname is not None:
        raw_event.close()

    if proj_fname is not None:
        print('Including SSP projections from : %s' % proj_fname)
        # append the ecg projs, so they are last in the list
        projs = mne.read_proj(proj_fname) + projs

    if isinstance(preload, str) and os.path.exists(preload):
        os.remove(preload)

    print("Writing ECG projections in %s" % ecg_proj_fname)
    mne.write_proj(ecg_proj_fname, projs)

    print("Writing ECG events in %s" % ecg_event_fname)
    mne.write_events(ecg_event_fname, events)


mne.utils.run_command_if_main()
