#!/usr/bin/env python
"""Compute SSP/PCA projections for EOG artifacts

You can do for example:

$mne_compute_proj_eog.py -i sample_audvis_raw.fif --l-freq 1 --h-freq 100 --rej-grad 3000 --rej-mag 4000 --rej-eeg 100
"""

# Authors : Alexandre Gramfort, Ph.D.
#           Martin Luessi, Ph.D.

import os
import sys
import mne


if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--in", dest="raw_in",
                    help="Input raw FIF file", metavar="FILE")
    parser.add_option("--tmin", dest="tmin",
                    help="Time before event in seconds",
                    default=-0.15)
    parser.add_option("--tmax", dest="tmax",
                    help="Time after event in seconds",
                    default=0.15)
    parser.add_option("-g", "--n-grad", dest="n_grad",
                    help="Number of SSP vectors for gradiometers",
                    default=2)
    parser.add_option("-m", "--n-mag", dest="n_mag",
                    help="Number of SSP vectors for magnetometers",
                    default=2)
    parser.add_option("-e", "--n-eeg", dest="n_eeg",
                    help="Number of SSP vectors for EEG",
                    default=2)
    parser.add_option("--l-freq", dest="l_freq",
                    help="Filter low cut-off frequency in Hz",
                    default=5)
    parser.add_option("--h-freq", dest="h_freq",
                    help="Filter high cut-off frequency in Hz",
                    default=35)
    parser.add_option("-p", "--preload", dest="preload",
                    help="Temporary file used during computation (to save memory)",
                    default=True)
    parser.add_option("-a", "--average", dest="average", action="store_true",
                    help="Compute SSP after averaging",
                    default=False)
    parser.add_option("--filtersize", dest="filter_length",
                    help="Number of taps to use for filtering",
                    default=2048)
    parser.add_option("-j", "--n-jobs", dest="n_jobs",
                    help="Number of jobs to run in parallel",
                    default=1)
    parser.add_option("--rej-grad", dest="rej_grad",
                    help="Gradiometers rejection parameter in fT/cm (peak to peak amplitude)",
                    default=2000)
    parser.add_option("--rej-mag", dest="rej_mag",
                    help="Magnetometers rejection parameter in fT (peak to peak amplitude)",
                    default=3000)
    parser.add_option("--rej-eeg", dest="rej_eeg",
                    help="EEG rejection parameter in uV (peak to peak amplitude)",
                    default=50)
    parser.add_option("--rej-eog", dest="rej_eog",
                    help="EOG rejection parameter in uV (peak to peak amplitude)",
                    default=1e9)
    parser.add_option("--avg-ref", dest="avg_ref", action="store_true",
                    help="Add EEG average reference proj",
                    default=False)
    parser.add_option("--no-proj", dest="no_proj", action="store_true",
                    help="Exclude the SSP projectors currently in the fiff file",
                    default=False)
    parser.add_option("--bad", dest="bad_fname",
                    help="Text file containing bad channels list (one per line)",
                    default=None)
    parser.add_option("--event-id", dest="event_id", type="int",
                    help="ID to use for events", default=999)

    options, args = parser.parse_args()

    raw_in = options.raw_in

    if raw_in is None:
        parser.print_help()
        sys.exit(-1)

    tmin = options.tmin
    tmax = options.tmax
    n_grad = options.n_grad
    n_mag = options.n_mag
    n_eeg = options.n_eeg
    l_freq = options.l_freq
    h_freq = options.h_freq
    average = options.average
    preload = options.preload
    filter_length = options.filter_length
    n_jobs = options.n_jobs
    reject = dict(grad=1e-13 * float(options.rej_grad),
                  mag=1e-15 * float(options.rej_mag),
                  eeg=1e-6 * float(options.rej_eeg),
                  eog=1e-6 * float(options.rej_eog))
    avg_ref = options.avg_ref
    no_proj = options.no_proj
    bad_fname = options.bad_fname
    event_id = options.event_id

    if bad_fname is not None:
        bads = [w.rstrip().split()[0] for w in open(bad_fname).readlines()]
        print 'Bad channels read : %s' % bads
    else:
        bads = []

    if raw_in.endswith('_raw.fif') or raw_in.endswith('-raw.fif'):
        prefix = raw_in[:-8]
    else:
        prefix = raw_in[:-4]

    eog_event_fname = prefix + '_eog-eve.fif'

    if average:
        eog_proj_fname = prefix + '_eog_avg_proj.fif'
    else:
        eog_proj_fname = prefix + '_eog_proj.fif'

    raw = mne.fiff.Raw(raw_in, preload=preload)

    projs, events = mne.preprocessing.compute_proj_eog(raw, tmin, tmax,
                            n_grad, n_mag, n_eeg, l_freq, h_freq, average,
                            filter_length, n_jobs, reject, bads,
                            avg_ref, no_proj, event_id)

    raw.close()

    if isinstance(preload, str) and os.path.exists(preload):
        os.remove(preload)

    print "Writing EOG projections in %s" % eog_proj_fname
    mne.write_proj(eog_proj_fname, projs)

    print "Writing EOG events in %s" % eog_event_fname
    mne.write_events(eog_event_fname, events)
