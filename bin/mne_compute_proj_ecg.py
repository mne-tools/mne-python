#!/usr/bin/env python
"""Compute SSP/PCA projections for ECG artifacts

You can do for example:

$mne_compute_proj_ecg.py -i sample_audvis_raw.fif -c "MEG 1531" --l-freq 1 --h-freq 100 --rej-grad 3000 --rej-mag 4000 --rej-eeg 100
"""

# Authors : Alexandre Gramfort, Ph.D.

import sys
import os
import mne


def compute_proj_ecg(in_fif_fname, tmin, tmax, n_grad, n_mag, n_eeg, l_freq,
                     h_freq, average, preload, filter_length, n_jobs, ch_name,
                     reject, avg_ref, bads):
    """Compute SSP/PCA projections for ECG artifacts

    Parameters
    ----------
    in_fif_fname: string
        Raw fif File
    XXX
    """
    # Reading fif File
    raw = mne.fiff.Raw(in_fif_fname, preload=preload)

    if in_fif_fname.endswith('_raw.fif') or in_fif_fname.endswith('-raw.fif'):
        prefix = in_fif_fname[:-8]
    else:
        prefix = in_fif_fname[:-4]

    ecg_event_fname = prefix + '_ecg-eve.fif'

    if average:
        ecg_proj_fname = prefix + '_ecg_avg_proj.fif'
    else:
        ecg_proj_fname = prefix + '_ecg_proj.fif'

    print 'Running ECG SSP computation'

    ecg_events, _, _ = mne.artifacts.find_ecg_events(raw, ch_name=ch_name)
    print "Writing ECG events in %s" % ecg_event_fname
    mne.write_events(ecg_event_fname, ecg_events)

    if avg_ref:
        print "Adding average EEG reference projection."
        eeg_proj = mne.fiff.proj.make_eeg_average_ref_proj(raw.info)
        raw.info['projs'].append(eeg_proj)

    print 'Computing ECG projector'

    # Handler rejection parameters
    if len(mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=False)) == 0:
        del reject['grad']
    if len(mne.fiff.pick_types(raw.info, meg='mag', eeg=False, eog=False)) == 0:
        del reject['mag']
    if len(mne.fiff.pick_types(raw.info, meg=False, eeg=True, eog=False)) == 0:
        del reject['eeg']
    if len(mne.fiff.pick_types(raw.info, meg=False, eeg=False, eog=True)) == 0:
        del reject['eog']

    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=True, eog=True,
                                exclude=raw.info['bads'] + bads)
    if l_freq is None and h_freq is not None:
        raw.high_pass_filter(picks, h_freq, filter_length, n_jobs)
    if l_freq is not None and h_freq is None:
        raw.low_pass_filter(picks, h_freq, filter_length, n_jobs)
    if l_freq is not None and h_freq is not None:
        raw.band_pass_filter(picks, l_freq, h_freq, filter_length, n_jobs)

    epochs = mne.Epochs(raw, ecg_events, None, tmin, tmax, baseline=None,
                        picks=picks, reject=reject, proj=True)

    projs_init = raw.info['projs']

    if average:
        evoked = epochs.average()
        projs = mne.compute_proj_evoked(evoked, n_grad=n_grad, n_mag=n_mag,
                                        n_eeg=n_eeg)
    else:
        projs = mne.compute_proj_epochs(epochs, n_grad=n_grad, n_mag=n_mag,
                                        n_eeg=n_eeg)

    if preload is not None and os.path.exists(preload):
        os.remove(preload)

    print "Writing ECG projections in %s" % ecg_proj_fname
    mne.write_proj(ecg_proj_fname, projs + projs_init)
    print 'Done.'


if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--in", dest="raw_in",
                    help="Input raw FIF file", metavar="FILE")
    parser.add_option("--tmin", dest="tmin",
                    help="time before event in seconds",
                    default=-0.2)
    parser.add_option("--tmax", dest="tmax",
                    help="time before event in seconds",
                    default=0.4)
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
                    default=None)  # XXX
    parser.add_option("--h-freq", dest="h_freq",
                    help="Filter high cut-off frequency in Hz",
                    default=None)  # XXX
    parser.add_option("-p", "--preload", dest="preload",
                    help="Temporary file used during computaion",
                    default='tmp.mmap')
    parser.add_option("-a", "--average", dest="average", action="store_true",
                    help="Compute SSP after averaging",
                    default=False)
    parser.add_option("--filtersize", dest="filter_length",
                    help="Number of SSP vectors for EEG",
                    default=2048)
    parser.add_option("-j", "--n-jobs", dest="n_jobs",
                    help="Number of jobs to run in parallel",
                    default=1)
    parser.add_option("-c", "--channel", dest="ch_name",
                    help="Channel to use for ECG detection (Required if no ECG found)",
                    default=None)
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
                    default=250)
    parser.add_option("--avg-ref", dest="avg_ref", action="store_true",
                    help="Add EEG average reference proj",
                    default=False)
    parser.add_option("--bad", dest="bad_fname",
                    help="Text file containing bad channels list (one per line)",
                    default=None)

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
    ch_name = options.ch_name
    reject = dict(grad=1e-13 * float(options.rej_grad),
                  mag=1e-15 * float(options.rej_mag),
                  eeg=1e-6 * float(options.rej_eeg),
                  eog=1e-6 * float(options.rej_eog))
    avg_ref = options.avg_ref
    bad_fname = options.bad_fname

    if bad_fname is not None:
        bads = [w.rstrip().split()[0] for w in open(bad_fname).readlines()]
        print 'Bad channels read : %s' % bads
    else:
        bads = []

    compute_proj_ecg(raw_in, tmin, tmax, n_grad, n_mag, n_eeg, l_freq, h_freq,
                     average, preload, filter_length, n_jobs, ch_name, reject,
                     avg_ref, bads)
