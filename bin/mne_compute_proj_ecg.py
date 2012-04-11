#!/usr/bin/env python
"""Compute SSP/PCA projections for ECG artifacts
"""

# Authors : Alexandre Gramfort, Ph.D.

import mne


def compute_proj_ecg(in_fif_fname, tmin, tmax, n_grad, n_mag, n_eeg, l_freq,
                     h_freq, average, preload, filter_length, n_jobs):
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

    ecg_proj_fname = prefix + '_ecg_proj.fif'
    ecg_event_fname = prefix + '_ecg-eve.fif'

    print 'Running ECG SSP computation'

    ecg_events, _, _ = mne.artifacts.find_ecg_events(raw)
    print "Writing ECG events in %s" % ecg_event_fname
    mne.write_events(ecg_event_fname, ecg_events)

    print 'Computing ECG projector'

    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=True)
    if l_freq is None and h_freq is not None:
        raw.high_pass_filter(picks, h_freq, filter_length, n_jobs)
    if l_freq is not None and h_freq is None:
        raw.low_pass_filter(picks, h_freq, filter_length, n_jobs)
    if l_freq is not None and h_freq is not None:
        raw.band_pass_filter(picks, l_freq, h_freq, filter_length, n_jobs)

    epochs = mne.Epochs(raw, ecg_events, None, tmin, tmax, baseline=None,
                        picks=picks)

    if average:
        evoked = epochs.average()
        projs = mne.compute_proj_evoked(evoked, n_grad=n_grad, n_mag=n_mag,
                                        n_eeg=n_eeg)
    else:
        projs = mne.compute_proj_epochs(epochs, n_grad=n_grad, n_mag=n_mag,
                                        n_eeg=n_eeg)

    print "Writing ECG projections in %s" % ecg_proj_fname
    mne.write_proj(ecg_proj_fname, projs)
    print 'Done.'


if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--in", dest="raw_in",
                    help="Input raw FIF file", metavar="FILE")
    parser.add_option("-b", "--tmin", dest="tmin",
                    help="time before event in seconds",
                    default=-0.2)
    parser.add_option("-c", "--tmax", dest="tmax",
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
    parser.add_option("-l", "--l-freq", dest="l_freq",
                    help="Filter low cut-off frequency in Hz",
                    default=None)  # XXX
    parser.add_option("-t", "--h-freq", dest="h_freq",
                    help="Filter high cut-off frequency in Hz",
                    default=None)  # XXX
    parser.add_option("-p", "--preload", dest="preload",
                    help="Temporary file used during computaion",
                    default='tmp.mmap')
    parser.add_option("-a", "--average", dest="average", action="store_true",
                    help="Compute SSP after averaging",
                    default=False)
    parser.add_option("-f", "--filter-length", dest="filter_length",
                    help="Number of SSP vectors for EEG",
                    default=2048)
    parser.add_option("-j", "--n-jobs", dest="n_jobs",
                    help="Number of jobs to run in parallel",
                    default=1)

    options, args = parser.parse_args()

    raw_in = options.raw_in
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

    compute_proj_ecg(raw_in, tmin, tmax, n_grad, n_mag, n_eeg, l_freq, h_freq,
                     average, preload, filter_length, n_jobs)
