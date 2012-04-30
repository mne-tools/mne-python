# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os

from .. import Epochs, compute_proj_evoked, compute_proj_epochs, \
               write_events, write_proj
from ..fiff import Raw, pick_types, make_eeg_average_ref_proj
from ..artifacts import find_ecg_events, find_eog_events


def _compute_exg_proj(mode, in_fif_fname, tmin, tmax,
                      n_grad, n_mag, n_eeg, l_freq, h_freq,
                      average, preload, filter_length, n_jobs, ch_name,
                      reject, bads, avg_ref, include_existing,
                      proj_fname, event_fname):
    """Compute SSP/PCA projections for ECG or EOG artifacts

    Parameters
    ----------
    mode: sting ('ECG', or 'EOG')
        What type of events to detect

    in_fif_fname: string
        Input Raw FIF file

    tmin: float
        Time before event in second

    tmax: float
        Time after event in seconds

    n_grad: int
        Number of SSP vectors for gradiometers

    n_mag: int
        Number of SSP vectors for magnetometers

    n_eeg: int
        Number of SSP vectors for EEG

    l_freq: float
        Filter low cut-off frequency in Hz

    h_freq: float
        Filter high cut-off frequency in Hz

    average: bool
        Compute SSP after averaging

    preload: string (or True)
        Temporary file used during computaion

    filter_length: int
        Number of taps to use for filtering

    n_jobs: int
        Number of jobs to run in parallel

    ch_name: string (or None)
        Channel to use for ECG event detection

    reject: dict
        Epoch rejection configuration (see Epochs)

    bads: list
        List with (additional) bad channels

    avg_ref: bool
        Add EEG average reference proj

    include_existing: bool
        Inlucde the SSP projectors currently in the fiff file

    proj_fname: string (or None)
        Filename to use for projectors (not saved if None)

    event_fname: string
        Filename to use for events (not saved if None)

    Returns
    -------
    proj : list
        Computed SSP projectors

    events : ndarray
        Detected events
    """
    # Reading fif File
    raw = Raw(in_fif_fname, preload=preload)

    if include_existing:
        projs = raw.info['projs']
    else:
        projs = []

    if avg_ref:
        print "Adding average EEG reference projection."
        eeg_proj = make_eeg_average_ref_proj(raw.info)
        projs.append(eeg_proj)

    if mode == 'ECG':
        print 'Running ECG SSP computation'
        events, _, _ = find_ecg_events(raw, ch_name=ch_name)
    elif mode == 'EOG':
        print 'Running EOG SSP computation'
        events = find_eog_events(raw)
    else:
        ValueError("mode must be 'ECG' or 'EOG'")

    print 'Computing projector'

    # Handler rejection parameters
    if len(pick_types(raw.info, meg='grad', eeg=False, eog=False)) == 0:
        del reject['grad']
    if len(pick_types(raw.info, meg='mag', eeg=False, eog=False)) == 0:
        del reject['mag']
    if len(pick_types(raw.info, meg=False, eeg=True, eog=False)) == 0:
        del reject['eeg']
    if len(pick_types(raw.info, meg=False, eeg=False, eog=True)) == 0:
        del reject['eog']

    picks = pick_types(raw.info, meg=True, eeg=True, eog=True,
                       exclude=raw.info['bads'] + bads)
    if l_freq is None and h_freq is not None:
        raw.high_pass_filter(picks, h_freq, filter_length, n_jobs)
    if l_freq is not None and h_freq is None:
        raw.low_pass_filter(picks, h_freq, filter_length, n_jobs)
    if l_freq is not None and h_freq is not None:
        raw.band_pass_filter(picks, l_freq, h_freq, filter_length, n_jobs)

    epochs = Epochs(raw, events, None, tmin, tmax, baseline=None,
                    picks=picks, reject=reject, proj=True)

    if average:
        evoked = epochs.average()
        ev_projs = compute_proj_evoked(evoked, n_grad=n_grad, n_mag=n_mag,
                                        n_eeg=n_eeg)
    else:
        ev_projs = compute_proj_epochs(epochs, n_grad=n_grad, n_mag=n_mag,
                                        n_eeg=n_eeg)
    projs.extend(ev_projs)

    if preload is not None and os.path.exists(preload):
        os.remove(preload)

    if event_fname is not None:
        print "Writing events in %s" % event_fname
        write_events(event_fname, events)

    if proj_fname is not None:
        print "Writing projections in %s" % proj_fname
        write_proj(proj_fname, projs)

    print 'Done.'

    return projs, events


def compute_proj_ecg(in_fif_fname, tmin=-0.2, tmax=0.4,
                     n_grad=2, n_mag=2, n_eeg=2, l_freq=1.0, h_freq=35.0,
                     average=False, preload="tmp.mmap",
                     filter_length=2048, n_jobs=1, ch_name=None,
                     reject=dict(grad=2000e-13, mag=3000e-15, eeg=50e-6,
                     eog=250e-6), bads=None,
                     avg_ref=False, include_existing=False,
                     ecg_proj_fname=None, ecg_event_fname=None):
    """Compute SSP/PCA projections for ECG artifacts

    Parameters
    ----------
    in_fif_fname: string
        Input Raw FIF file

    tmin: float
        Time before event in second

    tmax: float
        Time after event in seconds

    n_grad: int
        Number of SSP vectors for gradiometers

    n_mag: int
        Number of SSP vectors for magnetometers

    n_eeg: int
        Number of SSP vectors for EEG

    l_freq: float
        Filter low cut-off frequency in Hz

    h_freq: float
        Filter high cut-off frequency in Hz

    average: bool
        Compute SSP after averaging

    preload: string (or True)
        Temporary file used during computaion

    filter_length: int
        Number of taps to use for filtering

    n_jobs: int
        Number of jobs to run in parallel

    ch_name: string (or None)
        Channel to use for ECG detection (Required if no ECG found)

    reject: dict
        Epoch rejection configuration (see Epochs)

    bads: list
        List with (additional) bad channels

    avg_ref: bool
        Add EEG average reference proj

    include_existing: bool
        Inlucde the SSP projectors currently in the fiff file

    ecg_proj_fname: string (or None)
        Filename to use for projectors (not saved if None)

    ecg_event_fname: string (or None)
        Filename to use for events (not saved if None)

    Returns
    -------
    proj : list
        Computed SSP projectors

    ecg_events : ndarray
        Detected ECG events
    """

    projs, ecg_events = _compute_exg_proj('ECG', in_fif_fname, tmin, tmax,
                        n_grad, n_mag, n_eeg, l_freq, h_freq,
                        average, preload, filter_length, n_jobs, ch_name,
                        reject, bads, avg_ref, include_existing,
                        ecg_proj_fname, ecg_event_fname)


    return projs, ecg_events




