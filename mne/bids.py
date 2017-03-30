# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import pandas as pd

import mne
from mne.io.pick import channel_type

from datetime import datetime


def channel_tsv(raw, fname):
    """Create channel tsv."""

    map_chs = dict(grad='MEG', mag='MEG', stim='TRIG', eeg='EEG',
                   eog='EOG', misc='MISC')
    map_desc = dict(grad='sensor gradiometer', mag='magnetometer',
                    stim='analogue trigger',
                    eeg='electro-encephalography channel',
                    eog='electro-oculogram', misc='miscellaneous channel')

    status, ch_type, description = list(), list(), list()
    for idx, ch in enumerate(raw.info['ch_names']):
        status.append('bad' if ch in raw.info['bads'] else 'good')
        ch_type.append(map_chs[channel_type(raw.info, idx)])
        description.append(map_desc[channel_type(raw.info, idx)])

    onlinefilter = '%0.2f-%0.2f' % (raw.info['highpass'], raw.info['lowpass'])
    df = pd.DataFrame({'name': raw.info['ch_names'], 'type': ch_type,
                       'description': description,
                       'onlinefilter': onlinefilter,
                       'samplingrate': '%f' % raw.info['sfreq'],
                       'status': status
                       })
    df.to_csv(fname, sep='\t', index=False)


def events_tsv(raw, events, fname):
    """Create tsv file for events."""

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events[:, 0] -= raw.first_samp

    event_id_map = {v: k for k, v in event_id.items()}

    df = pd.DataFrame(events[:, [0, 2]], columns=['Sample', 'Condition'])
    df.Condition = df.Condition.map(event_id_map)

    df.to_csv(fname, sep='\t', index=False)


def scans_tsv(raw, fname):
    """Create tsv file for scans."""

    acq_time = datetime.fromtimestamp(raw.info['meas_date'][0]
                                      ).strftime('%Y-%m-%dT%H:%M:%S')

    df = pd.DataFrame({'filename': ['meg/%s' % raw.filenames[0]],
                       'acq_time': [acq_time]})

    print(df.head())

    df.to_csv(fname, sep='\t', index=False)


def folder_to_bids(path):
    """Walk over a folder of raw files and create bids compatible folder."""
    subject = '01'
    run = '01'
    task = 'audiovisual'

    events_fname = "sample_audvis_raw-eve.fif"
    raw_fname = "sub-01_task-audiovisual_run-01_meg.fif"
    events = mne.read_events(events_fname).astype(int)
    raw = mne.io.read_raw_fif(raw_fname)

    channel_tsv(raw, 'sub-%s_task-%s_run-%s_channel.tsv'
                % (subject, task, run))
    events_tsv(raw, events, 'sub-%s_task-%s_run-%s_events.tsv'
               % (subject, task, run))
    scans_tsv(raw, 'sub-%s_scans.tsv' % subject)
