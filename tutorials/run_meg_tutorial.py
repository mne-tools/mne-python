import sys
import os
from os.path import join

import mne
from mne.source_space import (setup_source_space, morph_source_spaces,
                              add_source_space_distances)

def run():
    args = sys.argv
    if len(args) <= 1:
        print 'Usage: run_meg_tutorial.sh <sample data directory>'
        return

    sample_dir = args[1]
    subjects_dir = join(sample_dir, 'subjects')
    meg_dir = join(sample_dir, join('MEG', 'sample'))

    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['MEG_DIR'] = meg_dir

    subject = 'sample'
    """
    src = setup_source_space(subject, fname=True, spacing='oct6',
                             overwrite=True)

    # If one wanted to use other source spaces, these types of options are
    # available
    src_fsaverage = setup_source_space('fsaverage', fname=True, spacing='ico5',
                                       overwrite=True)
    morph_source_spaces(src_fsaverage, subject_to='sample')
    setup_source_space(subjects_dir, fname=True, spacing='all', overwrite=True)

    # Add distances to source space (if desired, takes a long time)
    bem_dir = join(subjects_dir, join('sample', 'bem'))
    os.rename(join(bem_dir, 'sample-oct-6-src.fif'),
              join(bem_dir, 'sample-oct-6-orig-src.fif'))
    new_src = add_source_space_distances(src, dist_limit=0.007)
    new_src.save(join(bem_dir, 'sample-oct-6-src.fif'))

    """
    # Preprocessing
    raw = mne.io.Raw(join(meg_dir, 'sample_audvis_raw.fif'), preload=True)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']
    reject = dict(grad=3000e-13, mag=4000e-15, eeg=100e-6)
    proj, _ = mne.preprocessing.compute_proj_ecg(raw, l_freq=1, h_freq=100,
                                                 ch_name='MEG 1531',
                                                 reject=reject)
    raw.add_proj(proj)

    proj, _ = mne.preprocessing.compute_proj_eog(raw, l_freq=1, h_freq=35,
                                                 reject=reject, no_proj=True)
    raw.add_proj(proj)
    print 'Done'
    raw.filter(l_freq=None, h_freq=40)
    raw_resampled = raw.resample(150)
    #raw_resampled.save(join(meg_dir, 'sample_audvis_filt-0-40_raw.fif'))

    # Epoching
    events = mne.find_events(raw)
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}
    tmin, tmax = -0.2, 0.5
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True, eog=True)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    evoked = epochs.average()
    

is_main = (__name__ == '__main__')
if is_main:
    run()