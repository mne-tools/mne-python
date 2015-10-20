import sys
import os
from os.path import join

import mne
from mne.source_space import (setup_source_space, morph_source_spaces,
                              add_source_space_distances)
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator,
                              apply_inverse)


def run():

    args = sys.argv
    if len(args) <= 1:
        print 'Usage: run_meg_tutorial.sh <sample data directory>'
        return

    sample_dir = args[1]
    subjects_dir = join(sample_dir, 'subjects')
    meg_dir = join(sample_dir, 'MEG', 'sample')

    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['MEG_DIR'] = meg_dir

    subject = 'sample'

    src = setup_source_space(subject, fname=True, spacing='oct6',
                             overwrite=True)

    # If one wanted to use other source spaces, these types of options are
    # available
    src_fsaverage = setup_source_space('fsaverage', fname=True, spacing='ico5',
                                       overwrite=True, add_dist=False)
    morph_source_spaces(src_fsaverage, subject_to='sample')
    setup_source_space(subject, fname=True, spacing='all', overwrite=True,
                       add_dist=False)

    # Add distances to source space (if desired, takes a long time)
    bem_dir = join(subjects_dir, join('sample', 'bem'))
    os.rename(join(bem_dir, 'sample-oct-6-src.fif'),
              join(bem_dir, 'sample-oct-6-orig-src.fif'))
    new_src = add_source_space_distances(src, dist_limit=0.007)
    new_src.save(join(bem_dir, 'sample-oct-6-src.fif'))

    # Preprocessing
    raw = mne.io.Raw(join(meg_dir, 'sample_audvis_raw.fif'), preload=True)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']
    reject = dict(grad=3000e-13, mag=4000e-15, eeg=100e-6)
    ecg_proj, _ = mne.preprocessing.compute_proj_ecg(raw, l_freq=1, h_freq=100,
                                                     ch_name='MEG 1531',
                                                     reject=reject)

    eog_proj, _ = mne.preprocessing.compute_proj_eog(raw, l_freq=1, h_freq=35,
                                                     reject=reject,
                                                     no_proj=True)

    events = mne.find_events(raw)
    mne.write_events(join(meg_dir, 'sample_audvis_raw-eve.fif'), events)
    event_id = [1, 2, 3, 4]
    tmin, tmax = -0.2, 0.5
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True, eog=True)

    # Average with no filter
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    evoked = epochs.average()
    evoked.save(join(meg_dir, 'sample_audvis-no-filter-ave.fif'))

    raw.filter(l_freq=None, h_freq=40)
    raw_resampled = raw.resample(150)
    raw_resampled.save(join(meg_dir, 'sample_audvis_filt-0-40_raw.fif'),
                       overwrite=True)

    raw.add_proj(ecg_proj)
    raw.add_proj(eog_proj)

    resampled_events = mne.find_events(raw_resampled)
    mne.write_events(join(meg_dir, 'sample_audvis_filt-0-40_raw-eve.fif'),
                     resampled_events)

    # Average with filter
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    evoked = epochs.average()
    evoked.save(join(meg_dir, 'sample_audvis-ave.fif'))

    # Compute the noise covariance matrix
    noise_cov = mne.compute_raw_data_covariance(raw, picks=picks)
    noise_cov.save(join(meg_dir, 'audvis.cov'))

    # Compute the empty-room noise covariance matrix
    ernoise_raw = mne.io.Raw(join(meg_dir, 'ernoise_raw.fif'), preload=True)
    ernoise_raw.info['bads'] = ['MEG 2443']
    ernoise_raw.filter(l_freq=None, h_freq=40)
    picks = mne.pick_types(ernoise_raw.info, meg=True, eeg=True, stim=True,
                           eog=True)
    ernoise_cov = mne.compute_raw_data_covariance(ernoise_raw, picks=picks)
    ernoise_cov.save(join(meg_dir, 'ernoise.cov'))

###############################################################################
    # Compute forward solution a.k.a. lead field
    trans = join(meg_dir, 'sample_audvis_raw-trans.fif')
    bem = join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
    # for MEG only
    fname = 'sample_audvis-meg-oct-6-fwd.fif'
    fwd_meg = mne.make_forward_solution(raw.info, trans, src, bem,
                                        fname=fname, meg=True, eeg=False,
                                        mindist=5.0, n_jobs=2, overwrite=True)

    # for EEG only
    bem = join(subjects_dir, 'sample', 'bem',
               'sample-5120-5120-5120-bem-sol.fif')
    fname = 'sample_audvis-eeg-oct-6-fwd.fif'
    fwd_eeg = mne.make_forward_solution(raw.info, trans, src, bem,
                                        fname=fname, meg=False, eeg=True,
                                        mindist=5.0, n_jobs=2, overwrite=True)

    # for both EEG and MEG
    fname = 'sample_audvis-meg-eeg-oct-6-fwd.fif'
    fwd = mne.make_forward_solution(raw.info, trans, src, bem,
                                    fname=fname, meg=True, eeg=True,
                                    mindist=5.0, n_jobs=2, overwrite=True)

    # Create various sensitivity maps
    grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='free')
    grad_map.save('sample_audvis-grad-oct-6-fwd-sensmap', ftype='w')
    mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='free')
    mag_map.save('sample_audvis-mag-oct-6-fwd-sensmap', ftype='w')
    eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='free')
    eeg_map.save('sample_audvis-eeg-oct-6-fwd-sensmap', ftype='w')
    grad_map2 = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
    grad_map2.save('sample_audvis-grad-oct-6-fwd-sensmap-2', ftype='w')
    mag_map2 = mne.sensitivity_map(fwd, ch_type='mag', mode='ratio')
    mag_map2.save('sample_audvis-mag-oct-6-fwd-sensmap-3', ftype='w')

    # Compute some with the EOG + ECG projectors
    projs = ecg_proj + eog_proj + raw.info['projs']
    for map_type in ['radiality', 'angle', 'remaining', 'dampening']:
        eeg_map = mne.sensitivity_map(fwd, projs=projs, ch_type='eeg',
                                      mode=map_type)
        eeg_map.save('sample_audvis-eeg-oct-6-fwd-sensmap-' + map_type)

###############################################################################
    # Compute MNE inverse operators
    #
    # Note: The MEG/EEG forward solution could be used for all
    #
    inv_meg = make_inverse_operator(raw.info, fwd_meg, noise_cov, loose=0.2)
    fname = join(meg_dir, 'sample_audvis-meg-oct-6-meg-inv.fif')
    write_inverse_operator(fname, inv_meg)

    inv_eeg = make_inverse_operator(raw.info, fwd_eeg, noise_cov, loose=0.2)
    fname = join(meg_dir, 'sample_audvis-eeg-oct-6-eeg-inv.fif')
    write_inverse_operator(fname, inv_eeg)

    inv = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2)
    fname = join(meg_dir, 'sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif')
    write_inverse_operator(fname, inv)

    # inverse operator with fixed orientation (for testing)
    inv_fixed = make_inverse_operator(raw.info, fwd_meg, noise_cov, depth=None,
                                      fixed=True)
    fname = join(meg_dir, 'sample_audvis-meg-oct-6-meg-nodepth-fixed-inv.fif')
    write_inverse_operator(fname, inv_fixed)

    # produce two with diagonal noise (for testing)
    diag = noise_cov.as_diag()
    inv_meg_diag = make_inverse_operator(raw.info, fwd_meg, diag, loose=0.2)
    fname = join(meg_dir, 'sample_audvis-meg-oct-6-meg-diagnoise-inv.fif')
    write_inverse_operator(fname, inv_meg_diag)

    inv_eeg_diag = make_inverse_operator(raw.info, fwd, diag, loose=0.2)
    fname = join(meg_dir,
                 'sample_audvis-meg-eeg-oct-6-meg-eeg-diagnoise-inv.fif')
    write_inverse_operator(fname, inv_eeg_diag)

    # Produce stc files
    evoked.crop(0, 0.25)
    stc_meg = apply_inverse(evoked, inv_meg, method='MNE')
    stc_meg.save(join(meg_dir, 'sample_audvis-meg'))
    stc_eeg = apply_inverse(evoked, inv_eeg, method='MNE')
    stc_eeg.save(join(meg_dir, 'sample_audvis-eeg'))
    stc = apply_inverse(evoked, inv, method='MNE')
    stc.save(join(meg_dir, 'sample_audvis-meg-eeg'))

    # let's also morph to fsaverage
    stc_to = stc_meg.morph('fsaverage', grade=3, smooth=12)
    stc_to.save(join(meg_dir, 'fsaverage_audvis-meg'))
    stc_to = stc_eeg.morph('fsaverage', grade=3, smooth=12)
    stc_to.save(join(meg_dir, 'fsaverage_audvis-eeg'))
    stc_to = stc.morph('fsaverage', grade=3, smooth=12)
    stc_to.save(join(meg_dir, 'fsaverage_audvis-meg-eeg'))

###############################################################################
    # Do one dipole fitting
    evoked.crop(0.04, 0.095)
    bem = join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
    dip, _ = mne.fit_dipole(evoked, noise_cov, bem, trans)
    dip.save(join(meg_dir, 'sample_audvis_set1.dip'))

is_main = (__name__ == '__main__')
if is_main:
    run()
