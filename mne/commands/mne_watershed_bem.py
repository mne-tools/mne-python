"""
    Create BEM surfaces using the watershed algorithm included with
        FreeSurfer

"""

from __future__ import print_function
# import math
# import sys
import os
import shutil
import mne


def mne_watershed_bem(subject, subjects_dir, preflood, volume='T1', atlas=False, gcaatlas=False, overwrite=False):
    """
    """
    os.environ['SUBJECT'] = subject
    os.environ['SUBJECTS_DIR'] = subjects_dir
    # os.chdir(os.path.join(subjects_dir, subject, "mri"))

    subject_dir = subjects_dir+'/'+subject
    mri_dir = subject_dir+'/mri'
    T1_dir = mri_dir+'/'+volume
    T1_mgz = mri_dir+'/'+volume+'.mgz'
    bem_dir = subject_dir+'/bem'
    ws_dir = subject_dir+'/bem/watershed'

    if not os.path.exists(subject_dir):
        raise RuntimeError('Could not find the MRI data directory "%s"' % subject_dir)
    if not os.path.exists(bem_dir):
        os.makedirs(bem_dir)
    if (not os.path.exists(T1_dir) and not os.path.exists(T1_mgz)):
        raise RuntimeError('Could not find the MRI data')
    if os.path.exists(ws_dir):
        if not overwrite:
            raise RuntimeError('%s already exists. Use the overwrite option to recreate it' % ws_dir)
        else:
            shutil.rmtree(ws_dir)
    os.makedirs(ws_dir+'/ws')

    cmd = ['mri_watershed']

    if gcaatlas:
        cmd += ['-atlas -T1 -brain_atlas $FREESURFER_HOME/average/RB_all_withskull_2007-08-08.gca $subject_dir/mri/transforms/talairach_with_skull.lta']
    elif atlas:
        cmd += ['-atlas']

    if os.path.exists(T1_mgz):
        cmd += ['-useSRAS', '-surf', ws_dir+'/'+subject, T1_mgz, ws_dir+'/ws']
    else:
        cmd += ['-useSRAS', '-surf', ws_dir+'/'+subject, T1_dir, ws_dir+'/ws']

    print("--- Running mri_watershed for BEM segmentation with the following parameters:")
    print("SUBJECTS_DIR = %s" % subjects_dir)
    print("SUBJECT = %s" % subject)
    print("Result dir = %s" % ws_dir)

    env = os.environ.copy()
    mne.utils.run_subprocess(cmd, env=env)









is_main = (__name__ == '__main__')
if is_main:
    subject = os.environ['SUBJECT']
    subjects_dir = os.environ['SUBJECTS_DIR']
    mne_watershed_bem(subject, subjects_dir, preflood=None, volume='T1', atlas=False, gcaatlas=False, overwrite=False)
