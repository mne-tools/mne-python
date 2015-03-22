"""
    Create BEM surfaces using the watershed algorithm included with
        FreeSurfer

"""

from __future__ import print_function
import sys
import os
import shutil
import mne


def mne_watershed_bem(subject=None, subjects_dir=None, overwrite=False,
                      volume='T1', atlas=False, gcaatlas=False, preflood=None):
    """
    Create BEM surfaces using the watershed algorithm included with FreeSurfer

    Parameters
    ----------
    subject : string
        Subject name (SUBJECT environment variable)
    subjects_dir : string
        Directory containing subjects data. If None use the Freesurfer\
         SUBJECTS_DIR environment variable.
    overwrite : bool
        Write over existing files
    volume : string
        Defaults to T1
    atlas : bool
        Specify the --atlas option for mri_watershed
    gcaatlas : bool
        Use the subcortical atlas
    preflood : int
        Change the preflood height
    """
    if not os.environ['FREESURFER_HOME']:
        raise RuntimeError('FREESURFER_HOME environment variable not set')
    if subject:
        os.environ['SUBJECT'] = subject
    else:
        try:
            subject = os.environ['SUBJECT']
        except KeyError:
            raise RuntimeError('SUBJECT environment variable not defined')
    if subjects_dir:
        os.environ['SUBJECTS_DIR'] = subjects_dir
    else:
        try:
            subjects_dir = os.environ['SUBJECTS_DIR']
        except KeyError:
            raise RuntimeError('SUBJECTS_DIR environment variable not defined')
    env = os.environ.copy()

    subject_dir = os.path.join(subjects_dir, subject)
    mri_dir = os.path.join(subject_dir, 'mri')
    T1_dir = os.path.join(mri_dir, volume)
    T1_mgz = os.path.join(mri_dir, volume+'.mgz')
    bem_dir = os.path.join(subject_dir, 'bem')
    ws_dir = os.path.join(subject_dir, 'bem', 'watershed')

    if not os.path.exists(subject_dir):
        raise RuntimeError('Could not find the MRI data directory "%s"'
                           % subject_dir)
    if not os.path.exists(bem_dir):
        os.makedirs(bem_dir)
    if (not os.path.exists(T1_dir) and not os.path.exists(T1_mgz)):
        raise RuntimeError('Could not find the MRI data')
    if os.path.exists(ws_dir):
        if not overwrite:
            raise RuntimeError('%s already exists. Use the overwrite option to\
             recreate it' % ws_dir)
        else:
            shutil.rmtree(ws_dir)
    # put together the command
    cmd = ['mri_watershed']
    if preflood:
        cmd += [preflood]
    if gcaatlas:
        cmd += ['-atlas', '-T1', '-brain_atlas', env['FREESURFER_HOME'] +
                '/average/RB_all_withskull_2007-08-08.gca',
                subject_dir+'/mri/transforms/talairach_with_skull.lta']
    elif atlas:
        cmd += ['-atlas']
    if os.path.exists(T1_mgz):
        cmd += ['-useSRAS', '-surf', os.path.join(ws_dir, subject), T1_mgz,
                os.path.join(ws_dir, 'ws')]
    else:
        cmd += ['-useSRAS', '-surf', os.path.join(ws_dir, subject), T1_dir,
                os.path.join(ws_dir, 'ws')]
    # report and run
    print("\nRunning mri_watershed for BEM segmentation with the following \
        parameters:\n")
    print("SUBJECTS_DIR = %s" % subjects_dir)
    print("SUBJECT = %s" % subject)
    print("Result dir = %s\n" % ws_dir)
    os.makedirs(os.path.join(ws_dir, 'ws'))
    mne.utils.run_subprocess(cmd, env=env, stdout=sys.stdout)
    #
    os.chdir(ws_dir)
    if os.path.exists(T1_mgz):
        surfaces = [subject+'_brain_surface', subject+'_inner_skull_surface',
                    subject+'_outer_skull_surface', subject +
                    '_outer_skin_surface']
        for s in surfaces:
            cmd = ['mne_convert_surface', '--surf', s, '--mghmri', T1_mgz,
                   '--surfout', s]
            mne.utils.run_subprocess(cmd, env=env, stdout=sys.stdout)
    os.chdir(bem_dir)
    if os.path.exists(subject+'-head.fif'):
        os.remove(subject+'-head.fif')
    cmd = ['mne_surf2bem', '--surf', os.path.join(ws_dir,
           subject+'_outer_skin_surface'), '--id', '4', '--fif',
           subject+'-head.fif']
    mne.utils.run_subprocess(cmd, env=env, stdout=sys.stdout)
    print("Created %s/%s-head.fif" % (bem_dir, subject))
    print("\nComplete.")


def run():
    subject = None
    subjects_dir = None
    overwrite = True
    volume = 'T1'
    atlas = False
    gcaatlas = False
    preflood = None
    mne_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                      overwrite=overwrite, volume=volume, atlas=atlas,
                      gcaatlas=gcaatlas, preflood=preflood)

is_main = (__name__ == '__main__')
if is_main:
    run()
