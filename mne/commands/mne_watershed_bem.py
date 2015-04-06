"""

    Create BEM surfaces using the watershed algorithm included with
        FreeSurfer

"""

from __future__ import print_function
import sys
import os
import shutil
import mne
from mne.utils import (verbose, logger)


@verbose
def mne_watershed_bem(subject=None, subjects_dir=None, overwrite=False,
                      volume='T1', atlas=False, gcaatlas=False, preflood=None,
                      verbose=None):
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
    verbose : bool, str or None
        If not None, override default verbose level
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
    logger.info('\nRunning mri_watershed for BEM segmentation with the '
                'following parameters:\n\n'
                'SUBJECTS_DIR = %s\n'
                'SUBJECT = %s\n'
                'Result dir = %s\n' % (subjects_dir, subject, ws_dir))
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
    logger.info('Created %s/%s-head.fif\n\nComplete.' % (bem_dir, subject))


def run():
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-s", "--subject", dest="subject",
                      help="Subject name", default=None)
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=None)
    parser.add_option("-o", "--overwrite", dest="overwrite",
                      help="Write over existing files")
    parser.add_option("-v", "--volume", dest="volume",
                      help="Defaults to T1", default='T1')
    parser.add_option("-a", "--atlas", dest="atlas",
                      help="Specify the --atlas option for mri_watershed",
                      default=False)
    parser.add_option("-g", "--gcaatlas", dest="gcaatlas",
                      help="Use the subcortical atlas", default=False)
    parser.add_option("-p", "--preflood", dest="preflood",
                      help="Change the preflood height", default=None)
    parser.add_option("--verbose", dest="verbose",
                      help="If not None, override default verbose level",
                      default=None)

    options, args = parser.parse_args()

    subject = options.subject
    subjects_dir = options.subjects_dir
    overwrite = options.overwrite
    volume = options.volume
    atlas = options.atlas
    gcaatlas = options.gcaatlas
    preflood = options.preflood
    verbose = options.verbose

    mne_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                      overwrite=overwrite, volume=volume, atlas=atlas,
                      gcaatlas=gcaatlas, preflood=preflood, verbose=verbose)

is_main = (__name__ == '__main__')
if is_main:
    run()
