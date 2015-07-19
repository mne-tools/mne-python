#!/usr/bin/env python
"""Create 3-Layers BEM model from Flash MRI images

This function extracts the BEM surfaces (outer skull, inner skull, and
outer skin) from multiecho FLASH MRI data with spin angles of 5 and 30
degrees. The multiecho FLASH data are inputted in NIFTI format.
It was developed to work for Phillips MRI data, but could probably be
used for data from other scanners that have been converted to NIFTI format
(e.g., using MRIcron's dcm2nii). However,it has been tested only for
data from the Achieva scanner). This function assumes that the Freesurfer
segmentation of the subject has been completed. In particular, the T1.mgz
and brain.mgz MRI volumes should be, as usual, in the subject's mri
directory.

"""
from __future__ import print_function

# Authors:  Rey Rene Ramirez, Ph.D.   e-mail: rrramir at uw.edu
#           Alexandre Gramfort, Ph.D.

import sys
import math
import os
import os.path as op
import glob
import shutil

import mne
from mne.utils import (logger, get_subjects_dir, run_subprocess)


def make_flash_bem(subject, subjects_dir, flash05, flash30, noconvert=False,
                   unwarp=False, show=False):
    """
    Create 3-Layers BEM model from Flash MRI images

    Parameters
    ----------
    subject : string
        Subject name
    subjects_dir : string
        Directory containing subjects data (Freesurfer SUBJECTS_DIR)
    flash05 : string
        Full path of the NIFTI file for the
        FLASH sequence with a spin angle of 5 degrees
    flash30 : string
        Full path of the NIFTI file for the
        FLASH sequence with a spin angle of 30 degrees
    show : bool
        Show surfaces in 3D to visually inspect all three BEM
        surfaces (recommended)

    Notes
    -----
    This program assumes that both Freesurfer/FSL, and MNE,
    including MNE's Matlab Toolbox, are installed properly.
    For reference please read the MNE manual and wiki, and Freesurfer's wiki:
    http://www.nmr.mgh.harvard.edu/meg/manuals/
    http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/sofMNE.php
    http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php
    http://surfer.nmr.mgh.harvard.edu/
    http://surfer.nmr.mgh.harvard.edu/fswiki

    References:
    B. Fischl, D. H. Salat, A. J. van der Kouwe, N. Makris, F. Segonne,
    B. T. Quinn, and A. M. Dale, "Sequence-independent segmentation of magnetic
    resonance images," Neuroimage, vol. 23 Suppl 1, pp. S69-84, 2004.
    J. Jovicich, S. Czanner, D. Greve, E. Haley, A. van der Kouwe, R. Gollub,
    D. Kennedy, F. Schmitt, G. Brown, J. Macfall, B. Fischl, and A. Dale,
    "Reliability in multi-site structural MRI studies: effects of gradient
    non-linearity correction on phantom and human data," Neuroimage,
    vol. 30, Epp. 436-43, 2006.
    """
    env = os.environ.copy()

    if subject:
        env['SUBJECT'] = subject
    else:
        raise RuntimeError('The subject argument must be set')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    env['SUBJECTS_DIR'] = subjects_dir

    mri_dir = op.join(subjects_dir, subject, 'mri')
    bem_dir = op.join(subjects_dir, subject, 'bem')

    logger.info('\nProcessing the flash MRI data to produce BEM meshes with '
                'the following parameters:\n'
                'SUBJECTS_DIR = %s\n'
                'SUBJECT = %s\n'
                'Result dir = %s\n' % (subjects_dir, subject,
                                       op.join(bem_dir, 'flash')))
    # Step 1 : Data conversion to mgz format
    os.chdir(mri_dir)
    if not noconvert:
        if not op.exists('flash'):
            os.mkdir("flash")
        os.chdir("flash")
        if not op.exists('parameter_maps'):
            os.mkdir("parameter_maps")
        logger.info("--- Converting Flash 5")
        cmd = ['mri_convert', '-flip_angle', str(5 * math.pi / 180), '-tr 25',
               flash05, 'mef05.mgz']
        run_subprocess(cmd, env=env, stdout=sys.stdout)
        logger.info("--- Converting Flash 30")
        cmd = ['mri_convert', '-flip_angle', str(30 * math.pi / 180), '-tr 25',
               flash30, 'mef30.mgz']
        run_subprocess(cmd, env=env, stdout=sys.stdout)
    #
    os.chdir(op.join(mri_dir, "flash"))
    files = glob.glob("mef*.mgz")
    if unwarp:
        logger.info("--- Unwarp mgz data sets")
        for infile in files:
            outfile = infile.replace(".mgz", "u.mgz")
            cmd = ['grad_unwarp', '-i', infile, '-o', outfile, '-unwarp',
                   'true']
            run_subprocess(cmd, env=env, stdout=sys.stdout)
    # Step 2 : Create the parameter maps
    # (Clear everything if some of the data were reconverted)
    if not noconvert and op.exists("parameter_maps"):
            shutil.rmtree("parameter_maps")
            logger.info("Parameter maps directory cleared")
    if not op.exists("parameter_maps"):
        os.makedirs("parameter_maps")
    if unwarp:
        for i in range(len(files)):
            files[i] = files[i].replace(".mgz", "u.mgz")
    if len(os.listdir('parameter_maps')) == 0:
        logger.info("--- Creating the parameter maps")
        cmd = ['mri_ms_fitparms'] + files + ['parameter_maps']
        run_subprocess(cmd, env=env, stdout=sys.stdout)
    else:
        logger.info("Parameter maps were already computed")
    # Step 3 : Synthesize the flash 5 images
    logger.info("Synthesizing flash 5 images...")
    os.chdir('parameter_maps')
    if not op.exists('flash5.mgz'):
        cmd = ['mri_synthesize', '20 5 5', 'T1.mgz', 'PD.mgz', 'flash5.mgz']
        run_subprocess(cmd, env=env, stdout=sys.stdout)
        os.remove('flash5_reg.mgz')
    else:
        logger.info("Synthesized flash 5 volume is already there")
    # Step 4 : Register with MPRAGE
    logger.info("Registering flash 5 with MPRAGE...")
    if not op.exists('flash5_reg.mgz'):
        if op.exists(op.join(mri_dir, 'T1.mgz')):
            ref_volume = op.join(mri_dir, 'T1.mgz')
        else:
            ref_volume = op.join(mri_dir, 'T1')
        cmd = ['fsl_rigid_register', '-r', ref_volume, '-i', 'flash5.mgz',
               '-o', 'flash5_reg.mgz']
        run_subprocess(cmd, env=env, stdout=sys.stdout)
    else:
        logger.info("Registered flash 5 image is already there")
    # Step 5a : Convert flash5 into COR
    logger.info("Converting flash5 volume into COR format...")
    shutil.rmtree(op.join(mri_dir, 'flash5'), ignore_errors=True)
    os.makedirs(op.join(mri_dir, 'flash5'))
    cmd = ['mri_convert', 'flash5_reg.mgz', op.join(mri_dir, 'flash5')]
    run_subprocess(cmd, env=env, stdout=sys.stdout)
    # Step 5b and c : Convert the mgz volumes into COR
    os.chdir(mri_dir)
    convertT1 = False
    if op.isdir('T1'):
        if len(glob.glob('T1/COR*')) == 0:
            convertT1 = True
    else:
        convertT1 = True
    convertbrain = False
    if op.isdir('brain'):
        if len(glob.glob('brain/COR*')) == 0:
            convertbrain = True
    else:
        convertbrain = True
    if convertT1 == True:
        logger.info("Converting T1 volume into COR format...")
        if not op.isfile('T1.mgz'):
            raise RuntimeError ("Both T1 mgz and T1 COR volumes missing.")
        os.makedirs('T1')
        cmd = ['mri_convert', 'T1.mgz', 'T1']
        run_subprocess(cmd, env=env, stdout=sys.stdout)
    else:
        logger.info("T1 volume is already in COR format")
    if convertbrain == True:
        logger.info("Converting brain volume into COR format...")
        if not op.isfile('brain.mgz'):
            raise RuntimeError("Both brain mgz and brain COR volumes missing.")
        os.makedirs('brain')
        cmd = ['mri_convert', 'brain.mgz', 'brain']
        run_subprocess(cmd, env=env, stdout=sys.stdout)
    else:
        logger.info("Brain volume is already in COR format")
    # Finally ready to go
    logger.info("Creating the BEM surfaces...")
    cmd = ['mri_make_bem_surfaces', subject]
    run_subprocess(cmd, env=env, stdout=sys.stdout)
    logger.info("Converting the tri files into surf files...")
    os.chdir(bem_dir)
    if not op.exists('flash'):
        os.makedirs('flash')
    os.chdir('flash')
    surfs = ['inner_skull', 'outer_skull', 'outer_skin']
    for surf in surfs:
        shutil.move(op.join(bem_dir, surf+'.tri'), surf+'.tri')
        cmd = ['mne_convert_surface', '--tri', surf+'.tri', '--surfout', surf+'.surf', '--swap', '--mghmri', op.join(subjects_dir, subject, 'mri/flash/parameter_maps/flash5_reg.mgz')]
        run_subprocess(cmd, env=env, stdout=sys.stdout)
    #
    #
    '''
    print("--- Running mne_flash_bem")
    os.system('mne_flash_bem --noconvert')
    os.chdir(bem_dir)
    if not op.exists('flash'):
        os.mkdir("flash")
    os.chdir("flash")
    print("[done]")
    '''

    if show:
        fnames = ['outer_skin.surf', 'outer_skull.surf', 'inner_skull.surf']
        head_col = (0.95, 0.83, 0.83)  # light pink
        skull_col = (0.91, 0.89, 0.67)
        brain_col = (0.67, 0.89, 0.91)  # light blue
        colors = [head_col, skull_col, brain_col]
        from mayavi import mlab
        mlab.clf()
        for fname, c in zip(fnames, colors):
            points, faces = mne.read_surface(fname)
            mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
                                 faces, color=c, opacity=0.3)
        mlab.show()


def run():
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    subject = os.environ.get('SUBJECT')
    subjects_dir = os.environ.get('SUBJECTS_DIR')

    parser.add_option("-s", "--subject", dest="subject",
                      help="Subject name", default=subject)
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=subjects_dir)
    parser.add_option("-5", "--flash05", dest="flash05",
                      help=("Path to FLASH sequence with a spin angle of 5 "
                            "degrees in Nifti format"), metavar="FILE")
    parser.add_option("-3", "--flash30", dest="flash30",
                      help=("Path to FLASH sequence with a spin angle of 30 "
                            "degrees in Nifti format"), metavar="FILE")
    parser.add_option("-n", "--noconvert", dest="noconvert",
                      action="store_true", default=False,
                      help=("Assume that the images have already been "
                            "converted"))
    parser.add_option("-u", "--unwarp", dest="unwarp",
                      action="store_true", default=False,
                      help=("Run grad_unwarp with -unwarp <type> option on "
                            "each of the converted data sets"))
    parser.add_option("-v", "--view", dest="show", action="store_true",
                      help="Show BEM model in 3D for visual inspection",
                      default=False)

    options, args = parser.parse_args()

    if not options.noconvert:
        if options.flash05 is None or options.flash30 is None:
            parser.print_help()
            sys.exit(1)
        else:
            options.flash05 = op.abspath(options.flash05)
            options.flash30 = op.abspath(options.flash30)

    subject = options.subject
    subjects_dir = options.subjects_dir
    flash05 = options.flash05
    flash30 = options.flash30
    noconvert = options.noconvert
    unwarp = options.unwarp
    show = options.show

    make_flash_bem(subject=subject, subjects_dir=subjects_dir, flash05=flash05,
                   flash30=flash30, noconvert=noconvert, unwarp=unwarp,
                   show=show)

is_main = (__name__ == '__main__')
if is_main:
    run()
