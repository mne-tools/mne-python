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

import mne


def make_flash_bem(subject, subjects_dir, flash05, flash30, show=False):
    """Create 3-Layers BEM model from Flash MRI images

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
    os.environ['SUBJECT'] = subject
    os.chdir(os.path.join(subjects_dir, subject, "mri"))
    if not os.path.exists('flash'):
        os.mkdir("flash")
    os.chdir("flash")
    # flash_dir = os.getcwd()
    if not os.path.exists('parameter_maps'):
        os.mkdir("parameter_maps")
    print("--- Converting Flash 5")
    os.system('mri_convert -flip_angle %s -tr 25 %s mef05.mgz' %
              (5 * math.pi / 180, flash05))
    print("--- Converting Flash 30")
    os.system('mri_convert -flip_angle %s -tr 25 %s mef30.mgz' %
              (30 * math.pi / 180, flash30))
    print("--- Running mne_flash_bem")
    os.system('mne_flash_bem --noconvert')
    os.chdir(os.path.join(subjects_dir, subject, 'bem'))
    if not os.path.exists('flash'):
        os.mkdir("flash")
    os.chdir("flash")
    print("[done]")

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
    parser.add_option("-v", "--view", dest="show", action="store_true",
                      help="Show BEM model in 3D for visual inspection",
                      default=False)

    options, args = parser.parse_args()

    if options.flash05 is None or options.flash30 is None:
        parser.print_help()
        sys.exit(1)

    subject = options.subject
    subjects_dir = options.subjects_dir
    flash05 = os.path.abspath(options.flash05)
    flash30 = os.path.abspath(options.flash30)
    show = options.show

    make_flash_bem(subject, subjects_dir, flash05, flash30, show=show)

is_main = (__name__ == '__main__')
if is_main:
    run()
