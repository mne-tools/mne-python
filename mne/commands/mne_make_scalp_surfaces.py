#!/usr/bin/env python

# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
#          simplified bsd-3 license

"""Create high-resolution head surfaces for coordinate alignment.

example usage: mne make_scalp_surfaces --overwrite --subject sample
"""
from __future__ import print_function

import os
import copy
import os.path as op
import sys
import mne
from mne.utils import (run_subprocess, verbose, logger, ETSContext,
                       get_subjects_dir)


def _check_file(fname, overwrite):
    """Prevent overwrites."""
    if op.isfile(fname) and not overwrite:
        raise IOError('File %s exists, use --overwrite to overwrite it'
                      % fname)


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')

    parser.add_option('-o', '--overwrite', dest='overwrite',
                      action='store_true',
                      help='Overwrite previously computed surface')
    parser.add_option('-s', '--subject', dest='subject',
                      help='The name of the subject', type='str')
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='Force transformation of surface into bem.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true',
                      help='Print the debug messages.')
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=subjects_dir)
    parser.add_option("-n", "--no-decimate", dest="no_decimate",
                      help="Disable medium and sparse decimations "
                      "(dense only)", action='store_true')
    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    print(options.no_decimate)
    _run(subjects_dir, subject, options.force, options.overwrite,
         options.no_decimate, options.verbose)


@verbose
def _run(subjects_dir, subject, force, overwrite, no_decimate, verbose=None):
    this_env = copy.copy(os.environ)
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    this_env['SUBJECTS_DIR'] = subjects_dir
    this_env['SUBJECT'] = subject
    if 'FREESURFER_HOME' not in this_env:
        raise RuntimeError('The FreeSurfer environment needs to be set up '
                           'for this script')
    force = '--force' if force else '--check'
    subj_path = op.join(subjects_dir, subject)
    if not op.exists(subj_path):
        raise RuntimeError('%s does not exist. Please check your subject '
                           'directory path.' % subj_path)

    mri = 'T1.mgz' if op.exists(op.join(subj_path, 'mri', 'T1.mgz')) else 'T1'

    logger.info('1. Creating a dense scalp tessellation with mkheadsurf...')

    def check_seghead(surf_path=op.join(subj_path, 'surf')):
        surf = None
        for k in ['lh.seghead', 'lh.smseghead']:
            surf = op.join(surf_path, k)
            if op.exists(surf):
                break
        return surf

    my_seghead = check_seghead()
    if my_seghead is None:
        run_subprocess(['mkheadsurf', '-subjid', subject, '-srcvol', mri],
                       env=this_env)

    surf = check_seghead()
    if surf is None:
        raise RuntimeError('mkheadsurf did not produce the standard output '
                           'file.')

    dense_fname = '{0}/{1}/bem/{1}-head-dense.fif'.format(subjects_dir,
                                                          subject)
    logger.info('2. Creating %s ...' % dense_fname)
    _check_file(dense_fname, overwrite)
    surf = mne.bem._surfaces_to_bem(
        [surf], [mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_HEAD], [1])[0]
    mne.write_bem_surfaces(dense_fname, surf)
    levels = 'medium', 'sparse'
    tris = [] if no_decimate else [30000, 2500]
    if os.getenv('_MNE_TESTING_SCALP', 'false') == 'true':
        tris = [len(surf['tris'])]  # don't actually decimate
    for ii, (n_tri, level) in enumerate(zip(tris, levels), 3):
        logger.info('%i. Creating %s tessellation...' % (ii, level))
        logger.info('%i.1 Decimating the dense tessellation...' % ii)
        with ETSContext():
            points, tris = mne.decimate_surface(points=surf['rr'],
                                                triangles=surf['tris'],
                                                n_triangles=n_tri)
        dec_fname = dense_fname.replace('dense', level)
        logger.info('%i.2 Creating %s' % (ii, dec_fname))
        _check_file(dec_fname, overwrite)
        dec_surf = mne.bem._surfaces_to_bem(
            [dict(rr=points, tris=tris)],
            [mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_HEAD], [1], rescale=False)
        mne.write_bem_surfaces(dec_fname, dec_surf)

is_main = (__name__ == '__main__')
if is_main:
    run()
