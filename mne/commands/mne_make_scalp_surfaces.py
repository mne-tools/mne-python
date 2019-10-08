#!/usr/bin/env python

# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
#          simplified bsd-3 license

"""Create high-resolution head surfaces for coordinate alignment.

Examples
--------
.. code-block:: console

    $ mne make_scalp_surfaces --overwrite --subject sample

"""
import copy
import os
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
    from mne.commands.utils import get_optparser, _add_verbose_flag

    parser = get_optparser(__file__)
    subjects_dir = mne.get_config('SUBJECTS_DIR')

    parser.add_option('-o', '--overwrite', dest='overwrite',
                      action='store_true',
                      help='Overwrite previously computed surface')
    parser.add_option('-s', '--subject', dest='subject',
                      help='The name of the subject', type='str')
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='Force transformation of surface into bem.')
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=subjects_dir)
    parser.add_option("-n", "--no-decimate", dest="no_decimate",
                      help="Disable medium and sparse decimations "
                      "(dense only)", action='store_true')
    _add_verbose_flag(parser)
    options, args = parser.parse_args()

    subject = vars(options).get('subject', os.getenv('SUBJECT'))
    subjects_dir = options.subjects_dir
    if subject is None or subjects_dir is None:
        parser.print_help()
        sys.exit(1)
    _run(subjects_dir, subject, options.force, options.overwrite,
         options.no_decimate, options.verbose)


@verbose
def _run(subjects_dir, subject, force, overwrite, no_decimate, verbose=None):
    this_env = copy.deepcopy(os.environ)
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    this_env['SUBJECTS_DIR'] = subjects_dir
    this_env['SUBJECT'] = subject
    if 'FREESURFER_HOME' not in this_env:
        raise RuntimeError('The FreeSurfer environment needs to be set up '
                           'for this script')
    incomplete = 'warn' if force else 'raise'
    subj_path = op.join(subjects_dir, subject)
    if not op.exists(subj_path):
        raise RuntimeError('%s does not exist. Please check your subject '
                           'directory path.' % subj_path)

    mri = 'T1.mgz' if op.exists(op.join(subj_path, 'mri', 'T1.mgz')) else 'T1'

    logger.info('1. Creating a dense scalp tessellation with mkheadsurf...')

    def check_seghead(surf_path=op.join(subj_path, 'surf')):
        surf = None
        for k in ['lh.seghead', 'lh.smseghead']:
            this_surf = op.join(surf_path, k)
            if op.exists(this_surf):
                surf = this_surf
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

    bem_dir = op.join(subjects_dir, subject, 'bem')
    if not op.isdir(bem_dir):
        os.mkdir(bem_dir)
    dense_fname = op.join(bem_dir, '%s-head-dense.fif' % subject)
    logger.info('2. Creating %s ...' % dense_fname)
    _check_file(dense_fname, overwrite)
    # Helpful message if we get a topology error
    msg = '\n\nConsider using --force as an additional input parameter.'
    surf = mne.bem._surfaces_to_bem(
        [surf], [mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_HEAD], [1],
        incomplete=incomplete, extra=msg)[0]
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
            [mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_HEAD], [1], rescale=False,
            incomplete=incomplete, extra=msg)
        mne.write_bem_surfaces(dec_fname, dec_surf)


mne.utils.run_command_if_main()
