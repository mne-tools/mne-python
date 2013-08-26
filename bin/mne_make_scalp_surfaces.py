#!/usr/bin/env python

# Authors: Denis A. Engemann  <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
#          simplified bsd-3 license

"""
Create high-resulution head surfaces for coordinate alignment.

example usage: mne_make_scalp_surfaces.py --overwrite --subject sample
"""
import os
import os.path as op
import sys
from commands import getstatusoutput
import mne

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-o', '--overwrite', dest='overwrite',
                      action='store_true',
                      help='Overwrite previously computed surface')
    parser.add_option('-s', '--subject', dest='subject',
                      help='The name of the subject', type='str')
    options, args = parser.parse_args()
    if not options.subject:
        parser.print_help()
        sys.exit(-1)
    overwrite = options.overwrite
    subject = options.subject
    env = os.environ
    env['SUBJECT'] = subject

    def my_run_cmd(cmd, err_msg):
        if getstatusoutput(cmd)[0] != 0:
            print err_msg
            sys.exit(1)

    if not 'SUBJECTS_DIR' in env:
        print 'The environment variable SUBJECTS_DIR should be set'
        sys.exit(1)

    if not op.isabs(env['SUBJECTS_DIR']):
        env['SUBJECTS_DIR'] = op.abspath(env['SUBJECTS_DIR'])
    subj_dir = env['SUBJECTS_DIR']

    if not 'MNE_ROOT' in env:
        print 'MNE_ROOT environment variable is not set'
        sys.exit(1)

    if not 'FREESURFER_HOME' in env:
        print 'The FreeSurfer environment needs to be set up for this script'
        sys.exit(1)

    if op.exists(op.join(subj_dir, subject, 'mri', 'T1.mgz')):
        mri = 'T1.mgz'
    else:
        mri = 'T1'

    print '1. Creating a dense scalp tessellation with mkheadsurf...'

    my_seghead = None
    if op.exists(op.join(subj_dir, subject, 'surf',
                 'lh.seghead')):
        my_seghead = 'lh.seghead'
    elif op.exists(op.join(subj_dir, subject, 'surf',
                   'lh.smseghead')):
        my_seghead = 'lh.smseghead'
    if my_seghead is None:
        cmd = 'mkheadsurf -subjid %s -srcvol %s >/dev/null' % (subject, mri)
        print cmd
        my_run_cmd(cmd, 'mkheadsurf failed')
    else:
        print '%s/%s/surf/%s already there' % (subj_dir, subject, my_seghead)

    surf = None
    if op.exists(op.join(subj_dir, subject, 'surf',
                 'lh.seghead')):
        surf = op.join(subj_dir, subject, 'surf', 'lh.seghead')
    elif op.exists(op.join(subj_dir, subject, 'surf',
                   'lh.smseghead')):
        surf = op.join(subj_dir, subject, 'surf', 'lh.seghead')
    if surf is None:
        print 'mkheadsurf did not produce the standard output file.'
        sys.exit(1)

    fif = '{0}/{1}/bem/{1}-head-dense.fif'.format(subj_dir, subject)
    print '2. Creating $fif...'
    cmd = 'mne_surf2bem --surf %s --id 4 --check --fif %s' % (surf, fif)
    my_run_cmd(cmd, 'Failed to create %s, see above' % fif)
    levels = 'medium', 'sparse'
    for ii, (ntri, level) in enumerate(zip([30000, 2500], levels), 3):
        my_surf = mne.read_bem_surfaces(fif)[0]
        print '%i. Creating medium grade tessellation...' % ii
        print '%i.1 Decimating the dense tessellation...' % ii
        reduction = 1 - (float(ntri) / my_surf['ntri'])
        points, tris = mne.decimate_surface(points=my_surf['rr'],
                                            triangles=my_surf['tris'],
                                            reduction=reduction)
        out_fif = fif.replace('dense', level)
        print '%i.2 Creating %s' % (ii, out_fif)
        surf_fname = '/tmp/tmp-surf.fif'
        mne.write_surface(surf_fname, points, tris)
        cmd = 'mne_surf2bem --surf %s --id 4 --check --fif %s'
        cmd %= (surf_fname, out_fif)
        my_run_cmd(cmd, 'Failed to create %s, see above' % out_fif)
        os.remove(surf_fname)

    sys.exit(0)
