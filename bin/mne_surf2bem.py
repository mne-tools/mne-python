#!/usr/bin/env python
"""Example usage

mne_surf2bem.py --surf ${SUBJECTS_DIR}/${SUBJECT}/surf/lh.seghead --fif \
    ${SUBJECTS_DIR}/${SUBJECT}/bem/${SUBJECT}-head.fif --id=4

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import mne

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-s", "--surf", dest="surf",
                    help="Surface in Freesurfer format", metavar="FILE")
    parser.add_option("-f", "--fif", dest="fif",
                    help="FIF file produced", metavar="FILE")
    parser.add_option("-i", "--id", dest="id", default=4,
                    help=("Surface Id (e.g. 4 sur head surface)"))

    (options, args) = parser.parse_args()

    print "Converting %s to BEM FIF file." % options.surf

    points, tris = mne.read_surface(options.surf)
    points *= 1e-3
    surf = dict(coord_frame=5, id=int(options.id), nn=None, np=len(points),
                ntri=len(tris), rr=points, sigma=1, tris=tris)
    mne.write_bem_surface(options.fif, surf)
