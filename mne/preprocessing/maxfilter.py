# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD-3-Clause

import os

from ..bem import fit_sphere_to_headshape
from ..io import read_raw_fif
from ..utils import logger, verbose, warn


def _mxwarn(msg):
    """Warn about a bug."""
    warn('Possible MaxFilter bug: %s, more info: '
         'http://imaging.mrc-cbu.cam.ac.uk/meg/maxbugs' % msg)


@verbose
def apply_maxfilter(in_fname, out_fname, origin=None, frame='device',
                    bad=None, autobad='off', skip=None, force=False,
                    st=False, st_buflen=16.0, st_corr=0.96, mv_trans=None,
                    mv_comp=False, mv_headpos=False, mv_hp=None,
                    mv_hpistep=None, mv_hpisubt=None, mv_hpicons=True,
                    linefreq=None, cal=None, ctc=None, mx_args='',
                    overwrite=True, verbose=None):
    """Apply NeuroMag MaxFilter to raw data.

    Needs Maxfilter license, maxfilter has to be in PATH.

    Parameters
    ----------
    in_fname : str
        Input file name.
    out_fname : str
        Output file name.
    origin : array-like or str
        Head origin in mm. If None it will be estimated from headshape points.
    frame : str ('device' or 'head')
        Coordinate frame for head center.
    bad : str, list (or None)
        List of static bad channels. Can be a list with channel names, or a
        string with channels (names or logical channel numbers).
    autobad : str ('on', 'off', 'n')
        Sets automated bad channel detection on or off.
    skip : str or a list of float-tuples (or None)
        Skips raw data sequences, time intervals pairs in sec,
        e.g.: 0 30 120 150.
    force : bool
        Ignore program warnings.
    st : bool
        Apply the time-domain MaxST extension.
    st_buflen : float
        MaxSt buffer length in sec (disabled if st is False).
    st_corr : float
        MaxSt subspace correlation limit (disabled if st is False).
    mv_trans : str (filename or 'default') (or None)
        Transforms the data into the coil definitions of in_fname, or into the
        default frame (None: don't use option).
    mv_comp : bool (or 'inter')
        Estimates and compensates head movements in continuous raw data.
    mv_headpos : bool
        Estimates and stores head position parameters, but does not compensate
        movements (disabled if mv_comp is False).
    mv_hp : str (or None)
        Stores head position data in an ascii file
        (disabled if mv_comp is False).
    mv_hpistep : float (or None)
        Sets head position update interval in ms (disabled if mv_comp is
        False).
    mv_hpisubt : str ('amp', 'base', 'off') (or None)
        Subtracts hpi signals: sine amplitudes, amp + baseline, or switch off
        (disabled if mv_comp is False).
    mv_hpicons : bool
        Check initial consistency isotrak vs hpifit
        (disabled if mv_comp is False).
    linefreq : int (50, 60) (or None)
        Sets the basic line interference frequency (50 or 60 Hz)
        (None: do not use line filter).
    cal : str
        Path to calibration file.
    ctc : str
        Path to Cross-talk compensation file.
    mx_args : str
        Additional command line arguments to pass to MaxFilter.
    %(overwrite)s
    %(verbose)s

    Returns
    -------
    origin: str
        Head origin in selected coordinate frame.
    """
    # check for possible maxfilter bugs
    if mv_trans is not None and mv_comp:
        _mxwarn("Don't use '-trans' with head-movement compensation "
                "'-movecomp'")

    if autobad != 'off' and (mv_headpos or mv_comp):
        _mxwarn("Don't use '-autobad' with head-position estimation "
                "'-headpos' or movement compensation '-movecomp'")

    if st and autobad != 'off':
        _mxwarn("Don't use '-autobad' with '-st' option")

    # determine the head origin if necessary
    if origin is None:
        logger.info('Estimating head origin from headshape points..')
        raw = read_raw_fif(in_fname)
        r, o_head, o_dev = fit_sphere_to_headshape(raw.info, units='mm')
        raw.close()
        logger.info('[done]')
        if frame == 'head':
            origin = o_head
        elif frame == 'device':
            origin = o_dev
        else:
            raise RuntimeError('invalid frame for origin')

    if not isinstance(origin, str):
        origin = '%0.1f %0.1f %0.1f' % (origin[0], origin[1], origin[2])

    # format command
    cmd = ('maxfilter -f %s -o %s -frame %s -origin %s '
           % (in_fname, out_fname, frame, origin))

    if bad is not None:
        # format the channels
        if not isinstance(bad, list):
            bad = bad.split()
        bad = map(str, bad)
        bad_logic = [ch[3:] if ch.startswith('MEG') else ch for ch in bad]
        bad_str = ' '.join(bad_logic)

        cmd += '-bad %s ' % bad_str

    cmd += '-autobad %s ' % autobad

    if skip is not None:
        if isinstance(skip, list):
            skip = ' '.join(['%0.3f %0.3f' % (s[0], s[1]) for s in skip])
        cmd += '-skip %s ' % skip

    if force:
        cmd += '-force '

    if st:
        cmd += '-st '
        cmd += ' %d ' % st_buflen
        cmd += '-corr %0.4f ' % st_corr

    if mv_trans is not None:
        cmd += '-trans %s ' % mv_trans

    if mv_comp:
        cmd += '-movecomp '
        if mv_comp == 'inter':
            cmd += ' inter '

        if mv_headpos:
            cmd += '-headpos '

        if mv_hp is not None:
            cmd += '-hp %s ' % mv_hp

        if mv_hpisubt is not None:
            cmd += 'hpisubt %s ' % mv_hpisubt

        if mv_hpicons:
            cmd += '-hpicons '

    if linefreq is not None:
        cmd += '-linefreq %d ' % linefreq

    if cal is not None:
        cmd += '-cal %s ' % cal

    if ctc is not None:
        cmd += '-ctc %s ' % ctc

    cmd += mx_args

    if overwrite and os.path.exists(out_fname):
        os.remove(out_fname)

    logger.info('Running MaxFilter: %s ' % cmd)
    if os.getenv('_MNE_MAXFILTER_TEST', '') != 'true':  # fake maxfilter
        st = os.system(cmd)
    else:
        print(cmd)  # we can check the output
        st = 0
    if st != 0:
        raise RuntimeError('MaxFilter returned non-zero exit status %d' % st)
    logger.info('[done]')

    return origin
