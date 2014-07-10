# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from ..externals.six import string_types
import os
from warnings import warn
import logging

import numpy as np
from scipy import optimize, linalg

from ..io import Raw
from ..io.constants import FIFF
from ..utils import logger, verbose
from ..externals.six.moves import map
from ..externals.six.moves import zip


@verbose
def fit_sphere_to_headshape(info, verbose=None):
    """ Fit a sphere to the headshape points to determine head center for
        maxfilter.

    Parameters
    ----------
    info : dict
        Measurement info.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    radius : float
        Sphere radius in mm.
    origin_head: ndarray
        Head center in head coordinates (mm).
    origin_device: ndarray
        Head center in device coordinates (mm).

    """
    # get head digization points, excluding some frontal points (nose etc.)
    hsp = [p['r'] for p in info['dig'] if p['kind'] == FIFF.FIFFV_POINT_EXTRA
           and not (p['r'][2] < 0 and p['r'][1] > 0)]

    if len(hsp) == 0:
        raise ValueError('No head digitization points found')

    hsp = 1e3 * np.array(hsp)

    # initial guess for center and radius
    xradius = (np.max(hsp[:, 0]) - np.min(hsp[:, 0])) / 2
    yradius = (np.max(hsp[:, 1]) - np.min(hsp[:, 1])) / 2

    radius_init = (xradius + yradius) / 2
    center_init = np.array([0.0, 0.0, np.max(hsp[:, 2]) - radius_init])

    # optimization
    x0 = np.r_[center_init, radius_init]
    cost_fun = lambda x, hsp:\
        np.sum((np.sqrt(np.sum((hsp - x[:3]) ** 2, axis=1)) - x[3]) ** 2)

    disp = True if logger.level <= logging.INFO else False
    x_opt = optimize.fmin_powell(cost_fun, x0, args=(hsp,), disp=disp)

    origin_head = x_opt[:3]
    radius = x_opt[3]

    # compute origin in device coordinates
    trans = info['dev_head_t']
    if trans['from'] != FIFF.FIFFV_COORD_DEVICE\
        or trans['to'] != FIFF.FIFFV_COORD_HEAD:
            raise RuntimeError('device to head transform not found')

    head_to_dev = linalg.inv(trans['trans'])
    origin_device = 1e3 * np.dot(head_to_dev,
                                 np.r_[1e-3 * origin_head, 1.0])[:3]

    logger.info('Fitted sphere: r = %0.1f mm' % radius)
    logger.info('Origin head coordinates: %0.1f %0.1f %0.1f mm' %
                (origin_head[0], origin_head[1], origin_head[2]))
    logger.info('Origin device coordinates: %0.1f %0.1f %0.1f mm' %
                (origin_device[0], origin_device[1], origin_device[2]))

    return radius, origin_head, origin_device


def _mxwarn(msg):
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

    """ Apply NeuroMag MaxFilter to raw data.

        Needs Maxfilter license, maxfilter has to be in PATH

    Parameters
    ----------
    in_fname : string
        Input file name

    out_fname : string
        Output file name

    origin : array-like or string
        Head origin in mm. If None it will be estimated from headshape points.

    frame : string ('device' or 'head')
        Coordinate frame for head center

    bad : string, list (or None)
        List of static bad channels. Can be a list with channel names, or a
        string with channels (names or logical channel numbers)

    autobad : string ('on', 'off', 'n')
        Sets automated bad channel detection on or off

    skip : string or a list of float-tuples (or None)
        Skips raw data sequences, time intervals pairs in sec,
        e.g.: 0 30 120 150

    force : bool
        Ignore program warnings

    st : bool
        Apply the time-domain MaxST extension

    st_buflen : float
        MaxSt buffer length in sec (disabled if st is False)

    st_corr : float
        MaxSt subspace correlation limit (disabled if st is False)

    mv_trans : string (filename or 'default') (or None)
        Transforms the data into the coil definitions of in_fname, or into the
        default frame (None: don't use option)

    mv_comp : bool (or 'inter')
        Estimates and compensates head movements in continuous raw data

    mv_headpos : bool
        Estimates and stores head position parameters, but does not compensate
        movements (disabled if mv_comp is False)

    mv_hp : string (or None)
        Stores head position data in an ascii file
        (disabled if mv_comp is False)

    mv_hpistep : float (or None)
        Sets head position update interval in ms (disabled if mv_comp is False)

    mv_hpisubt : string ('amp', 'base', 'off') (or None)
        Subtracts hpi signals: sine amplitudes, amp + baseline, or switch off
        (disabled if mv_comp is False)

    mv_hpicons : bool
        Check initial consistency isotrak vs hpifit
        (disabled if mv_comp is False)

    linefreq : int (50, 60) (or None)
        Sets the basic line interference frequency (50 or 60 Hz)
        (None: do not use line filter)

    cal : string
        Path to calibration file

    ctc : string
        Path to Cross-talk compensation file

    mx_args : string
        Additional command line arguments to pass to MaxFilter

    overwrite : bool
        Overwrite output file if it already exists

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).


    Returns
    -------
    origin: string
        Head origin in selected coordinate frame
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
        raw = Raw(in_fname)
        r, o_head, o_dev = fit_sphere_to_headshape(raw.info)
        raw.close()
        logger.info('[done]')
        if frame == 'head':
            origin = o_head
        elif frame == 'device':
            origin = o_dev
        else:
            RuntimeError('invalid frame for origin')

    if not isinstance(origin, string_types):
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
    st = os.system(cmd)
    if st != 0:
        raise RuntimeError('MaxFilter returned non-zero exit status %d' % st)
    logger.info('[done]')

    return origin
