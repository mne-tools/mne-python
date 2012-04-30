# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import subprocess
from warnings import warn

import numpy as np
import scipy as sp
from scipy.optimize import fmin_powell

from ..fiff import Raw
from ..fiff.constants import FIFF


def fit_sphere_to_headshape(info):
    """ Fit a sphere to the headshape points to determine head center for
        maxfilter.

    Parameters
    ----------
    info: dict
        Measurement info

    Returns
    -------
    radius : float
        Sphere radius in mm

    origin_head: ndarray
        Head center in head coordinates (mm)

    origin_device: ndarray
        Head center in device coordinates (mm)

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

    x_opt = fmin_powell(cost_fun, x0, args=(hsp,))

    origin_head = x_opt[:3]
    radius = x_opt[3]

    # compute origin in device coordinates
    trans = info['dev_head_t']
    if trans['from'] != FIFF.FIFFV_COORD_DEVICE\
        or trans['to'] != FIFF.FIFFV_COORD_HEAD:
            raise RuntimeError('device to head transform not found')

    head_to_dev = sp.linalg.inv(trans['trans'])
    origin_device = 1e3 * np.dot(head_to_dev,
                                 np.r_[1e-3 * origin_head, 1.0])[:3]

    print 'Fitted sphere: r = %0.1f mm' % radius
    print ('Origin head coordinates: %0.1f %0.1f %0.1f mm' %
           (origin_head[0], origin_head[1], origin_head[2]))
    print ('Origin device coordinates: %0.1f %0.1f %0.1f mm' %
           (origin_device[0], origin_device[1], origin_device[2]))

    return radius, origin_head, origin_device


def apply_maxfilter(in_fname, out_fname, origin=None, frame='device',
                    in_order=None, out_order=None, bad=None, autobad=None,
                    badlimit=None, skip=None, data_format=None, force=False,
                    st=False, st_buflen=None, st_corr=None, mv_trans=None,
                    mv_comp=False, mv_headpos=False, mv_hp=None,
                    mv_hpistep=None, mv_hpisubt=None, mv_hpicons=False,
                    linefreq=None, lpfilt=None, site=None, cal=None,
                    ctc=None, magbad=False, regularize=None, iterate=None,
                    ds=None):

    """ Apply NeuroMag MaxFilter to raw data.

        Needs Maxfilter license, maxfilter has to be in PATH

    Parameters:
    -----------
    in_fname: string
        Input file name

    out_fname: string
        Output file name

    origin: ndarray
        Head origin in mm. If None it will be estimated from headshape points.

    frame: string ('device' or 'head')
        Coordinate frame for head center

    in_order: int (or None)
        Order of the inside expansion (None: use default)

    out_order: int (or None)
        Order of the outside expansion (None: use default)

    bad: string (or None)
        List of static bad channels (logical chnos, e.g.: 0323 1042 2631)

    autobad: string ('on', 'off', 'n') (or None)
        Sets automated bad channel detection on or off (None: use default)

    badlimit: float (or None)
        Threshold for bad channel detection (>ave+x*SD) (None: use default)

    skip: string (or None)
        Skips raw data sequences, time intervals pairs in sec,
        e.g.: 0 30 120 150

    data_format: string ('short', 'long', 'float') (or None)
        Output data format (None: use default)

    force: bool
        Ignore program warnings

    st: bool
        Apply the time-domain MaxST extension

    st_buflen: float (or None)
        MaxSt buffer length in sec (None: use default)

    st_corr: float (or None)
        MaxSt subspace correlation limit (None: use default)

    mv_trans: string (filename or 'default') (or None)
        Transforms the data into the coil definitions of in_fname, or into the
        default frame (None: don't use option)

    mv_comp: bool (or 'inter')
        Estimates and compensates head movements in continuous raw data

    mv_headpos: bool
        Estimates and stores head position parameters, but does not compensate
        movements

    mv_hp: string (or None)
        Stores head position data in an ascii file

    mv_hpistep: float (or None)
        Sets head position update interval in ms (None: use default)

    mv_hpisubt: string ('amp', 'base', 'off') (or None)
        Subtracts hpi signals: sine amplitudes, amp + baseline, or switch off
        (None: use default)

    mv_hpicons: bool
        Check initial consistency isotrak vs hpifit

    linefreq: int (50, 60) (or None)
        Sets the basic line interference frequency (50 or 60 Hz)
        (None: do not use line filter)

    lpfilt: float (or None or True)
        Corner frequency for IIR low-pass filtering
        (None: don't use option: True: use default frequency)

    site: string (or None)
        Loads sss_cal_<name>.dat and ct_sparse_<name>.fif

    cal: string (filename or 'off') (or None)
        Uses the fine-calibration in <filename>, or switch off.

    ctc: string (filename or 'off') (or None)
        Uses the cross-talk matrix in <filename>, or switch off

    magbad: bool
        Marks all magnetometers bad

    regularize: bool (or None)
        Sets the component selection on or off (None: use default)

    iterate: int (or None)
        Uses iterative pseudo-inverse, n iteration loops default n=10;
        n=0 forces direct pseudo-inverse. (None: use default)

    ds: int (or None)
        Applies downsampling with low-pass FIR filtering f is optional
        downsampling factor (None: don't use option)

    Returns
    -------
    origin: ndarray
        Head origin in selected coordinate frame (mm)
    """

    # check for possible maxfilter bugs
    def _mxwarn(msg):
        warn('Possible MaxFilter bug: %s, more info: '
             'http://imaging.mrc-cbu.cam.ac.uk/meg/maxbugs')

    if mv_trans is not None and mv_comp is not None:
        _mxwarn("Don't use '-trans' with head-movement compensation "
                "'-movecomp'")

    if autobad is not None and (mv_headpos is not None or mv_comp is not None):
        _mxwarn("Don't use '-autobad' with head-position estimation "
                "'-headpos' or movement compensation '-movecomp'")

    if st and autobad is not None:
        _mxwarn("Don't use '-autobad' with '-st' option")

    if lpfilt is not None:
        _mxwarn("Don't use '-lpfilt' to low-pass filter data")

    # determine the head origin if necessary
    if origin is None:
        print 'Estimating head origin from headshape points..'
        raw = Raw(in_fname)
        r, o_head, o_dev = fit_sphere_to_headshape(raw.info)
        raw.close()
        print '[done]'
        if frame == 'head':
            origin = o_head
        elif frame == 'device':
            origin = o_dev
        else:
            RuntimeError('invalid frame for origin')

    # format command
    cmd = ('maxfilter -f %s -o %s -frame %s -origin %d %d %d '
           % (in_fname, out_fname, frame, origin[0], origin[1], origin[2]))

    if in_order is not None:
        cmd += '-in %d ' % in_order

    if out_order is not None:
        cmd += '-out %d ' % out_order

    if bad is not None:
        cmd += '-bad %s ' % bad

    if autobad is not None:
        cmd += '-autobad %s ' % autobad

    if badlimit is not None:
        cmd += '-badlimit %0.4f ' % badlimit

    if skip is not None:
        cmd += '-skip %s ' % skip

    if data_format is not None:
        cmd += '-format %s ' % data_format

    if force:
        cmd += '-force '

    if st:
        cmd += '-st '

    if st_buflen is not None:
        if not st:
            raise RuntimeError('st_buflen cannot be used if st == False')
        cmd += ' %d ' % st_buflen

    if st_corr is not None:
        if not st:
            raise RuntimeError('st_corr cannot be used if st == False')
        cmd += '-corr %0.4f ' % st_corr

    if mv_trans is not None:
        cmd += '-trans %s ' % mv_trans

    if mv_comp is not None:
        cmd += '-movecomp '
        if mv_comp == 'inter':
            cmd += ' inter '

    if mv_headpos:
        cmd += '-headpos '

    if mv_hp:
        cmd += '-hp %s' % mv_hp

    if mv_hpisubt is not None:
        cmd += 'hpisubt %s ' % mv_hpisubt

    if mv_hpicons:
        cmd += '-hpicons '

    if linefreq is not None:
        cmd += '-linefreq %d ' % linefreq

    if lpfilt is not None:
        cmd += '-lpfilt '
        if not isinstance(lpfilt, bool):
            cmd += '%0.1f ' % lpfilt

    if site is not None:
        cmd += '-site %s ' % site

    if cal is not None:
        cmd += '-cal %s ' % cal

    if ctc is not None:
        cmd += '-ctc %s ' % ctc

    if magbad:
        cmd += '-magbad '

    if regularize is not None:
        if regularize:
            cmd += '-regularize on '
        else:
            cmd += '-regularize off '

    if iterate is not None:
        cmd += '-iterate %d' % iterate

    if ds is not None:
        cmd += '-ds %d ' % ds

    print 'Running MaxFilter: %s ' % cmd
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    print out
    print '[done]'
