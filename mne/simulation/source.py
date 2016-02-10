# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

from ..source_estimate import SourceEstimate, VolSourceEstimate
from ..source_space import (_ensure_src, _make_discrete_source_space,
                            SourceSpaces)
from ..utils import check_random_state, warn
from ..externals.six.moves import zip
from ..forward import make_forward_solution, convert_forward_solution
from ..fixes import in1d

def select_source_in_label(src, label, random_state=None):
    """Select source positions using a label

    Parameters
    ----------
    src : list of dict
        The source space
    label : Label
        the label (read with mne.read_label)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    lh_vertno : list
        selected source coefficients on the left hemisphere
    rh_vertno : list
        selected source coefficients on the right hemisphere
    """
    lh_vertno = list()
    rh_vertno = list()

    rng = check_random_state(random_state)

    if label.hemi == 'lh':
        src_sel_lh = np.intersect1d(src[0]['vertno'], label.vertices)
        idx_select = rng.randint(0, len(src_sel_lh), 1)
        lh_vertno.append(src_sel_lh[idx_select][0])
    else:
        src_sel_rh = np.intersect1d(src[1]['vertno'], label.vertices)
        idx_select = rng.randint(0, len(src_sel_rh), 1)
        rh_vertno.append(src_sel_rh[idx_select][0])

    return lh_vertno, rh_vertno


def simulate_sparse_stc(src, n_dipoles, times,
                        data_fun=lambda t: 1e-7 * np.sin(20 * np.pi * t),
                        labels=None, random_state=None):
    """Generate sparse (n_dipoles) sources time courses from data_fun

    This function randomly selects n_dipoles vertices in the whole cortex
    or one single vertex in each label if labels is not None. It uses data_fun
    to generate waveforms for each vertex.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space.
    n_dipoles : int
        Number of dipoles to simulate.
    times : array
        Time array
    data_fun : callable
        Function to generate the waveforms. The default is a 100 nAm, 10 Hz
        sinusoid as ``1e-7 * np.sin(20 * pi * t)``. The function should take
        as input the array of time samples in seconds and return an array of
        the same length containing the time courses.
    labels : None | list of Labels
        The labels. The default is None, otherwise its size must be n_dipoles.
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    rng = check_random_state(random_state)
    src = _ensure_src(src, verbose=False)
    data = np.zeros((n_dipoles, len(times)))
    for i_dip in range(n_dipoles):
        data[i_dip, :] = data_fun(times)

    if labels is None:
        # can be vol or surface source space
        offsets = np.linspace(0, n_dipoles, len(src) + 1).astype(int)
        n_dipoles_ss = np.diff(offsets)
        # don't use .choice b/c not on old numpy
        vs = [s['vertno'][np.sort(rng.permutation(np.arange(s['nuse']))[:n])]
              for n, s in zip(n_dipoles_ss, src)]
        datas = data
    else:
        if n_dipoles != len(labels):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % min(n_dipoles, len(labels)))
        labels = labels[:n_dipoles] if n_dipoles < len(labels) else labels

        vertno = [[], []]
        lh_data = [np.empty((0, data.shape[1]))]
        rh_data = [np.empty((0, data.shape[1]))]
        for i, label in enumerate(labels):
            lh_vertno, rh_vertno = select_source_in_label(src, label, rng)
            vertno[0] += lh_vertno
            vertno[1] += rh_vertno
            if len(lh_vertno) != 0:
                lh_data.append(data[i][np.newaxis])
            elif len(rh_vertno) != 0:
                rh_data.append(data[i][np.newaxis])
            else:
                raise ValueError('No vertno found.')
        vs = [np.array(v) for v in vertno]
        datas = [np.concatenate(d) for d in [lh_data, rh_data]]
        # need to sort each hemi by vertex number
        for ii in range(2):
            order = np.argsort(vs[ii])
            vs[ii] = vs[ii][order]
            if len(order) > 0:  # fix for old numpy
                datas[ii] = datas[ii][order]
        datas = np.concatenate(datas)

    tmin, tstep = times[0], np.diff(times[:2])[0]
    assert datas.shape == data.shape
    cls = SourceEstimate if len(vs) == 2 else VolSourceEstimate
    stc = cls(datas, vertices=vs, tmin=tmin, tstep=tstep)
    return stc


def simulate_stc(src, labels, stc_data, tmin, tstep, value_fun=None):
    """Simulate sources time courses from waveforms and labels

    This function generates a source estimate with extended sources by
    filling the labels with the waveforms given in stc_data.

    By default, the vertices within a label are assigned the same waveform.
    The waveforms can be scaled for each vertex by using the label values
    and value_fun. E.g.,

    # create a source label where the values are the distance from the center
    labels = circular_source_labels('sample', 0, 10, 0)

    # sources with decaying strength (x will be the distance from the center)
    fun = lambda x: exp(- x / 10)
    stc = generate_stc(fwd, labels, stc_data, tmin, tstep, fun)

    Parameters
    ----------
    src : list of dict
        The source space
    labels : list of Labels
        The labels
    stc_data : array (shape: len(labels) x n_times)
        The waveforms
    tmin : float
        The beginning of the timeseries
    tstep : float
        The time step (1 / sampling frequency)
    value_fun : function
        Function to apply to the label values

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.
    """
    if len(labels) != len(stc_data):
        raise ValueError('labels and stc_data must have the same length')

    vertno = [[], []]
    stc_data_extended = [[], []]
    hemi_to_ind = {'lh': 0, 'rh': 1}
    for i, label in enumerate(labels):
        hemi_ind = hemi_to_ind[label.hemi]
        src_sel = np.intersect1d(src[hemi_ind]['vertno'],
                                 label.vertices)
        if value_fun is not None:
            idx_sel = np.searchsorted(label.vertices, src_sel)
            values_sel = np.array([value_fun(v) for v in
                                   label.values[idx_sel]])

            data = np.outer(values_sel, stc_data[i])
        else:
            data = np.tile(stc_data[i], (len(src_sel), 1))

        vertno[hemi_ind].append(src_sel)
        stc_data_extended[hemi_ind].append(np.atleast_2d(data))

    # format the vertno list
    for idx in (0, 1):
        if len(vertno[idx]) > 1:
            vertno[idx] = np.concatenate(vertno[idx])
        elif len(vertno[idx]) == 1:
            vertno[idx] = vertno[idx][0]
    vertno = [np.array(v) for v in vertno]

    # the data is in the order left, right
    data = list()
    if len(vertno[0]) != 0:
        idx = np.argsort(vertno[0])
        vertno[0] = vertno[0][idx]
        data.append(np.concatenate(stc_data_extended[0])[idx])

    if len(vertno[1]) != 0:
        idx = np.argsort(vertno[1])
        vertno[1] = vertno[1][idx]
        data.append(np.concatenate(stc_data_extended[1])[idx])

    data = np.concatenate(data)

    stc = SourceEstimate(data, vertices=vertno, tmin=tmin, tstep=tstep)
    return stc


def make_forward_dipole(dipole, bem, info, trans=None, n_jobs=1, verbose=None):
    """Convert dipole object to source estimate and calculate forward operator

    The instance of Dipole is converted to a discrete volume source space,
    which is then combined with a BEM (can also be a sphere model dict) and
    the sensor information in info to form a forward operator.

    The source estimate object (with the forward operator) can be projected to
    sensor-space using :func:`mne.simulation.evoked.simulate_evoked`.

    Parameters
    ----------
    dipole : Dipole object
        Instance of Dipole containing position, orientation and amplitude of
        one or more dipoles. Several dipoles may occur simultaneously, but
        when more than one time point is defined, the spacing between them
        (i.e. the sampling rate) must be constant.
    bem : str | dict
        The BEM filename (str) or a loaded sphere model (dict).
    info : dict
        Sensor-info etc., e.g., from real evoked file.
    trans : str | None
        The head<->MRI transform filename. Must be provided unless BEM
        is a sphere model.
    n_jobs : int
        Number of jobs to run in parallel (used in making forward solution).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : instance of VolSourceEstimate
        The dipoles converted to a discrete set of points and associated
        time courses.
    fwd : instance of Forward
        The forward solution.

    See Also
    --------
    mne.simulation.simulate_evoked

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Make copies to avoid mangling original dipole
    times = dipole.times.copy()
    pos = dipole.pos.copy()
    amplitude = dipole.amplitude.copy()
    ori = dipole.ori.copy()

    # Convert positions to discrete source space (allows duplicate rr & nn)
    # NB information about dipole orientation enters here, then no more
    sources = dict(rr=pos, nn=ori)
    # Dipole objects must be in the head frame
    sp = _make_discrete_source_space(sources, coord_frame='head')
    src = SourceSpaces([sp])  # dict with working_dir, command_line not nec

    # Forward operator created for channels in info (use pick_info to restrict)
    # Use defaults for most params, including min_dist
    fwd = make_forward_solution(info, trans, src, bem, fname=None,
                                n_jobs=n_jobs, verbose=verbose)
    # Convert from free orientations to fixed (in-place)
    convert_forward_solution(fwd, surf_ori=False, force_fixed=True,
                             copy=False, verbose=None)

    # Check for omissions due to proximity to inner skull in
    # make_forward_solution, which will result in an exception
    if fwd['src'][0]['nuse'] != len(pos):
        inuse = fwd['src'][0]['inuse'].astype(np.bool)
        head = ('The following dipoles are outside the inner skull boundary')
        msg = len(head)*'#' + '\n' + head + '\n'
        for (t, pos) in zip(times[np.logical_not(inuse)],
                            pos[np.logical_not(inuse)]):
            msg += '    t={:.0f} ms, pos=({:.0f}, {:.0f}, {:.0f}) mm\n'.\
                format(t*1000., pos[0]*1000., pos[1]*1000., pos[2]*1000.)
        msg += len(head)*'#'
        print(msg)
        raise ValueError('One or more dipoles outside the inner skull')

    # multiple dipoles (rr and nn) per time instant allowed
    # uneven sampling in time returns list
    timepoints = np.unique(times)
    if len(timepoints) > 1:
        tdiff = np.diff(timepoints)
        if not np.allclose(tdiff, tdiff[0]):
            warn('Unique time points of dipoles unevenly spaced: returned '
                 'stc will be a list, one for each time point.')
            tstep = -1.0
        else:
            tstep = tdiff[0]
    elif len(timepoints) == 1:
        tstep = 0.001

    # Build the data matrix, essentially a block-diagonal with
    # n_rows: number of dipoles in total (dipole.amplitudes)
    # n_cols: number of unique time points in dipole.times
    # amplitude with identical value of times go together in one col (others=0)
    data = np.zeros((len(amplitude), len(timepoints)))  # (n_d, n_t)
    row = 0
    for tpind, tp in enumerate(timepoints):
        amp = amplitude[in1d(times, tp)]
        data[row:row+len(amp), tpind] = amp
        row += len(amp)

    if tstep > 0:
        stc = VolSourceEstimate(data, vertices=fwd['src'][0]['vertno'],
                                tmin=timepoints[0],
                                tstep=tstep, subject=None)
    else:  # Must return a list of stc, one for each time point
        stc = []
        for col, tp in enumerate(timepoints):
            stc += [VolSourceEstimate(data[:, col][:, np.newaxis],
                                      vertices=fwd['src'][0]['vertno'],
                                      tmin=tp, tstep=0.001, subject=None)]
    return stc, fwd
