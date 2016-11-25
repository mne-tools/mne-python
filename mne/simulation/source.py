# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

from ..source_estimate import SourceEstimate, VolSourceEstimate
from ..source_space import _ensure_src
from ..utils import check_random_state, warn

from ..externals.six import string_types
from ..externals.six.moves import zip


def select_source_in_label(src, label, random_state=None, location='random',
                           subject=None, subjects_dir=None, surf='sphere'):
    """Select source positions using a label.

    Parameters
    ----------
    src : list of dict
        The source space
    label : Label
        the label (read with mne.read_label)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.
    location : str
        The label location to choose. Can be 'random' (default) or 'center'
        to use :func:`mne.Label.center_of_mass` (restricting to vertices
        both in the label and in the source space). Note that for 'center'
        mode the label values are used as weights.

        .. versionadded:: 0.13

    subject : string | None
        The subject the label is defined for.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    surf : str
        The surface to use for Euclidean distance center of mass
        finding. The default here is "sphere", which finds the center
        of mass on the spherical surface to help avoid potential issues
        with cortical folding.

        .. versionadded:: 0.13

    Returns
    -------
    lh_vertno : list
        selected source coefficients on the left hemisphere
    rh_vertno : list
        selected source coefficients on the right hemisphere
    """
    lh_vertno = list()
    rh_vertno = list()
    if not isinstance(location, string_types) or \
            location not in ('random', 'center'):
        raise ValueError('location must be "random" or "center", got %s'
                         % (location,))

    rng = check_random_state(random_state)
    if label.hemi == 'lh':
        vertno = lh_vertno
        hemi_idx = 0
    else:
        vertno = rh_vertno
        hemi_idx = 1
    src_sel = np.intersect1d(src[hemi_idx]['vertno'], label.vertices)
    if location == 'random':
        idx = src_sel[rng.randint(0, len(src_sel), 1)[0]]
    else:  # 'center'
        idx = label.center_of_mass(
            subject, restrict_vertices=src_sel, subjects_dir=subjects_dir,
            surf=surf)
    vertno.append(idx)
    return lh_vertno, rh_vertno


def simulate_sparse_stc(src, n_dipoles, times,
                        data_fun=lambda t: 1e-7 * np.sin(20 * np.pi * t),
                        labels=None, random_state=None, location='random',
                        subject=None, subjects_dir=None, surf='sphere'):
    """Generate sparse (n_dipoles) sources time courses from data_fun.

    This function randomly selects ``n_dipoles`` vertices in the whole
    cortex or one single vertex (randomly in or in the center of) each
    label if ``labels is not None``. It uses ``data_fun`` to generate
    waveforms for each vertex.

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
    location : str
        The label location to choose. Can be 'random' (default) or 'center'
        to use :func:`mne.Label.center_of_mass`. Note that for 'center'
        mode the label values are used as weights.

        .. versionadded:: 0.13

    subject : string | None
        The subject the label is defined for.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
        Only used with ``location='center'``.

        .. versionadded:: 0.13

    surf : str
        The surface to use for Euclidean distance center of mass
        finding. The default here is "sphere", which finds the center
        of mass on the spherical surface to help avoid potential issues
        with cortical folding.

        .. versionadded:: 0.13

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.

    See Also
    --------
    simulate_raw
    simulate_evoked
    simulate_stc

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    rng = check_random_state(random_state)
    src = _ensure_src(src, verbose=False)
    subject_src = src[0].get('subject_his_id')
    if subject is None:
        subject = subject_src
    elif subject_src is not None and subject != subject_src:
        raise ValueError('subject argument (%s) did not match the source '
                         'space subject_his_id (%s)' % (subject, subject_src))
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
            lh_vertno, rh_vertno = select_source_in_label(
                src, label, rng, location, subject, subjects_dir, surf)
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
    stc = cls(datas, vertices=vs, tmin=tmin, tstep=tstep, subject=subject)
    return stc


def simulate_stc(src, labels, stc_data, tmin, tstep, value_fun=None):
    """Simulate sources time courses from waveforms and labels.

    This function generates a source estimate with extended sources by
    filling the labels with the waveforms given in stc_data.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space
    labels : list of Labels
        The labels
    stc_data : array (shape: len(labels) x n_times)
        The waveforms
    tmin : float
        The beginning of the timeseries
    tstep : float
        The time step (1 / sampling frequency)
    value_fun : function | None
        Function to apply to the label values to obtain the waveform
        scaling for each vertex in the label. If None (default), uniform
        scaling is used.

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.

    See Also
    --------
    simulate_raw
    simulate_evoked
    simulate_sparse_stc
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
    for v, hemi in zip(vertno, ('left', 'right')):
        d = len(v) - len(np.unique(v))
        if d > 0:
            raise RuntimeError('Labels had %s overlaps in the %s hemisphere, '
                               'they must be non-overlapping' % (d, hemi))

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

    subject = src[0].get('subject_his_id')
    stc = SourceEstimate(data, vertices=vertno, tmin=tmin, tstep=tstep,
                         subject=subject)
    return stc
