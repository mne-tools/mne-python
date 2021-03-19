import numpy as np

from .event import make_fixed_length_events
from .utils import (check_version)


def lds_raw(raw, win_size=0.5, step_size=0, l2_penalty=None,
            normalize=False, solver='auto', random_state=None,
            sample_weight=None, return_times=False):
    """Compute sliding window linear dynamical system of Raw data.

    Takes a sliding window, possibly with overlap and computes
    a model of the data as:

        x(t+1) = Ax(t)

    The ``A`` matrix is computed for each window of the entire
    Raw object.

    Parameters
    ----------
    raw : instance of Raw
    win_size : float
    step_size : float
    l2_penalty :
    normalize :
    solver :
    random_state :
    sample_weight :
    return_times : bool
        Whether to return the time points of the associated
        ``lds_seq``. Will treat the middle point of each window
        as the time in seconds.

    Returns
    -------
    lds_seq : np.ndarray (n_chs, n_chs, n_windows)
        The sequence of ``A`` matrix representing the
        time-varying linear dynamical system.

    References
    ----------
    - Li A, Gunnarsdottir KM, Inati S, Zaghloul K, Gale J, Bulacio J,
        Martinez-Gonzalez J, Sarma SV. Linear time-varying model characterizes
        invasive EEG signals generated from complex epileptic networks.
        Annu Int Conf IEEE Eng Med Biol Soc. 2017 Jul;2017:2802-2805.
        doi: 10.1109/EMBC.2017.8037439. PMID: 29060480; PMCID: PMC7294983.
    """
    from sklearn.linear_model import Ridge

    check_ok = check_version('sklearn', '0.24')
    if not check_ok:
        raise RuntimeError(f'In order to run this function, '
                           f'scikit-learn v0.24 and above must be installed.')

    # initialize the least-squares model engine via sklearn
    clf = Ridge(
        alpha=l2_penalty,
        normalize=normalize,
        solver=solver,
        random_state=random_state,
    )

    # generate epochs based on window size and step size
    events = make_fixed_length_events(raw, duration=win_size, overlap=step_size)
    n_chs = len(raw.ch_names)

    lds_seq = np.zeros((n_chs, n_chs, len(events)))
    for idx, (start_sample, duration, _) in enumerate(events):
        # obtain a data window of all channels
        end_sample = start_sample + duration
        snaps = raw.get_data(start=start_sample, end=end_sample)

        # setup time-shifted matrices
        X, Y = snaps[:, :-1], snaps[:, 1:]

        # n_samples X n_features and n_samples X n_targets
        clf.fit(X.T, Y.T, sample_weight=sample_weight)

        # n_targets X n_features
        A = clf.coef_
        lds_seq[..., idx] = A
    return lds_seq
