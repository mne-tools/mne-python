# Authors: Lukas Breuer <l.breuer@fz-juelich.de>
#          Juergen Dammers <j.dammers@fz-juelich.de>
#          Denis A. Engeman <denis.engemann@gemail.com>
#
# License: BSD (3-clause)

import copy
import numpy as np

from ..utils import logger, verbose


@verbose
def infomax(data, w_init=None, learning_rate=None, block=None,
            w_change=1e-12, anneal_deg=60., anneal_step=0.9,
            extended=True, max_iter=200, random_state=None,
            verbose=True):

    """Run the (extended) Infomax ICA decomposition on raw data

    Parameters
    ----------
    data : np.ndarray, shape (n_features, n_times)
        The data to unmix.
    w_init : np.ndarray, shape (n_features, n_features)
        The initialized unmixing matrix. Defaults to None. If None, the
        identity matrix is used.
    learning_rate : float
        This quantity indicates the relative size of the change in weights.
        Note. Smaller learining rates will slow down the procedure.
        Defaults to 0.010d / alog(n_features ^ 2.0)
    block : int
        The block size of randomly chosen data segment.
        Defaults to floor(sqrt(n_times / 3d))
    w_change : float
        The change at which to stop iteration. Defaults to 1e-12.
    anneal_deg : float
        The angle at which (in degree) the learning rate will be reduced.
        Defaults to 60.0
    anneal_step : float
        The factor by which the learning rate will be reduced once
        ``anneal_deg`` is exceeded:
            learning_rate *= anneal_step
        Defaults to 0.9
    extended : bool
        Wheather to use the extended infomax algorithm or not. Defaults to
        True.
    max_iter : int
        The maximum number of iterations. Defaults to 200.
    verbose : bool, str, int, or None
        if not None, override default verbose level (see mne.verbose).

    Returns
    -------
    unmixing_matrix : np.ndarray of float, shape (n_features, n_features)
        The linear unmixing operator.
    """
    if random_state is None:
        seed = 42
        rng = np.random.RandomState(seed=seed)
    elif isinstance(random_state, int):
        seed = random_state
        rng = np.random.RandomState(seed=seed)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    rng2 = copy.deepcopy(rng)  # the other gets updated each iteration

    import math
    if extended is True:
        from scipy.stats import kurtosis

    # define some default parameter
    default_max_weight = 1e8
    default_restart_fac = 0.9
    default_min_learning_rate = 1e-10
    default_blowup = 1e4
    default_blowup_fac = 0.5
    default_nsmall_angle = 20
    degconst = 180.0 / np.pi

    # for extended Infomax
    if extended is True:
        default_kurtsize = 6000
        default_extmomentum = 0.5
        default_signsbias = 0.02
        default_signcount_threshold = 25
        default_signcount_step = 2
        default_n_sub = 1

    # check data shape
    n_times, n_features = data.shape

    if n_features < 2 or n_times < n_features:
        raise ValueError('Number of components is to small')
    n_features_square = n_features ** 2

    # check input parameter
    # heuristic default - may need adjustment for
    # large or tiny data sets
    if learning_rate is None:
        learning_rate = 0.01 / math.log(n_features ** 2.0)

    if block is None:
        block = int(math.floor(math.sqrt(n_times / 3.0)))

    # collect parameter
    n_blocks = n_times // block
    lastt = (n_blocks - 1) * block + 1

    # initialize training
    if w_init is None:
        # initialize weights as identity matrix
        weights = np.identity(n_features, dtype=np.float64)
    else:
        weights = w_init

    BI = block * np.identity(n_features, dtype=np.float64)
    bias = np.zeros((n_features, 1), dtype=np.float64)
    onesrow = np.ones((1, block), dtype=np.float64)
    startweights = weights.copy()
    oldweights = startweights.copy()
    step = 0
    count_small_angle = 0
    wts_blowup = False
    blockno = 0

    # for extended Infomax
    if extended:
        signs = np.diag(-1. * np.ones(n_features))
        extblocks = 1
        signcount = 0
        n_sub = default_n_sub
        if default_kurtsize < n_times:
            kurtsize = default_kurtsize
        else:
            kurtsize = n_times
        extmomentum = default_extmomentum
        signsbias = default_signsbias
        signcount_threshold = default_signcount_threshold
        signcount_step = default_signcount_step
        old_kurt = np.zeros(n_features, dtype=np.float64)
        oldsigns = np.zeros((n_features, n_features))

    # trainings loop
    olddelta, oldchange = 1., 0.
    while step < max_iter:

        # shuffel data at each step
        rng.seed(step)  # --> permutation is fixed but differs at each step
        permute = range(n_times)
        rng.shuffle(permute)

        # ICA training block
        # loop across block samples
        for t in range(0, lastt, block):
            u = np.dot(data[permute[t:t + block], :],
                       weights) + np.dot(bias, onesrow).T

            if extended:
                # extended ICA update
                y = np.tanh(u)
                weights += learning_rate * np.dot(weights, BI -
                                                  np.dot(np.dot(u.T, y), signs)
                                                  - np.dot(u.T, u))
                bias += (learning_rate * (np.sum(y, axis=0, dtype=np.float64)
                         * -2)).reshape(n_features, 1)

            else:
                # logistic ICA weights update
                y = 1.0 / (1.0 + np.exp(-u))
                weights += learning_rate * np.dot(weights, BI +
                                                  np.dot(u.T, (1.0 - 2.0 * y)))
                bias += ((learning_rate * np.sum((1.0 - 2.0 * y), axis=0,
                                                 dtype=np.float64))
                         .reshape(n_features, 1))

            # check change limit
            max_weight_val = np.max(np.abs(weights))
            if max_weight_val > default_max_weight:
                wts_blowup = True

            blockno += 1
            if wts_blowup:
                break

            # ICA kurtosis estimation
            if extended:

                n = np.fix(blockno / extblocks)

                if np.abs(n) * extblocks == blockno:
                    if kurtsize < n_times:
                        rp = np.floor(rng2.uniform(0, 1, kurtsize) *
                                      (n_times - 1))
                        tpartact = np.dot(data[rp.astype(int), :], weights).T
                    else:
                        tpartact = np.dot(data, weights).T

                    # estimate kurtosis
                    kurt = kurtosis(tpartact, axis=1, fisher=True)

                    if extmomentum != 0:
                        kurt = (extmomentum * old_kurt + (1.0 - extmomentum) *
                                kurt)
                        old_kurt = kurt

                    # estimate weighted signs
                    sings_tmp = (kurt + signsbias) / np.abs(kurt + signsbias)
                    signs.flat[::n_features + 1] = sings_tmp

                    ndiff = ((signs.flat[::n_features + 1] -
                              oldsigns.flat[::n_features + 1]) != 0).sum()
                    if ndiff == 0:
                        signcount += 1
                    else:
                        signcount = 0
                    oldsigns = signs

                    if signcount >= signcount_threshold:
                        extblocks = np.fix(extblocks * signcount_step)
                        signcount = 0

        # here we continue after the for
        # loop over the ICA training blocks
        # if weights in bounds:
        if not wts_blowup:
            oldwtchange = weights - oldweights
            step += 1
            angledelta = 0.0
            delta = oldwtchange.reshape(1, n_features_square)
            change = np.sum(delta * delta, dtype=np.float64)
            # XXX debug
            info = "       ...step %4d of %4d: lrate = %g, wchange = %g, max weight val = %0.2f" \
                   % (step, max_iter, learning_rate, change, max_weight_val)
            print(info)
            if step > 1:
                angledelta = math.acos(np.sum(delta * olddelta) /
                                       math.sqrt(change * oldchange))
                angledelta *= degconst

            # anneal learning rate
            oldweights = weights.copy()

            if angledelta > anneal_deg:
                learning_rate *= anneal_step  # anneal learning rate
                olddelta = delta  # accumulate angledelta until anneal_deg ...
                oldchange = change  # ... reached learning_rates
                count_small_angle = 0  # reset count when angle delta is large
            else:
                if step == 1:  # on first step only
                    olddelta = delta  # initialize
                    oldchange = change
                count_small_angle += 1
                if count_small_angle > default_nsmall_angle:
                    max_iter = step

            # apply stopping rule
            if step > 2 and change < w_change:
                step = max_iter
            elif change > default_blowup:
                learning_rate *= default_blowup_fac

        # restart if weights blow up
        # (for lowering learning_rate)
        else:
            step = 0  # start again
            wts_blowup = 0  # re-initialize variables
            blockno = 1
            learning_rate *= default_restart_fac  # with lower learning rate
            weights = startweights.copy()
            oldweights = startweights.copy()
            olddelta = np.zeros((1, n_features_square), dtype=np.float64)
            bias = np.zeros((n_features, 1), dtype=np.float64)

            # for extended Infomax
            if extended:
                signs = np.identity(n_features)
                signs.flat[::n_sub + 1] = -1
                oldsigns = np.zeros((n_features, n_features))

            if learning_rate <= default_min_learning_rate:
                raise ValueError('Error in Infomax ICA: weight matrix may not '
                                 'be invertible!')

    # keep in mind row/col convention outside this routine

    return weights.T
