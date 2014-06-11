import numpy as np

from .constants import FIFF


def get_current_comp(info):
    """Get the current compensation in effect in the data
    """
    comp = None
    first_comp = -1
    for k, chan in enumerate(info['chs']):
        if chan['kind'] == FIFF.FIFFV_MEG_CH:
            comp = int(chan['coil_type']) >> 16
            if first_comp < 0:
                first_comp = comp
            elif comp != first_comp:
                raise ValueError('Compensation is not set equally on '
                                 'all MEG channels')
    return comp


def set_current_comp(info, comp):
    """Set the current compensation in effect in the data
    """
    comp_now = get_current_comp(info)
    for k, chan in enumerate(info['chs']):
        if chan['kind'] == FIFF.FIFFV_MEG_CH:
            rem = chan['coil_type'] - (comp_now << 16)
            chan['coil_type'] = int(rem + (comp << 16))


def _make_compensator(info, kind):
    """Auxiliary function for make_compensator
    """
    for k in range(len(info['comps'])):
        if info['comps'][k]['kind'] == kind:
            this_data = info['comps'][k]['data']

            #   Create the preselector
            presel = np.zeros((this_data['ncol'], info['nchan']))
            for col, col_name in enumerate(this_data['col_names']):
                ind = [k for k, ch in enumerate(info['ch_names'])
                       if ch == col_name]
                if len(ind) == 0:
                    raise ValueError('Channel %s is not available in '
                                     'data' % col_name)
                elif len(ind) > 1:
                    raise ValueError('Ambiguous channel %s' % col_name)
                presel[col, ind[0]] = 1.0

            #   Create the postselector
            postsel = np.zeros((info['nchan'], this_data['nrow']))
            for c, ch_name in enumerate(info['ch_names']):
                ind = [k for k, ch in enumerate(this_data['row_names'])
                       if ch == ch_name]
                if len(ind) > 1:
                    raise ValueError('Ambiguous channel %s' % ch_name)
                elif len(ind) == 1:
                    postsel[c, ind[0]] = 1.0
            this_comp = np.dot(postsel, np.dot(this_data['data'], presel))
            return this_comp

    raise ValueError('Desired compensation matrix (kind = %d) not'
                     ' found' % kind)


def make_compensator(info, from_, to, exclude_comp_chs=False):
    """Returns compensation matrix eg. for CTF system.

    Create a compensation matrix to bring the data from one compensation
    state to another.

    Parameters
    ----------
    info : dict
        The measurement info.
    from_ : int
        Compensation in the input data.
    to : int
        Desired compensation in the output.
    exclude_comp_chs : bool
        Exclude compensation channels from the output.

    Returns
    -------
    comp : array | None.
        The compensation matrix. Might be None if no compensation
        is needed (from == to).
    """
    if from_ == to:
        return None

    if from_ == 0:
        C1 = np.zeros((info['nchan'], info['nchan']))
    else:
        C1 = _make_compensator(info, from_)

    if to == 0:
        C2 = np.zeros((info['nchan'], info['nchan']))
    else:
        C2 = _make_compensator(info, to)

    #   s_orig = s_from + C1*s_from = (I + C1)*s_from
    #   s_to   = s_orig - C2*s_orig = (I - C2)*s_orig
    #   s_to   = (I - C2)*(I + C1)*s_from = (I + C1 - C2 - C2*C1)*s_from
    comp = np.eye(info['nchan']) + C1 - C2 - np.dot(C2, C1)

    if exclude_comp_chs:
        pick = [k for k, c in enumerate(info['chs'])
                if c['kind'] != FIFF.FIFFV_REF_MEG_CH]

        if len(pick) == 0:
            raise ValueError('Nothing remains after excluding the '
                             'compensation channels')

        comp = comp[pick, :]

    return comp


# @verbose
# def compensate_to(data, to, verbose=None):
#     """
#     %
#     % [newdata] = mne_compensate_to(data,to)
#     %
#     % Apply compensation to the data as desired
#     %
#     """
#
#     newdata = data.copy()
#     now = get_current_comp(newdata['info'])
#
#     #   Are we there already?
#     if now == to:
#         logger.info('Data are already compensated as desired')
#
#     #   Make the compensator and apply it to all data sets
#     comp = make_compensator(newdata['info'], now, to)
#     for k in range(len(newdata['evoked'])):
#         newdata['evoked'][k]['epochs'] = np.dot(comp,
#                                               newdata['evoked'][k]['epochs'])
#
#     #  Update the compensation info in the channel descriptors
#     newdata['info']['chs'] = set_current_comp(newdata['info']['chs'], to)
#     return newdata


# def set_current_comp(chs, value):
#     """Set the current compensation value in the channel info structures
#     """
#     new_chs = chs
#
#     lower_half = int('FFFF', 16) # hex2dec('FFFF')
#     for k in range(len(chs)):
#         if chs[k]['kind'] == FIFF.FIFFV_MEG_CH:
#             coil_type = float(chs[k]['coil_type']) & lower_half
#             new_chs[k]['coil_type'] = int(coil_type | (value << 16))
#
#     return new_chs
