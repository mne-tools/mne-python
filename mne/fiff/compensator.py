# import numpy as np

from .constants import FIFF


def get_current_comp(info):
    """Get the current compensation in effect in the data
    """
    comp = 0
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


# def findall(L, value, start=0):
#     """Returns indices of all occurence of value in list L starting from start
#     """
#     c = L.count(value)
#     if c == 0:
#         return list()
#     else:
#         ind = list()
#         i = start-1
#         for _ in range(c):
#             i = L.index(value, i+1)
#             ind.append(i)
#         return ind


# def _make_compensator(info, kind):
#     """Auxiliary function for make_compensator
#     """
#     for k in range(len(info['comps'])):
#         if info['comps'][k]['kind'] == kind:
#             this_data = info['comps'][k]['data']
#
#             #   Create the preselector
#             presel = np.zeros((this_data['ncol'], info['nchan']))
#             for col, col_name in enumerate(this_data['col_names']):
#                 ind = findall(info['ch_names'], col_name)
#                 if len(ind) == 0:
#                     raise ValueError, 'Channel %s is not available in data' % \
#                                                                       col_name
#                 elif len(ind) > 1:
#                     raise ValueError, 'Ambiguous channel %s' % col_name
#                 presel[col, ind] = 1.0
#
#             #   Create the postselector
#             postsel = np.zeros((info['nchan'], this_data['nrow']))
#             for c, ch_name in enumerate(info['ch_names']):
#                 ind = findall(this_data['row_names'], ch_name)
#                 if len(ind) > 1:
#                     raise ValueError, 'Ambiguous channel %s' % ch_name
#                 elif len(ind) == 1:
#                     postsel[c, ind] = 1.0
#
#             this_comp = np.dot(postsel, np.dot(this_data['data'], presel))
#             return this_comp
#
#     return []


# def make_compensator(info, from_, to, exclude_comp_chs=False):
#     """ XXX : bug !!! 2 make_compensator functions
#     %
#     % [comp] = mne_make_compensator(info,from,to,exclude_comp_chs)
#     %
#     % info              - measurement info as returned by the fif reading routines
#     % from              - compensation in the input data
#     % to                - desired compensation in the output
#     % exclude_comp_chs  - exclude compensation channels from the output (optional)
#     %
#
#     %
#     % Create a compensation matrix to bring the data from one compensation
#     % state to another
#     %
#     """
#
#     if from_ == to:
#         comp = np.zeros((info['nchan'], info['nchan']))
#         return comp
#
#     if from_ == 0:
#         C1 = np.zeros((info['nchan'], info['nchan']))
#     else:
#         try:
#             C1 = _make_compensator(info, from_)
#         except Exception as inst:
#             raise ValueError, 'Cannot create compensator C1 (%s)' % inst
#
#         if len(C1) == 0:
#             raise ValueError, ('Desired compensation matrix (kind = %d) not'
#                                ' found' % from_)
#
#
#     if to == 0:
#         C2 = np.zeros((info['nchan'], info['nchan']))
#     else:
#         try:
#             C2 = _make_compensator(info, to)
#         except Exception as inst:
#             raise ValueError, 'Cannot create compensator C2 (%s)' % inst
#
#         if len(C2) == 0:
#             raise ValueError, ('Desired compensation matrix (kind = %d) not '
#                                'found' % to)
#
#
#     #   s_orig = s_from + C1*s_from = (I + C1)*s_from
#     #   s_to   = s_orig - C2*s_orig = (I - C2)*s_orig
#     #   s_to   = (I - C2)*(I + C1)*s_from = (I + C1 - C2 - C2*C1)*s_from
#     comp = np.eye(info['nchan']) + C1 - C2 - C2*C1
#
#     if exclude_comp_chs:
#         pick = np.zeros((info['nchan'], info['nchan']))
#         npick = 0
#         for k, chan in info['chs']:
#             if chan['kind'] != FIFF.FIFFV_REF_MEG_CH:
#                 npick += 1
#                 pick[npick] = k
#
#         if npick == 0:
#             raise ValueError, ('Nothing remains after excluding the '
#                                'compensation channels')
#
#         comp = comp[pick[1:npick], :]
#
#     return comp


# def compensate_to(data, to):
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
#         print 'Data are already compensated as desired'
#
#     #   Make the compensator and apply it to all data sets
#     comp = make_compensator(newdata['info'], now, to)
#     for k in range(len(newdata['evoked'])):
#         newdata['evoked'][k]['epochs'] = np.dot(comp,
#                                                 newdata['evoked'][k]['epochs'])
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
