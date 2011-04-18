# import numpy as np
#
# from .fiff.proj import make_projector_info
# from .fiff.compensator import get_current_comp
# from .fiff.compensator import compensate_to, make_compensator

# XXX

# def cancel_noise(data, dest_comp=0):
#     """Do projection and compensation as needed
#
#        Return the appropriate operators
#
#        [res,proj,comp] = mne_ex_cancel_noise(data,dest_comp)
#
#        res     - Data after noise cancellation
#        proj    - The projection operator applied
#        comp    - The compensator which brings uncompensated data to the
#                  desired compensation grade (will be useful in forward
#                  calculations)
#
#     """
#     #
#     #   Compensate the data and make a compensator for forward modelling
#     #
#     comp = []
#     proj = []
#     comp_now = get_current_comp(data['info'])
#     if comp_now == dest_comp:
#         res = data
#     else:
#         res = compensate_to(data, dest_comp)
#         print 'The data are now compensated to grade %d.' % dest_comp
#
#     if dest_comp > 0:
#         comp = make_compensator(res['info'], 0, dest_comp)
#         print 'Appropriate forward operator compensator created.'
#     else:
#         print 'No forward operator compensator needed.'
#
#     #   Do the projection
#     if data['info']['projs'] is None:
#         print 'No projector included with these data.'
#     else:
#         #   Activate the projection items
#         for k in range(len(res['info']['projs'])):
#             res['info']['projs'][k]['active'] = True;
#
#         #   Create the projector
#         proj, nproj = make_projector_info(res['info'])
#         if nproj == 0:
#             print 'The projection vectors do not apply to these channels'
#             proj = []
#         else:
#             print 'Created an SSP operator (subspace dimension = %d)' % nproj
#             res['evoked']['epochs'] = np.dot(proj, res['evoked']['epochs'])
#             print 'Projector applied to the data'
#
#     return res, proj, comp
