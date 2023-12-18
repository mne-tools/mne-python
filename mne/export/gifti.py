# Authors: MNE Developers
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import nibabel as nib
import numpy as np

def _get_decimated_surfaces(src):
    """Helper function to make exporting GIFTI surfaces easier
    Parameters
    -----------
    src : instance of SourceSpaces
    Returns
    -----------
    surfaces : list of dict
        The decimated surfaces present in the source space. Each dict
        which contains 'rr' and 'tris' keys for vertices positions and
        triangle indices.
    Notes
        .. versionadded:: 1.7
    """
    surfaces = []
    for s in src:
        if s['type'] != 'surf':
            continue
        rr = s['rr']
        use_tris = s['use_tris']
        vertno = s['vertno']
        ss = {}
        ss['rr'] = rr[vertno]
        reindex = np.full(len(rr), -1, int)
        reindex[vertno] = np.arange(len(vertno))
        ss['tris'] = reindex[use_tris]
        assert (ss['tris'] >= 0).all()
        surfaces.append(ss)
    return surfaces


def export_gifti(fname, stc, src, scale=1, scale_rr=1e3):
    """Function for exporting STC to GIFTI file
    Parameters
    ------------
    fname : string
        filename basename to save files as
    stc : stc object
        instance of stc to export
    src : source solution (surface) object
        the source space of the forward solution
    scale : int
        scale of functional values
    scale_rr : float
        Value to scale source solution
    Notes
    ------------
    Creates gifti files for source solution and time courses of STC
    .. versionadded:: 1.7
    """

    ss = _get_decimated_surfaces(src)
    
    # Create lists to put DataArrays into
    lh = []
    rh = []
    
    # Coerce rr to be in mm (MNE uses meters)
    ss[0]['rr'] *= scale_rr
    ss[1]['rr'] *= scale_rr
    
    lh.append(nib.gifti.gifti.GiftiDataArray(data=ss[0]['rr'], intent='NIFTI_INTENT_POINTSET', datatype='NIFTI_TYPE_FLOAT32'))
    rh.append(nib.gifti.gifti.GiftiDataArray(data=ss[1]['rr'], intent='NIFTI_INTENT_POINTSET', datatype='NIFTI_TYPE_FLOAT32'))
    
    # Make the topology DataArray
    lh.append(nib.gifti.gifti.GiftiDataArray(data=ss[0]['tris'], intent='NIFTI_INTENT_TRIANGLE', datatype='NIFTI_TYPE_INT32'))
    rh.append(nib.gifti.gifti.GiftiDataArray(data=ss[1]['tris'], intent='NIFTI_INTENT_TRIANGLE', datatype='NIFTI_TYPE_INT32'))
    
    # Make the output GIFTI for anatomicals
    topo_gi_lh = nib.gifti.gifti.GiftiImage(darrays=lh)
    topo_gi_rh = nib.gifti.gifti.GiftiImage(darrays=rh)
    
    #actually save the files
    nib.save(topo_gi_lh, f"{fname}-lh.gii")
    nib.save(topo_gi_rh, f"{fname}-rh.gii")
    
    # Make the Time Series data arrays
    lh_ts = []
    rh_ts = []
    
    for t in range(stc.shape[1]):
        lh_ts.append(
            nib.gifti.gifti.GiftiDataArray(
                data=stc.lh_data[:, t] * scale,
                intent='NIFTI_INTENT_POINTSET',
                datatype='NIFTI_TYPE_FLOAT32'
            )
        )
        
        rh_ts.append(
            nib.gifti.gifti.GiftiDataArray(
                data=stc.rh_data[:, t] * scale,
                intent='NIFTI_INTENT_POINTSET',
                datatype='NIFTI_TYPE_FLOAT32'
            )
        )
        
    #save the time series
    ts_gi_lh = nib.gifti.gifti.GiftiImage(darrays=lh_ts)
    ts_gi_rh = nib.gifti.gifti.GiftiImage(darrays=rh_ts)
    nib.save(ts_gi_lh, f"{fname}-lh.time.gii")
    nib.save(ts_gi_rh, f"{fname}-rh.time.gii")