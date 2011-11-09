import numpy as np
from scipy import linalg

from .source_estimate import read_stc


def read_label(filename):
    """Read FreeSurfer Label file

    Parameters
    ----------
    filename : string
        Path to label file.

    Returns
    -------
    label : dict
        Label dictionaries with keys:
            comment        comment from the first line of the label file
            vertices       vertex indices (0 based, column 1)
            pos            locations in meters (columns 2 - 4 divided by 1000)
            values         values at the vertices (column 5)

    """
    fid = open(filename, 'r')
    comment = fid.readline().replace('\n', '')
    nv = int(fid.readline())
    data = np.empty((5, nv))
    for i, line in enumerate(fid):
        data[:, i] = line.split()

    label = dict()
    label['comment'] = comment[1:]
    label['vertices'] = np.array(data[0], dtype=np.int32)
    label['pos'] = 1e-3 * data[1:4].T
    label['values'] = data[4]
    if filename.endswith('lh.label'):
        label['hemi'] = 'lh'
    elif filename.endswith('rh.label'):
        label['hemi'] = 'rh'
    else:
        raise ValueError('Cannot find which hemisphere it is. File should end'
                         ' with lh.label or rh.label')
    fid.close()

    return label


def label_time_courses(labelfile, stcfile):
    """Extract the time courses corresponding to a label file from an stc file

    Parameters
    ----------
    labelfile : string
        Path to the label file

    stcfile : string
        Path to the stc file. The name of the stc file (must be on the
        same subject and hemisphere as the stc file)

    Returns
    -------
    values : 2d array
        The time courses
    times : 1d array
        The time points
    vertices : array
        The indices of the vertices corresponding to the time points
    """
    stc = read_stc(stcfile)
    lab = read_label(labelfile)

    vertices = np.intersect1d(stc['vertices'], lab['vertices'])
    idx = [k for k in range(len(stc['vertices']))
                   if stc['vertices'][k] in vertices]

    if len(vertices) == 0:
        raise ValueError('No vertices match the label in the stc file')

    values = stc['data'][idx]
    times = stc['tmin'] + stc['tstep'] * np.arange(stc['data'].shape[1])

    return values, times, vertices


def label_sign_flip(label, src):
    """Compute sign for label averaging

    Parameters
    ----------
    label : dict
        A label read with the read_label function
    src : list of dict
        The source space over which is defined the label

    Returns
    -------
    flip : array
        Sign flip vector (contains 1 or -1)
    """
    if len(src) != 2:
        raise ValueError('Only source spaces with 2 hemisphers are accepted')

    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    # get source orientations
    if label['hemi'] == 'lh':
        vertno_sel = np.intersect1d(lh_vertno, label['vertices'])
        ori = src[0]['nn'][vertno_sel]
    elif label['hemi'] == 'rh':
        vertno_sel = np.intersect1d(rh_vertno, label['vertices'])
        ori = src[1]['nn'][vertno_sel]
    else:
        raise Exception("Unknown hemisphere type")

    _, _, Vh = linalg.svd(ori, full_matrices=False)

    # Comparing to the direction of the first right singular vector
    flip = np.sign(np.dot(ori, Vh[:, 0]))
    return flip
