import numpy as np

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
