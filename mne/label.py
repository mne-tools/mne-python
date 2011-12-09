import os
import numpy as np
from scipy import linalg

from .source_estimate import read_stc, mesh_edges
from .surface import read_surface


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
    if filename.endswith('lh.label') or filename.startswith('lh.'):
        label['hemi'] = 'lh'
    elif filename.endswith('rh.label') or filename.startswith('rh.'):
        label['hemi'] = 'rh'
    else:
        raise ValueError('Cannot find which hemisphere it is. File should end'
                         ' with lh.label or rh.label')
    fid.close()

    return label


def write_label(filename, label):
    """Write a FreeSurfer label

    Parameters
    ----------
    filename : string
        Path to label file to produce.
    label : dict
        The label structure.
    """
    if not filename.endswith('lh.label') or not filename.endswith('rh.label'):
        filename += '-' + label['hemi'] + '.label'

    print 'Saving label to : %s' % filename

    fid = open(filename, 'wb')
    n_vertices = len(label['vertices'])
    data = np.zeros((n_vertices, 5), dtype=np.float)
    data[:, 0] = label['vertices']
    data[:, 1:4] = 1e3 * label['pos']
    data[:, 4] = label['values']
    fid.write("#%s\n" % label['comment'])
    fid.write("%d\n" % n_vertices)
    for d in data:
        fid.write("%d %f %f %f %f\n" % tuple(d))

    print '[done]'
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
        The source space over which the label is defined

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


def stc_to_label(stc, src, smooth=5):
    """Compute a label from the non-zero sources in an stc object.

    Parameters
    ----------
    stc : SourceEstimate
        The source estimates
    src : list of dict or string
        The source space over which are defined the source estimates.
        If it's a string it should the subject name (e.g. fsaverage).

    Returns
    -------
    labels : list of dict
        The generated labels. One per hemisphere containing sources.
    """
    from scipy import sparse

    if not stc.is_surface():
        raise ValueError('SourceEstimate should be surface source '
                         'estimates')

    if isinstance(src, str):
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            raise ValueError('SUBJECTS_DIR environment variable not set')
        surf_path_from = os.path.join(subjects_dir, src, 'surf')
        rr_lh, tris_lh = read_surface(os.path.join(surf_path_from,
                                      'lh.white'))
        rr_rh, tris_rh = read_surface(os.path.join(surf_path_from,
                                      'rh.white'))
        rr = [rr_lh, rr_rh]
        tris = [tris_lh, tris_rh]
    else:
        if len(src) != 2:
            raise ValueError('source space should contain the 2 hemispheres')
        tris = [src[0]['tris'], src[1]['tris']]
        rr = [1e3 * src[0]['rr'], 1e3 * src[1]['rr']]

    labels = []
    cnt = 0
    for hemi, this_vertno, this_tris, this_rr in \
                                    zip(['lh', 'rh'], stc.vertno, tris, rr):
        if len(this_vertno) == 0:
            continue
        e = mesh_edges(this_tris)
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        this_data = stc.data[cnt:cnt + len(this_vertno)]
        cnt += len(this_vertno)
        e = e + sparse.eye(n_vertices, n_vertices)
        idx_use = this_vertno[np.any(this_data, axis=1)]
        if len(idx_use) == 0:
            continue
        for k in range(smooth):
            e_use = e[:, idx_use]
            data1 = e_use * np.ones(len(idx_use))
            idx_use = np.where(data1)[0]

        label = dict()
        label['comment'] = 'Label from stc'
        label['vertices'] = idx_use
        label['pos'] = this_rr[idx_use]
        label['values'] = np.ones(len(idx_use))
        label['hemi'] = hemi
        labels.append(label)

    return labels
