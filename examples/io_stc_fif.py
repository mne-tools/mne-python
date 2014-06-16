import numpy as np
from mne.io.write import start_file, end_file
from mne.io import fiff_open
from mne.io.constants import FIFF
from mne.io.tag import find_tag
from mne.io.tree import dir_tree_find
from mne.io.write import (start_block, end_block, write_int,
                          write_double, write_float_matrix, write_string)
from mne.source_estimate import SourceEstimate, VolSourceEstimate

FIFF.FIFFB_MNE_STC = 365
FIFF.FIFFB_MNE_STC_DATA = 366
FIFF.FIFF_MNE_STC_TMIN = 3901
FIFF.FIFF_MNE_STC_TSTEP = 3902
FIFF.FIFF_MNE_STC_KIND = 3903
FIFF.FIFF_MNE_STC_DATA = 3904
FIFF.FIFF_MNE_STC_VERTNO = 3905
FIFF.FIFF_MNE_STC_NORI = 3906


def write_stc_fif(fname, stc):
    with start_file(fname) as fid:
        start_block(fid, FIFF.FIFFB_MNE_STC)

        write_double(fid, FIFF.FIFF_MNE_STC_TMIN, stc.tmin)
        write_double(fid, FIFF.FIFF_MNE_STC_TSTEP, stc.tstep)

        if stc.subject is not None:
            write_string(fid, FIFF.FIFFB_SUBJECT, stc.subject)

        # write data blocks

        if isinstance(stc, SourceEstimate):
            for vertno, data in zip(stc.vertno, [stc.lh_data, stc.rh_data]):
                start_block(fid, FIFF.FIFFB_MNE_STC_DATA)
                write_int(fid, FIFF.FIFF_MNE_STC_NORI, 1) # nb of orientation
                write_int(fid, FIFF.FIFF_MNE_STC_KIND, FIFF.FIFFV_MNE_SPACE_SURFACE)
                write_float_matrix(fid, FIFF.FIFF_MNE_STC_DATA, data)
                write_int(fid, FIFF.FIFF_MNE_STC_VERTNO, vertno)
                end_block(fid, FIFF.FIFFB_MNE_STC_DATA)
        elif isinstance(stc, VolSourceEstimate):
            start_block(fid, FIFF.FIFFB_MNE_STC_DATA)
            write_int(fid, FIFF.FIFF_MNE_STC_NORI, 1) # nb of orientation
            write_int(fid, FIFF.FIFF_MNE_STC_KIND, FIFF.FIFFV_MNE_SPACE_VOLUME)
            write_float_matrix(fid, FIFF.FIFF_MNE_STC_DATA, stc.data)
            write_int(fid, FIFF.FIFF_MNE_STC_VERTNO, stc.vertno)
            end_block(fid, FIFF.FIFFB_MNE_STC_DATA)
        else:
            raise RuntimeError('Unknown STC type.')

        end_block(fid, FIFF.FIFFB_MNE_STC)
        end_file(fid)


def read_stc_fif(fname):
    fid, tree, _ = fiff_open(fname)
    stcs = dir_tree_find(tree, FIFF.FIFFB_MNE_STC)

    if len(stcs) == 0:
        raise ValueError('No STC found.')

    this = stcs[0]
    tag = find_tag(fid, this, FIFF.FIFF_MNE_STC_TMIN)
    tmin = tag.data

    tag = find_tag(fid, this, FIFF.FIFF_MNE_STC_TSTEP)
    tstep = tag.data

    tag = find_tag(fid, this, FIFF.FIFFB_SUBJECT)
    subject = None
    if tag is not None:
        subject = tag.data

    stcs_data = dir_tree_find(tree, FIFF.FIFFB_MNE_STC_DATA)
    vertno = []
    surf_data = []
    vol_data = []

    for node in stcs_data:
        #   Find all the necessary data
        tag_kind = find_tag(fid, node, FIFF.FIFF_MNE_STC_KIND)
        tag = find_tag(fid, node, FIFF.FIFF_MNE_STC_DATA)
        if int(tag_kind.data) == FIFF.FIFFV_MNE_SPACE_VOLUME:
            vol_data.append(tag.data)
        elif int(tag_kind.data) == FIFF.FIFFV_MNE_SPACE_SURFACE:
            surf_data.append(tag.data)
        else:
            raise ValueError('Unknown STC data type.')
        tag = find_tag(fid, node, FIFF.FIFF_MNE_STC_VERTNO)
        vertno.append(tag.data)

    fid.close()

    if surf_data:
        data = np.concatenate(surf_data, axis=0)
        stc = SourceEstimate(data, vertices=vertno, tmin=tmin, tstep=tstep,
                             subject=subject)
    elif vol_data:
        data = np.concatenate(vol_data, axis=0)
        stc = VolSourceEstimate(data, vertices=vertno, tmin=tmin, tstep=tstep,
                                subject=subject)

    return stc

# some testing

import mne
from mne.datasets import sample

data_path = sample.data_path()
stc_fnames = []
stc_fnames.append(data_path + '/MEG/sample/mne_dSPM_vol_inverse-vl.stc')
stc_fnames.append(data_path + '/MEG/sample/sample_audvis-meg-lh.stc')

for stc_fname in stc_fnames:
    stc = mne.read_source_estimate(stc_fname)
    write_stc_fif('toto-stc.fif', stc)
    stc2 = read_stc_fif('toto-stc.fif')
    stc.save('pouet')

    if isinstance(stc.vertno, list):
        stc3 = mne.read_source_estimate('pouet')  # surface
    else:
        stc3 = mne.read_source_estimate('pouet-vl.stc')

    s = stc3 - stc2
    assert np.max(s.data) == 0
    assert np.min(s.data) == 0
