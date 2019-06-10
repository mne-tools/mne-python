from copy import deepcopy
import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
import pytest
from scipy.fftpack import fft
from scipy import sparse

from mne.datasets import testing
from mne import (stats, SourceEstimate, VectorSourceEstimate,
                 VolSourceEstimate, Label, read_source_spaces,
                 read_evokeds, MixedSourceEstimate, find_events, Epochs,
                 read_source_estimate, extract_label_time_course,
                 spatio_temporal_tris_connectivity,
                 spatio_temporal_src_connectivity,
                 spatial_inter_hemi_connectivity,
                 spatial_src_connectivity, spatial_tris_connectivity,
                 SourceSpaces, VolVectorSourceEstimate)
from mne.source_estimate import grade_to_tris, _get_vol_mask

from mne.minimum_norm import (read_inverse_operator, apply_inverse,
                              apply_inverse_epochs)
from mne.label import read_labels_from_annot, label_sign_flip
from mne.utils import (_TempDir, requires_pandas, requires_sklearn,
                       requires_h5py, run_tests_if_main, requires_nibabel)
from mne.io import read_raw_fif
from mne.source_tfr import SourceTFR


def test_source_tfr():
    # compare kernelized and normal data shapes
    kernel_stfr = SourceTFR((np.ones([1800, 300]), np.ones([300, 40, 30])),
                            vertices=np.ones([1800, 1]), tmin=0, tstep=1)

    full_stfr = SourceTFR(np.ones([1800, 40, 30]), vertices=np.ones([1800, 1]), tmin=0, tstep=1)

    # check if data is in correct shape
    assert_equal(kernel_stfr.shape, full_stfr.shape)
    assert_array_equal(kernel_stfr.data.shape, full_stfr.data.shape)
