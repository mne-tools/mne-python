# -*- coding: utf-8 -*-

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from mne.stats import combine_adjacency
from mne.utils import requires_sklearn


@requires_sklearn
@pytest.mark.parametrize('shape', [
    (1,),
    (2,),
    (1, 1),
    (1, 2),
    (2, 1),
    (3, 4),
    (1, 1, 1),
    (1, 1, 2),
    (3, 4, 5),
])
def test_adjacency_equiv(shape):
    """Test adjacency equivalence for lattice adjacency."""
    from sklearn.feature_extraction import grid_to_graph
    # sklearn requires at least two dimensions
    sk_shape = shape if len(shape) > 1 else (shape + (1,))
    conn_sk = grid_to_graph(*sk_shape).toarray()
    conn = combine_adjacency(*shape)
    want_shape = (np.prod(shape),) * 2
    assert conn.shape == conn_sk.shape == want_shape
    assert (conn.data == 1.).all()
    conn = conn.toarray()
    # we end up with some duplicates that can turn into 2's and 3's,
    # eventually we might want to keep these as 1's but it's easy enough
    # with a .astype(bool) (also matches sklearn output) so let's leave it
    # for now
    assert np.in1d(conn, [0, 1, 2, 3]).all()
    assert conn.shape == conn_sk.shape
    assert_array_equal(conn, conn_sk)
