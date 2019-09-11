# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)
import pytest
import numpy as np
from mne._digitization import Digitization
from mne._digitization.base import _format_dig_points

dig_dict_list = [
    dict(kind=_, ident=_, r=np.empty((3,)), coord_frame=_)
    for _ in [1, 2, 42]
]

digpoints_list = _format_dig_points(dig_dict_list)


@pytest.mark.parametrize('data', [
    pytest.param(digpoints_list, id='list of digpoints'),
    pytest.param(dig_dict_list, id='list of digpoint dicts',
                 marks=pytest.mark.xfail(raises=ValueError)),
    pytest.param(['foo', 'bar'], id='list of strings',
                 marks=pytest.mark.xfail(raises=ValueError)),
])
def test_digitization_constructor(data):
    """Test Digitization constructor."""
    dig = Digitization(data)
    assert dig == data

    dig[0]['kind'] = data[0]['kind'] - 1  # modify something in dig
    assert dig != data


def test_delete_elements():
    """Test deleting some Digitization elements."""
    dig = Digitization(digpoints_list)
    original_length = len(dig)
    del dig[0]
    assert len(dig) == original_length - 1
