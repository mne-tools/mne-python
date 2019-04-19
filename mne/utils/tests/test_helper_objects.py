# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)
import pytest
from mne.utils._bunch import MNEObjectsList


class MyIntList(MNEObjectsList):  # noqa: D101
    def __init__(self, elements=None):
        super(MyIntList, self).__init__(elements=elements, kls=int)


@pytest.mark.parametrize('data', [
    pytest.param([1, 2], id='list of ints'),
    pytest.param(['foo', 'bar'], id='list of strings',
                 marks=pytest.mark.xfail(raises=ValueError)),
])
def test_mne_objects_list_constructor(data):
    """Test MyIntList constructor."""
    my_int_list = MyIntList(data)
    assert my_int_list == data


@pytest.mark.parametrize('data, expected_len, expected_bool', [
    pytest.param(None, 0, False, id='None'),
    pytest.param([], 0, False, id='emtpy list'),
    pytest.param([1, 2], 2, True, id='list of ints'),
    pytest.param(['foo', 'bar'], 2, True, id='list of strings',
                 marks=pytest.mark.xfail(raises=ValueError)),
])
def test_emptylist_none_behaviour_in_conditionals(
        data, expected_len, expected_bool):
    """Test MyIntList constructor."""
    my_int_list = MyIntList(data)
    assert len(my_int_list) == expected_len
    if my_int_list:
        assert expected_bool
    else:
        assert not expected_bool
