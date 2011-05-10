import os.path as op

from ..misc import parse_config

ave_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test.ave')

def test_parse_ave():
    """Test parsing of .ave file
    """
    conditions = parse_config(ave_fname)
    assert len(conditions) == 4
