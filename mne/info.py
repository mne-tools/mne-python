# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: Simplified BSD

import copy


class Info(dict):
    """ Info class to nicely represent info dicts
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __repr__(self):
        """Summarize info instead of printing all"""
        strs = ['<Info | %s non-empty fields']
        non_empty = 0
        for k, v in self.items():
            this_len = (len(v) if hasattr(v, '__len__') else
                       ('%s' % v if v is not None else None))
            entries = ((' | %d items' % this_len) if isinstance(this_len, int)
                       else (' | %s' % this_len if this_len else ''))
            if entries:
                non_empty += 1
            strs.append('%s : %s%s' % (k, str(type(v))[7:-2], entries))
        strs_non_empty = sorted(s for s in strs if '|' in s)
        strs_empty = sorted(s for s in strs if '|' not in s)
        st = '\n    '.join(strs_non_empty + strs_empty)
        st += '\n>'
        st %= non_empty
        return st

    def copy(self):
        """Return deep copy of info"""
        return copy.deepcopy(self)
