import sys

__all__ = ['PY3', 'b', 'basestring_', 'bytes', 'next', 'is_unicode']

PY3 = True if sys.version_info[0] == 3 else False

if sys.version_info[0] < 3:
    b = bytes = str
    basestring_ = basestring
else:

    def b(s):
        if isinstance(s, str):
            return s.encode('latin1')
        return bytes(s)
    basestring_ = (bytes, str)
    bytes = bytes
text = str

if sys.version_info[0] < 3:

    def next(obj):
        return obj.next()
else:
    next = next


def is_unicode(obj):
    if sys.version_info[0] < 3:
        return isinstance(obj, unicode)
    else:
        return isinstance(obj, str)


def coerce_text(v):
    if not isinstance(v, basestring_):
        if sys.version_info[0] < 3:
            attr = '__unicode__'
        else:
            attr = '__str__'
        if hasattr(v, attr):
            return unicode(v)
        else:
            return bytes(v)
    return v
