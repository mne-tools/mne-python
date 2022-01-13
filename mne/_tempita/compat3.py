__all__ = ['PY3', 'b', 'basestring_', 'bytes', 'next', 'is_unicode']

PY3 = True


def b(s):  # noqa
    if isinstance(s, str):
        return s.encode('latin1')
    return bytes(s)


basestring_ = (bytes, str)
text = str
bytes = bytes
next = next


def is_unicode(obj):
    return isinstance(obj, str)


def coerce_text(v):
    if not isinstance(v, basestring_):
        return bytes(v)
    return v
