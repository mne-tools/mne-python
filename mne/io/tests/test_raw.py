# Generic tests that all raw classes should run
from numpy.testing import assert_allclose


def _test_concat(reader, *args):
    """Test concatenation of raw classes that allow not preloading"""
    data = None
    for preload in (True, False):
        raw1 = reader(*args, preload=preload)
        raw2 = reader(*args, preload=preload)
        raw1.append(raw2)
        raw1.preload_data()
        if data is None:
            data = raw1[:, :][0]
        assert_allclose(data, raw1[:, :][0])
    for first_preload in (True, False):
        raw = reader(*args, preload=first_preload)
        data = raw[:, :][0]
        for preloads in ((True, True), (True, False), (False, False)):
            for last_preload in (True, False):
                print(first_preload, preloads, last_preload)
                raw1 = raw.crop(0, 0.4999)
                if preloads[0]:
                    raw1.preload_data()
                raw2 = raw.crop(0.5, None)
                if preloads[1]:
                    raw2.preload_data()
                raw1.append(raw2)
                if last_preload:
                    raw1.preload_data()
                assert_allclose(data, raw1[:, :][0])
