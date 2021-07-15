class BrowserDataManager:
    def __init__(self, inst):
        self._data = None
        self._times = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, times):
        self._times = times

    def pick(self, picks):
        pass

    def apply_downsampling(self):
        pass

    def apply_filtering(self):
        pass

    def apply_clipping(self):
        pass
