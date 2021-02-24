from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSlider


class QFloatSlider(QSlider):
    valueChanged = pyqtSignal(float)

    def __init__(self, ori, parent=None):
        super().__init__(ori, parent)
        self._float_min = 0.0
        self._float_max = 1.0
        self._float_value = 0.5
        self._precision = 10000
        self._int_min = int(self._float_min * self._precision)
        self._int_max = int(self._float_min * self._precision)
        self._int_value = int(self._float_value * self._precision)
        super().valueChanged.connect(self._convert)

    def _convert(self, value):
        int_value = value
        float_value = int_value / self._precision
        self.valueChanged.emit(float_value)

    def minimum(self):
        self._int_min = super.minimum()
        self._float_min = self._int_min / self._precision
        return self._float_min

    def setMinimum(self, value):
        self._float_min = value
        self._int_min = int(self._float_min * self._precision)
        super().setMinimum(self._int_min)

    def maximum(self):
        self._int_max = super.maximum()
        self._float_max = self._int_max / self._precision
        return self._float_max

    def setMaximum(self, value):
        self._float_max = value
        self._int_max = int(self._float_max * self._precision)
        super().setMaximum(self._int_max)

    def value(self):
        self._int_value = super.value()
        self._float_value = self._int_value / self._precision
        return self._float_value

    def setValue(self, value):
        self._float_value = value
        self._int_value = int(self._float_value * self._precision)
        super().setValue(self._int_value)
