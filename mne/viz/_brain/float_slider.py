from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSlider


class QFloatSlider(QSlider):
    valueChanged = pyqtSignal(float)

    def __init__(self, ori, parent=None):
        super().__init__(ori, parent)
        self._precision = 10000
        super().valueChanged.connect(self._convert)

    def _convert(self, value):
        self.valueChanged.emit(value / self._precision)

    def minimum(self):
        return super().minimum() / self._precision

    def setMinimum(self, value):
        super().setMinimum(int(value * self._precision))

    def maximum(self):
        return super().maximum() / self._precision

    def setMaximum(self, value):
        super().setMaximum(int(value * self._precision))

    def value(self):
        return super().value() / self._precision

    def setValue(self, value):
        super().setValue(int(value * self._precision))
