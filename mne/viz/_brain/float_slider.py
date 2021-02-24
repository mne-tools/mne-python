def float_slider_class():
    """Return the QFloatSlider class."""
    from PyQt5.QtCore import pyqtSignal
    from PyQt5.QtWidgets import QSlider

    class QFloatSlider(QSlider):
        """Slider that handles float values."""

        valueChanged = pyqtSignal(float)

        def __init__(self, ori, parent=None):
            """Initialize the slider."""
            super().__init__(ori, parent)
            self._precision = 10000
            super().valueChanged.connect(self._convert)

        def _convert(self, value):
            self.valueChanged.emit(value / self._precision)

        def minimum(self):
            """Get the minimum."""
            return super().minimum() / self._precision

        def setMinimum(self, value):
            """Set the minimum."""
            super().setMinimum(int(value * self._precision))

        def maximum(self):
            """Get the maximum."""
            return super().maximum() / self._precision

        def setMaximum(self, value):
            """Set the maximum."""
            super().setMaximum(int(value * self._precision))

        def value(self):
            """Get the current value."""
            return super().value() / self._precision

        def setValue(self, value):
            """Set the current value."""
            super().setValue(int(value * self._precision))

    return QFloatSlider
