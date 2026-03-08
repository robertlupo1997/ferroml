"""
FerroML Calibration

Post-hoc probability calibration methods for improving the reliability
of classifier probability estimates.

Classes
-------
TemperatureScalingCalibrator
    Temperature scaling for multi-class probability calibration.
    Learns a single temperature parameter on a held-out calibration set
    to rescale logits before softmax.

Example
-------
>>> from ferroml.calibration import TemperatureScalingCalibrator
>>> import numpy as np
>>>
>>> # logits from a trained classifier and true labels
>>> logits = np.array([[2.0, 0.5], [0.3, 1.8], [1.5, 0.2]])
>>> y_true = np.array([0, 1, 0])
>>> calibrator = TemperatureScalingCalibrator()
>>> calibrator.fit(logits, y_true)
>>> calibrated = calibrator.transform(logits)  # calibrated probabilities
"""

from ferroml import ferroml as _native

TemperatureScalingCalibrator = _native.calibration.TemperatureScalingCalibrator

__all__ = [
    "TemperatureScalingCalibrator",
]
