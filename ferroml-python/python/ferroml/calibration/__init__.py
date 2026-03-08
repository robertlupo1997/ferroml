"""
FerroML Calibration

Post-hoc probability calibration methods.

Classes
-------
TemperatureScalingCalibrator
    Temperature scaling for multi-class probability calibration
"""

from ferroml import ferroml as _native

TemperatureScalingCalibrator = _native.calibration.TemperatureScalingCalibrator

__all__ = [
    "TemperatureScalingCalibrator",
]
