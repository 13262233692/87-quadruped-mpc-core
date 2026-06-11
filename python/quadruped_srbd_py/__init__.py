"""
Quadruped SRBD - Python Package
Single Rigid Body Dynamics and Swing Trajectory for quadruped robots
"""

from .srbd_wrapper import SRBDWrapper, DiscretizationResult, DiscretizationMethod
from .srbd_wrapper import STATE_DIM, INPUT_DIM, NUM_LEGS
from .swing_wrapper import SwingTrajectoryPlannerWrapper, BezierCurveWrapper

__all__ = [
    'SRBDWrapper',
    'DiscretizationResult',
    'DiscretizationMethod',
    'SwingTrajectoryPlannerWrapper',
    'BezierCurveWrapper',
    'STATE_DIM',
    'INPUT_DIM',
    'NUM_LEGS',
]
__version__ = '0.3.0'
