"""
Quadruped SRBD - Python Package
Single Rigid Body Dynamics for quadruped robots
"""

from .srbd_wrapper import SRBDWrapper, DiscretizationResult, DiscretizationMethod
from .srbd_wrapper import STATE_DIM, INPUT_DIM, NUM_LEGS

__all__ = [
    'SRBDWrapper',
    'DiscretizationResult',
    'DiscretizationMethod',
    'STATE_DIM',
    'INPUT_DIM',
    'NUM_LEGS',
]
__version__ = '0.2.0'
