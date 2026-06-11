"""
High-level Python wrapper for the SRBD C++ core.
Provides a more pythonic interface and additional utilities.
"""

import numpy as np
from typing import Tuple, List, Optional

import sys
import os

_cpp_module = None

def _find_and_import_cpp_module():
    global _cpp_module
    if _cpp_module is not None:
        return _cpp_module
    
    try:
        from . import _quadruped_srbd as cpp
        _cpp_module = cpp
        return cpp
    except ImportError:
        pass
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(module_dir))
    
    candidate_dirs = [
        os.path.join(project_root, 'build', 'python', 'Release'),
        os.path.join(project_root, 'build', 'python'),
        os.path.join(project_root, 'build', 'Release'),
    ]
    
    for d in candidate_dirs:
        if os.path.isdir(d):
            sys.path.insert(0, d)
            try:
                import quadruped_srbd as cpp
                _cpp_module = cpp
                return cpp
            except ImportError:
                sys.path.pop(0)
    
    raise ImportError(
        "Cannot import quadruped_srbd C++ extension. "
        "Please build the project first with CMake."
    )

cpp = _find_and_import_cpp_module()


STATE_DIM = 13
INPUT_DIM = 12
NUM_LEGS = 4

DiscretizationMethod = cpp.DiscretizationMethod


class DiscretizationResult:
    """
    Result of discretization, including stability analysis.
    """
    def __init__(self, A_d: np.ndarray, B_d: np.ndarray, 
                 matrix_exp_series_terms: int,
                 spectral_radius: float, is_stable: bool):
        self.A_d = A_d
        self.B_d = B_d
        self.matrix_exp_series_terms = matrix_exp_series_terms
        self.spectral_radius = spectral_radius
        self.is_stable = is_stable
    
    def __repr__(self):
        return (f"DiscretizationResult(stable={self.is_stable}, "
                f"spectral_radius={self.spectral_radius:.6f}, "
                f"terms={self.matrix_exp_series_terms})")


class SRBDWrapper:
    """
    High-level wrapper for the Single Rigid Body Dynamics model.
    
    State vector layout (13D):
        [0:3]   - position (x, y, z)
        [3:7]   - orientation quaternion (w, x, y, z)
        [7:10]  - linear velocity (vx, vy, vz)
        [10:13] - angular velocity (wx, wy, wz)
    
    Input vector layout (12D):
        4 legs × 3D contact forces (Fx, Fy, Fz)
    """

    def __init__(
        self,
        mass: float = 12.0,
        inertia: Optional[np.ndarray] = None,
        gravity: Optional[np.ndarray] = None
    ):
        self._model = cpp.SRBDModel()
        self._model.set_mass(mass)
        
        if inertia is None:
            inertia = np.diag([0.2, 0.3, 0.4])
        self._model.set_inertia(np.array(inertia, dtype=np.float64))
        
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])
        self._model.set_gravity(np.array(gravity, dtype=np.float64))

    @property
    def mass(self) -> float:
        return self._model.get_mass()

    @mass.setter
    def mass(self, value: float):
        self._model.set_mass(value)

    @property
    def inertia(self) -> np.ndarray:
        return self._model.get_inertia()

    @inertia.setter
    def inertia(self, value: np.ndarray):
        self._model.set_inertia(np.array(value, dtype=np.float64))

    @property
    def gravity(self) -> np.ndarray:
        return self._model.get_gravity()

    @gravity.setter
    def gravity(self, value: np.ndarray):
        self._model.set_gravity(np.array(value, dtype=np.float64))

    def build_state(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        lin_vel: np.ndarray,
        ang_vel: np.ndarray
    ) -> np.ndarray:
        """
        Build a state vector from components.
        
        Args:
            pos: Position (3,)
            quat: Quaternion in [w, x, y, z] format (4,)
            lin_vel: Linear velocity (3,)
            ang_vel: Angular velocity (3,)
            
        Returns:
            State vector (13,)
        """
        return self._model.build_state(
            np.array(pos, dtype=np.float64),
            np.array(quat, dtype=np.float64),
            np.array(lin_vel, dtype=np.float64),
            np.array(ang_vel, dtype=np.float64)
        )

    def get_com_position(self, state: np.ndarray) -> np.ndarray:
        return self._model.get_com_position(state)

    def get_orientation(self, state: np.ndarray) -> np.ndarray:
        return self._model.get_orientation(state)

    def get_linear_velocity(self, state: np.ndarray) -> np.ndarray:
        return self._model.get_linear_velocity(state)

    def get_angular_velocity(self, state: np.ndarray) -> np.ndarray:
        return self._model.get_angular_velocity(state)

    def continuous_dynamics(
        self,
        state: np.ndarray,
        forces: np.ndarray,
        foot_positions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute continuous-time state derivative.
        
        Args:
            state: State vector (13,)
            forces: Contact forces (12,)
            foot_positions: List of 4 foot positions in world frame (each 3,)
            
        Returns:
            State derivative (13,)
        """
        fp = [np.array(p, dtype=np.float64) for p in foot_positions]
        return self._model.continuous_dynamics(
            np.array(state, dtype=np.float64),
            np.array(forces, dtype=np.float64),
            fp
        )

    def continuous_AB(
        self,
        state: np.ndarray,
        foot_positions: List[np.ndarray],
        contact: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute continuous-time A and B matrices (state-space model).
        
        dx/dt = A·x + B·u
        
        Args:
            state: State vector (13,)
            foot_positions: List of 4 foot positions (each 3,)
            contact: List of 4 booleans indicating foot contact
            
        Returns:
            Tuple of (A matrix (13,13), B matrix (13,12))
        """
        fp = [np.array(p, dtype=np.float64) for p in foot_positions]
        A, B = self._model.continuous_AB_matrices(
            np.array(state, dtype=np.float64),
            fp,
            list(contact)
        )
        return A, B

    def continuous_A(
        self,
        state: np.ndarray,
        foot_positions: List[np.ndarray],
        contact: List[bool]
    ) -> np.ndarray:
        fp = [np.array(p, dtype=np.float64) for p in foot_positions]
        return self._model.continuous_A_matrix(
            np.array(state, dtype=np.float64),
            fp,
            list(contact)
        )

    def continuous_B(
        self,
        state: np.ndarray,
        foot_positions: List[np.ndarray],
        contact: List[bool]
    ) -> np.ndarray:
        fp = [np.array(p, dtype=np.float64) for p in foot_positions]
        return self._model.continuous_B_matrix(
            np.array(state, dtype=np.float64),
            fp,
            list(contact)
        )

    def discrete_AB(
        self,
        state: np.ndarray,
        foot_positions: List[np.ndarray],
        contact: List[bool],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time A and B matrices using zero-order hold.
        
        x_{k+1} = A_d · x_k + B_d · u_k
        
        Args:
            state: State vector (13,)
            foot_positions: List of 4 foot positions (each 3,)
            contact: List of 4 booleans
            dt: Time step
            
        Returns:
            Tuple of (A_d (13,13), B_d (13,12))
        """
        fp = [np.array(p, dtype=np.float64) for p in foot_positions]
        A_d = self._model.discrete_A_matrix(
            np.array(state, dtype=np.float64), fp, list(contact), dt
        )
        B_d = self._model.discrete_B_matrix(
            np.array(state, dtype=np.float64), fp, list(contact), dt
        )
        return A_d, B_d

    def integrate_rk4(
        self,
        state: np.ndarray,
        forces: np.ndarray,
        foot_positions: List[np.ndarray],
        dt: float
    ) -> np.ndarray:
        """
        Integrate state forward using RK4 method.
        
        Args:
            state: Current state (13,)
            forces: Contact forces (12,)
            foot_positions: List of 4 foot positions (each 3,)
            dt: Time step
            
        Returns:
            Next state (13,)
        """
        fp = [np.array(p, dtype=np.float64) for p in foot_positions]
        return self._model.integrate_rk4(
            np.array(state, dtype=np.float64),
            np.array(forces, dtype=np.float64),
            fp,
            dt
        )

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Build skew-symmetric cross product matrix from vector."""
        return cpp.skew_symmetric(np.array(v, dtype=np.float64))

    def rotation_matrix(self, state: np.ndarray) -> np.ndarray:
        """Get rotation matrix from state orientation."""
        return self._model.rotation_matrix(np.array(state, dtype=np.float64))

    def discretize(
        self,
        state: np.ndarray,
        foot_positions: List[np.ndarray],
        contact: List[bool],
        dt: float,
        method: DiscretizationMethod = DiscretizationMethod.MATRIX_EXPONENTIAL
    ) -> DiscretizationResult:
        """
        Discretize the continuous state-space model.
        
        Args:
            state: State vector (13,)
            foot_positions: List of 4 foot positions (each 3,)
            contact: List of 4 booleans
            dt: Time step
            method: Discretization method
            
        Returns:
            DiscretizationResult with A_d, B_d, and stability info
        """
        fp = [np.array(p, dtype=np.float64) for p in foot_positions]
        result = self._model.discretize(
            np.array(state, dtype=np.float64),
            fp, list(contact), dt, method
        )
        return DiscretizationResult(
            A_d=result.A_d,
            B_d=result.B_d,
            matrix_exp_series_terms=result.matrix_exp_series_terms,
            spectral_radius=result.spectral_radius,
            is_stable=result.is_stable
        )

    def spectral_radius(self, A: np.ndarray) -> float:
        """Compute spectral radius of a matrix."""
        return self._model.spectral_radius(np.array(A, dtype=np.float64))

    def check_stability(self, A_disc: np.ndarray) -> bool:
        """Check if discrete-time system is stable (spectral radius <= 1)."""
        return self._model.check_stability(np.array(A_disc, dtype=np.float64))

    def matrix_exponential(self, A: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute matrix exponential using scaling and squaring.
        
        Returns:
            Tuple of (exp_A, num_terms_used)
        """
        exp_A, terms = self._model.matrix_exponential(
            np.array(A, dtype=np.float64)
        )
        return exp_A, terms

    def set_discretization_method(self, method: DiscretizationMethod):
        """Set the default discretization method."""
        self._model.set_discretization_method(method)

    def get_discretization_method(self) -> DiscretizationMethod:
        """Get the current default discretization method."""
        return self._model.get_discretization_method()
