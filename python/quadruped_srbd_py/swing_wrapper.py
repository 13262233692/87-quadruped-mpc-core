"""
High-level Python wrapper for the Swing Trajectory C++ core.
Provides a Pythonic interface for Bezier curve swing leg trajectory generation.
"""

import numpy as np
from typing import Tuple, List, Optional

from .srbd_wrapper import cpp


BezierOrder = cpp.BezierOrder
LegPhase = cpp.LegPhase
LegIndex = cpp.LegIndex


class BezierCurveWrapper:
    """
    Pythonic wrapper for BezierCurve.
    
    Supports cubic (order 3) and quintic (order 5) Bezier curves
    with de Casteljau evaluation, derivatives, and ground penetration detection.
    """

    def __init__(self, control_points: Optional[List[np.ndarray]] = None,
                 _cpp_curve=None):
        if _cpp_curve is not None:
            self._curve = _cpp_curve
        elif control_points is not None:
            cp = [np.array(p, dtype=np.float64) for p in control_points]
            self._curve = cpp.BezierCurve(cp)
        else:
            self._curve = cpp.BezierCurve()

    @staticmethod
    def cubic(p0: np.ndarray, p1: np.ndarray, 
              p2: np.ndarray, p3: np.ndarray) -> 'BezierCurveWrapper':
        curve = cpp.BezierCurve.cubic(
            np.array(p0, dtype=np.float64),
            np.array(p1, dtype=np.float64),
            np.array(p2, dtype=np.float64),
            np.array(p3, dtype=np.float64)
        )
        return BezierCurveWrapper(_cpp_curve=curve)

    @staticmethod
    def quintic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                p3: np.ndarray, p4: np.ndarray, p5: np.ndarray) -> 'BezierCurveWrapper':
        curve = cpp.BezierCurve.quintic(
            np.array(p0, dtype=np.float64),
            np.array(p1, dtype=np.float64),
            np.array(p2, dtype=np.float64),
            np.array(p3, dtype=np.float64),
            np.array(p4, dtype=np.float64),
            np.array(p5, dtype=np.float64)
        )
        return BezierCurveWrapper(_cpp_curve=curve)

    def evaluate(self, t: float) -> np.ndarray:
        return self._curve.evaluate(t)

    def derivative(self, t: float) -> np.ndarray:
        return self._curve.derivative(t)

    def second_derivative(self, t: float) -> np.ndarray:
        return self._curve.second_derivative(t)

    def sample(self, num_samples: int = 50) -> List[np.ndarray]:
        return self._curve.sample(num_samples)

    def arc_length(self, num_segments: int = 100) -> float:
        return self._curve.arc_length(num_segments)

    @property
    def control_points(self) -> List[np.ndarray]:
        return self._curve.control_points()

    @property
    def order(self) -> int:
        return self._curve.order()

    @property
    def num_control_points(self) -> int:
        return self._curve.num_control_points()

    @property
    def start_point(self) -> np.ndarray:
        return self._curve.start_point()

    @property
    def end_point(self) -> np.ndarray:
        return self._curve.end_point()

    def set_control_point(self, index: int, point: np.ndarray):
        self._curve.set_control_point(index, np.array(point, dtype=np.float64))

    @property
    def min_z(self) -> float:
        return self._curve.min_z()

    @property
    def max_z(self) -> float:
        return self._curve.max_z()

    def has_ground_penetration(self, ground_height: float = 0.0) -> bool:
        return self._curve.has_ground_penetration(ground_height)

    def elevate_degree(self) -> 'BezierCurveWrapper':
        return BezierCurveWrapper(_cpp_curve=self._curve.elevate_degree())

    def __repr__(self):
        return f"BezierCurve(order={self.order}, n_points={self.num_control_points})"


class SwingTrajectoryConfig:
    """
    Configuration for swing trajectory generation.
    """

    def __init__(self, **kwargs):
        self._config = cpp.SwingTrajectoryConfig()
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                raise AttributeError(f"Unknown config parameter: {key}")

    @property
    def swing_height(self) -> float:
        return self._config.swing_height

    @swing_height.setter
    def swing_height(self, value: float):
        self._config.swing_height = value

    @property
    def swing_duration(self) -> float:
        return self._config.swing_duration

    @swing_duration.setter
    def swing_duration(self, value: float):
        self._config.swing_duration = value

    @property
    def bezier_order(self):
        return self._config.bezier_order

    @bezier_order.setter
    def bezier_order(self, value):
        self._config.bezier_order = value

    @property
    def ground_height(self) -> float:
        return self._config.ground_height

    @ground_height.setter
    def ground_height(self, value: float):
        self._config.ground_height = value

    @property
    def ground_clearance(self) -> float:
        return self._config.ground_clearance

    @ground_clearance.setter
    def ground_clearance(self, value: float):
        self._config.ground_clearance = value

    @property
    def penetration_penalty_gain(self) -> float:
        return self._config.penetration_penalty_gain

    @penetration_penalty_gain.setter
    def penetration_penalty_gain(self, value: float):
        self._config.penetration_penalty_gain = value

    def __repr__(self):
        return (f"SwingTrajectoryConfig(swing_height={self.swing_height}, "
                f"bezier_order={self.bezier_order}, "
                f"swing_duration={self.swing_duration})")


class SwingTrajectoryResult:
    """
    Result of swing trajectory generation.
    """

    def __init__(self, _cpp_result=None):
        if _cpp_result is not None:
            self._result = _cpp_result
        else:
            self._result = None

    @property
    def bezier_body(self) -> BezierCurveWrapper:
        return BezierCurveWrapper(_cpp_curve=self._result.bezier_body)

    @property
    def bezier_world(self) -> BezierCurveWrapper:
        return BezierCurveWrapper(_cpp_curve=self._result.bezier_world)

    @property
    def trajectory_body(self) -> List[np.ndarray]:
        return self._result.trajectory_body

    @property
    def trajectory_world(self) -> List[np.ndarray]:
        return self._result.trajectory_world

    @property
    def ground_penetration_max(self) -> float:
        return self._result.ground_penetration_max

    @property
    def had_penetration(self) -> bool:
        return self._result.had_penetration

    @property
    def penalty_iterations(self) -> int:
        return self._result.penalty_iterations

    @property
    def is_valid(self) -> bool:
        return self._result.is_valid

    def __repr__(self):
        return (f"SwingTrajectoryResult(valid={self.is_valid}, "
                f"had_penetration={self.had_penetration}, "
                f"max_penetration={self.ground_penetration_max:.6f})")


class SwingTrajectoryPlannerWrapper:
    """
    High-level wrapper for the Swing Trajectory Planner.
    
    Generates Bezier curve swing leg trajectories with:
    - Cubic or quintic Bezier curves
    - Body-to-world coordinate transform (considers body tilt)
    - Ground penetration detection and penalty correction
    - Per-leg state machine (stance/swing phase tracking)
    """

    def __init__(self, config: Optional[SwingTrajectoryConfig] = None):
        self._planner = cpp.SwingTrajectoryPlanner()
        if config is not None:
            self._planner.set_config(config._config)
        self._config = config or SwingTrajectoryConfig()

    @property
    def config(self) -> SwingTrajectoryConfig:
        return self._config

    @config.setter
    def config(self, value: SwingTrajectoryConfig):
        self._config = value
        self._planner.set_config(value._config)

    def generate_swing_trajectory(
        self,
        start_body: np.ndarray,
        target_body: np.ndarray,
        body_quaternion: np.ndarray,
        body_position: np.ndarray
    ) -> SwingTrajectoryResult:
        """
        Generate swing trajectory from start to target in body frame.
        
        Args:
            start_body: Start foot position in body frame (3,)
            target_body: Target foot position in body frame (3,)
            body_quaternion: Body orientation [w, x, y, z] (4,)
            body_position: Body position in world frame (3,)
            
        Returns:
            SwingTrajectoryResult with body and world frame trajectories
        """
        result_cpp = self._planner.generate_swing_trajectory(
            np.array(start_body, dtype=np.float64),
            np.array(target_body, dtype=np.float64),
            np.array(body_quaternion, dtype=np.float64),
            np.array(body_position, dtype=np.float64)
        )
        return SwingTrajectoryResult(_cpp_result=result_cpp)

    def generate_swing_trajectory_from_state(
        self,
        start_body: np.ndarray,
        target_body: np.ndarray,
        state: np.ndarray
    ) -> SwingTrajectoryResult:
        """
        Generate swing trajectory using a 13D state vector.
        
        Args:
            start_body: Start foot position in body frame (3,)
            target_body: Target foot position in body frame (3,)
            state: 13D SRBD state vector
            
        Returns:
            SwingTrajectoryResult with body and world frame trajectories
        """
        result_cpp = self._planner.generate_swing_trajectory_state(
            np.array(start_body, dtype=np.float64),
            np.array(target_body, dtype=np.float64),
            np.array(state, dtype=np.float64)
        )
        return SwingTrajectoryResult(_cpp_result=result_cpp)

    def build_body_frame_bezier(
        self,
        start: np.ndarray,
        target: np.ndarray
    ) -> BezierCurveWrapper:
        """
        Build a Bezier curve in body frame (no ground penalty applied).
        """
        curve = self._planner.build_body_frame_bezier(
            np.array(start, dtype=np.float64),
            np.array(target, dtype=np.float64)
        )
        return BezierCurveWrapper(_cpp_curve=curve)

    def body_to_world(
        self,
        point_body: np.ndarray,
        body_quaternion: np.ndarray,
        body_position: np.ndarray
    ) -> np.ndarray:
        """
        Transform a point from body frame to world frame.
        
        Args:
            point_body: Point in body frame (3,)
            body_quaternion: Body orientation [w, x, y, z] (4,)
            body_position: Body position in world frame (3,)
        """
        return self._planner.body_to_world(
            np.array(point_body, dtype=np.float64),
            np.array(body_quaternion, dtype=np.float64),
            np.array(body_position, dtype=np.float64)
        )

    def get_leg_phase(self, leg_index: int):
        return self._planner.get_leg_phase(leg_index)

    def set_leg_phase(self, leg_index: int, phase):
        self._planner.set_leg_phase(leg_index, phase)

    def get_swing_phase_progress(self, leg_index: int) -> float:
        return self._planner.get_swing_phase_progress(leg_index)

    def reset_leg(self, leg_index: int):
        self._planner.reset_leg(leg_index)

    def update_leg_state(
        self,
        leg_index: int,
        dt: float,
        current_foot_body: np.ndarray,
        target_foot_body: np.ndarray,
        state: np.ndarray
    ):
        self._planner.update_leg_state(
            leg_index, dt,
            np.array(current_foot_body, dtype=np.float64),
            np.array(target_foot_body, dtype=np.float64),
            np.array(state, dtype=np.float64)
        )

    def get_current_foot_position(
        self, leg_index: int, state: np.ndarray
    ) -> np.ndarray:
        return self._planner.get_current_foot_position(
            leg_index, np.array(state, dtype=np.float64)
        )

    def get_current_foot_velocity(
        self, leg_index: int, state: np.ndarray
    ) -> np.ndarray:
        return self._planner.get_current_foot_velocity(
            leg_index, np.array(state, dtype=np.float64)
        )
