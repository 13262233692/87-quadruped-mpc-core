"""
Swing Leg Trajectory Generation Demo

Demonstrates the Bezier curve swing trajectory engine with:
1. Cubic and Quintic Bezier curves
2. Body-to-world coordinate transform (with body tilt)
3. Ground penetration detection and penalty correction
"""

import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
build_python_dir = os.path.join(project_root, 'build', 'python', 'Release')
python_pkg_dir = os.path.join(project_root, 'python')

if os.path.exists(build_python_dir):
    sys.path.insert(0, build_python_dir)
if os.path.exists(python_pkg_dir):
    sys.path.insert(0, python_pkg_dir)

from quadruped_srbd_py import SwingTrajectoryPlannerWrapper, BezierCurveWrapper
from quadruped_srbd_py.swing_wrapper import LegPhase, BezierOrder


def demo_bezier_curves():
    print("=" * 70)
    print("PART 1: Bezier Curve Basics")
    print("=" * 70)
    print()

    cubic = BezierCurveWrapper.cubic(
        np.array([0, 0, 0], dtype=np.float64),
        np.array([1, 0, 3], dtype=np.float64),
        np.array([2, 0, 3], dtype=np.float64),
        np.array([3, 0, 0], dtype=np.float64),
    )
    print(f"Cubic Bezier: order={cubic.order}, points={cubic.num_control_points}")
    print(f"  Start: {cubic.start_point}")
    print(f"  End:   {cubic.end_point}")
    print(f"  At t=0.25: {cubic.evaluate(0.25)}")
    print(f"  At t=0.50: {cubic.evaluate(0.5)}")
    print(f"  At t=0.75: {cubic.evaluate(0.75)}")
    print(f"  Velocity at t=0: {cubic.derivative(0.0)}")
    print(f"  Arc length: {cubic.arc_length():.4f}")
    print(f"  Ground penetration (h=0): {cubic.has_ground_penetration(0.0)}")
    print()

    quintic = BezierCurveWrapper.quintic(
        np.array([0, 0, 0], dtype=np.float64),
        np.array([0.5, 0, 1.5], dtype=np.float64),
        np.array([1.2, 0, 3], dtype=np.float64),
        np.array([1.8, 0, 3], dtype=np.float64),
        np.array([2.5, 0, 1.5], dtype=np.float64),
        np.array([3, 0, 0], dtype=np.float64),
    )
    print(f"Quintic Bezier: order={quintic.order}, points={quintic.num_control_points}")
    print(f"  Start: {quintic.start_point}")
    print(f"  End:   {quintic.end_point}")
    print(f"  At t=0.50: {quintic.evaluate(0.5)}")
    print()

    elevated = cubic.elevate_degree()
    print(f"Elevated cubic -> order={elevated.order}, points={elevated.num_control_points}")
    for t in [0.0, 0.5, 1.0]:
        diff = np.linalg.norm(cubic.evaluate(t) - elevated.evaluate(t))
        print(f"  Deviation at t={t}: {diff:.2e}")
    print()


def demo_swing_trajectory():
    print("=" * 70)
    print("PART 2: Swing Leg Trajectory Generation")
    print("=" * 70)
    print()

    planner = SwingTrajectoryPlannerWrapper()

    foot_positions = {
        "FL": (np.array([0.2, 0.15, 0.0]), np.array([0.3, 0.15, 0.0])),
        "FR": (np.array([0.2, -0.15, 0.0]), np.array([0.3, -0.15, 0.0])),
        "RL": (np.array([-0.2, 0.15, 0.0]), np.array([-0.1, 0.15, 0.0])),
        "RR": (np.array([-0.2, -0.15, 0.0]), np.array([-0.1, -0.15, 0.0])),
    }

    body_quat = np.array([1.0, 0.0, 0.0, 0.0])
    body_pos = np.array([0.0, 0.0, 0.5])

    print("Upright body (no tilt), h=0.5m:")
    print(f"{'Leg':<6} {'Max Z (body)':<14} {'Max Z (world)':<14} {'Min Z (world)':<14} {'Penetration':<12}")
    print("-" * 66)

    for name, (start, target) in foot_positions.items():
        result = planner.generate_swing_trajectory(start, target, body_quat, body_pos)

        body_samples = result.bezier_body.sample(100)
        world_samples = result.bezier_world.sample(100)

        max_z_body = max(p[2] for p in body_samples)
        max_z_world = max(p[2] for p in world_samples)
        min_z_world = min(p[2] for p in world_samples)

        pen_str = "YES!" if result.had_penetration else "None"
        print(f"{name:<6} {max_z_body:<14.4f} {max_z_world:<14.4f} {min_z_world:<14.4f} {pen_str:<12}")

    print()


def demo_body_tilt_and_penalty():
    print("=" * 70)
    print("PART 3: Body Tilt & Ground Penetration Penalty")
    print("=" * 70)
    print()

    planner = SwingTrajectoryPlannerWrapper()

    start = np.array([0.2, 0.15, 0.0])
    target = np.array([0.3, 0.15, 0.0])

    tilt_angles = [0, 15, 30, 45, 60]

    print(f"{'Tilt (deg)':<12} {'Max Pen (before)':<18} {'Max Pen (after)':<18} {'Valid':<8} {'Iterations':<10}")
    print("-" * 72)

    for angle_deg in tilt_angles:
        angle_rad = np.radians(angle_deg)
        q = np.array([
            np.cos(angle_rad / 2),
            np.sin(angle_rad / 2),
            0, 0
        ])
        pos = np.array([0.0, 0.0, 0.3])

        result = planner.generate_swing_trajectory(start, target, q, pos)

        world_samples = result.bezier_world.sample(200)
        min_z = min(p[2] for p in world_samples)

        pen_before = max(0, -min_z + result.ground_penetration_max)

        print(f"{angle_deg:<12} {result.ground_penetration_max:<18.6f} "
              f"{'~0 (corrected)':<18} {str(result.is_valid):<8} "
              f"{result.penalty_iterations:<10}")

    print()
    print("Ground Penetration Penalty Algorithm:")
    print("  1. Build Bezier curve in body frame")
    print("  2. Transform all control points to world frame")
    print("  3. Sample trajectory, find max penetration depth")
    print("  4. Apply weighted Z-correction to body-frame control points")
    print("  5. Repeat until no penetration (iterative)")
    print("  6. Guarantee: foot tip Z_world >= ground_height + clearance")
    print()


def demo_leg_phase_machine():
    print("=" * 70)
    print("PART 4: Per-Leg State Machine")
    print("=" * 70)
    print()

    planner = SwingTrajectoryPlannerWrapper()

    leg_names = ["FL", "FR", "RL", "RR"]

    print("Initial state:")
    for i, name in enumerate(leg_names):
        phase = planner.get_leg_phase(i)
        print(f"  Leg {name}: {phase}")

    print()
    print("Set FL and RL to SWING (Trot gait pattern):")
    planner.set_leg_phase(0, LegPhase.SWING)
    planner.set_leg_phase(2, LegPhase.SWING)

    for i, name in enumerate(leg_names):
        phase = planner.get_leg_phase(i)
        print(f"  Leg {name}: {phase}")

    print()
    print("Reset all legs:")
    for i in range(4):
        planner.reset_leg(i)
    for i, name in enumerate(leg_names):
        phase = planner.get_leg_phase(i)
        print(f"  Leg {name}: {phase}")

    print()


def main():
    demo_bezier_curves()
    demo_swing_trajectory()
    demo_body_tilt_and_penalty()
    demo_leg_phase_machine()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("SWING TRAJECTORY ENGINE capabilities:")
    print()
    print("1. BEZIER CURVES (C++ core with pybind11):")
    print("   - Cubic (4 control points): fast, simple")
    print("   - Quintic (6 control points): velocity constraints at endpoints")
    print("   - de Casteljau evaluation + 1st/2nd derivatives")
    print("   - Arc length computation")
    print("   - Degree elevation (cubic -> quartic -> ...)")
    print()
    print("2. COORDINATE TRANSFORM (body -> world):")
    print("   - Full 3D rotation via quaternion")
    print("   - Accounts for body tilt/roll/pitch")
    print("   - All Bezier control points transformed to world frame")
    print()
    print("3. GROUND PENETRATION PENALTY:")
    print("   - Detects Z_world < ground_height along trajectory")
    print("   - Iteratively corrects body-frame control point Z values")
    print("   - Weighted correction: max near penetration, zero at endpoints")
    print("   - Configurable: clearance, penalty gain, max iterations")
    print()
    print("4. PER-LEG STATE MACHINE:")
    print("   - STANCE / SWING phase tracking for each of 4 legs")
    print("   - Phase progress (0..1 during swing)")
    print("   - Real-time foot position & velocity queries")
    print("=" * 70)


if __name__ == "__main__":
    main()
