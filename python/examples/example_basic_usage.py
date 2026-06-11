"""
Example: SRBD Model Basic Usage

Demonstrates:
1. Creating an SRBD model
2. Building state vectors
3. Computing continuous dynamics
4. Building A and B state-space matrices
5. RK4 integration
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

from quadruped_srbd_py import SRBDWrapper


def main():
    print("=" * 60)
    print("Quadruped SRBD Model - Basic Example")
    print("=" * 60)

    model = SRBDWrapper(mass=12.0)
    print(f"\nRobot mass: {model.mass} kg")
    print(f"Inertia matrix:\n{model.inertia}")
    print(f"Gravity: {model.gravity}")

    pos = np.array([0.0, 0.0, 0.5])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    lin_vel = np.array([0.0, 0.0, 0.0])
    ang_vel = np.array([0.0, 0.0, 0.0])

    state = model.build_state(pos, quat, lin_vel, ang_vel)
    print(f"\nInitial state (13D):\n{state}")

    foot_positions = [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, -0.1, 0.0]),
        np.array([-0.2, 0.1, 0.0]),
        np.array([-0.2, -0.1, 0.0]),
    ]

    contact = [True, True, True, True]

    mg = model.mass * 9.81
    forces = np.zeros(12)
    for i in range(4):
        forces[i * 3 + 2] = mg / 4.0

    print(f"\nContact forces (12D): {forces}")

    state_deriv = model.continuous_dynamics(state, forces, foot_positions)
    print(f"\nState derivative (continuous dynamics):")
    print(f"  d(pos)/dt = {state_deriv[0:3]}")
    print(f"  d(quat)/dt = {state_deriv[3:7]}")
    print(f"  d(vel)/dt = {state_deriv[7:10]}")
    print(f"  d(ang_vel)/dt = {state_deriv[10:13]}")

    A, B = model.continuous_AB(state, foot_positions, contact)
    print(f"\nContinuous A matrix ({A.shape[0]}x{A.shape[1]}):")
    print(f"  Norm: {np.linalg.norm(A):.4f}")
    print(f"  Top-left 6x6:\n{A[:6, :6]}")
    
    print(f"\nContinuous B matrix ({B.shape[0]}x{B.shape[1]}):")
    print(f"  Norm: {np.linalg.norm(B):.4f}")
    print(f"  Velocity-to-force block (rows 7-10):")
    print(B[7:10, :])

    dt = 0.01
    A_d, B_d = model.discrete_AB(state, foot_positions, contact, dt)
    print(f"\nDiscrete A matrix (dt={dt}s):")
    print(f"  Deviation from identity: {np.linalg.norm(A_d - np.eye(13)):.6f}")

    print(f"\nDiscrete B matrix (dt={dt}s):")
    print(f"  Norm: {np.linalg.norm(B_d):.6f}")

    print("\n" + "=" * 60)
    print("Free fall simulation (RK4)")
    print("=" * 60)

    zero_forces = np.zeros(12)
    s = state.copy()
    dt_sim = 0.001
    steps = 100

    for i in range(steps):
        s = model.integrate_rk4(s, zero_forces, foot_positions, dt_sim)

    t_final = steps * dt_sim
    z_analytical = state[2] + 0.5 * (-9.81) * t_final**2
    vz_analytical = -9.81 * t_final

    print(f"After {t_final:.3f}s:")
    print(f"  Numerical z: {s[2]:.6f} m")
    print(f"  Analytical z: {z_analytical:.6f} m")
    print(f"  Error: {abs(s[2] - z_analytical):.2e} m")
    print(f"  Numerical vz: {s[9]:.6f} m/s")
    print(f"  Analytical vz: {vz_analytical:.6f} m/s")
    print(f"  Error: {abs(s[9] - vz_analytical):.2e} m/s")

    skew = SRBDWrapper.skew_symmetric(np.array([1.0, 2.0, 3.0]))
    print(f"\nSkew-symmetric matrix test:")
    print(skew)
    print(f"  Is skew-symmetric: {np.allclose(skew, -skew.T)}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
