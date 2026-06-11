import sys
import os
import numpy as np

build_dir = r"D:\SOLO-10\87-quadruped-mpc-core\build\python\Release"
sys.path.insert(0, build_dir)

print("Step 1: Importing C++ module...")
try:
    import quadruped_srbd as cpp
    print("SUCCESS: Module imported")
    print(f"  STATE_DIM = {cpp.STATE_DIM}")
    print(f"  INPUT_DIM = {cpp.INPUT_DIM}")
    print(f"  NUM_LEGS = {cpp.NUM_LEGS}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 2: Creating SRBDModel...")
try:
    model = cpp.SRBDModel()
    print("SUCCESS: Model created")
    print(f"  mass = {model.get_mass()}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Setting mass...")
try:
    model.set_mass(12.0)
    print(f"SUCCESS: mass = {model.get_mass()}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Testing skew_symmetric...")
try:
    v = np.array([1.0, 2.0, 3.0])
    skew = model.skew_symmetric(v)
    print(f"SUCCESS: skew matrix =\n{skew}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Building state...")
try:
    pos = np.array([0.0, 0.0, 0.5])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    lin_vel = np.zeros(3)
    ang_vel = np.zeros(3)
    state = model.build_state(pos, quat, lin_vel, ang_vel)
    print(f"SUCCESS: state = {state}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Testing continuous_dynamics...")
try:
    foot_positions = [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, -0.1, 0.0]),
        np.array([-0.2, 0.1, 0.0]),
        np.array([-0.2, -0.1, 0.0]),
    ]
    forces = np.zeros(12)
    deriv = model.continuous_dynamics(state, forces, foot_positions)
    print(f"SUCCESS: state_deriv = {deriv}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 7: Testing continuous_A_matrix...")
try:
    contact = [True, True, True, True]
    A = model.continuous_A_matrix(state, foot_positions, contact)
    print(f"SUCCESS: A matrix shape = {A.shape}")
    print(f"  A norm = {np.linalg.norm(A)}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 8: Testing continuous_B_matrix...")
try:
    B = model.continuous_B_matrix(state, foot_positions, contact)
    print(f"SUCCESS: B matrix shape = {B.shape}")
    print(f"  B norm = {np.linalg.norm(B)}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")
