import sys
import numpy as np

build_dir = r"D:\SOLO-10\87-quadruped-mpc-core\build\python\Release"
sys.path.insert(0, build_dir)

print("Step 1: Import module...")
import quadruped_srbd as cpp
print("OK")

print("\nStep 2: Create model...")
model = cpp.SRBDModel()
print("OK")

print("\nStep 3: get_mass...")
m = model.get_mass()
print(f"OK, mass={m}")

print("\nStep 4: get_inertia...")
try:
    I = model.get_inertia()
    print(f"OK, inertia shape={I.shape}")
    print(I)
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: get_gravity...")
try:
    g = model.get_gravity()
    print(f"OK, gravity={g}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: set_mass...")
try:
    model.set_mass(15.0)
    print(f"OK, new mass={model.get_mass()}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll basic tests passed!")
