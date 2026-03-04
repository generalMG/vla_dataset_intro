import numpy as np
import math

def lookat_euler(eye, target):
    # eye and target are 3D
    # camera looks along -Z, up is +Y
    d = np.array(target) - np.array(eye)
    d = d / np.linalg.norm(d)
    
    # yaw (rotation around Z axis in world coords?)
    # Wait, euler XYZ: rotate X, then Y, then Z.
    
    # let's just use scipy
    from scipy.spatial.transform import Rotation as R
    
    # We want camera's local -Z to align with d
    # local +Z = -d
    z = -d
    
    # Up vector in world
    up = np.array([0, 0, 1.0])
    
    # local +X = cross(up, z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    
    # local +Y = cross(z, x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    
    # Rotation matrix from Local to World
    # R * [0,0,1]^T = z, etc
    # So columns of R are x, y, z
    mat = np.column_stack((x, y, z))
    
    r = R.from_matrix(mat)
    angles = r.as_euler('xyz', degrees=True)
    return angles

print("Top:", lookat_euler([0.5, 0.0, 1.2], [0.5, 0.0, 0.0]))
print("Front:", lookat_euler([1.2, 0.0, 0.65], [0.5, 0.0, 0.0]))
print("Side:", lookat_euler([0.5, -1.0, 0.65], [0.5, 0.0, 0.0]))
