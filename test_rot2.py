import numpy as np

def _compute_camera_rotation(eye, target):
    eye = np.array(eye, dtype=float)
    tgt = np.array(target, dtype=float)
    forward = tgt - eye
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(z_axis, up)) > 0.999:
        if z_axis[2] > 0: # looking up
            up = np.array([1.0, 0.0, 0.0])
        else: # looking down
            up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    mat = np.column_stack((x_axis, y_axis, z_axis))

    sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(mat[2, 1], mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0
    return [np.degrees(x), np.degrees(y), np.degrees(z)]

print(f"Top: {_compute_camera_rotation([0.5, 0.0, 1.2], [0.5, 0.0, 0.0])}")
print(f"Front: {_compute_camera_rotation([1.2, 0.0, 0.65], [0.5, 0.0, 0.0])}")
print(f"Side: {_compute_camera_rotation([0.5, -1.0, 0.65], [0.5, 0.0, 0.0])}")
