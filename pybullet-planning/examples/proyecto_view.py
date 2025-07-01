from math import pi
import pybullet as p
import cv2
import numpy as np
from pybullet_tools.utils import (
    connect, disconnect, load_model, create_box, set_pose,
    Pose, Point, BLOCK_URDF, DRAKE_IIWA_URDF, wait_if_gui,
    add_data_path, quat_from_euler,draw_pose, euler_from_quat,
    get_link_pose, point_from_pose, quat_from_pose, get_image, multiply
)

def quat_multiply(q1, q2):
    """Multiplica dos cuaterniones (q2 * q1)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
        w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
        w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
    ]

# Conexión
connect(use_gui=True)
add_data_path()

# Suelo
floor = p.loadURDF('plane.urdf')

# Pedestal
pedestal = create_box(0.5, 0.5, 1.0, color=(0.3, 0.3, 0.3, 1))
set_pose(pedestal, Pose(Point(x=0.0, y=0.0, z=0.5)))

# Robot


robot = load_model(DRAKE_IIWA_URDF, fixed_base=True)
set_pose(robot, Pose(Point(x=0.0, y=0.0, z=1.0)))

joint_positions = [0, pi/4, 0, -pi/2, 0, 0, 0]

# Aplicar a los joints activos (1 al 7)
for i, q in enumerate(joint_positions, start=1):
    p.resetJointState(robot, i, q)

# Mesa
mesa = load_model('models/table_collision/table.urdf', fixed_base=True)
set_pose(mesa, Pose(Point(x=1.1, y=0.0, z=0.0), [0, 0, pi/2]))

# Mesa
mesa = load_model('models/hospital_bed.urdf', fixed_base=True)
set_pose(mesa, Pose(Point(x=-1.1, y=-1.8, z=0.0), [0, 0, -pi/2]))


objectos = []
for i in range(3):
    block = load_model(BLOCK_URDF, fixed_base=False)
    x_offset = 1.0
    y_offset = (i - 1) * 0.3
    set_pose(block, Pose(Point(x=x_offset, y=y_offset, z=0.73)))

    objectos.append(block)

# Estantería
try:
    estanteria = load_model('models/bookcase.urdf')
    set_pose(estanteria, Pose(Point(x=-0.5, y=1, z=0.0)))
except:
    print("⚠️ Estantería no encontrada, omitiendo.")

# ----------- CÁMARA EN EL EFECTOR FINAL ---------------
end_effector_index = 9  # usual para KUKA IIWA

cam_pose = get_link_pose(robot, end_effector_index)
cam_pos = point_from_pose(cam_pose)
cam_rot = quat_from_pose(cam_pose)

offset_pose = Pose(Point(x=0, y=0, z=0.1))  # solo traslación
kinect_pose = multiply(cam_pose, offset_pose)

kinect_model = load_model('models/kinect/kinect.urdf', fixed_base=True)
set_pose(kinect_model, kinect_pose)

cam_rot_matrix = p.getMatrixFromQuaternion(cam_rot)
cam_forward = [cam_rot_matrix[0], cam_rot_matrix[3], cam_rot_matrix[6]]
cam_up = [cam_rot_matrix[2], cam_rot_matrix[5], cam_rot_matrix[8]]
target_pos = [cam_pos[i] + 0.2 * cam_forward[i] for i in range(3)]

image = get_image(camera_pos=cam_pos, target_pos=target_pos, vertical_fov=20.0)

rgb_img = image.rgbPixels.astype(np.uint8)

# Si tiene 4 canales (RGBA), y quieres mostrar en BGR:
if rgb_img.shape[-1] == 4:
    rgb_img = rgb_img[:, :, :3]  # Elimina canal alfa
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

cv2.imshow("Vista", rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------------

wait_if_gui()
disconnect()
