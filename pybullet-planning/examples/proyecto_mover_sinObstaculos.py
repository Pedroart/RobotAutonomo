import cv2
import numpy as np
import pybullet as p
import time
from math import pi
from pybullet_tools.utils import (
    connect, disconnect, load_model, create_box, set_pose,
    Pose, Point, BLOCK_URDF, DRAKE_IIWA_URDF, add_data_path,
    quat_from_euler, get_link_pose, point_from_pose, quat_from_pose,
    multiply, get_image, stable_z, WorldSaver, update_state, disable_real_time,
    draw_global_system, wait_if_gui, get_joint_positions, set_joint_positions, get_movable_joints,
    inverse_kinematics, plan_direct_joint_motion
)

from pybullet_tools.kuka_primitives import (
    BodyPose, BodyConf, Command,BodyPath,
    get_grasp_gen, get_ik_fn,
    get_free_motion_gen, get_holding_motion_gen
)

import threading

def total_path_length(command: Command):
    return sum(len(bp.path) for bp in command.body_paths if hasattr(bp, 'path'))


class EnfermeriaRobot:
    def __init__(self):
        
        self.camera_frame = None
        self.camera_lock = threading.Lock()
        self.camera_thread_running = True
        connect(use_gui=True)
        disable_real_time()
        add_data_path()
        draw_global_system()
        self.floor = p.loadURDF('plane.urdf')
        self._setup_environment()
        self._setup_robot()

    def _setup_environment(self):
        self.pedestal = create_box(0.5, 0.5, 1.0, color=(0.3, 0.3, 0.3, 1))
        set_pose(self.pedestal, Pose(Point(x=0.0, y=0.0, z=0.5)))

        self.mesa = load_model('models/table_collision/table.urdf', fixed_base=True)
        set_pose(self.mesa, Pose(Point(x=1.1, y=0.0, z=0.0), [0, 0, pi/2]))

        self.cama = load_model('models/hospital_bed.urdf', fixed_base=True)
        set_pose(self.cama, Pose(Point(x=-1.1, y=-1.8, z=0.0), [0, 0, -pi/2]))

        self.estanteria = load_model('models/bookcase.urdf', fixed_base=True)
        set_pose(self.estanteria, Pose(Point(x=-0.5, y=1, z=0.0)))

        self.objetos = []
        for i in range(1):
            block = load_model(BLOCK_URDF, fixed_base=False)
            x_offset = 0.5
            y_offset = (i - 1) * 0.3
            set_pose(block, Pose(Point(x=x_offset, y=y_offset, z=0.73)))
            self.objetos.append(block)

    def _setup_robot(self):
        self.robot = load_model(DRAKE_IIWA_URDF, fixed_base=True)
        set_pose(self.robot, Pose(Point(x=0.0, y=0.0, z=1.0)))
        joint_positions = [0, pi/4, 0, -pi/2, 0, 0, 0]
        for i, q in enumerate(joint_positions, start=1):
            p.resetJointState(self.robot, i, q)

        self.end_effector_index = 9
        #self.update_kinect()
        '''
        self.kinect_model = load_model('models/kinect/kinect.urdf', fixed_base=False)
    
        # Desactivar colisiones y hacer sin masa
        p.changeDynamics(self.kinect_model, -1, mass=0, lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setCollisionFilterGroupMask(self.kinect_model, -1, 0, 0)


        self.kinect_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.end_effector_index,
            childBodyUniqueId=self.kinect_model,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.1],  # offset desde el end effector
            childFramePosition=[0, 0, 0]      # origen del modelo Kinect
        )
        '''

    
    '''
    def update_kinect(self):
        cam_pose = get_link_pose(self.robot, self.end_effector_index)
        offset_pose = Pose(Point(x=0, y=0, z=0.1))
        kinect_pose = multiply(cam_pose, offset_pose)
        if hasattr(self, 'kinect_model'):
            set_pose(self.kinect_model, kinect_pose)
        else:
            self.kinect_model = load_model('models/kinect/kinect.urdf', fixed_base=True)
            set_pose(self.kinect_model, kinect_pose)
    '''

    def show_camera_view(self):
        cam_pose = get_link_pose(self.robot, self.end_effector_index)
        cam_pos = point_from_pose(cam_pose)
        cam_rot = quat_from_pose(cam_pose)

        rot_matrix = p.getMatrixFromQuaternion(cam_rot)
        cam_forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        cam_up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]
        target_pos = [cam_pos[i] + 0.2 * cam_forward[i] for i in range(3)]

        image = get_image(camera_pos=cam_pos, target_pos=target_pos, vertical_fov=20.0)
        rgb_img = image.rgbPixels.astype(np.uint8)

        if rgb_img.shape[-1] == 4:
            rgb_img = rgb_img[:, :, :3]
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        cv2.imshow("Vista Kinect", rgb_img)
        cv2.waitKey(1)

    def start_camera_loop(self):
        def camera_loop():
            while self.camera_thread_running:
                cam_pose = get_link_pose(self.robot, self.end_effector_index)
                cam_pos = point_from_pose(cam_pose)
                cam_rot = quat_from_pose(cam_pose)

                rot_matrix = p.getMatrixFromQuaternion(cam_rot)
                cam_forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
                target_pos = [cam_pos[i] + 0.2 * cam_forward[i] for i in range(3)]

                image = get_image(camera_pos=cam_pos, target_pos=target_pos, vertical_fov=20.0)
                rgb_img = image.rgbPixels.astype(np.uint8)

                if rgb_img.shape[-1] == 4:
                    rgb_img = rgb_img[:, :, :3]
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                with self.camera_lock:
                    self.camera_frame = rgb_img

                time.sleep(0.1)

        threading.Thread(target=camera_loop, daemon=True).start()

    def start_camera_display_loop(self):
        def display_loop():
            while self.camera_thread_running:
                with self.camera_lock:
                    frame = self.camera_frame.copy() if self.camera_frame is not None else None

                if frame is not None:
                    cv2.imshow("Vista en Tiempo Real", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.camera_thread_running = False
                        break

                time.sleep(0.05)

        threading.Thread(target=display_loop, daemon=True).start()


    def plan_pick_and_place(self, obj, destino_pose):
        grasp_gen = get_grasp_gen(self.robot, 'top')
        ik_fn = get_ik_fn(self.robot, fixed=[self.floor, self.mesa, self.cama, self.estanteria], teleport=False)
        free_motion_fn = get_free_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama, obj], teleport=False)
        holding_motion_fn = get_holding_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama], teleport=False)

        pose0 = BodyPose(obj)
        conf0 = BodyConf(self.robot)
        saved_world = WorldSaver()

        best_command = None
        best_length = float('inf')

        for grasp, in grasp_gen(obj):
            saved_world.restore()
            pose0.assign()

            result1 = ik_fn(obj, pose0, grasp)
            if result1 is None:
                continue
            conf1, path2 = result1

            result2 = free_motion_fn(conf0, conf1)
            if result2 is None:
                continue
            path1, = result2

            pose1 = BodyPose(obj)
            pose1.assign()
            set_pose(obj, destino_pose)

            result3 = holding_motion_fn(conf1, conf0, obj, grasp)
            if result3 is None:
                continue
            path3, = result3

            command = Command(path1.body_paths + path2.body_paths + path3.body_paths)

            steps = total_path_length(command)

            if steps < best_length:
                best_length = steps
                best_command = command

        return best_command


    def plan_and_execute_motion_to_pose(self, target_pose: Pose):
        movable_joints = get_movable_joints(self.robot)
        current_conf = BodyConf(self.robot)
        
        saved_world = WorldSaver()

        # IK sin grasp
        q_target = inverse_kinematics(self.robot, self.end_effector_index, target_pose)
        if q_target is None:
            print("❌ IK falló")
            return

        path = plan_direct_joint_motion(self.robot, movable_joints, q_target, obstacles=[self.floor, self.mesa, self.cama, self.estanteria])
        if path is None:
            print("❌ Planificación de movimiento fallida")
            return
        saved_world.restore()
        update_state()
        command = Command([BodyPath(self.robot, path)])
        self.execute_with_camera_update(command.refine(num_steps=30), time_step=0.05)

    def plan_free_motion_to_object(self, obj, destino_pose):
        # Obtener IK y planificación sin grasp
        ik_fn = get_ik_fn(self.robot, fixed=[self.floor, self.mesa, self.cama, self.estanteria], teleport=False)
        free_motion_fn = get_free_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama, obj], teleport=False)

        pose0 = BodyPose(obj)  # Pose actual del objeto
        conf0 = BodyConf(self.robot)  # Configuración actual del robot
        saved_world = WorldSaver()

        best_command = None
        best_length = float('inf')

        saved_world.restore()
        pose0.assign()

        # No grasp
        result1 = ik_fn(obj, pose0, grasp=None)
        if result1 is None:
            print("❌ IK falló")
            return None
        conf1, path_to_target = result1

        result2 = free_motion_fn(conf0, conf1)
        if result2 is None:
            print("❌ Planificación falló")
            return None
        path_free, = result2

        # Crear comando solo con movimiento libre
        command = Command(path_free.body_paths + path_to_target.body_paths)
        steps = total_path_length(command)

        if steps < best_length:
            best_length = steps
            best_command = command

        return best_command


    def pick_and_place(self, obj_index, destino_pose):
        objeto = self.objetos[obj_index]
        saved_world = WorldSaver()
        command = self.plan_pick_and_place(objeto, destino_pose)
        if command is None:
            print("❌ No se pudo planear")
            return
        saved_world.restore()
        update_state()
        #command.refine(num_steps=10)
        #self.execute_with_camera_update(command)

        command.refine(num_steps=10).execute(time_step=0.1)
        #self.update_kinect()


    def ejecutar_trayectoria_cartesiana(self, poses: list[Pose], delay: float = 0.02, steps_per_segment: int = 100):
        """
        Ejecuta una trayectoria cartesiana suavemente, interpolando entre configuraciones.
        """
        from pybullet_tools.utils import (
            get_movable_joints, draw_pose, remove_handles
        )

        joint_path = []
        handles = []
        joints = get_movable_joints(self.robot)

        # Resolver IK para cada pose
        for i, pose in enumerate(poses):
            q = inverse_kinematics(self.robot, self.end_effector_index, pose)
            if q is None:
                print(f"❌ IK falló para pose {i}")
                continue
            joint_path.append(q)
            handles.append(draw_pose(pose, length=0.1))

        if len(joint_path) < 2:
            print("❌ Se necesitan al menos 2 configuraciones.")
            return

        print(f"✅ Ejecutando trayectoria con interpolación entre {len(joint_path)} poses...")

        # Ejecutar interpolando entre cada par consecutivo de configuraciones
        for i in range(len(joint_path) - 1):
            q_start = joint_path[i]
            q_end = joint_path[i + 1]
            for alpha in np.linspace(0, 1, steps_per_segment):
                q_interp = interpolate_configs(q_start, q_end, alpha)
                set_joint_positions(self.robot, joints, q_interp)
                p.stepSimulation()
                time.sleep(delay)

        remove_handles(handles)



    def move_object_to(self, obj_index, pose):
        if 0 <= obj_index < len(self.objetos):
            set_pose(self.objetos[obj_index], pose)

    def shutdown(self):
        self.camera_thread_running = False
        cv2.destroyAllWindows()
        wait_if_gui()
        disconnect()

def quat_from_euler_safe(euler):
    if not isinstance(euler, (list, tuple)) or len(euler) != 3:
        raise ValueError("❌ 'euler' debe tener 3 valores: [roll, pitch, yaw]")
    return p.getQuaternionFromEuler(euler)

def interpolate_configs(q1, q2, alpha):
    """
    Interpola línea recta entre dos configuraciones articulares.
    q1, q2: listas de ángulos articulares.
    alpha: valor entre 0 y 1.
    """
    return [(1 - alpha) * a + alpha * b for a, b in zip(q1, q2)]


if __name__ == '__main__':
    from pybullet_tools.utils import Pose, Point, quat_from_euler
    from math import pi

    robot = EnfermeriaRobot()
    # esperar entorno...
    
    from math import cos, sin, pi
    poses = []
    radio = 0.5
    q = [0, pi/2, 0]
    for t in np.linspace(0, 2*pi, 20):
        x = 0 + radio * cos(t)
        y = 0 + radio * sin(t)
        z = 1.2 + 0.01 * t  # leve subida
        poses.append(Pose(Point(x, y, z), q))
    
    robot.ejecutar_trayectoria_cartesiana(poses)
    robot.shutdown()


