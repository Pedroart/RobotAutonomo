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
    draw_global_system, wait_if_gui
)

from pybullet_tools.kuka_primitives import (
    BodyPose, BodyConf, Command,
    get_grasp_gen, get_ik_fn,
    get_free_motion_gen, get_holding_motion_gen
)


class EnfermeriaRobot:
    def __init__(self):
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
        for i in range(3):
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
        self.update_kinect()

    def update_kinect(self):
        cam_pose = get_link_pose(self.robot, self.end_effector_index)
        offset_pose = Pose(Point(x=0, y=0, z=0.1))
        kinect_pose = multiply(cam_pose, offset_pose)
        if hasattr(self, 'kinect_model'):
            set_pose(self.kinect_model, kinect_pose)
        else:
            self.kinect_model = load_model('models/kinect/kinect.urdf', fixed_base=True)
            set_pose(self.kinect_model, kinect_pose)

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


    def plan_pick_and_place(self, obj, destino_pose):
        grasp_gen = get_grasp_gen(self.robot, 'top')
        ik_fn = get_ik_fn(self.robot, fixed=[self.floor, self.mesa, self.cama, self.estanteria], teleport=False)
        free_motion_fn = get_free_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama, obj], teleport=False)
        holding_motion_fn = get_holding_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama], teleport=False)

        pose0 = BodyPose(obj)
        conf0 = BodyConf(self.robot)
        saved_world = WorldSaver()

        for grasp, in grasp_gen(obj):
            saved_world.restore()
            pose0.assign()  # ← asegura pose inicial
            result1 = ik_fn(obj, pose0, grasp)
            if result1 is None:
                continue
            conf1, path2 = result1
            result2 = free_motion_fn(conf0, conf1)
            if result2 is None:
                continue
            path1, = result2

            # Simular destino sin mover aún
            pose1 = BodyPose(obj)
            pose1.assign()  # copia pose actual
            set_pose(obj, destino_pose)
            result3 = holding_motion_fn(conf1, conf0, obj, grasp)
            if result3 is None:
                continue
            path3, = result3
            return Command(path1.body_paths + path2.body_paths + path3.body_paths)
        
        return None


    def execute_with_camera_update(self, command: Command, time_step=0.01):
        for bp in command.body_paths:
            # Filtrar solo BodyPath válidos (que tengan atributo path)
            if not hasattr(bp, 'path') or not hasattr(bp, 'body'):
                continue
            body = bp.body
            for conf in bp.path:
                for j, q in enumerate(conf):
                    p.resetJointState(body, j, q)
                self.update_kinect()
                self.show_camera_view()  # opcional
                p.stepSimulation()
                time.sleep(time_step)





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





    def move_object_to(self, obj_index, pose):
        if 0 <= obj_index < len(self.objetos):
            set_pose(self.objetos[obj_index], pose)

    def shutdown(self):
        cv2.destroyAllWindows()
        wait_if_gui()
        disconnect()

if __name__ == '__main__':
    robot = EnfermeriaRobot()
    

    # Esperar a que PyBullet cargue todo
    print("🕒 Esperando carga del entorno...")
    for _ in range(10):
        robot.show_camera_view()
        p.stepSimulation()
        time.sleep(0.01)  # 1s total

    time.sleep(1.0) 
    update_state()

    destino = Pose(Point(x=0.5, y=0.3, z=1))
    robot.pick_and_place(obj_index=0, destino_pose=destino)

    robot.shutdown()

