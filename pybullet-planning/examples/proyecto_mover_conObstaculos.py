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
    draw_global_system, wait_if_gui, draw_pose
)

from pybullet_tools.kuka_primitives import (
    BodyPose, BodyConf, Command,
    get_grasp_gen, get_ik_fn,
    get_free_motion_gen, get_holding_motion_gen
)

import threading

def total_path_length(command: Command):
    return sum(len(bp.path) for bp in command.body_paths if hasattr(bp, 'path'))


def try_multiple_orientations(obj, base_position, grasp, ik_fn, num_angles=20):
    pitch_angles = np.linspace(-np.pi/4, np.pi/4, num_angles)
    yaw_angles = np.linspace(0, 2*np.pi, num_angles)

    for pitch in pitch_angles:
        for yaw in yaw_angles:
            orientation = [0, pitch, yaw]  # roll fijo, se varía pitch y yaw
            destino_pose = Pose(base_position, orientation)
            set_pose(obj, destino_pose)
            pose_destino = BodyPose(obj)
            result = ik_fn(obj, pose_destino, grasp)
            if result is not None:
                print(f"✅ IK exitosa con pitch={pitch:.2f}, yaw={yaw:.2f}")
                return result
    print("❌ No se logró ninguna IK con rotaciones")
    return None



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
        for i in range(3):
            block = load_model(BLOCK_URDF, fixed_base=False)
            x_offset = 0.5
            y_offset = (i - 1) * 0.3
            set_pose(block, Pose(Point(x=x_offset, y=y_offset, z=0.73)))
            self.objetos.append(block)

    def _setup_robot(self):
        self.robot = load_model(DRAKE_IIWA_URDF, fixed_base=True)
        set_pose(self.robot, Pose(Point(x=0.0, y=0.0, z=1.0)))
        joint_positions = [0, pi/4, 0, 0, 0, 0, 0]
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
        from pybullet_tools.utils import draw_pose, set_pose
        from time import time

        grasp_gen = get_grasp_gen(self.robot, 'top')
        ik_fn = get_ik_fn(self.robot, fixed=[self.floor, self.mesa, self.cama, self.estanteria], teleport=False)
        free_motion_fn = get_free_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama, obj], teleport=False)
        holding_motion_fn = get_holding_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama], teleport=False)

        pose0 = BodyPose(obj)
        conf0 = BodyConf(self.robot)
        saved_world = WorldSaver()

        best_command = None
        best_length = float('inf')
        best_partial_command = None
        best_partial_steps = float('inf')

        draw_pose(destino_pose, length=0.2)

        for i, (grasp,) in enumerate(grasp_gen(obj)):
            print(f"\n🟢 Probando grasp {i}")
            saved_world.restore()
            pose0.assign()

            # Fase 1: IK para grasp
            result1 = ik_fn(obj, pose0, grasp)
            if result1 is None:
                print("❌ Fase 1: IK para agarrar falló")
                continue
            conf1, path2 = result1
            print("✅ Fase 1 completada")

            # Fase 2: trayectoria libre hacia grasp
            result2 = free_motion_fn(conf0, conf1)
            if result2 is None:
                print("❌ Fase 2: movimiento libre hacia grasp falló")
                continue
            path1, = result2
            print("✅ Fase 2 completada")

            # Guardar mejor intento parcial (grasp válido pero sin colocar)
            partial_command = Command(path1.body_paths + path2.body_paths)
            steps_partial = total_path_length(partial_command)
            if steps_partial < best_partial_steps:
                best_partial_steps = steps_partial
                best_partial_command = partial_command
                print("💾 Guardando como mejor intento parcial")

            # Fase 2b: regreso opcional (omitido, pero conservado como referencia)
            path_back = []

            # Fase 3a: generar nueva pose objetivo con orientación del grasp
            position = destino_pose[0]
            orientation = quat_from_euler(0, 0, 0)
            set_pose(obj, (position, orientation))
            pose_destino = BodyPose(obj)

            # Fase 3b: IK para colocar
            result_dest = ik_fn(obj, pose_destino, grasp)
            if result_dest is None:
                print("❌ Fase 3a: IK para soltar en destino falló")
                continue
            conf_dest, _ = result_dest
            print("✅ Fase 3a completada")

            # Fase 3c: trayectoria con objeto hacia destino
            result3 = holding_motion_fn(conf1, conf_dest, obj, grasp)
            if result3 is None:
                print("❌ Fase 3b: trayecto con objeto hacia destino falló")
                continue
            path3, = result3
            print("✅ Fase 3b completada")

            # Combinar todo
            command = Command(path1.body_paths + path2.body_paths + path_back + path3.body_paths)
            steps = total_path_length(command)
            print(f"🎯 Plan válido con {steps} pasos")

            if steps < best_length:
                best_length = steps
                best_command = command
                print("✅ Guardando como mejor plan completo")

        if best_command is not None:
            print("✅ Planificación completa con éxito.")
            return best_command
        elif best_partial_command is not None:
            print("⚠️ Ejecutando intento parcial (sin colocar el objeto).")
            return best_partial_command
        else:
            print("❌ Ningún plan parcial ni completo fue posible.")
            return None


    def plan_pick_and_place2(self, obj, destino_pose):
        from pybullet_tools.utils import draw_pose, set_pose, get_pose
        from time import time


        draw_pose(destino_pose, length=0.2)

        grasp_gen = get_grasp_gen(self.robot, 'top')
        ik_fn = get_ik_fn(self.robot, fixed=[self.floor, self.mesa, self.cama, self.estanteria], teleport=False)
        free_motion_fn = get_free_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama, obj], teleport=False)
        holding_motion_fn = get_holding_motion_gen(self.robot, fixed=[self.floor, self.mesa, self.cama], teleport=False)

        pose0 = BodyPose(obj)
        conf0 = BodyConf(self.robot)
        saved_world = WorldSaver()

        best_command = None
        best_length = float('inf')

        for i, (grasp,) in enumerate(grasp_gen(obj)):
            print(f"\n🟢 Probando grasp {i}")
            saved_world.restore()
            pose0.assign()

            # Fase 1: IK para agarrar en la posición actual
            result1 = ik_fn(obj, pose0, grasp)
            if result1 is None:
                print("❌ Fase 1: IK para agarrar falló")
                continue
            conf1, path_to_grasp = result1
            print("✅ Fase 1 completada")

            # Fase 2: trayectoria libre desde inicio hasta posición de grasp
            result2 = free_motion_fn(conf0, conf1)
            if result2 is None:
                print("❌ Fase 2: trayectoria libre hacia grasp falló")
                continue
            path_free = result2[0]
            print("✅ Fase 2 completada")

            # Fase 3: simular colocación invirtiendo el plan de grasp desde destino
            print("🔄 Intentando plan inverso desde destino")
            saved_world.restore()  # Restauramos para evitar cambios anteriores

            # Teletransportamos el objeto al destino con misma orientación del grasp
            '''position = destino_pose[0]
            orientation = grasp.grasp_pose[1]  # mantener orientación de grasp
            set_pose(obj, (position, orientation))
            pose_destino = BodyPose(obj)'''
            # DEBUG: colocar objeto en su misma pose original (pick)
            original_pose = pose0.pose  # (position, orientation)
            set_pose(obj, original_pose)
            pose_destino = BodyPose(obj)

            # IK para grasp en destino
            result3 = ik_fn(obj, pose_destino, grasp)
            if result3 is None:
                print("❌ Fase 3: IK en destino falló")
                continue
            conf_dest, path_dest = result3
            print("✅ Fase 3 IK inverso completado")

            # Plan con el objeto hacia la posición de grasp (desde destino hasta conf1)
            result4 = holding_motion_fn(conf_dest, conf1, obj, grasp)
            if result4 is None:
                print("❌ Fase 4: holding motion inverso falló")
                continue
            path_place_inv = result4[0]
            print("✅ Fase 4 completada")

            # 🔁 Invertimos el path para simular colocación
            path_place = path_place_inv.reverse() if hasattr(path_place_inv, 'reverse') else list(reversed(path_place_inv))

            # Construimos el comando completo
            command = Command(path_free.body_paths + path_to_grasp.body_paths + path_place.body_paths)
            steps = total_path_length(command)
            print(f"🎯 Plan válido con {steps} pasos")

            if steps < best_length:
                best_length = steps
                best_command = command
                print("✅ Guardando como mejor plan")

        if best_command is not None:
            print("✅ Planificación completada exitosamente.")
            return best_command
        else:
            print("❌ No se pudo generar ningún plan válido.")
            return None



    def plan_pick_and_place3(self, obj, destino_pose):
        from pybullet_tools.utils import draw_pose, set_pose, get_pose
        from time import time


        draw_pose(destino_pose, length=0.2)

        otros_objetos = [o for o in self.objetos if o != obj]


        grasp_gen = get_grasp_gen(self.robot, 'top')
        ik_fn = get_ik_fn(self.robot, fixed=[*otros_objetos, self.mesa, self.cama, self.estanteria], teleport=False)
        free_motion_fn = get_free_motion_gen(self.robot, fixed=[*otros_objetos,self.mesa, self.cama, obj], teleport=False)
        holding_motion_fn = get_holding_motion_gen(self.robot, fixed=[*otros_objetos, self.mesa, self.cama], teleport=True)

        pose0 = BodyPose(obj)
        conf0 = BodyConf(self.robot)
        saved_world = WorldSaver()

        for i, (grasp,) in enumerate(grasp_gen(obj)):
            print(f"\n🟢 Probando grasp {i}")
            saved_world.restore()
            pose0.assign()

            # Fase 1: IK para agarrar
            result1 = ik_fn(obj, pose0, grasp)
            if result1 is None:
                print("❌ Fase 1: IK falló")
                continue
            conf1, path_to_grasp = result1
            print("✅ Fase 1 completada")

            # Fase 2: movimiento libre hasta el grasp
            result2 = free_motion_fn(conf0, conf1)
            if result2 is None:
                print("❌ Fase 2: trayectoria falló")
                continue
            path_free = result2[0]
            print("✅ Fase 2 completada")

            # Fase 3: mover con objeto al destino
            orientation = destino_pose
            set_pose(obj, orientation)
            pose_destino = BodyPose(obj)

            
            position = destino_pose[0]  # asumiendo destino_pose es una tupla
            result3 = try_multiple_orientations(obj, position, grasp, ik_fn)
            if result3 is None:
                print("❌ Fase 3: IK en destino falló")
                continue
            conf_dest, _ = result3

            print("✅ Fase 3 IK colocación completada")

            result4 = holding_motion_fn(conf1, conf_dest, obj, grasp)
            if result4 is None:
                print("❌ Fase 4: movimiento con objeto falló")
                continue
            path_place = result4[0]
            print("✅ Fase 4 completada")

            # Comando final: pick + move + place
            command = Command(path_free.body_paths + path_to_grasp.body_paths + path_place.body_paths)
            print("✅ Planificación completada exitosamente")
            return command

        print("❌ No se pudo generar un plan válido")
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
                #self.update_kinect()
                #self.show_camera_view()  # opcional
                p.stepSimulation()
                time.sleep(time_step)

    def pick_and_place(self, obj_index, destino_pose):


        objeto = self.objetos[obj_index]
        saved_world = WorldSaver()
        command = self.plan_pick_and_place3(objeto, destino_pose)
        if command is None:
            print("❌ No se pudo planear")
            return
        saved_world.restore()
        update_state()
        #command.refine(num_steps=10)
        #self.execute_with_camera_update(command)

        command.refine(num_steps=10).execute(time_step=0.01)
        #self.update_kinect()




    def move_object_to(self, obj_index, pose):
        if 0 <= obj_index < len(self.objetos):
            set_pose(self.objetos[obj_index], pose)

    def shutdown(self):
        self.camera_thread_running = False
        cv2.destroyAllWindows()
        wait_if_gui()
        disconnect()


if __name__ == '__main__':
    robot = EnfermeriaRobot()
    

    # Esperar a que PyBullet cargue todo
    print("🕒 Esperando carga del entorno...")
    for _ in range(30):
        #robot.show_camera_view()
        p.stepSimulation()
        time.sleep(0.1)  # 1s total

    
    robot.start_camera_loop()           # Captura frames
    robot.start_camera_display_loop()   # Muestra frames en tiempo real

    time.sleep(1.0) 
    update_state()

    destino = Pose(Point(x=0.8, y=-0.2, z=0.9))
    robot.pick_and_place(obj_index=2, destino_pose=destino)

    robot.shutdown()

