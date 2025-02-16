from collections import OrderedDict

import numpy as np
import random
import os
import json
import glob
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.objects import BoxObject
from robosuite.models.objects.xml_objects import BottleObject
from robosuite.models.objects import CylinderObject
from robosuite.models.objects.xml_objects  import CanObject
from robosuite.models.objects import CapsuleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class Lift(SingleArmEnv):
    
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=200,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=600,
        camera_widths=800,
        camera_depths=500,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):

        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube1_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward
           
            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube1):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward
        



    def _load_model(self):

        super()._load_model()

        # Adjust base pose accordingly
        #arm position
        #xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos = np.array([0.0, 3, 0.3])  # change to your desired position
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        # 定義各種材質
        materials= [
            #"WoodRed": 
            CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
            #"WoodGreen": 
            CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
            #"WoodBlue": 
            CustomMaterial(
            texture="WoodBlue",
            tex_name="WoodBlue",
            mat_name="WoodBlue_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
            #"PlasterPink": 
            CustomMaterial(
            texture="WoodDark",
            tex_name="WoodDark",
            mat_name="WoodDark",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
            
           CustomMaterial(
            texture="Lemon",
            tex_name="Lemon",
            mat_name="Lemon",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
        ]
        
        selected_material = random.choice(materials) 
        
        
        Objects = [
            BoxObject(
            name="box1",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BoxObject(
            name="box2",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BoxObject(
            name="box3",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BoxObject(
            name="box4",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BoxObject(
            name="box5",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
            BallObject(
            name="ball1",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BallObject(
            name="ball2",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BallObject(
            name="ball3",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BallObject(
            name="ball4",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
        BallObject(
            name="ball5",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials),
        ),
            CapsuleObject(
            name="Capsule1",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CapsuleObject(
            name="Capsule2",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CapsuleObject(
            name="Capsule3",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CapsuleObject(
            name="Capsule4",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CapsuleObject(
            name="Capsule5",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CylinderObject(
            name="cylinder1",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CylinderObject(
            name="cylinder2",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CylinderObject(
            name="cylinder3",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
        ),
        CylinderObject(
            name="cylinder4",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
            
        ),
        CylinderObject(
            name="cylinder5",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],  # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=random.choice(materials) ,
            
        ),
        ]
          
        selected_obj = random.choice(Objects)
        Objects.remove(selected_obj)
        self.cube1 = selected_obj
        # 再次选择，由于selected_obj已经被移除，所以不会被再次选中
        
        selected_obj = random.choice(Objects)
        Objects.remove(selected_obj)
        self.cube2 = selected_obj
        selected_obj = random.choice(Objects)
        Objects.remove(selected_obj)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube1)
            self.placement_initializer.add_objects(self.cube2)
            
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube1,
                x_range=[0.5, -0.5],
                y_range=[0.3, -0.3],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube2,
                x_range=[0.5, -0.5],
                y_range=[0.3, -0.3],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube1,self.cube2]
        )

    def _setup_references(self):

        super()._setup_references()

        # Additional object references from this env
        self.cube1_body_id = self.sim.model.body_name2id(self.cube1.root_body)
        self.cube2_body_id = self.sim.model.body_name2id(self.cube2.root_body) 
        
    def _setup_observables(self):
 
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
        # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

        # cube-related observables
            @sensor(modality=modality)
            def cube1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube1_body_id])

            @sensor(modality=modality)
            def cube1_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube1_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube1_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube1_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube1_pos" in obs_cache
                    else np.zeros(3)
                )
                 
        # cube2-related observables
           
            @sensor(modality=modality)
            def cube2_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube2_body_id])

            @sensor(modality=modality)
            def cube2_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube2_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube2_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube2_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube2_pos" in obs_cache
                    else np.zeros(3)
                )


            sensors = [cube1_pos, cube1_quat, gripper_to_cube1_pos, cube2_pos, cube2_quat, gripper_to_cube2_pos]
            names = [s.__name__ for s in sensors]

        # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

            
    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            # 随机化位置
            cube1_pos = [random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35), self.table_offset[2] + 0.01]
            cube1_quat = [1, 0, 0, 0]  # No rotation

            cube2_pos = [random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35), self.table_offset[2] + 0.01]
            cube2_quat = [1, 0, 0, 0]  # No rotation

            # 设置位置和方向
            self.sim.data.set_joint_qpos(self.cube1.joints[0], np.concatenate([np.array(cube1_pos), np.array(cube1_quat)]))
            self.sim.data.set_joint_qpos(self.cube2.joints[0], np.concatenate([np.array(cube2_pos), np.array(cube2_quat)]))
            
            # 保证每次重置只保存一次信息，避免重复调用
            '''
            if not hasattr(self, 'info_saved') or not self.info_saved:
                self.save_objects_info([self.cube1, self.cube2])
                self.info_saved = True'''
            
    def save_objects_info(self, objects_list):
        directory_path = "/home/allen880523/cgu_robot/robosuite/json/"
        json_files = glob.glob(os.path.join(directory_path, '*.json'))
        max_num = 0
        for file in json_files:
            try:
                num = int(os.path.basename(file).split('.')[0])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
        next_num = max_num + 1
        new_filename = os.path.join(directory_path, f"{next_num}.json")

        objects_info = []
        for obj in objects_list:
            obj_id = self.sim.model.body_name2id(obj.root_body)
            position = self.sim.data.body_xpos[obj_id].tolist()
            orientation = self.sim.data.body_xquat[obj_id].tolist()
            obj_dict = {
                "name": obj.name,
                "type": obj.__class__.__name__,
                "material": obj.material.name if hasattr(obj, "material") and hasattr(obj.material, "name") else "unknown",
                "position": position,
                "orientation": orientation,
            }
            objects_info.append(obj_dict)

        with open(new_filename, "w") as file:
            json.dump([objects_info], file, indent=4)
            
    


    def visualize(self, vis_settings):
 
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube1)
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube2)

    def _check_success(self):

        cube_height = self.sim.data.body_xpos[self.cube1_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        cube_height2 = self.sim.data.body_xpos[self.cube2_body_id][2]
        table_height2 = self.model.mujoco_arena.table_offset[2]            

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04, cube_height2 > table_height + 0.04, 
        
        
