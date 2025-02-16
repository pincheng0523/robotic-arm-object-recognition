from collections import OrderedDict

import numpy as np
import random
import os
import json
import glob

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena


from robosuite.models.objects import BallObject
from robosuite.models.objects import CylinderObject
from robosuite.models.objects import BoxObject
from robosuite.models.objects import CylinderObject
from robosuite.models.objects import LemonObject
from robosuite.models.objects import Lemon1Object
from robosuite.models.objects import Lemon2Object
from robosuite.models.objects import Lemon3Object
from robosuite.models.objects import Lemon4Object
from robosuite.models.objects import Lemon5Object
from robosuite.models.objects import Lemon6Object
from robosuite.models.objects import Lemon7Object


from robosuite.models.objects.xml_objects import BottleObject
from robosuite.models.objects.xml_objects import Bottle1Object
from robosuite.models.objects.xml_objects import Bottle2Object
from robosuite.models.objects.xml_objects import Bottle3Object
from robosuite.models.objects.xml_objects import Bottle4Object
from robosuite.models.objects.xml_objects import Bottle5Object
from robosuite.models.objects.xml_objects import Bottle6Object
from robosuite.models.objects.xml_objects import Bottle7Object

from robosuite.models.objects import CerealObject
from robosuite.models.objects import Cereal1Object
from robosuite.models.objects import Cereal2Object
from robosuite.models.objects import Cereal3Object
from robosuite.models.objects import Cereal4Object
from robosuite.models.objects import Cereal5Object
from robosuite.models.objects import Cereal6Object
from robosuite.models.objects import Cereal7Object

from robosuite.models.objects import RoundNutObject
from robosuite.models.objects import RoundNut1Object
from robosuite.models.objects import RoundNut2Object
from robosuite.models.objects import RoundNut3Object
from robosuite.models.objects import RoundNut4Object
from robosuite.models.objects import RoundNut5Object
from robosuite.models.objects import RoundNut6Object
from robosuite.models.objects import RoundNut7Object

from robosuite.models.objects import SquareNutObject
from robosuite.models.objects import SquareNut1Object
from robosuite.models.objects import SquareNut2Object
from robosuite.models.objects import SquareNut3Object
from robosuite.models.objects import SquareNut4Object
from robosuite.models.objects import SquareNut5Object
from robosuite.models.objects import SquareNut6Object
from robosuite.models.objects import SquareNut7Object

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class Lift5(SingleArmEnv):
    
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
        table_arena_xml=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.table_arena_xml = table_arena_xml

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
            
            cube2_pos = self.sim.data.body_xpos[self.cube2_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube2_pos)
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
            xml=self.table_arena_xml  # 使用传递的背景 XML 文件名
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        
        randlemon = [LemonObject, Lemon1Object, Lemon2Object,Lemon3Object,Lemon4Object,Lemon5Object,Lemon6Object,Lemon7Object]
        randbottle = [BottleObject,Bottle1Object,Bottle2Object,Bottle3Object,Bottle4Object,Bottle5Object,Bottle6Object,Bottle7Object]
        randcereal = [CerealObject,Cereal1Object,Cereal2Object,Cereal3Object,Cereal4Object,Cereal5Object,Cereal6Object,Cereal7Object]
        randRoundNutObject = [RoundNutObject,RoundNut1Object,RoundNut2Object,RoundNut3Object,RoundNut4Object,RoundNut5Object,RoundNut6Object,RoundNut7Object]
        randomSquareNutObject = [SquareNutObject,SquareNut1Object,SquareNut2Object,SquareNut3Object,SquareNut4Object,SquareNut5Object,SquareNut6Object,SquareNut7Object]
        
        selectlemon1 = random.choice(randlemon)
        selectlemon2 = random.choice(randlemon)
        selectlemon3 = random.choice(randlemon)
        selectlemon4 = random.choice(randlemon)
        selectlemon5 = random.choice(randlemon)
        
        selectbottle1 = random.choice(randbottle)
        selectbottle2 = random.choice(randbottle)
        selectbottle3 = random.choice(randbottle)
        selectbottle4 = random.choice(randbottle)
        selectbottle5 = random.choice(randbottle)
        
        seclectcereal1 = random.choice(randcereal)
        seclectcereal2 = random.choice(randcereal)
        seclectcereal3 = random.choice(randcereal)
        seclectcereal4 = random.choice(randcereal)
        seclectcereal5 = random.choice(randcereal)
        
        selectrnut1 = random.choice(randRoundNutObject)
        selectrnut2 = random.choice(randRoundNutObject)
        selectrnut3 = random.choice(randRoundNutObject)
        selectrnut4 = random.choice(randRoundNutObject)
        selectrnut5 = random.choice(randRoundNutObject)
        
        selectsnut1 = random.choice(randomSquareNutObject)
        selectsnut2 = random.choice(randomSquareNutObject)
        selectsnut3 = random.choice(randomSquareNutObject)
        selectsnut4 = random.choice(randomSquareNutObject)
        selectsnut5 = random.choice(randomSquareNutObject)
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
            texture="orange",
            tex_name="orange",
            mat_name="orange",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
           CustomMaterial(
            texture="Brass",
            tex_name="Brass",
            mat_name="Brass",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
           CustomMaterial(
            texture="PlasterGray",
            tex_name="PlasterGray",
            mat_name="PlasterGray",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
           CustomMaterial(
            texture="purple",
            tex_name="purple",
            mat_name="purple",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "0.4", "shininess": "0.1"}
        ),
        
        ]
        
        selected_material = random.choice(materials) 
        Objects = [
        BoxObject(
            name="box1",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],    # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        BoxObject(
            name="box2",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],   # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        BoxObject(
            name="box3",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],   # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
           material=selected_material
        ),
        BoxObject(
            name="box4",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],   # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        BoxObject(
            name="box5",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],   # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        
        BallObject(
            name="ball1",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],   # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        BallObject(
            name="ball2",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],    # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        BallObject(
            name="ball3",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],    # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        BallObject(
            name="ball4",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],    # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        BallObject(
            name="ball5",
            size_min=[0.04, 0.04, 0.04],  # [0.015, 0.015, 0.015],
            size_max=[0.04, 0.04, 0.04],    # [0.018, 0.018, 0.018])
            # rgba=[1, 0, 0, 1],
            material=selected_material
        ),
        
        selectlemon1(name = 'lemon1'),
        selectlemon2(name = 'lemon2'),
        selectlemon3(name = 'lemon3'),
        selectlemon4(name = 'lemon4'),
        selectlemon5(name = 'lemon5'),
        
        selectbottle1(name = 'bottle1'),
        selectbottle2(name = 'bottle2'),
        selectbottle3(name = 'bottle3'),
        selectbottle4(name = 'bottle4'),
        selectbottle5(name = 'bottle5'),
        
        seclectcereal1(name = 'cereal1'),
        seclectcereal2(name = 'cereal2'),
        seclectcereal3(name = 'cereal3'),
        seclectcereal4(name = 'cereal4'),
        seclectcereal5(name = 'cereal5'),
        
        
        
        selectrnut1(name = 'rnut1'),
        selectrnut2(name = 'rnut2'),
        selectrnut3(name = 'rnut3'),
        selectrnut4(name = 'rnut4'),
        selectrnut5(name = 'rnut5'),
        
        selectsnut1(name = 'snut1'),
        selectsnut2(name = 'snut2'),
        selectsnut3(name = 'snut3'),
        selectsnut4(name = 'snut4'),
        selectsnut5(name = 'snut5'),
        
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

        self.cube3 = selected_obj
        selected_obj = random.choice(Objects)
        Objects.remove(selected_obj)

        self.cube4 = selected_obj
        selected_obj = random.choice(Objects)
        Objects.remove(selected_obj)
        
        self.cube5 = selected_obj
        
       
       


        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube1)
            self.placement_initializer.add_objects(self.cube2)
            self.placement_initializer.add_objects(self.cube3)
            self.placement_initializer.add_objects(self.cube4)
            self.placement_initializer.add_objects(self.cube5)

            
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube1,
                x_range=[-0.35, 0.35],
                y_range=[-0.35, 0.35],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube2,
                x_range=[-0.35, 0.35],
                y_range=[-0.35, 0.35],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube3,
                x_range=[-0.35, 0.35],
                y_range=[-0.35, 0.35],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            
            
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube4,
                x_range=[-0.35, 0.35],
                y_range=[-0.35, 0.35],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube5,
                x_range=[-0.35, 0.35],
                y_range=[-0.35, 0.35],
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
            mujoco_objects=[self.cube1,self.cube2,self.cube3,self.cube4,self.cube5]
        )

    def _setup_references(self):

        super()._setup_references()

        # Additional object references from this env
        self.cube1_body_id = self.sim.model.body_name2id(self.cube1.root_body)
        self.cube2_body_id = self.sim.model.body_name2id(self.cube2.root_body)
        self.cube3_body_id = self.sim.model.body_name2id(self.cube3.root_body)
        self.cube4_body_id = self.sim.model.body_name2id(self.cube4.root_body)
        self.cube5_body_id = self.sim.model.body_name2id(self.cube5.root_body)
      

    
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

            @sensor(modality=modality)
            def cube3_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube3_body_id])

            @sensor(modality=modality)
            def cube3_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube3_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube3_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube3_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube3_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            
            def cube4_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube4_body_id])

            @sensor(modality=modality)
            def cube4_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube4_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube4_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube4_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube4_pos" in obs_cache
                    else np.zeros(3)
                )
                
            @sensor(modality=modality)
            
            def cube5_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube5_body_id])

            @sensor(modality=modality)
            def cube5_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube5_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube5_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube5_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube5_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cube1_pos, cube1_quat, gripper_to_cube1_pos, cube2_pos, cube2_quat, gripper_to_cube2_pos,cube3_pos, cube3_quat, gripper_to_cube3_pos ,cube4_pos, cube4_quat, gripper_to_cube4_pos,cube5_pos, cube5_quat, gripper_to_cube5_pos]
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
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
 
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube1)
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube2)
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube3)
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube4)
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube5)
            
    def save_objects_info(self, objects_list, xml_file):
        directory_path = "/home/allen880523/cgu_robot/robosuite/test4/"
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
            
            
                     
 
        
        # Determine the material based on the object type
            if isinstance(obj, (LemonObject, Lemon1Object, Lemon2Object,Lemon3Object,Lemon4Object,Lemon5Object,Lemon6Object,Lemon7Object,
                     BottleObject,Bottle1Object,Bottle2Object,Bottle3Object,Bottle4Object,Bottle5Object,Bottle6Object,Bottle7Object,
                     CerealObject,Cereal1Object,Cereal2Object,Cereal3Object,Cereal4Object,Cereal5Object,Cereal6Object,Cereal7Object,
                     RoundNutObject,RoundNut1Object,RoundNut2Object,RoundNut3Object,RoundNut4Object,RoundNut5Object,RoundNut6Object,RoundNut7Object,
                     SquareNutObject,SquareNut1Object,SquareNut2Object,SquareNut3Object,SquareNut4Object,SquareNut5Object,SquareNut6Object,SquareNut7Object)):
                if isinstance(obj, (LemonObject, BottleObject, CerealObject,RoundNutObject,SquareNutObject)):
                    material = "redwood_mat"
                elif isinstance(obj, (Lemon1Object, Bottle1Object, Cereal1Object, RoundNut1Object, SquareNut1Object)):
                    material = "greenwood_mat"
                elif isinstance(obj, (Lemon2Object, Bottle2Object, Cereal2Object, RoundNut2Object, SquareNut2Object)):
                    material = "WoodBlue_mat"
                elif isinstance(obj, (Lemon3Object, Bottle3Object, Cereal3Object, RoundNut3Object,SquareNut3Object)):
                    material = "WoodDark"
                elif isinstance(obj, (Lemon4Object, Bottle4Object, Cereal4Object, RoundNut4Object,SquareNut4Object)):
                    material = "orange"
                elif isinstance(obj, (Lemon5Object, Bottle5Object, Cereal5Object, RoundNut5Object,SquareNut5Object)):
                    material = "Brass"
                elif isinstance(obj, (Lemon6Object, Bottle6Object, Cereal6Object, RoundNut6Object,SquareNut6Object)):
                    material = "purple"
                elif isinstance(obj, (Lemon7Object, Bottle7Object, Cereal7Object, RoundNut7Object,SquareNut7Object)):
                    material = "PlasterGray"
                else:
                    material = "unknown"
            else:
                material = obj.material.name if hasattr(obj, "material") and hasattr(obj.material, "name") else "unknown"
                
            #print(f"Object: {obj.name}, Type: {type(obj)}, Material: {material}")
        
            obj_dict = {
                "name": obj.name,
                "type": obj.__class__.__name__,
                "material": material,
                "position": position,
                "orientation": orientation,
            }
            objects_info.append(obj_dict)
            
        env_info = {
            "objects": objects_info,
            "background_xml":  xml_file # Save the current background XML file name
            }

        with open(new_filename, "w") as file:
           json.dump(env_info, file, indent=4)

    def _check_success(self):

        cube_height = self.sim.data.body_xpos[self.cube1_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        cube_height2 = self.sim.data.body_xpos[self.cube2_body_id][2]
        table_height2 = self.model.mujoco_arena.table_offset[2]
        cube_height3 = self.sim.data.body_xpos[self.cube3_body_id][2]
        table_height3 = self.model.mujoco_arena.table_offset[2]

        cube_height4 = self.sim.data.body_xpos[self.cube4_body_id][2]
        table_height4 = self.model.mujoco_arena.table_offset[2]
        
        cube_height5 = self.sim.data.body_xpos[self.cube4_body_id][2]
        table_height5 = self.model.mujoco_arena.table_offset[2]
            

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04, cube_height2 > table_height + 0.04, cube_height3 > table_height + 0.04, cube_height4 > table_height + 0.04,cube_height5 > table_height + 0.04
        
        
