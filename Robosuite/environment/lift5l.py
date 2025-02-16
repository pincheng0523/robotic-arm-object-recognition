from collections import OrderedDict

import numpy as np
import random
import json
import glob
import os
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena


from robosuite.models.objects import BallObject
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


class Lift5l(SingleArmEnv):
    
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
           
            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube1):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward
        
        
    def _load_model(self):
        super()._load_model()

        # 调整机械臂基座位置
        xpos = np.array([0.0, 3, 0.3])  # 设置到所需的位置
        self.robots[0].robot_model.set_base_xpos(xpos)

    # 加载桌面工作空间模型
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,    
            table_offset=self.table_offset,
            xml=self.table_arena_xml  # 使用传递的背景 XML 文件名
        )

        # 场景设置为零原点
        mujoco_arena.set_origin([0, 0, 0])

        # 初始化材质
        self.init_materials()

    # 读取JSON文件
        json_directory = "/home/allen880523/cgu_robot/robosuite/test4/"
        min_json_file = self.find_min_numbered_json_file(json_directory)
        if min_json_file:
            self.cube1, self.cube2, self.cube3, self.cube4, self.cube5 = self.load_and_transform_objects(min_json_file)

        # 检查cube1和cube2是否被正确初始化
        if self.cube1 is None or self.cube2 is None or self.cube3 is None:
            raise ValueError("Failed to initialize cube1 or cube2 from JSON file")

        # 创建位置初始化器
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
                mujoco_objects=[self.cube1, self.cube2],
                x_range=[-0.35, 0.35],
                y_range=[-0.35, 0.35],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # 任务包含场景、机器人和感兴趣的物体
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube1, self.cube2, self.cube3, self.cube4, self.cube5]
        )

        
    def init_materials(self):
        # 初始化所有的材料
        self.materials_dict = {
            "redwood_mat": CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            "greenwood_mat": CustomMaterial(texture="WoodGreen", tex_name="greenwood", mat_name="greenwood_mat", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            "WoodBlue_mat": CustomMaterial(texture="WoodBlue", tex_name="WoodBlue", mat_name="WoodBlue_mat", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            "WoodDark": CustomMaterial(texture="WoodDark", tex_name="WoodDark", mat_name="WoodDark", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            "orange": CustomMaterial(texture="orange", tex_name="orange", mat_name="orange", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            "Brass": CustomMaterial(texture="Brass", tex_name="Brass", mat_name="Brass", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            "PlasterGray": CustomMaterial(texture="PlasterGray", tex_name="PlasterGray", mat_name="PlasterGray", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            "purple": CustomMaterial(texture="purple", tex_name="purple", mat_name="purple", tex_attrib={"type": "cube"}, mat_attrib={"specular": "0.4", "shininess": "0.1"}),
            
        }

    @staticmethod
    def find_min_numbered_json_file(directory):
        json_files = glob.glob(os.path.join(directory, '*.json'))
        if not json_files:
            return None   
        return min(json_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    
    def load_and_transform_objectspos(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    
    # 打印出數據的結構以進行調試
        #print("Loaded data:", data)
    
    # 確保數據是字典
        if not isinstance(data, dict):
            raise ValueError(f"Expected data to be a dict, but got {type(data)}")
    
        objects_data = data.get("objects", [])
        if not isinstance(objects_data, list):
            raise ValueError(f"Expected objects data to be a list, but got {type(objects_data)}")
    
        object_infos = []
        for item in objects_data:  # 假設數據存儲在列表的第一個元素中
        # 確保每個項目是字典
            if not isinstance(item, dict):
                raise ValueError(f"Expected item to be a dict, but got {type(item)}")
        
            name = item["name"]
            position = item["position"]
            orientation = item["orientation"]
            object_infos.append((name, position, orientation))  # 存儲名稱、位置和方向
    
        return object_infos

    
    def load_and_transform_objects(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not data or not isinstance(data, dict):
            raise ValueError("The JSON file is empty or not properly formatted")
        
        objects_data = data.get("objects", [])
        if not objects_data:
            raise ValueError("No objects information found in the JSON file")

        #objects_data = data[0]  # 假定第一个元素包含所需数据

        cube1 = None
        cube2 = None
        cube3 = None
        cube4 = None
        cube5 = None

        for item in objects_data:
           if isinstance(item, dict):  # 确保 item 是字典类型
                material_instance = self.materials_dict.get(item.get("material"), None)
                obj_type = eval(item["type"])  # 请确保这里使用 eval 是安全的

            # 检查对象类型，决定是否需要 material 参数
                if obj_type in [LemonObject, Lemon1Object, Lemon2Object, Lemon3Object, Lemon4Object, Lemon5Object, Lemon6Object, Lemon7Object,
                            BottleObject, Bottle1Object, Bottle2Object, Bottle3Object, Bottle4Object, Bottle5Object, Bottle6Object, Bottle7Object,
                            CerealObject, Cereal1Object, Cereal2Object, Cereal3Object, Cereal4Object, Cereal5Object, Cereal6Object, Cereal7Object,
                            RoundNutObject, RoundNut1Object, RoundNut2Object, RoundNut3Object, RoundNut4Object, RoundNut5Object, RoundNut6Object, RoundNut7Object,
                            SquareNutObject, SquareNut1Object, SquareNut2Object, SquareNut3Object, SquareNut4Object, SquareNut5Object, SquareNut6Object, SquareNut7Object]:
                    obj = obj_type(name=item["name"])
                else:
                    obj = obj_type(name=item["name"], size_min=[0.04, 0.04, 0.04], size_max=[0.04, 0.04, 0.04], material=material_instance)

                if not cube1:
                    cube1 = obj
                elif not cube2:
                    cube2 = obj
                    
                elif not cube3:
                    cube3 = obj
                
                elif not cube4:
                    cube4 = obj
                    
                else:
                    cube5 = obj
                    
                    
                if cube1 and cube2 and cube3 and cube4 and cube5:
                    break  # 假设只处理两个对象

        if cube1 is None or cube2 is None or cube3 is None:
            raise ValueError("Could not find enough objects in JSON to initialize both cube1 and cube2")

        return cube1, cube2, cube3, cube4, cube5




       

    def _setup_references(self):

        super()._setup_references()
        
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
        super()._reset_internal()

        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # 讀取位置資訊
            json_directory = "/home/allen880523/cgu_robot/robosuite/test4/"
            min_json_file = self.find_min_numbered_json_file(json_directory)
            if min_json_file:
                object_infos = self.load_and_transform_objectspos(min_json_file)
                
                # 依據讀取的位置資訊設定物體位置
                for info in object_infos:
                    obj_name, pos, quat = info
                    if obj_name == self.cube1.name:
                        self.sim.data.set_joint_qpos(self.cube1.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
                    elif obj_name == self.cube2.name:
                        self.sim.data.set_joint_qpos(self.cube2.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
                    elif obj_name == self.cube3.name:
                        self.sim.data.set_joint_qpos(self.cube3.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
                    elif obj_name == self.cube4.name:
                        self.sim.data.set_joint_qpos(self.cube4.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
                    elif obj_name == self.cube5.name:
                        self.sim.data.set_joint_qpos(self.cube5.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
                        

                

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
        
        
