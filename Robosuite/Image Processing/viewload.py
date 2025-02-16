import os
import numpy as np
import glob
import robosuite as suite
from PIL import Image
import json

num_iterations = 500 # number of times to create the environment and save an image
base_directory = "/home/allen880523/cgu_robot/robosuite/pp/wood2-1/"  # base directory where to save images
json_directory = "/home/allen880523/cgu_robot/robosuite/test1/"

def find_min_numbered_json_file(directory):
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        return None
    return min(json_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

for j in range(num_iterations):
    min_json_file = find_min_numbered_json_file(json_directory)
    if not min_json_file:
        raise ValueError("No JSON files found in the directory")

    with open(min_json_file, 'r') as f:
        env_info = json.load(f)

    background_xml = env_info.get("background_xml")
    if not background_xml:
        raise ValueError("No background XML information found in the JSON file")

    env = suite.make(
        env_name="Lift2l",  # 使用自定义的 Lift2l 环境
        robots="Panda",
        render_camera="birdview",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["birdview"],
        table_arena_xml=background_xml  # 传递背景 XML 文件名
    )

    env.reset()

    action = np.random.randn(env.robots[0].dof)
    obs, reward, done, info = env.step(action)
    
    image_data = obs['birdview_image']
    image = Image.fromarray(image_data)
    
    file_index = 1
    while os.path.exists(f"{base_directory}{file_index}.png"):
        file_index += 1

    image.save(f"{base_directory}{file_index}.png")

    os.remove(min_json_file)

    env.render()

