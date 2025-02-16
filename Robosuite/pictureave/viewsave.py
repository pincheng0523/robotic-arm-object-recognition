import os
import numpy as np
import robosuite as suite
from PIL import Image
import json
import glob

num_iterations = 1  # number of times to create the environment and save an image
base_directory = "/home/allen880523/cgu_robot/robosuite/light/"  # base directory where to save images
json_directory = "/home/allen880523/cgu_robot/robosuite/test1/"

for j in range(num_iterations):
    # create environment instance
    env = suite.make(
        env_name="Lift2",
        robots="Panda",
        render_camera="birdview",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["birdview"],
    )

    # reset the environment
    env.reset()

    action = np.random.randn(env.robots[0].dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    
    # get the image from the observation
    image_data = obs['birdview_image']  # get the image from the observation
    image = Image.fromarray(image_data)  # convert numpy array to PIL image
    
    # Check for file name availability and increment if needed
    file_index = 1
    while os.path.exists(f"{base_directory}{file_index}.png"):
        file_index += 1

    image.save(f"{base_directory}{file_index}.png")  # save the image to a file with a new name if needed

    env.render()  # render on display
    
    if 'cube1' in dir(env) and 'cube2' in dir(env):
        env.save_objects_info([env.cube1, env.cube2],env.model.mujoco_arena.xml_file)  # Save environment info including background XML

