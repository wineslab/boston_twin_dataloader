# import os
# gpu_num = "1"  # Use "" to use the CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.classes.BostonTwin import BostonTwin
from pathlib import Path
import random
import matplotlib.pyplot as plt

dataset_dir = Path("bostontwin")
bostwin = BostonTwin(dataset_dir)

new_scene_name = "test" 
center = [-71.08730583197658, 42.33713805318744]  # center of the scene
radius = 500 

# Define the bounding box for the Greater Boston area
min_lon, max_lon = -71.1912, -70.7488
min_lat, max_lat = 42.2279, 42.3995

# Generate random coordinates within the bounding box
center = [
    random.uniform(min_lon, max_lon),
    random.uniform(min_lat, max_lat)
]
                                
ele = bostwin.get_elevation_map(scene_name=new_scene_name,
                                center_lon=center[0],
                                center_lat=center[1],
                                side_m=radius,
                                resolution=1,
                                )

plt.imshow(ele, cmap='viridis', origin='lower')
plt.colorbar(label='Elevation (m)')
plt.title('Elevation Map')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.savefig("out.pdf")


