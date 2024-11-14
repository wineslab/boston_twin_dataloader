import os
gpu_num = "1"  # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from src.classes.BostonTwin import BostonTwin
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
import numpy as np

dataset_dir = Path("bostontwin")
bostwin = BostonTwin(dataset_dir)

new_scene_name = "test" 
center = [-71.08730583197658, 42.33713805318744]  # center of the scene
radius = 500 

# # Define the bounding box for the Greater Boston area
# min_lon, max_lon = -71.1912, -70.7488
# min_lat, max_lat = 42.2279, 42.3995

# # Generate random coordinates within the bounding box
# center = [
#     random.uniform(min_lon, max_lon),
#     random.uniform(min_lat, max_lat)
# ]

bostwin.generate_scene_from_radius(scene_name=new_scene_name,
                                   center_lon=center[0],
                                   center_lat=center[1],
                                   side_m=radius,
                                   load=True,
                                  )
                                
ele = bostwin.get_elevation_map(resolution=5)
print(ele.shape)

sionna_scene, scene_antennas = bostwin.load_bostontwin(new_scene_name)

sionna_scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,  # relative to wavelength
                             horizontal_spacing=0.5,  # relative to wavelength
                             pattern="iso",
                             polarization="V")
sionna_scene.rx_array = sionna_scene.tx_array

tx0 = Transmitter(name='tx0',
                  position=[0, 0, 20],  # Center of the scene
                  orientation=[np.pi*5/6, 0, 0],
                  power_dbm=44)
sionna_scene.add(tx0)

cm = sionna_scene.coverage_map(max_depth=1,
                        diffraction=True, # Disable to see the effects of diffraction
                        cm_cell_size=(5., 5.), # Grid size of coverage map cells in m
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(1e2))

# extract the actaul coverage map out of the scene. 








