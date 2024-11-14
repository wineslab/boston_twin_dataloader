import os
gpu_num = "1"  # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.classes.BostonTwin import BostonTwin
from pathlib import Path

dataset_dir = Path("bostontwin")
bostwin = BostonTwin(dataset_dir)

new_scene_name = "test" 
center = [-71.08730583197658, 42.33713805318744]  # center of the scene
radius = 500 
bostwin.generate_scene_from_radius(scene_name=new_scene_name,
                                   center_lon=center[0],
                                   center_lat=center[1],
                                   side_m=radius,
                                  )

sionna_scene, scene_antennas = bostwin.load_bostontwin(new_scene_name)
# bostwin.export_scene_models(out_path=Path("out"))

