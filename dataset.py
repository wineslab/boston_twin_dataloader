import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import Transmitter, PlanarArray

from src.classes.BostonTwin import BostonTwin

# Configure environment
def configure_environment(gpu_num="1"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Initialize BostonTwin
def initialize_bostwin(dataset_dir="bostontwin"):
    dataset_path = Path(dataset_dir)
    return BostonTwin(dataset_path)

# Generate scene for a specific location and radius
def generate_scene(bostwin, scene_name, center, radius, resolution=5):
    bostwin.generate_scene_from_radius(
        scene_name=scene_name,
        center_lon=center[0],
        center_lat=center[1],
        side_m=radius,
        load=True,
    )
    elevation_map = bostwin.get_elevation_map(resolution=resolution)
    return elevation_map

# Save elevation map visualization
def save_elevation_map(elevation_map, output_path="elevation_map.pdf"):
    plt.imshow(elevation_map, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Elevation Map')
    plt.savefig(output_path)
    plt.close()

# Configure the scene with transmitters and receivers
def configure_scene(bostwin, scene_name):
    sionna_scene, _ = bostwin.load_bostontwin(scene_name)
    antenna_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    sionna_scene.tx_array = antenna_array
    sionna_scene.rx_array = antenna_array

    tx0 = Transmitter(
        name="tx0",
        position=[0, 0, 20],  # Center of the scene
        orientation=[np.pi * 5 / 6, 0, 0],
        power_dbm=44,
    )
    sionna_scene.add(tx0)
    return sionna_scene

# Generate the coverage map
def generate_coverage_map(sionna_scene, cell_size=(5.0, 5.0), num_samples=int(1e6)):
    start_time = time.time()
    coverage_map = sionna_scene.coverage_map(
        max_depth=10,
        diffraction=True,
        cm_cell_size=cell_size,
        combining_vec=None,
        precoding_vec=None,
        num_samples=num_samples,
    )
    end_time = time.time()
    print(f"Coverage map generation took {end_time - start_time} seconds")
    return coverage_map

# Save coverage map visualization
def save_coverage_map(coverage_map, output_path="coverage_map.pdf"):
    fig = coverage_map.show(metric="path_gain")
    plt.savefig(output_path)
    plt.close()

# Generate dataset for training
def generate_dataset(
    bostwin, 
    num_samples=100, 
    output_dir="dataset", 
    elevation_resolution=5, 
    coverage_cell_size=(5.0, 5.0), 
    area_radius=100
):
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        scene_name = f"scene_{i}"
        center = [np.random.uniform(-71.09, -71.07), np.random.uniform(42.33, 42.34)]
        
        # Generate scene and elevation map
        elevation_map = generate_scene(
            bostwin, scene_name, center, area_radius, resolution=elevation_resolution
        )
        np.save(f"{output_dir}/elevation_map_{i}.npy", elevation_map)
        
        # Save elevation map visualization
        save_elevation_map(elevation_map, f"{output_dir}/elevation_map_{i}.pdf")
        
        # Configure and generate coverage map
        sionna_scene = configure_scene(bostwin, scene_name)
        coverage_map = generate_coverage_map(
            sionna_scene, cell_size=coverage_cell_size
        )
        path_gain = coverage_map.path_gain.numpy().squeeze()
        np.save(f"{output_dir}/coverage_map_{i}.npy", path_gain)
        
        # Optional: Save visualization of coverage map
        save_coverage_map(coverage_map, f"{output_dir}/coverage_map_{i}.pdf")
        
        print(f"Sample {i + 1}/{num_samples} generated.")

# Main function
if __name__ == "__main__":
    configure_environment(gpu_num="1")
    
    bostwin = initialize_bostwin("bostontwin")
    generate_dataset(
        bostwin, 
        num_samples=100, 
        output_dir="training_data", 
        elevation_resolution=5, 
        coverage_cell_size=(5.0, 5.0), 
        area_radius=100
    )
