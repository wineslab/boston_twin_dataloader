import os
import mitsuba as mi
mi.set_variant("llvm_ad_rgb")
import sionna
import tensorflow as tf
tf.random.set_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import geopandas as gpd
import numpy as np
import numpy.typing
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from src.classes.BostonTwin import BostonTwin
from pathlib import Path
import geopandas as gpd
import numpy as np
import numpy.typing

dataset_dir = Path("bostontwin")
bostwin = BostonTwin(dataset_dir)

scene_names = bostwin.get_scene_names()

scene_names.sort()

print(f"There are {len(scene_names)} scenes available in {dataset_dir}.\n{scene_names}")
scene_name = "BOS_G_5"
sionna_scene, scene_antennas = bostwin.load_bostontwin(scene_name)
n_antennas = scene_antennas.shape[0]
print(f"There are {n_antennas} antennas in the {scene_name} scene.")
n_txs = 30  # number of transmitters
if n_txs>n_antennas:
    raise ValueError(f"There are {n_antennas} in the scene. Please select a number of transmitters lower than {n_antennas}")
n_rxs = n_antennas - n_txs  # set the antennas that are not transmitters to be receivers 

# select the antennas from scene_antennas using their indices 
ant_ids = np.arange(n_antennas)
tx_ids = np.sort(np.random.choice(ant_ids, size=n_txs, replace=False))  # pick n_txs random indices for the transmitters
rx_ids = np.setdiff1d(ant_ids,tx_ids)  # the rest are receivers 
print(f"Transmitter IDs: {tx_ids}\nReceivers IDs:{rx_ids}")


# Configure antenna array for all transmitters
sionna_scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")

# Configure antenna array for all receivers
sionna_scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")


# remove existing Radio Devices
[sionna_scene.remove(tx) for tx in sionna_scene.transmitters]
[sionna_scene.remove(rx) for rx in sionna_scene.receivers]

# add the antennas to the BostonTwin
tx_rx_dict = bostwin.add_scene_antennas(tx_ids,rx_ids)

n_antennas = scene_antennas.shape[0]
print(f"There are {n_antennas} antennas in the scene.")

n_txs = 5  # number of transmitters
n_rxs = 5  # number of receivers

# select the antennas from scene_antennas using their indices 
ant_ids = np.arange(n_antennas)
tx_ids = np.sort(np.random.choice(ant_ids, size=n_txs, replace=False))  # pick n_txs random indices for the transmitters
rx_ids = np.sort(np.random.choice(np.setdiff1d(ant_ids,tx_ids), size=n_txs, replace=False))  # the rest are receivers 
print(f"Transmitter IDs: {tx_ids}\nReceivers IDs:{rx_ids}")

# Configure antenna array for all transmitters
sionna_scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")

# Configure antenna array for all receivers
sionna_scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")


tx_rx_dict = bostwin.add_scene_antennas(tx_ids,rx_ids)


paths = sionna_scene.compute_paths(
                    max_depth=2,
                    num_samples=1e5,
                    reflection=True,
                    diffraction=True,
                    scattering=True,
                   )


cm = sionna_scene.coverage_map(max_depth=2,
                        diffraction=True, # Disable to see the effects of diffraction
                        cm_cell_size=(5., 5.), # Grid size of coverage map cells in m
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(1e5))

[sionna_scene.get(tx).position for tx in sionna_scene.transmitters]

sionna_scene.preview(coverage_map=cm, show_devices=True)
