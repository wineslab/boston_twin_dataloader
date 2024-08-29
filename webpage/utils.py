import pyproj
import shapely
from shapely.geometry import shape
from shapely.ops import transform
import math
import pyvista as pv
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, LineString, mapping
import os
from pyproj import Transformer
import open3d as o3d
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import shutil
import geopandas as gpd
import json


def osm_scene(dest_dir, 
              coordinates,
              center_coord):
    spp_default = 4096
    resx_default = 1024
    resy_default = 768

    camera_settings = {
        "rotation": (0, 0, -90),  # Assuming Z-up orientation
        "fov": 42.854885
    }

    # Define material colors. This is RGB 0-1 formar https://rgbcolorpicker.com/0-1
    material_colors = {
        "mat-itu_concrete": (0.539479, 0.539479, 0.539480),
        "mat-itu_marble": (0.701101, 0.644479, 0.485150),
        "mat-itu_metal": (0.219526, 0.219526, 0.254152),
        "mat-itu_wood": (0.043, 0.58, 0.184),
        "mat-itu_wet_ground": (0.91,0.569,0.055),
    }
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26915")
    center_26915 = transformer.transform(center_coord[0],center_coord[1])
    sionna_center_x = center_26915[0]
    sionna_center_y = center_26915[1]
    sionna_center_z = 0

    scene = ET.Element("scene", version="2.1.0")
    # Add defaults
    ET.SubElement(scene, "default", name="spp", value=str(spp_default))
    ET.SubElement(scene, "default", name="resx", value=str(resx_default))
    ET.SubElement(scene, "default", name="resy", value=str(resy_default))
    # Add integrator
    integrator = ET.SubElement(scene, "integrator", type="path")
    ET.SubElement(integrator, "integer", name="max_depth", value="12")

    # Define materials
    for material_id, rgb in material_colors.items():
        bsdf_twosided = ET.SubElement(scene, "bsdf", type="twosided", id=material_id)
        bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
        ET.SubElement(bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance")

    # %%
    # Get coordinates in meter for the area of interst polygon (This will be used in next steps)
    wsg84 = pyproj.CRS("epsg:4326")
    lambert = pyproj.CRS("epsg:26915")
    transformer = pyproj.Transformer.from_crs(wsg84, lambert, always_xy=True)
    coords = [transformer.transform(x, y) for x, y, z in coordinates]

    aoi_polygon = shapely.geometry.Polygon(coords)

    # %%
    # Store center of the selected area to be used in calculations later on
    center_x = aoi_polygon.centroid.x
    center_y = aoi_polygon.centroid.y

    # %%
    # Create Directories
    os.mkdir(f'{dest_dir}')
    os.mkdir(f'{dest_dir}/mesh')  

    # %%
    # Utility Function
    def points_2d_to_poly(points, z):
        """Convert a sequence of 2d coordinates to a polydata with a polygon."""
        faces = [len(points), *range(len(points))]
        poly = pv.PolyData([p + (z,) for p in points], faces=faces)
        return poly

    # %% [markdown]
    # ### Create Ground mesh and add to the scene

    # %%
    wsg84 = pyproj.CRS("epsg:4326")
    lambert = pyproj.CRS("epsg:26915")
    transformer = pyproj.Transformer.from_crs(wsg84, lambert, always_xy=True)
    coords = [transformer.transform(x, y) for x, y, z in coordinates]

    ground_polygon = shapely.geometry.Polygon(coords)
    z_coordinates = np.full(len(ground_polygon.exterior.coords), 0)  # Assuming the initial Z coordinate is zmin
    exterior_coords = ground_polygon.exterior.coords
    oriented_coords = list(exterior_coords)
    # Ensure counterclockwise orientation
    if ground_polygon.exterior.is_ccw:
        oriented_coords.reverse()
    points = [(coord[0]-center_x, coord[1]-center_y) for coord in oriented_coords]
    # bounding polygon
    boundary_points_polydata = points_2d_to_poly(points, z_coordinates[0])
    edge_polygon = boundary_points_polydata
    footprint_plane = edge_polygon.delaunay_2d()
    footprint_plane.points[:] = (footprint_plane.points - footprint_plane.center)*1.5 + footprint_plane.center
    pv.save_meshio(f"{dest_dir}/mesh/ground.ply",footprint_plane)

    material_type = "mat-itu_wet_ground"
    sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
    ET.SubElement(sionna_shape, "string", name="filename", value=f"{dest_dir}/mesh/ground.ply")
    bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
    ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")

    # %% [markdown]
    # #### Create Buildings mesh and add to the scene

    # %% [markdown]
    # First download 2D buildings from openstreetmap using OSMNX (https://osmnx.readthedocs.io/en/stable/)

    # %%
    import osmnx as ox
    wsg84 = pyproj.CRS("epsg:4326")
    lambert = pyproj.CRS("epsg:4326")
    transformer = pyproj.Transformer.from_crs(wsg84, lambert, always_xy=True)
    coords = [transformer.transform(x, y) for x, y, z in coordinates]

    osm_polygon = shapely.geometry.Polygon(coords)
    # Query the OpenStreetMap data
    buildings = ox.features.features_from_polygon(osm_polygon, tags={'building': True})
    # buildings = ox.geometries.geometries_from_polygon(osm_polygon, tags={'building': True})
    # Filter buildings that intersect with the polygon
    filtered_buildings = buildings[buildings.intersects(osm_polygon)]

    # %% [markdown]
    # Following code uses building footprints and extrude them to create a triangular mesh and add in Sionna scene one by one.

    # %%
    buildings_list = filtered_buildings.to_dict('records')
    source_crs = pyproj.CRS(filtered_buildings.crs)
    target_crs = pyproj.CRS('EPSG:26915')
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform
    for idx, building in enumerate(buildings_list):
        # Convert building geometry to a shapely polygon
        building_polygon = shape(building['geometry'])
        if building_polygon.geom_type != 'Polygon':
            continue
        building_polygon = transform(transformer, building_polygon)
        if math.isnan(float(building['building:levels'])):
            building_height = 3.5
        else:
            building_height = int(building['building:levels']) * 3.5
        z_coordinates = np.full(len(building_polygon.exterior.coords), 0)  # Assuming the initial Z coordinate is zmin
        exterior_coords = building_polygon.exterior.coords
        oriented_coords = list(exterior_coords)
        # Ensure counterclockwise orientation
        if building_polygon.exterior.is_ccw:
            oriented_coords.reverse()
        points = [(coord[0]-center_x, coord[1]-center_y) for coord in oriented_coords]
        # bounding polygon
        boundary_points_polydata = points_2d_to_poly(points, z_coordinates[0])
        edge_polygon = boundary_points_polydata
        footprint_plane = edge_polygon.delaunay_2d()
        footprint_plane = footprint_plane.triangulate()
        footprint_3D = footprint_plane.extrude((0, 0, building_height), capping=True)
        footprint_3D.save(f"{dest_dir}/mesh/building_{idx}.ply")
        local_mesh = o3d.io.read_triangle_mesh(f"{dest_dir}/mesh/building_{idx}.ply")
        o3d.io.write_triangle_mesh(f"{dest_dir}/mesh/building_{idx}.ply", local_mesh)
        material_type = "mat-itu_marble"
        # Add shape elements for PLY files in the folder
        sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-building_{idx}")
        ET.SubElement(sionna_shape, "string", name="filename", value=f"{dest_dir}/mesh/building_{idx}.ply")
        bsdf_ref = ET.SubElement(sionna_shape, "ref", id= material_type, name="bsdf")
        ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")

    # %% [markdown]
    # #### Create Roads mesh and add to the scene

    # %%
    def convert_lane_to_numeric(lane):
        try:
            return int(lane)
        except ValueError:
            try:
                return float(lane)
            except ValueError:
                return None

            # Helper function to calculate edge geometry if missing
    def calculate_edge_geometry(u, v, data):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        return LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])

    # %% [markdown]
    # Get the road network linestring and plot

    # %%
    G = ox.graph_from_polygon(polygon = osm_polygon, simplify= False, retain_all=True,truncate_by_edge=True,network_type = 'all_private')
    graph = ox.project_graph(G, to_crs='epsg:26915')
    ox.plot_graph(graph, show=False, save=True, filepath='graph.png')


    # %% [markdown]
    # Now convert each line segment into road mesh using lanes as parameter to set road width

    # %%
    # Create a list to store GeoDataFrames for each road segment
    gdf_roads_list = []
    # Set the fixed Z coordinate for the buffer polygons
    Z0 = .25  # You can adjust this value based on the desired elevation of the roads
    # Create a list to store the meshes
    mesh_list = []
    mesh_collection = pv.PolyData()
    boundary = Polygon(points)
    # Iterate over each edge in the graph
    for u, v, key, data in graph.edges(keys=True, data=True):
        # Check if the edge has geometry, otherwise create geometries from the nodes
        if 'geometry' not in data:
            data['geometry'] = calculate_edge_geometry(u, v, data)

        # Get the lanes attribute for the edge
        lanes = data.get('lanes', 1)  # Default to 1 lane if lanes attribute is not available

        if not isinstance(lanes, list):
            lanes = [lanes]
            
        # Convert lane values to numeric (integers or floats) using the helper function
        num_lanes = [convert_lane_to_numeric(lane) for lane in lanes]

        # Filter out None values (representing non-numeric lanes) and calculate the road width
        num_lanes = [lane for lane in num_lanes if lane is not None]
        road_width = num_lanes[0] * 3.5
        # Buffer the LineString with the road width and add Z coordinate
        line_buffer = data['geometry'].buffer(road_width)
        # Convert the buffer polygon to a PyVista mesh
        exterior_coords = line_buffer.exterior.coords
        z_coordinates = np.full(len(line_buffer.exterior.coords), Z0)
        oriented_coords = list(exterior_coords)
        # Ensure counterclockwise orientation
        if line_buffer.exterior.is_ccw:
            oriented_coords.reverse()
        points = [(coord[0]-center_x, coord[1]-center_y) for coord in oriented_coords]
        # bounding polygon
        boundary_points_polydata = points_2d_to_poly(points, z_coordinates[0])
        mesh = boundary_points_polydata.delaunay_2d()
        # Add the mesh to the list
        mesh_collection = mesh_collection + mesh
        mesh_list.append(mesh)
    output_file = f"{dest_dir}/mesh/road_mesh_combined.ply"
    pv.save_meshio(output_file,mesh_collection)
    material_type = "mat-itu_concrete"
    # Add shape elements for PLY files in the folder
    sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-roads_{idx}")
    ET.SubElement(sionna_shape, "string", name="filename", value=f"{dest_dir}/mesh/road_mesh_combined.ply")
    bsdf_ref = ET.SubElement(sionna_shape, "ref", id= material_type, name="bsdf")
    ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")
    tree = ET.ElementTree(scene)
    xml_string = ET.tostring(scene, encoding="utf-8")
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="    ")  # Adjust the indent as needed

    with open(f"{dest_dir}/simple_OSM_scene.xml", "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_pretty)

def generate_tile_corners(start_lat, start_lon, end_lat, end_lon, tile_size=200, overlap=0.0):
    m_to_deg = 1 / 111000

    tiles = []
    offset = 0
    lat = start_lat
    while lat < end_lat:
        lon = start_lon + offset
        while lon < end_lon:
            top_left = [lon, lat]
            top_right = [lon + tile_size * m_to_deg, lat]
            bottom_left = [lon, lat + tile_size * m_to_deg]
            bottom_right = [lon + tile_size * m_to_deg, lat + tile_size * m_to_deg]
            tiles.append([top_left, top_right, bottom_right, bottom_left])
            lon += tile_size * m_to_deg
        lat += tile_size * m_to_deg
        offset = tile_size * m_to_deg * overlap if offset == 0 else 0
    return tiles

def save_tiles_to_files(tiles, filename):
    for i, tile in enumerate(tiles):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": mapping(Polygon(tile))
                }
            ]
        }
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{i}{ext}"

        with open(new_filename, 'w') as f:
            json.dump(geojson, f)

def read_tile_from_file(filename):
    with open(filename, 'r') as f:
        geojson = json.load(f)
    tile = list(shape(geojson['features'][0]['geometry']).exterior.coords)
    return tile

if __name__ == "__main__":
    start_lat, start_lon = 42.3601, -71.0589  # Boston coordinates
    end_lat, end_lon = start_lat + 0.1, start_lon + 0.1  # 0.1 degree ~ 11km

    tiles = generate_tile_corners(start_lat, start_lon, end_lat, end_lon)
    save_tiles_to_files(tiles, 'tiles.geojson')
    tiles = read_tile_from_file('tiles.geojson') 
    tiles = tiles[:5]

    for i, points in enumerate(tiles):
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        center = [sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]
        try:
            osm_scene(dest_dir=f'NEU_{i}',
                    coordinates=points,
                    center_coord=center)
        except Exception as e:
            print(f"Skipping scene {i} due to error: {e}")
            dir_path = os.path.join(os.getcwd(), f'NEU_{i}')
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)