import requests

def get_osm_data(lat, lon, radius=500, output_file='map.osm'):
    """
    Download OSM data for a given latitude, longitude, and radius, and save it to a file.

    Parameters:
    - lat (float): Latitude of the center point.
    - lon (float): Longitude of the center point.
    - radius (int): Radius in meters for the area to download.
    - output_file (str): Filename for saving the OSM data.

    Returns:
    - str: Path to the downloaded OSM file.
    """
    # Convert radius to degrees (approximately, 1 degree ~ 111,320 meters)
    lat_offset = radius / 111320.0
    lon_offset = radius / (111320.0 * abs(math.cos(math.radians(lat))))

    # Calculate the bounding box (min_lon, min_lat, max_lon, max_lat)
    min_lat = lat - lat_offset
    max_lat = lat + lat_offset
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset

    # Overpass API query to get the data within the bounding box
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:xml];
    (
      node({min_lat},{min_lon},{max_lat},{max_lon});
      <;
      >;
    );
    out body;
    """

    # Make the request to the Overpass API
    response = requests.get(overpass_url, params={'data': overpass_query})

    # Check if the request was successful
    if response.status_code == 200:
        # Save the response content to a file
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print(f"OSM data saved to {output_file}")
        return output_file
    else:
        raise Exception(f"Failed to retrieve OSM data. Status code: {response.status_code}")

# Example usage
if __name__ == "__main__":
    import math
    
    # Example coordinates (Northeastern University's EXP Building)
    latitude = 42.340082
    longitude = -71.089488
    
    # Get OSM data for the specified location
    osm_file = get_osm_data(latitude, longitude, radius=500, output_file='northeastern_exp.osm')
    print(f"OSM file downloaded: {osm_file}")
