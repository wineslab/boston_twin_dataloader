from flask import Flask, render_template, request, jsonify
import os

# Set the base directory to the 'webpage' folder
base_dir = os.path.dirname(__file__)

# Initialize Flask app with specified template and static folders
app = Flask(__name__, template_folder=os.path.join(base_dir), static_folder=os.path.join(base_dir))

@app.route('/')
def index():
    # Render index.html located in the 'webpage' folder
    return render_template('index.html')

@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    altitude = data.get('altitude')
    print(f"Latitude: {lat}, Longitude: {lon}, Altitude: {altitude}")
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(debug=True)
