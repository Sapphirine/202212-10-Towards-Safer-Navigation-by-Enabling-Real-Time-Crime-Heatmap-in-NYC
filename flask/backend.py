import os

from math import sin, cos, sqrt, atan2, dist
from datetime import time, datetime, timedelta
from flask import Flask, request, render_template, Response
from flask_sse import sse

import numpy as np
import pandas as pd
from tqdm import tqdm

app = Flask(__name__)
app.config['REDIS_URL'] = 'redis://localhost'
app.register_blueprint(sse, url_prefix='/stream')

# Set some default coordinates for the map
DEFAULT_COORDS = (40.802871, -73.966169)
df = pd.read_csv('dataset_encode.csv')
df[['date', 'time']] = df['Timestamp'].str.split(' ', expand=True)
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['time'] = df.time.dt.time


def interpolate(start, end, interval=0.0001, result=None):
    if result is None:
        result = [start]
    if dist(start, end) < interval:
        result.append(end)
    else:
        direction = atan2(end[1] - start[1], end[0] - start[0])
        new_lat = start[0] + interval * cos(direction)
        new_lon = start[1] + interval * sin(direction)
        new_loc = [new_lat, new_lon]
        result.append(new_loc)
        interpolate(new_loc, end, interval=interval, result=result)
    return result


def compute_value(geolocation, df, rad=0.002):
    lat_center, lng_center = geolocation

    ret = df[ \
        (df['Latitude'] > lat_center - rad) & (df['Latitude'] < lat_center + rad) \
        & (df['Longitude'] > lng_center - rad) & (df['Longitude'] < lng_center + rad) \
        ]

    return len(ret)


def process_line_heatmap(path_responses):
    # Fake data for testing
    # data = pd.read_csv('sample_heatmap.csv')
    # data.iloc[:, 2] = np.random.permutation(data.iloc[:, 2].values)
    # return data.to_numpy().tolist()[:1000]

    interpolate_path = []
    interpolate_paths = []
    for path_response in path_responses:
        for i in range(len(path_response) - 1):
            interpolate_path = interpolate_path + interpolate(path_response[i], path_response[i + 1], interval=0.0001)
        interpolate_paths.append(pd.DataFrame(interpolate_path).drop_duplicates().to_numpy())
    interpolate_paths = np.vstack(interpolate_paths)

    ret_arr = []
    for loc in tqdm(interpolate_path):
        ret_arr.append(compute_value(loc, df, rad=0.002))

    heatmap = []
    for loc, count in zip(interpolate_path, ret_arr):
        heatmap.append([*loc, count])

    # Map heatmap points to grid
    heatmap = np.array(heatmap)
    # grid_interval = 0.0001
    grid_interval = 0.0001
    lat_bounds = np.min(heatmap[:, 0]), np.max(heatmap[:, 0])
    lng_bounds = np.min(heatmap[:, 1]), np.max(heatmap[:, 1])

    grid_lat_pts = int((lat_bounds[1] - lat_bounds[0]) // grid_interval + 1)
    grid_lng_pts = int((lng_bounds[1] - lng_bounds[0]) // grid_interval + 1)
    heatmap_grid = np.zeros((grid_lat_pts, grid_lng_pts))
    n_pts_grid = np.zeros_like(heatmap_grid)

    for point in heatmap:
        lat, lng, val = point
        lat_idx = int((lat - np.min(heatmap[:, 0])) // grid_interval)
        lng_idx = int((lng - np.min(heatmap[:, 1])) // grid_interval)
        heatmap_grid[lat_idx, lng_idx] += val
        n_pts_grid[lat_idx, lng_idx] += 1

    n_pts_grid[n_pts_grid == 0] = 1
    heatmap_grid = heatmap_grid / n_pts_grid
    # Normalize heatmap_grid
    # heatmap_grid = (heatmap_grid - np.min(heatmap_grid)) / (np.max(heatmap_grid) - np.min(heatmap_grid))

    heatmap_new = []
    for lat_idx, lat in enumerate(np.arange(lat_bounds[0], lat_bounds[1], grid_interval)):
        for lng_idx, lng in enumerate(np.arange(lng_bounds[0], lng_bounds[1], grid_interval)):
            heatmap_new.append([lat, lng, heatmap_grid[lat_idx, lng_idx]])

    return heatmap_new


@app.route('/')
def index():
    # Initialize the map with the default coordinates
    # Note: Get API key from https://developers.google.com/maps/documentation/javascript/get-api-key
    #       and put the key in environmental variable GOOGLE_MAPS_API_KEY
    return render_template('main.html', gmaps_key=os.environ['GOOGLE_MAPS_KEY'], coords=DEFAULT_COORDS)


@app.route('/compute_path_heatmap', methods=['POST'])
def compute_path_heatmap():
    # Get the coordinates from the request
    coords = request.get_json()
    data = process_line_heatmap(coords)
    with app.app_context():
        destination = f'heatmap_update'
        sse.publish(data, type=destination)
    return 'ok'


@app.route('/refresh_heatmap', methods=['GET'])
def update_heatmap():
    # conv_model_input = {'age': {'45-64': 0, '25-44': 1, '<18': 2, '18-24': 3, '65+': 4},
    #                     'race': {'Black': 0,
    #                              'White': 1,
    #                              'White Hispanic': 2,
    #                              'Black Hispanic': 3,
    #                              'Asian / Pacific Islander': 4,
    #                              'American Indian/Alaskan Native': 5,
    #                              },
    #                     'gender': {'Male': 0, 'Female': 1}}
    # parameter = request.json
    # # Model Input
    # age = conv_model_input['age'][parameter['ageGroup']]
    # race = conv_model_input['race'][parameter['race']]
    # gender = conv_model_input['gender'][parameter['gender']]
    # # TODO: Add gps, time

    import pandas as pd
    data = pd.read_csv('sample_heatmap.csv').to_numpy().tolist()[:1000]
    with app.app_context():
        destination = f'heatmap_update'
        sse.publish(data, type=destination)
    return 'ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
