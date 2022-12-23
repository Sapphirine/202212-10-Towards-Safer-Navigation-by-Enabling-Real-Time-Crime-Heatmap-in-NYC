# Towards Safer Navigation by Enabling Real-Time Crime Heatmap in NYC

## About
This repository is the source code for the class project of EE6893 Big Data at Columbia University.

This project is aiming to integrate crime prediction heatmap on navigation platforms. By integrating a crime heatmap with a navigation system, users can be informed to potential threats in real-time. This is helpful for travelers or persons who are alone at night. A heatmap-based crime prediction and navigation system can help individuals feel safer and more secure in a city or neighborhood.

Group Members: Kaiyuan Hou (kh3119), Yanchen Liu (yl4189)

## Setup environment
* Create virtual environment with conda\
`conda create -n bigdata -c conda-forge python==3.8 flask pandas numpy tqdm pytorch`
* Activate virtual environment \
`conda activate bigdata`

* install flask sse \
`pip install flask_sse`
* Setup global environment\
`export GOOGLE_MAPS_KEY= <Your Google API key>`
* <a href="https://redis.io/docs/getting-started/installation/" target="_blank">install redis and start service</a>.\
On macOS: `brew install redis` & then
`brew services start redis`

**Note:** We have only tested on macOS, please following instruction on official website to setup redis service.

## How to run
* ``cd flask`` and then `python backend.py`
* view the frontend at: `http://127.0.0.1:8000`
* Select a start location on the map and click, you can adjust pin by dragging. The second click is to pick the destination, similarly you can drag it as well. Third click is confirming the start point and end point, the heatmap overlay will show up in seconds.

## Demo Video
<video src="https://user-images.githubusercontent.com/86454568/209273109-c7ae83d9-b950-4553-ab45-7cd5d6d40f8d.mp4" width="320" height="240" controls></video>



<!-- https://user-images.githubusercontent.com/86454568/209273109-c7ae83d9-b950-4553-ab45-7cd5d6d40f8d.mp4 -->

