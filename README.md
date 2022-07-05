# Vehicle Tracking and Counting System With Python OpenCV4 TF2-YOLOv3 and DeepSORT

## Installation
`git clone https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT.git`  

`cd Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT` 

`pip3 install -r requirements.txt`  

Download yolov3.weights file from https://pjreddie.com/media/files/yolov3.weights and store it in model_data/ folder

## Quick Start
1. cd to this directory
2. Move Video Input (VehicleTest.mp4) to inputs/ directory
3. Adjust parameters in config.yaml
4. Run `python3 main.py`


## Pipeline
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/flowchart.png)

## Detailed Steps

### 1. Stabilize
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/stabilize.gif)

### 2. Polygon ROI Crop
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/masked.gif)

### 3. Detect, Track, Count
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/move.gif)

### 4. Transform back to original 
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/final.gif)
