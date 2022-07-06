# Vehicle Tracking and Counting System With Python3.7, OpenCV4, TF2-YOLOv3, and DeepSORT
A Python based Vehicle Tracking and Counting System using OpenCV 4, Tensorflow YOLOv3 and
DeepSORT. This software aims to track southward moving vehicles and count the number of vehicles
exiting the road.

## Installation
Clone this repository: `git clone https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT.git`  

Change directory to this repository: `cd Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT` 

Install Python Libraries: `pip3 install -r requirements.txt`  

Download weights files: `wget https://pjreddie.com/media/files/yolov3.weights -P model_data/`   
(OR Download yolov3.weights file from https://pjreddie.com/media/files/yolov3.weights and store it in model_data/ folder)

## Quick Start
1. cd to this directory
2. Move Video Input (VehicleTest.mp4) to inputs/ directory
3. Adjust parameters in config.yaml
4. Run `python3 main.py`
5. Output will be saved in outputs/ folder  

## Pipeline
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/flowchart2.png)

## Detailed Steps

### 1. Stabilize
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/stabilize.gif)

### 2. Polygon ROI Crop
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/masked.gif)

### 3. Define exit lines
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/exit_lanes2.png)

### 4. Detect, Track, Count
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/move.gif)

### 5. Transform back to original 
![alt text](https://github.com/zhengkang128/Vehicle-Tracking-and-Counting-System-With-Python-OpenCV-4-TF-YOLOv3-and-DeepSORT/blob/main/docs/final.gif)
