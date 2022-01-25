# social-distancing
Real-time social distancing classifier

# About
This project attempts to enforce social distancing rules in these challenging times of the Covid-19 pandemic. The goal is to recognize people in a given frame of video in real
time and classify each of them into one of the following three categories of social distancing:
_safe_, _potential_risk_, _risk_. The models used to detect people are: YOLOv4, YOLOv4-tiny and SSD.

# Devs
- Vladimir Jovin, SW-30/2018
- Jovan Petljanski, SW-31/2018
- Milovan Milovanović, SW-41/2018

# Prerequisites
- Python 3.9
- [YOLOv4 weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) inside the /YOLO directory of this project
- Dependencies listed in _requirements.txt_

These commands can be used in order to install the dependencies within a virtual environment in the root project directory, assuming Windows and Python 3.9 installed:
1) `py -m venv env`
2) `.\env\Scripts\activate`
3) `pip install -r requirements.txt`

# Usage
There are four (optional) command-line arguments that can be defined when running the program:
1) `-n` or `--net` to choose the desired model, with 3 possible values: `yolov4` for YOLOv4, `ssd` for SSD, or `yolo-tiny` for YOLOv4-tiny. The default model is YOLOv4.
2) `-c` or `--conf` to set the confidence value for models (between 0 and 1). The default value is 0.3.
3) `-m` or `--min` to define the minimum distance (in meters) for a person to be classified as _potential_risk_ (if it's less than this value, the person is classified as _risk_). The default value is 2.
4) `-M` or `--max` to define the maximum distance (in meters) for a person to be classified as _potential_risk_ (if it's greater than this value, the person is classified as _safe_). The default value is 3.

To run the program: `py main.py [<arguments>]`

Example: `py main.py -n ssd -c 0.4 -m 1 -M 4`

When execution has finished, the program will have generated an entry in the `results.txt` file in the root directory:

![image](https://user-images.githubusercontent.com/29868001/150995799-09450989-52a5-422c-ab56-f3ae3694b715.png)
