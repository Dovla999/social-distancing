import argparse
import cv2

ap = argparse.ArgumentParser()

ap.add_argument('-n', '--net', type=str,
                default="yolov4",
                help="desired model, default yolov4, pass ssd for SSD, pass yolo-tiny for yolo-tinyv4")

ap.add_argument('-c', '--conf', type=float,
                help="confidence for models")

ap.add_argument('-m', '--min', type=float,
                default=2,
                help="min distance for person to be classified in potential_risk")
ap.add_argument('-M', '--max', type=float,
                default=3,
                help="max distance for person to be classified in potential_risk")

args = vars(ap.parse_args())

##################### SSD ######################
PROTOTXT_SSD = "SSD/MobileNetSSD_deploy.prototxt"
MODEL_SSD = "SSD/MobileNetSSD_deploy.caffemodel"
LABELS_SSD = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
              "tvmonitor"]
################################################


##################### YOLOV4-tiny ##############
YOLO_TINY_CFG = "YOLO_T/yolov4-tiny.cfg"
YOLO_TINY_WEIGHTS = "YOlO_T/yolov4-tiny.weights"
LABELS_TINY = open("YOLO_T/coco.names.txt").read().strip().splitlines(keepends=False)
################################################


##################### YOLO #####################
YOLO_CFG = "YOLO/yolov4.cfg"
YOLO_WEIGHTS = "YOLO/yolov4.weights"
LABELS_YOLO = open("YOLO/coco.names.txt").read().strip().splitlines(keepends=False)
################################################


net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)