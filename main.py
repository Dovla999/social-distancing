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

LABELS = LABELS_YOLO

CONF = 0.3
if args.get('conf'):
    CONF = float(args.get('conf'))

DISTANCE_RISK = float(args.get('min'))
DISTANCE_POTENTIAL_RISK = float(args.get('max'))

if args.get('net') == "ssd":
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_SSD, MODEL_SSD)
    LABELS = LABELS_SSD

    # configure your ssd parameters here

    if not args.get('conf'):
        CONF = 0.4

if args.get('net') == "yolo-tiny":
    net = cv2.dnn.readNetFromDarknet(YOLO_TINY_CFG, YOLO_TINY_WEIGHTS)
    LABELS = LABELS_TINY
##########################################################################
INP_VIDEO_PATH = 'TownCentreXVID.mp4'
OUT_VIDEO_PATH = 'TownCentreXVIDDetected.mp4'

GROUND_TRUTH = "TownCentre-groundtruth.top"

gt = {}

person_idx = LABELS.index("person")

with open(GROUND_TRUTH, mode='r') as gf_file:
    lines = gf_file.readlines()
    for line in lines:
        tokens = line.replace("\n", "").split(',')
        tokens = [float(token) for token in tokens]
        if int(tokens[1]) not in gt:
            gt[int(tokens[1])] = []

        gt[int(tokens[1])].append(
            [int(tokens[-4]), int(tokens[-3]), int(tokens[-2]), int(tokens[-1]), person_idx, 0, 0]
        )
