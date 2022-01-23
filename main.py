import argparse


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
