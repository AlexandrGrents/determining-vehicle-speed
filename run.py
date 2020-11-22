# import some common libraries
import numpy as np
import os
import json
import cv2
import random
import sys
import matplotlib 
import argparse
matplotlib.use('TKAgg')

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
from sort.sort import Sort
from detect import detect_on_video

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

CLASS_NAMES = ['car', 'minibus', 'trolleybus', 'tram', 'truck', 'bus', 'middle_bus', 'ambulance', 'fire_truck', 'middle_truck', 'tractor', 'uncategorized', 'van', 'person']



parser = argparse.ArgumentParser(description="Detect vehicles end they speed in the video stream.")

parser.add_argument("--video-file", help="path to the requested .mp4 video. Default: input.mp4", default = "input.mp4", type=str)
parser.add_argument("--save-to", help="path where to save results. Default: output.mp4", default = "output.mp4", type=str)
parser.add_argument("--image-mask", help="path to the image mask. Default: mask.png",default = "mask.png", type=str)

args = parser.parse_args()


if __name__=="__main__":
	
	register_coco_instances("my_dataset", {'thing_classes': CLASS_NAMES}, "", "")
	dataset_metadata = MetadataCatalog.get("my_dataset")

	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")) # получение используемой модели 
	cfg.MODEL.WEIGHTS = "model_final.pth" # путь к найденным лучшим весам модели
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # установить порог распознавания объекта в 50% (объекты, распознанные с меньшей вероятностью не будут учитываться)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) # число классов для распознавания

	detector = DefaultPredictor(cfg) 
	tracker = Sort(max_age = 40)

	detect_on_video(args.video_file, args.save_to, detector, tracker, mask_file = args.image_mask, to_mp4 = True)