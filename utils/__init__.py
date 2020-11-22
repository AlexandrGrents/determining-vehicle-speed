import cv2
import numpy as np


def draw_label(image, value, pos, color = None, size = 1, shadow = False, shadow_color = None):
	size = int(size)
	weight = size/2
	offset = int(size/2)

	if shadow:
		if shadow_color is None:
			shadow_color = [0, 0, 0]
		shadow_size = size + 4
		shadow_offset = offset + 2
		cv2.putText(image, str(value), (pos[0]+shadow_offset, pos[1]+shadow_offset), cv2.FONT_HERSHEY_SIMPLEX, weight, shadow_color, shadow_size)

	if color is None:
		color = [255, 255, 255]
	cv2.putText(image, str(value), (pos[0]+offset, pos[1]+offset), cv2.FONT_HERSHEY_SIMPLEX, weight, color, size)

def draw_box(image, box, color = None):
	if color is None:
		color = [0, 255, 0]
	cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
