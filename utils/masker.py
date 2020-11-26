import cv2
import numpy as np

class Masker(object):
	def __init__(self, filename, size = None):
		self.mask = cv2.imread(filename)
		if size is None:
			h, w, _ = self.mask.shape
			self.w = w
			self.h = h
			self.mask = cv2.resize(self.mask, dsize=(w, h))
		else:
			self.w = size[0]
			self.h = size[1]
		self.mask = self.mask.astype(np.bool)

	def change_maskfile(self, filename, size = None):
		self.mask = cv2.imread(filename)
		h, w, _ = self.mask.shape
		if size is None:
			self.w = w
			self.h = h
			self.mask = cv2.resize(self.mask, dsize=(w, h))
		else:
			self.w = size[0]
			self.h = size[1]
		self.mask = self.mask.astype(np.bool)

	def apply(self, image):
		return image * self.mask
