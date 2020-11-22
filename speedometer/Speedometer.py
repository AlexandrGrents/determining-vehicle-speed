import json

from .CoordConverter import CoordConverter


def box_to_pos(box, de = 4):
	return [round((box[0] + box[2])/2), round((box[1] + box[3])/2)]

class Speedometer(object):
	def __init__(self, filename = None, fps = 25, size = None, kalman_coef = None):
		self.last_positions = {}
		self.last_speeds = {}
		self.last_momental_speed = {}
		self.last_frame_nums = {}
		self.frame_num = 0
		self.time_per_frame = 1/fps
		if size:
			self.w, self.h = size[0], size[1]
		if kalman_coef:
			self.kalman_coef = kalman_coef
		else: 
			self.kalman_coef = 0.1

		if not filename is None:
			self.load_coefs(filename)

	def set_params(self, fps = None, size = None, kalman_coef = None):
		if fps:
			self.time_per_frame = 1/fps
		if size:
			self.w, self.h = size[0], size[1]
		if kalman_coef:
			self.kalman_coef = kalman_coef


	def load_coefs(self, filename = 'coef.json'):
		with open(filename,'r') as file:
			data = json.load(file)
			self.converter = CoordConverter(zones = data['zones'], borders = data['borders'])

	def update(self, detections):
		speeds = {}
		for detection in detections:
			box, track_id = detection[:4], detection[4]
			pos = box_to_pos(box)
			pos[0] = pos[0] / self.w
			pos[1] = pos[1] / self.h
			last_pos = self.last_positions.get(track_id)
			last_frame_num = self.last_frame_nums.get(track_id)
			if last_pos is None or last_frame_num is None:
				speed = 0
				kalman_speed = 0
			else:
				dist = self.converter.calc_converted_dist(last_pos, pos)
				speed = dist / ((self.frame_num - last_frame_num) * self.time_per_frame)
				kalman_speed = self.kalman_coef * speed + (1 - self.kalman_coef) * self.last_speeds[track_id]
			speeds[track_id] = kalman_speed

			self.last_frame_nums[track_id] = self.frame_num

			self.last_positions[track_id] = pos
			self.last_speeds[track_id] = kalman_speed
			self.last_momental_speed[track_id] = speed

		self.frame_num+=1
		return speeds

	def get_position(self, track_id):
		return [int(x) for x in self.converter.convert(*self.last_positions[track_id])]

	def get_momental_speed(self, track_id):
		return self.last_momental_speed[track_id]

	@staticmethod
	def ms_to_kmh(speed):
		return speed*3.6