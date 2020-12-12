import cv2
import numpy as np
from tqdm import tqdm

from .utils import draw_label, draw_box
from .utils.masker import Masker
from .speedometer import Speedometer


blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


def detect_on_frame(frame, detector, tracker, speedometer):
	outputs = detector(frame)

	# Получаем bbox (рамки) и вероятность принадлежности классу для каждого найденного объекта
	boxes = outputs['instances'].pred_boxes.tensor.to("cpu").numpy()
	scores = outputs['instances'].scores.to("cpu").numpy()
	class_ids = outputs['instances'].pred_classes.to("cpu").numpy()

	detect_count = scores.shape[0]

	# Обновляем трекер, получаем track_id (id объекта на предыдущих кадрах) для каждого найденного объекта
	for_tracker = np.concatenate([boxes, scores[:, None]], axis=1)
	dets, associaties = tracker.update(for_tracker, make_associaties=True)

	speeds = speedometer.update(dets)
	boxes = boxes.tolist()
	scores = scores.tolist()
	class_ids = class_ids.tolist()
	track_ids = associaties.tolist()

	for i in range(detect_count-1, -1, -1):
		if track_ids[i] == 0:
			boxes.pop(i)
			scores.pop(i)
			class_ids.pop(i)
			track_ids.pop(i)
			detect_count -=1

	return {'track_ids': track_ids, 'boxes': boxes, 'scores': scores, 'class_ids':class_ids, 'speeds': speeds, 'detect_count': detect_count}


def drow_on_frame(frame, detections, class_names):
	for i in range(detections['detect_count']):
		box = detections['boxes'][i]
		track_id = detections['track_id'][i]
		class_label = class_names[detections['class_ids'][i]]
		speed_kmh = Speedometer.ms_to_kmh(detections['speeds'][track_id])

		draw_box(frame, box)
		draw_label(frame, track_id, box[:2], size=2, shadow=True)
		draw_label(frame, round(speed_kmh, 2), (box[0], box[1] + 10), color=(255, 100, 100), shadow=True)
		draw_label(frame, class_label, (box[0], box[1] + 25), color=(100, 255, 100), shadow=True)
	return frame


def detect_on_video(video_path, out_path, detector, tracker, coef_file='coef.json', mask_file = 'mask.png' , to_mp4 = False, class_names = None):
	if class_names is None:
		class_names = ['car', 'minibus', 'trolleybus', 'tram', 'truck', 'bus', 'middle_bus', 'ambulance', 'fire_truck', 'middle_truck', 'tractor', 'uncategorized', 'van', 'person']

	# Создаём считыватель исходного видео
	istream = cv2.VideoCapture(video_path)

	# Получаем данные о исходном видео
	w = int(istream.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(istream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(istream.get(cv2.CAP_PROP_FPS))
	frame_count = int(istream.get(cv2.CAP_PROP_FRAME_COUNT))

	# Создаём писателя для результирующего видео
	if to_mp4:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
	else:
		fourcc = cv2.VideoWriter_fourcc(*'VP80')
	writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), True)

	# Создаём экземпляр класса Masker для выделения проезжей части на каждом кадре
	masker = Masker(mask_file, size=(w, h))
	speedometer = Speedometer(coef_file, fps=fps, size=(w, h))
	detections_on_video = []
	# Обрабатываем видео покадрово
	for frame_id in tqdm(range(frame_count)):
		# Считываем кадр, создаём кадр для видео с результатом
		ret, frame = istream.read()
		if not ret:
			break

		# Выделяем проезжую часть
		masked_frame = masker.apply(frame)

		# Распознаём объекты на кадре
		detections = detect_on_frame(masked_frame, detector, tracker, speedometer)

		# Добавляем кадр с разметкой к результирующему видео
		writer.write(drow_on_frame(frame, detections, class_names))
		detections['frame_id'] = frame_id
		detections_on_video.append(detections)

	# Сохраняем видео с результатом
	writer.release()
	return detections_on_video
