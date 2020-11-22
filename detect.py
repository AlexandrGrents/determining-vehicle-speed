import cv2
import numpy as np
from tqdm import tqdm

from utils import draw_label, draw_box
from utils.masker import Masker
from speedometer import Speedometer 


blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


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
	masker = Masker(mask_file, (w, h))

	# Создаём трекер для отслеживания автомобилей на разных кадрах
	tracker = Sort()
	speedometer = Speedometer(coef_file, fps=fps, size=(w, h))


	# Обрабатываем видео покадрово
	for frame_id in tqdm(range(frame_count)):
		# Считываем кадр, создаём кадр для видео с результатом
		ret, frame = istream.read()
		if not ret:
			break
		else:
			out_frame = frame

		# Выделяем проезжую часть
		masked_frame = masker.apply(frame)

		# Распознаём объекты на кадре
		outputs = detector(masked_frame)

		# Получаем bbox (рамки) и вероятность принадлежности классу для каждого найденного объекта
		boxes = outputs['instances'].pred_boxes.tensor.to("cpu").numpy()
		scores = outputs['instances'].scores.to("cpu").numpy()
		classes = outputs['instances'].pred_classes.to("cpu").numpy()

		# Обновляем трекер, получаем track_id (id объекта на предыдущих кадрах) для каждого найденного объекта
		for_tracker = np.concatenate([boxes, scores[:, None]], axis=1)
		dets, associaties = tracker.update(for_tracker, make_associaties = True)

		speeds = spmeter.update(dets)


		for i in range(scores.shape[0]):
			box = boxes[i].astype(np.int16)
			track_id = associaties[i]

			if track_id == 0:  continue
			
			speed = spmeter.ms_to_kmh(speeds[track_id])
			class_id = classes[i]
			class_label = class_names[class_id]

			draw_box(out_frame, box)

			draw_label(out_frame, track_id, box[:2], size=2, shadow=True)
			draw_label(out_frame, round(speed, 2), (box[0], box[1] + 10), color=(255, 100, 100), shadow=True)
			draw_label(out_frame, class_label, (box[0], box[1] + 25), color=(100, 255, 100), shadow=True)

		# Добавляем кадр с разметкой к результирующему видео
		writer.write(out_frame)

	# Сохраняем видео с результатом
	writer.release()
	return True