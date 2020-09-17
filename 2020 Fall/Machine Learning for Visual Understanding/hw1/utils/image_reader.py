import cv2, pdb
import numpy as np
import torch

def image_reader(fname,crop=False,detector=None):

	image = cv2.imread(fname)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	if crop:
		bboxes = detector.detect_faces(image, conf_th=0.9, scales=[1])

		sx = int((bboxes[0][0]+bboxes[0][2])/2)
		sy = int((bboxes[0][1]+bboxes[0][3])/2)
		ss = int(max((bboxes[0][3]-bboxes[0][1]),(bboxes[0][2]-bboxes[0][0]))/2)

		x1 = int(sx-ss*1.2)
		x2 = int(sx+ss*1.2)
		y1 = int(sy-ss)
		y2 = int(sy+ss*1.4)

		try:
			image = image[y1:y2,x1:x2]
		except:
			raise ValueError('Face image is too tight')

	image = cv2.resize(image, dsize=(224, 224))

	image = np.expand_dims(image,axis=0)
	image = np.transpose(image,(0,3,1,2))

	return torch.FloatTensor(image)

