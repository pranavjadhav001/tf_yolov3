import numpy as np
from tensorflow import keras
import os
import pandas as pd
from PIL import Image
from utils import letterbox_image_label

class DataGenerator(keras.utils.Sequence):
	def __init__(self,csv_path,img_dir,label_dir,anchors,batch_size,image_size=416,scales=[13,26,52],):
		self.annotations = pd.read_csv(csv_path)
		self.img_dir = img_dir
		self.label_dir = label_dir
		self.image_size = image_size
		self.scales = scales
		self.anchors = np.array(anchors).reshape(9,2)
		self.ignore_thresh = 0.5
		self.batch_size = batch_size
		self.shuffle = True
		self.num_classes = 20
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.annotations) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] 
		img_batch = np.empty(shape=(self.batch_size,self.image_size,self.image_size,3),dtype=np.float32)
		target_batch = [np.zeros((self.batch_size,9 // 3, s, s, 5+self.num_classes)) for s in self.scales]
		# Find list of IDs
		for cnt,i in enumerate(indexes):
			label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
			img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
			image,target = self.convert_to_targets(img_path, label_path)
			img_batch[cnt,:] = image
			target_batch[0][cnt,:] = target[0]
			target_batch[1][cnt,:] = target[1]
			target_batch[2][cnt,:] = target[2]
		# Generate data
		return img_batch, target_batch

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.annotations))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def iou_width_height(self,boxes1, boxes2):
		"""
		Parameters:
			boxes1 (tensor): width and height of the first bounding boxes
			boxes2 (tensor): width and height of the second bounding boxes
		Returns:
			tensor: Intersection over union of the corresponding boxes
		"""
		intersection = np.minimum(boxes1[..., 0], boxes2[..., 0]) * np.minimum(
			boxes1[..., 1], boxes2[..., 1]
		)
		union = (
			boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
		)
		return intersection / union

	def convert_to_targets(self,image_path,label_path):
		bboxes = np.array(np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist())
		image = Image.open(image_path).convert("RGB")
		image = np.array(image)
		image,bboxes = letterbox_image_label(image,bboxes)
		targets = [np.zeros((9 // 3, s, s, 5+self.num_classes)) for s in self.scales]
		for box in bboxes:
			iou_anchors = self.iou_width_height(box[2:4],self.anchors)
			anchor_indices = np.argsort(iou_anchors)[::-1]
			x, y, width, height, class_label = box
			has_anchor = [False] * 3  # each scale should have one anchor
			for anchor_idx in anchor_indices:
				scale_idx = anchor_idx // 3
				anchor_on_scale = anchor_idx % 3
				new_s = self.scales[scale_idx]
				i, j = int(new_s * y), int(new_s * x)  # which cell
				anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
				if (anchor_taken == 0)  and not has_anchor[scale_idx]:
					targets[scale_idx][anchor_on_scale, i, j, 0] = 1
					x_cell, y_cell = new_s * x - j, new_s * y - i  # both between [0,1]
					width_cell, height_cell = (
						width * new_s,
						height * new_s,
					)  # can be greater than 1 since it's relative to cell
					box_coordinates = [x_cell, y_cell, width_cell, height_cell]
					targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
					targets[scale_idx][anchor_on_scale, i, j, 4+int(class_label)] = 1 
					has_anchor[scale_idx] = True
				elif not anchor_taken and iou_anchors[anchor_idx] > 0.5:
					targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
		return image, targets