import numpy as np
from tensorflow import keras
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import math
from PIL import Image
from utils import letterbox_image_label

class DataGenerator(keras.utils.Sequence):
    def __init__(self,img_dir,label_dir,anchors,batch_size,image_size=416,scales=[13,26,52],):
        self.img_paths = glob.glob(os.path.join(img_dir,"*.jpg"))
        self.label_paths = glob.glob(os.path.join(label_dir,"*.xml"))
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.scales = scales
        self.anchors = np.array(anchors).reshape(9,2)
        self.ignore_thresh = 0.5
        self.batch_size = batch_size
        self.class_map = {'face':0,'face_mask':1}
        self.shuffle = True
        self.num_classes = 2
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        img_paths = [self.img_paths[i] for i in indexes]
        batch_label_path = []
        batch_img_path = []
        for i in img_paths:
            label_path = os.path.join(self.label_dir,i.split('//')[0].split('.')[-2]+'.xml')
            if label_path in self.label_paths:
                batch_label_path.append(label_path)
                batch_img_path.append(i)
            else:
                continue                
        img_batch = np.empty(shape=(len(batch_img_path),self.image_size,self.image_size,3),dtype=np.float32)
        target_batch = [np.zeros((len(batch_img_path),9 // 3, s, s, 5+self.num_classes)) for s in self.scales]
        # Find list of IDs
        for cnt,(img_path,label_path) in enumerate(zip(batch_img_path,batch_label_path)):
            image,target = self.convert_to_targets(img_path, label_path)
            img_batch[cnt,:] = image
            target_batch[0][cnt,:] = target[0]
            target_batch[1][cnt,:] = target[1]
            target_batch[2][cnt,:] = target[2]
        # Generate data
        return img_batch, target_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_paths))
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
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
                
        ih,iw,_ = image.shape
        xmltree = ET.parse(label_path).getroot()
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bboxes = []
        for obj in xmltree.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            name = obj.find('name').text.lower().strip().lower()
            bbox = obj.find('bndbox')
            # get face rect
            bndbox = [int(bbox.find(pt).text) for pt in pts]
            #img = cv2.rectangle(img, tuple(bndbox[:2]), tuple(bndbox[2:]), (0,0,255), 2)
            #img = cv2.putText(img, name, tuple(bndbox[:2]), 3, 1, (0,255,0), 1)
            bboxes.append([int((bndbox[2]+bndbox[0])/2), int((bndbox[1]+bndbox[3])/2),\
                           int((bndbox[2]-bndbox[0])), int((bndbox[3]-bndbox[1])), self.class_map[name]])
        bboxes = np.array(bboxes, dtype=np.float32)
        bboxes[:,0] = bboxes[:,0]/iw
        bboxes[:,1] = bboxes[:,1]/ih
        bboxes[:,2] = bboxes[:,2]/iw
        bboxes[:,3] = bboxes[:,3]/ih
        image,bboxes = letterbox_image_label(image,bboxes)
        targets = [np.zeros((9 // 3, s, s, 5+self.num_classes),dtype=np.float32) for s in self.scales]
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
                    targets[scale_idx][anchor_on_scale, i, j, 5+int(class_label)] = 1 
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > 0.5:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        return image/255.0, targets