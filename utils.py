import cv2
import numpy as np
import math

def letterbox_image(img,desired_size=(416,416)):
    h,w,_ = img.shape
    new_h,new_w = desired_size
    h_ratio,w_ratio = new_h/h,new_w/w
    scale = min(h_ratio,w_ratio)
    new_img = cv2.resize(img,(int(scale*w),int(scale*h)))
    pad_img = np.pad(new_img,((math.floor((new_h-new_img.shape[0])/2), math.ceil((new_h-new_img.shape[0])/2)),\
                             (math.floor((new_w-new_img.shape[1])/2), math.ceil((new_w-new_img.shape[1])/2)),(0, 0)))
    return pad_img

def letterbox_image_label(img,bboxes,desired_size=(416,416)):
    h,w,_ = img.shape
    new_h,new_w = desired_size
    h_ratio,w_ratio = new_h/h,new_w/w
    scale = min(h_ratio,w_ratio)
    new_img = cv2.resize(img,(int(scale*w),int(scale*h)))
    pad_img = np.pad(new_img,((math.floor((new_h-new_img.shape[0])/2), math.ceil((new_h-new_img.shape[0])/2)),\
                             (math.floor((new_w-new_img.shape[1])/2), math.ceil((new_w-new_img.shape[1])/2)),(0, 0)))
    bboxes[...,0] = bboxes[...,0]*img.shape[1]
    bboxes[...,1] = bboxes[...,1]*img.shape[0]
    bboxes[...,2] = bboxes[...,2]*img.shape[1]
    bboxes[...,3] = bboxes[...,3]*img.shape[0]
    new_bboxes = bboxes.copy()
    new_bboxes[...,0] = bboxes[...,0] - bboxes[...,2]/2
    new_bboxes[...,1] = bboxes[...,1] - bboxes[...,3]/2
    new_bboxes[...,2] = bboxes[...,0] + bboxes[...,2]/2
    new_bboxes[...,3] = bboxes[...,1] + bboxes[...,3]/2
    new_bboxes[...,0:4] = new_bboxes[...,0:4]*scale
    new_bboxes[...,0] = new_bboxes[...,0]+math.floor((new_w-new_img.shape[1])/2)
    new_bboxes[...,1] = new_bboxes[...,1]+math.floor((new_h-new_img.shape[0])/2)
    new_bboxes[...,2] = new_bboxes[...,2]+math.floor((new_w-new_img.shape[1])/2)
    new_bboxes[...,3] = new_bboxes[...,3]+math.floor((new_h-new_img.shape[0])/2)
    new_bboxes = new_bboxes.astype(int)
    bboxes = new_bboxes.copy().astype(np.float32)
    bboxes[...,0] = ((new_bboxes[...,0] + new_bboxes[...,2])//2)/desired_size[0]
    bboxes[...,1] = ((new_bboxes[...,1] + new_bboxes[...,3])//2)/desired_size[0]
    bboxes[...,2] = (new_bboxes[...,2] - new_bboxes[...,0])/desired_size[0]
    bboxes[...,3] = (new_bboxes[...,3] - new_bboxes[...,1])/desired_size[0]
    return pad_img, bboxes

def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = np.maximum(box1_x1, box2_x1)
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)

    intersection = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)
    
    box1_area = np.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = np.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return  -np.log(1e-16 + 1. / x - 1.)