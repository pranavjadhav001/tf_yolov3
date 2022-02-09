import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy,MeanSquaredError,CategoricalCrossentropy
import numpy as np
from utils import intersection_over_union

class YOLOLOSS:
    def __init__(self,anchors):
        self.bce_loss = BinaryCrossentropy(from_logits=True) 
        self.mse_loss = MeanSquaredError()
        self.cce_loss = CategoricalCrossentropy() 
        self.anchors = anchors
        self.anchors = np.array(self.anchors).reshape(3,1,3,1,1,2)
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        
    def loss_per_scale(self,predictions,target,anchors):
        object_score = target[...,0] == 1
        no_object_score = target[...,0] == 0
        anchors = anchors.copy()

        no_object_loss = self.bce_loss(target[...,0][no_object_score],predictions[...,0][no_object_score])

        box_preds = tf.concat([tf.sigmoid(predictions[...,1:3]),tf.exp(predictions[...,3:5])*anchors],axis=-1)
        iou_pred_target = intersection_over_union(box_preds[object_score],target[...,1:5][object_score])
        object_loss = self.mse_loss(iou_pred_target*target[...,0:1][object_score],predictions[...,0:1][object_score])

        xy_loss = self.mse_loss(target[...,1:3][object_score],tf.sigmoid(predictions[...,1:3][object_score]))
        wh_loss = self.mse_loss(tf.math.log(
            (1e-16 + target[..., 3:5] / anchors))[object_score],predictions[...,3:5][object_score])
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            target[...,5:][object_score],predictions[...,5:][object_score]))
        return self.lambda_class*class_loss + self.lambda_noobj*no_object_loss + \
            self.lambda_obj*object_loss + xy_loss + self.lambda_box*wh_loss
    
    def main_loss(self,predictions,targets):
        loss1 = self.loss_per_scale(predictions[0],targets[0],self.anchors[0])
        loss2 = self.loss_per_scale(predictions[1],targets[1],self.anchors[1])
        loss3 = self.loss_per_scale(predictions[2],targets[2],self.anchors[2])
        return loss1 + loss2 + loss3