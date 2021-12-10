import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy,MeanSquaredError,CategoricalCrossentropy
import numpy as np
from utils import intersection_over_union

class YOLOLOSS:
    def __init__(self):
        self.bce_loss = BinaryCrossentropy(from_logits=False) 
        self.mse_loss = MeanSquaredError()
        self.cce_loss = CategoricalCrossentropy() 
        self.anchors = [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]]  # N
        self.anchors = np.array(self.anchors).reshape(3,1,3,1,1,2)
        self.lambda_class = 1
        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_box = 1
        
    def loss_per_scale(self,predictions,target,anchors):
        object_score = target[...,0] == 1
        no_object_score = target[...,0] == 0
        anchors = anchors.copy()        
        no_object_loss = self.bce_loss(tf.sigmoid(predictions[...,0][no_object_score]),\
                                       target[...,0][no_object_score])
        box_preds = tf.concat([tf.sigmoid(predictions[...,1:3]),tf.exp(predictions[...,3:5])*anchors],axis=-1)
        iou_pred_target = intersection_over_union(box_preds[object_score],target[...,1:5][object_score])
        object_loss = self.mse_loss(tf.sigmoid(predictions[...,0:1][object_score]),\
                                    iou_pred_target*target[...,0:1][object_score])
        new_predictions = tf.concat([tf.sigmoid(predictions[...,1:3]),predictions[...,3:5]],axis=-1)
        new_target = tf.concat([target[...,1:3],tf.math.log(
            (1e-16 + target[..., 3:5] / anchors))],axis=-1)
        box_loss = self.mse_loss(new_predictions[object_score],new_target[object_score])
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.sigmoid(predictions[...,5:][object_score]),\
            target[...,5:][object_score]))
        print(no_object_loss.numpy(),object_loss.numpy(),box_loss.numpy(),class_loss.numpy())
        return self.lambda_class*class_loss + self.lambda_noobj*no_object_loss + \
            self.lambda_obj*object_loss + self.lambda_box*box_loss
    
    def main_loss(self,predictions,targets):
        loss1 = self.loss_per_scale(predictions[0],targets[0],self.anchors[0])
        loss2 = self.loss_per_scale(predictions[1],targets[1],self.anchors[1])
        loss3 = self.loss_per_scale(predictions[2],targets[2],self.anchors[2])
        return loss1 + loss2 + loss3