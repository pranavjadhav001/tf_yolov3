from loss import YOLOLOSS
import time
from face_dataset import DataGenerator
from model import get_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

ANCHORS = [
    [(0.207, 0.411), (0.299, 0.34), (0.42, 0.57)],
    [(0.064, 0.10), (0.105, 0.15), (0.154, 0.245)],
    [(0.011, 0.02), (0.021, 0.038), (0.038, 0.065)],
]
datagen = DataGenerator(img_dir='/content/drive/MyDrive/face_dataset/val',label_dir='/content/drive/MyDrive/face_dataset/val',anchors=ANCHORS,batch_size=8)

model= get_model(input_shape=(416,416,3),num_classes=2)
custom_loss = YOLOLOSS()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(1, 200):
    print('new epoch')
    batch_loss = []
    start = time.time()
    for batch, (images, labels) in enumerate(datagen):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            pred_loss = []
            total_loss = custom_loss.main_loss(outputs,labels)
            batch_loss.append(total_loss)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables))
    print('time taken:',(time.time()-start),'loss:',np.mean(batch_loss))