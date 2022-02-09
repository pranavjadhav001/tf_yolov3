from keras.models import load_model

ANCHORS = [
    [(0.207, 0.411), (0.299, 0.34), (0.42, 0.57)],
    [(0.064, 0.10), (0.105, 0.15), (0.154, 0.245)],
    [(0.011, 0.02), (0.021, 0.038), (0.038, 0.065)],
]
datagen = DataGenerator(img_dir='/content/drive/MyDrive/face_dataset/val',\
    label_dir='/content/drive/MyDrive/face_dataset/val',anchors=ANCHORS,batch_size=8)

images,predictions = datagen[0]
scales=[13,26,52]
is_preds = True
predictions = model.predict(images)
all_bboxes = []
batch,anchor_dim,_,_,p = predictions[0].shape
for s,pred,anchor in zip(scales,predictions,ANCHORS):
    if is_preds :
        anchor = np.array(anchor).reshape(1, anchor_dim, 1, 1, 2)
        pred[...,0:3] = tf.sigmoid(pred[...,0:3])
        pred[...,3:5] = tf.exp(pred[...,3:5])*anchor
    cell_x,cell_y = np.meshgrid(np.arange(s),np.arange(s),indexing='ij')
    pred[...,1:2] = (pred[...,1:2]+np.expand_dims(np.tile(cell_y.reshape(1,1,s,s),(batch,anchor_dim,1,1)),axis=-1))/s
    pred[...,2:3] = (pred[...,2:3]+np.expand_dims(np.tile(cell_x.reshape(1,1,s,s),(batch,anchor_dim,1,1)),axis=-1))/s
    pred[...,3:5] = pred[...,3:5]/s
    pred = pred.reshape(batch,anchor_dim*s*s,p)
    pred[...,1:5] = pred[...,1:5]*416
    new_pred = pred.copy()
    new_pred[...,1] = pred[...,1] - pred[...,3]//2 
    new_pred[...,2] = pred[...,2] - pred[...,4]//2 
    new_pred[...,3] = pred[...,1] + pred[...,3]//2 
    new_pred[...,4] = pred[...,2] + pred[...,4]//2 
    all_bboxes.append(new_pred)
new_bboxes = np.concatenate((all_bboxes[0],all_bboxes[1],all_bboxes[2]),axis=1).astype(np.float32)
for image,bbox in zip(images,new_bboxes):
    selected_indices  = tf.image.non_max_suppression(bbox[...,1:5],bbox[...,0],\
                                               iou_threshold=0.5,max_output_size=2,score_threshold=0.5)
    selected_boxes = tf.gather(bbox, selected_indices)
    try:
      for i in selected_boxes:
          x1,y1,x2,y2 = i[1:5]
          image = cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
      plt.imshow(image)
      plt.show()
    except ValueError:
      pass