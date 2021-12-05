import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,Subtract,Add
from tensorflow.keras.layers import BatchNormalization,Activation,ZeroPadding2D,Concatenate
from tensorflow.keras.layers import LeakyReLU,Lambda
from tensorflow.keras.layers import UpSampling2D,Conv2D,Conv2DTranspose,MaxPooling2D,AveragePooling2D
from tensorflow.keras.models import Sequential,Model,load_model

def conv_bn_block(x,**kwargs):
    x = Conv2D(**kwargs,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    return x

def residual_block(x,in_filters=32,n_repeats=1):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = conv_bn_block(x,filters=in_filters*2,kernel_size=(3,3),strides=2,padding='valid')
    for _ in range(n_repeats):
        x_add = x
        x = conv_bn_block(x,filters=in_filters,kernel_size=(1,1),strides=1,padding='same')
        x = conv_bn_block(x,filters=in_filters*2,kernel_size=(3,3),strides=1,padding='same')
        x = Add()([x,x_add])
    return x

def conv_pred_block(x,num_filters):
    x = conv_bn_block(x,filters=num_filters,kernel_size=(1,1),strides=1,padding='same')
    x = conv_bn_block(x,filters=num_filters*2,kernel_size=(3,3),strides=1,padding='same')
    x = conv_bn_block(x,filters=num_filters,kernel_size=(1,1),strides=1,padding='same')
    x = conv_bn_block(x,filters=num_filters*2,kernel_size=(3,3),strides=1,padding='same')
    x = conv_bn_block(x,filters=num_filters,kernel_size=(1,1),strides=1,padding='same')
    return x

def get_model(num_classes=11,input_shape=(256,256,3)):
    image = Input(shape=input_shape,name='input_image')
    x1 = conv_bn_block(image,filters=32,kernel_size=(3,3),strides=1,padding='same')
    x2 = residual_block(x1,in_filters=32,n_repeats=1)
    x3 = residual_block(x2,in_filters=64,n_repeats=2)
    x4 = residual_block(x3,in_filters=128,n_repeats=8)
    x5 = residual_block(x4,in_filters=256,n_repeats=8)
    x6 = residual_block(x5,in_filters=512,n_repeats=4)
    x7 = conv_pred_block(x6,512)
    out1 = conv_bn_block(x7,filters=1024,kernel_size=(3,3),strides=(1,1),padding='same')
    out1 = Conv2D((num_classes+5)*3,use_bias=False,kernel_size=(1,1),padding='same')(out1)
    out1 = Reshape((3,13,13,num_classes+5))(out1)
    x8 = conv_bn_block(x7,filters=256,kernel_size=(1,1),strides=1,padding='same')
    x9 = UpSampling2D((2,2))(x8)
    x10 = Concatenate()([x9,x5])
    x11 =  conv_pred_block(x10,256)
    out2 = conv_bn_block(x11,filters=512,kernel_size=(3,3),strides=(1,1),padding='same')
    out2 = Conv2D((num_classes+5)*3,use_bias=False,kernel_size=(1,1),padding='same')(out2)
    out2 = Reshape((3,26,26,num_classes+5))(out2)
    x12 = conv_bn_block(x11,filters=128,kernel_size=(1,1),strides=1,padding='same')
    x13 = UpSampling2D((2,2))(x12)
    x14 = Concatenate()([x13,x4])
    x15 =  conv_pred_block(x14,128)
    out3 = conv_bn_block(x15,filters=256,kernel_size=(3,3),strides=(1,1),padding='same')
    out3 = Conv2D((num_classes+5)*3,use_bias=False,kernel_size=(1,1),padding='same')(out3)
    out3 = Reshape((3,52,52,num_classes+5))(out3)
    model = Model(inputs=[image],outputs=[out1,out2,out3])
    return model