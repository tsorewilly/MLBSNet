#from nets.Repunet import convnet_unet
#from nets.fcn import FCN_Vgg16_32s
#from nets.unetplusplus import UNetPlusPlus
#from nets.deeplab import Deeplabv3
#from nets.Repconvnet import Rep_get_convnet_encoder
#from nets.pspnet import pspnet  #Change the code and data to return, change the accuracy evaluation, change pr[0]
#from nets.repvgg import repvgg_model_convert
#from nets.helper_functions import UNetPlusPlus,U_Net

from nets.MLBSNet import MLBSNet  #Change the input size and data enhancement procedures
import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import tensorflow.keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.font_manager as font_manager
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import copy
from metricsold import Iou_score, f_score
from nets.repvgg import repvgg_model_convert
import math
import datetime
#from tensorflow.tensorflow.keras.utils.data_utils import get_file

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
K.set_image_data_format('channels_last')

NCLASSES = 1
HEIGHT = 256
WIDTH = 256
HEIGHT1 = 256
WIDTH1 = 256
loss1 = None

def generate_arrays_from_file(lines,batch_size,aug = None):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # Get a batch_size of data
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
                # Read a certain path of the first column in train.txt
            name = lines[i].split(';')[0]
            # read image from file
            img = Image.open(r"./gw_904/jpg" + '/' + name)
            img = img.resize((WIDTH,HEIGHT))
            tmp_im_array = np.array(img)[:,:,0]
            tmp_im_array = tmp_im_array/255
            tmp_im_array = tmp_im_array[np.newaxis,:,:]

            #X_train.append(img)
            name = (lines[i].split(';')[1]).replace("\n", "") # read image from file

            img = Image.open(r"./gw_904/png" + '/' + name)
            img = img.resize((int(WIDTH1),int(HEIGHT1)))
            tmp_lb_array = np.array(img)[:,:,0]
            tmp_lb_array = tmp_lb_array[np.newaxis,:,:] #Class #1


            if len(X_train) == 0:
                X_train = tmp_im_array
                Y_train = tmp_lb_array
            else:
                X_train = np.concatenate((X_train,tmp_im_array),axis=0)
                Y_train = np.concatenate((Y_train,tmp_lb_array),axis=0)
            #Y_train.append(img[:,:,0])
            # Restart after reading one cycle
            i = (i+1) % n

        X_train = X_train[:,:,:,np.newaxis]
        Y_train = Y_train[:,:,:,np.newaxis] #Labels are stored in Y_train, and dimension #1 needs to be added
        # print(X_train.shape)
        # print(Y_train.shape)
        if (aug is not None):
        #and (random.random() > 0.5):
            X_train = next(aug.flow(X_train,batch_size = batch_size,shuffle=False,seed = i))
            Y_train = next(aug.flow(Y_train,batch_size = batch_size,shuffle=False,seed = i))
            #Image.fromarray(X_train[0,:,:,0]*255).show()
        #print(Y_train.shape)
        #print(np.array(X_train).shape,np.array(Y_train).shape)

        # Return data in batchsize, reshuffle after each return round
        yield(X_train, Y_train)
        #yield (X_train,[Y_train,Y_train,Y_train,Y_train]) #Multi-class

aug = ImageDataGenerator( #define a data augmentation generator
     rotation_range = 45,       #define the rotation range
     zoom_range = 0.2,          #Randomly scale the image size proportionally
     width_shift_range = 0.2,   #image horizontal shift range
     height_shift_range = 0.2,  #vertical shift range of the image
     #brightness_range=(0,0.1),
     #shear_range = 0.05,       #horizontal or vertical projection transformation
     horizontal_flip = True,    #Flip the image horizontally
     fill_mode = "reflect"      #fill pixels, appear after rotation or translation
)

def new_sqrt(y_true,y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    p_t = y_true * (K.ones_like(y_true) - y_pred)+ (K.ones_like(y_true) - y_true) * y_pred + K.epsilon()
    sqrt_loss = tf.sqrt(p_t)
    return K.mean(sqrt_loss)

def my_huber_loss(y_true,y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error)/2
    big_error_loss = threshold*(tf.abs(error) - (0.5*threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

def new_xy_loss(y_true, y_pred):

    alpha = K.constant(0.00001, dtype=tf.float32)
    #alpha1 = K.constant(0.0007, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    y_true_p = y_true*y_pred
    #y_true_p = y_true_p + (K.ones_like(y_true) - y_true)
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred))
    #y_pred_p = y_pred_p + y_true
    y_p = y_pred_p + y_true_p
    #xy_loss = alpha1*K.sum(y_true/y_true_p) + alpha*K.sum((K.ones_like(y_true) - y_true)/y_pred_p)
    #p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    #xy_loss = alpha*K.sum(K.ones_like(y_true)/y_p) - K.log(p_t)
    #xy_loss = alpha * K.sum(K.ones_like(y_true) / y_p)
    xy_loss = y_true/y_pred
    xy_loss = alpha*xy_loss
    #e = math.exp(1)
    #xy_loss = 1 / (tf.pow(e, -xy_loss) + 1)

    #return xy_loss
    return K.mean(xy_loss)

def new_xy_loss_v1(y_true, y_pred):

    alpha = K.constant(0.000001, dtype=tf.float32)
    #alpha1 = K.constant(0.0007, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    y_true_p = y_true*y_pred
    #y_true_p = y_true_p + (K.ones_like(y_true) - y_true)
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred))
    #y_pred_p = y_pred_p + y_true
    y_p = y_pred_p + y_true_p + K.epsilon()
    #xy_loss = alpha1*K.sum(y_true/y_true_p) + alpha*K.sum((K.ones_like(y_true) - y_true)/y_pred_p)
    #p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    #xy_loss = alpha*K.sum(K.ones_like(y_true)/y_p) - K.log(p_t)
    xy_loss = alpha * K.sum(K.ones_like(y_true) / y_p)
    #e = math.exp(1)
    #xy_loss = 1 / (tf.pow(e, -xy_loss) + 1)

    #return xy_loss
    return K.mean(xy_loss)

def new_xy_loss_v2(y_true, y_pred):

    alpha = K.constant(0.000001, dtype=tf.float32)
    alpha1 = K.constant(0.95, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    y_true_p = y_true*y_pred
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred))
    y_p = y_pred_p + y_true_p + K.epsilon()
    y_p = tf.pow(y_p, gamma)
    alpha_t = y_true * alpha1 + (K.ones_like(y_true) - y_true) * (1 - alpha1)
    xy_loss1 = K.sum((K.ones_like(y_true) / y_p)*alpha_t)
    xy_loss2 = K.sum((y_p/K.ones_like(y_true))*alpha_t)
    xy_loss = (2*xy_loss2)/xy_loss1

    #xy_loss = alpha * K.sum(K.ones_like(y_true) / y_p)
    return xy_loss
    #return K.mean(xy_loss)

def new_xy_loss_v3(y_true, y_pred):
    alpha = K.constant(0.000001, dtype=tf.float32)
    alpha1 = K.constant(0.7, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    y_true_p = y_true*y_pred
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred))
    y_true_p1 = y_true*(K.ones_like(y_true)-y_pred)
    y_pred_p1 = (K.ones_like(y_true) - y_true)*y_pred
    y_p = y_pred_p + y_true_p + K.epsilon()
    y_p1 = y_pred_p1 + y_true_p1 + K.epsilon()
    #y_p = tf.pow(y_p, gamma)
    alpha_t = y_true * alpha1 + (K.ones_like(y_true) - y_true) * (1 - alpha1)

    # xy_loss1 = K.sum((K.ones_like(y_true) / y_p)*alpha_t)
    # xy_loss2 = K.sum((y_p/K.ones_like(y_true))*alpha_t)
    # xy_loss = (2*xy_loss2)/xy_loss1
    xy_loss1 = K.sum((K.ones_like(y_true) / y_p))
    xy_loss2 = K.sum(K.ones_like(y_true) / y_p1)
    #xy_loss = alpha * K.sum(K.ones_like(y_true) / y_p)
    xy_loss = xy_loss1/xy_loss2
    #return xy_loss
    return K.mean(xy_loss)

def new_xy_loss_v4(y_true, y_pred):

    alpha = K.constant(0.45, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    y_true_p = y_true*y_pred + y_true*alpha
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p

    return K.mean(xy_loss)

def new_xy_loss_v44(y_true, y_pred):
    xy_loss1 = []
    w1 = K.constant(0.25, dtype=tf.float32)
    w2 = K.constant(0.75, dtype=tf.float32)
    for i in range(2):
        y_true1 = y_true[i]
        y_pred1 = y_pred[i]

        alpha = K.constant(0.35, dtype=tf.float32)
        y_true1 = K.cast(y_true1,dtype='float32')
        y_pred1 = K.cast(y_pred1,dtype='float32')
        gw_n = K.cast(K.sum(y_true1),dtype='float32')
        bg_n = K.cast(K.sum(K.ones_like(y_true1) - y_true1), dtype='float32')
        y_true_p = y_true1*y_pred1 + y_true1*alpha
        y_pred_p = ((K.ones_like(y_true1) - y_true1)*(K.ones_like(y_true1)-y_pred1)) + (K.ones_like(y_true1) - y_true1)*alpha
        y_p = y_pred_p + y_true_p + K.epsilon()
        xy_loss = K.ones_like(y_true1) / y_p
        # gw = K.sum(y_true1*xy_loss)/gw_n
        # bg = K.sum((K.ones_like(y_true1) - y_true1)*xy_loss)/bg_n
        # gw = gw*w1
        # bg = bg*w2
        # xy_loss = gw + bg
        xy_loss1.append(xy_loss)

    xy_loss1 = K.sum(xy_loss1)
    return K.mean(xy_loss1)

def new_xy_loss_v5(y_true, y_pred):
    alpha = K.constant(0.45, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    print(y_true.shape)
    print(y_pred.shape)
    y_true_p = y_true*y_pred + y_true*alpha
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    alpha_t = y_true * alpha1 + (K.ones_like(y_true) - y_true) * (1 - alpha1)
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = (K.ones_like(y_true) / y_p)*alpha_t

    return K.mean(xy_loss)

def new_xy_loss_v6(y_true, y_pred):
    alpha = K.constant(0.4, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    y_true_p = y_true*y_pred + y_true*alpha
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p

    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    smooth = 1
    dice = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice

    return K.mean(xy_loss) + K.mean(diceloss)

def new_xy_loss_v7(y_true, y_pred):
    alpha = K.constant(0.3, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    p_t1 = (K.ones_like(y_true) - y_pred) * gamma
    y_true_p = y_true*y_pred + y_true*alpha
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p
    p_t1 = K.sum(p_t1)
    xy_loss = K.sum(xy_loss)
    smooth = 1
    xy_loss = (xy_loss + smooth)/(p_t1 + smooth)

    return K.mean(xy_loss)

def new_xy_loss_v9(y_true, y_pred):
    alpha = K.constant(0.80, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(0.2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')

    y_true_p = y_true*y_pred + y_true*alpha
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p

    y_true_p1 = y_true*(K.ones_like(y_true)-y_pred) + y_true*alpha
    y_pred_p1 = ((K.ones_like(y_true) - y_true)*y_pred) + (K.ones_like(y_true) - y_true)*alpha
    y_p1 = y_pred_p1 + y_true_p1 + K.epsilon()
    xy_loss1 = K.ones_like(y_true) / y_p1
    xy_loss = xy_loss + xy_loss1*gamma

    return K.mean(xy_loss)

def new_xy_loss_v10(y_true, y_pred):
    alpha = K.constant(0.45, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')

    y_true_p = y_true*y_pred + y_true*alpha
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p
    alpha_t = y_true * (1 - alpha1) + (K.ones_like(y_true) - y_true) * alpha1

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    xy_loss = K.pow((K.ones_like(y_true) - p_t), gamma)*xy_loss*alpha_t
    return K.mean(xy_loss)

def new_xy_loss_v11(y_true, y_pred):
    alpha = K.constant(0.45, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')

    y_t1 = y_true * y_pred
    y_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    y_p1 = (K.ones_like(y_true) - y_true) * y_pred
    y_p2 = (K.ones_like(y_true) - y_true) * y_pred + y_true

    y_t1 = tf.math.greater_equal(y_t1, 0.5)
    y_t2 = tf.math.less(y_t2, 0.5)
    y_t1 = tf.where(y_t1, 1, 0)
    y_t2 = tf.where(y_t2, 1, 0)
    y_t1 = tf.cast(y_t1, tf.float32)
    y_t2 = tf.cast(y_t2, tf.float32)
    y_t11 = y_t1
    y_t22 = y_t2
    y_t1 =y_pred*y_t1
    y_t2 =  y_t2*y_pred
    y_t2 = (y_t22 *K.ones_like(y_true)-y_t2) + y_t22 *K.ones_like(y_true)*0.5

    y_p1 = tf.math.greater_equal(y_p1, 0.5)
    y_p2 = tf.math.less(y_p2, 0.5)
    y_p1 = tf.where(y_p1, 1, 0)
    y_p2 = tf.where(y_p2, 1, 0)
    y_p1 = tf.cast(y_p1, tf.float32)
    y_p2 = tf.cast(y_p2, tf.float32)
    y_p11 = y_p1
    y_p22 = y_p2
    y_p1 = y_pred*y_p1
    y_p2 = y_p2*y_pred
    y_p2 = y_p22 *K.ones_like(y_true)-y_p2
    y_p1 = y_p11 *K.ones_like(y_true)*0.5 + y_p1

    y_pt1 = y_p1 + y_t2
    y_pt2 = y_p2 + y_t1
    y_pt2 = (K.ones_like(y_true) - (y_p22 + y_t11)) + y_pt2 + K.epsilon()
    y_pt2 = K.ones_like(y_true) / y_pt2

    y_pt = y_pt2 +y_pt1
    y_pt_loss = K.mean(y_pt)
    #print(y_pt_loss)

    return y_pt_loss

def new_xy_loss_v13(y_true, y_pred):
    alpha = K.constant(0.45, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    n = K.cast(K.sum(y_true),dtype='float32')
    n1 = K.cast(K.sum(K.ones_like(y_true) - y_true),dtype='float32')
    y_true_p = y_true*y_pred + y_true*alpha
    #y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    y_pred_p = (K.ones_like(y_true) - y_true)
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p
    #xy_loss = xy_loss-(K.ones_like(y_true) - y_true)*0.99

    xy_loss = (K.sum(xy_loss)-n1)/n

    return K.mean(xy_loss)

def new_xy_loss_v14(y_true, y_pred):
    alpha = K.constant(0.45, dtype=tf.float32)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    n = K.cast(K.sum(y_true),dtype='float32')
    n1 = K.cast(K.sum(K.ones_like(y_true) - y_true),dtype='float32')
    y_true_p = y_true
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha
    #y_pred_p = (K.ones_like(y_true) - y_true)
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p
    #xy_loss = xy_loss-(K.ones_like(y_true) - y_true)*0.99

    xy_loss = (K.sum(xy_loss)-n)/n1

    return K.mean(xy_loss)

def new_xy_loss_v16(y_true, y_pred):
    alpha = K.constant(0.45, dtype=tf.float32)
    alpha1 = K.constant(0.75, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    n = K.cast(K.sum(y_true),dtype='float32')
    n1 = K.cast(K.sum(K.ones_like(y_true) - y_true),dtype='float32')
    y_true_p = y_true*y_pred + y_true*alpha
    y_pred_p = ((K.ones_like(y_true) - y_true)*(K.ones_like(y_true)-y_pred)) + (K.ones_like(y_true) - y_true)*alpha1
    #y_pred_p = (K.ones_like(y_true) - y_true)*alpha1
    y_p = y_pred_p + y_true_p + K.epsilon()
    xy_loss = K.ones_like(y_true) / y_p
    #xy_loss = xy_loss-(K.ones_like(y_true) - y_true)*0.99

    #xy_loss = (K.sum(xy_loss)-n1)/n

    return K.mean(xy_loss)

def tversky_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1-y_pred))
    false_pos = K.sum((1-y_true)*y_pred)
    alpha = 0.7
    smooth = 1
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    tversky = 1 - tversky
    return K.mean(tversky)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    threshold = 1
    #y_pred = K.cast(K.greater(y_pred,threshold), dtype='float32')
    #y_true = tf.to_uint8(y_true)
    #y_pred = K.to_uint8(y_pred)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_true_f = y_true
    # y_pred_f = y_pred
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def jaccard(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    p = K.cast(1.5, dtype='float32')
    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(K.pow(y_true, p) + K.pow(y_pred, p))
    smooth = 1
    dice  = (intersection + smooth)/(sumtotal - intersection + smooth)
    diceloss = 1 - dice

def jaccard_v1(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    p = K.cast(1.5, dtype='float32')
    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(K.pow(y_true, p) + K.pow(y_pred, p))
    smooth = 1
    dice  = (intersection + smooth)/(sumtotal - intersection + smooth)
    diceloss = 1 - dice

    gamma = K.cast(5, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)
    d_dice = gamma*((b_1 + b_2 + smooth)/(K.sum(K.ones_like(y_true) - y_pred) + smooth))
    diceloss = diceloss + d_dice

    return K.mean(diceloss)

def dice_loss(y_true, y_pred):
    threshold = 0.25
    #y_pred = tensorflow.keras.cast(tensorflow.keras.greater(y_pred,threshold), dtype='float32')
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    smooth = 1
    dice = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice
    return K.mean(diceloss)

def dice_loss2(y_true, y_pred):
    threshold = 0.25
    #y_pred = tensorflow.keras.cast(tensorflow.keras.greater(y_pred,threshold), dtype='float32')
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    p_t1 = (K.ones_like(y_true) - y_pred) * y_true
    p_t2 = (K.ones_like(y_true) - y_true) * y_pred
    p_t3 = p_t1 + p_t2
    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + p_t3)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice
    return K.mean(diceloss)

def difficult_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = 2

    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*y_pred


    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    smooth = 1
    dice = (2*intersection + b_1*gamma + smooth)/(sumtotal + b_2*gamma +smooth)
    diceloss = 1 - dice
    return K.mean(diceloss)

def difficult_dice_loss_v1(y_true, y_pred):

    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = 2

    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*y_pred


    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    smooth = 1
    dice = (2*intersection + b_2*gamma + smooth)/(sumtotal + b_1*gamma +smooth)
    diceloss = 1 - dice
    return K.mean(diceloss)

def difficult_dice_loss_v2(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = K.cast(50, dtype='float32')

    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)


    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    smooth = 1
    dice = (2*intersection + smooth)/(sumtotal + b_1*gamma + b_2*gamma + smooth)
    diceloss = 1 - dice
    return K.mean(diceloss)

def difficult_dice_loss_v3(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = K.cast(0.2, dtype='float32')
    gamma1 = K.cast(25, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)
    d_dice = gamma1*((b_1 + b_2 + smooth)/(K.sum(K.ones_like(y_true) - y_pred) + smooth))


    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    dice = (2*intersection + smooth)/(sumtotal + smooth)
    diceloss = 1 - dice
    diceloss = gamma*diceloss + (1-gamma)*d_dice
    return K.mean(diceloss)

def difficult_dice_loss_v4(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    alpha = K.constant(0.25, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)

    gamma1 = K.cast(10, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)
    d_dice = gamma1*((b_1 + b_2 + smooth)/(K.sum(K.ones_like(y_true) - y_pred) + smooth))


    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    dice = (2*intersection + smooth)/(sumtotal + smooth)
    diceloss = 1 - dice
    diceloss = diceloss + d_dice


    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

    return K.mean(diceloss) + K.mean(focal_loss)

def difficult_dice_loss_v5(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = K.cast(10, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)

    c_t1 = (K.ones_like(y_true) - y_true) * y_pred
    c_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    c_1 = tf.math.greater_equal(c_t1, 0.5)
    c_11 = tf.math.less_equal(c_t2, 0.6)
    c_2 = tf.math.greater_equal(c_t1, 0.4)
    c_22 = tf.math.less(c_t2, 0.5)
    c_1 = tf.where(c_1, 1, 0)
    c_11 = tf.where(c_11, 1, 0)
    c_2 = tf.where(c_2, 1, 0)
    c_22 = tf.where(c_22, 1, 0)
    c_1 = tf.cast(c_1, tf.float32)
    c_2 = tf.cast(c_2, tf.float32)
    c_11 = tf.cast(c_11, tf.float32)
    c_22 = tf.cast(c_22, tf.float32)
    c_1 = c_1*c_11
    c_2 = c_2*c_22
    c_1 = c_1*y_pred
    c_2 = c_2*(K.ones_like(y_true) - y_pred)
    c_1 = K.sum(c_1)
    c_2 = K.sum(c_2)

    d_dice = gamma*((b_1 + b_2 + c_1 + c_2 + smooth)/(K.sum(K.ones_like(y_true) - y_pred) + smooth))

    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    dice = (2*intersection + smooth)/(sumtotal + smooth)
    diceloss = 1 - dice
    diceloss = diceloss + d_dice
    return K.mean(diceloss)

def difficult_dice_loss_v6(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = K.cast(3.5, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)
    d_dice = (gamma*(b_1 + b_2) + smooth)/(K.sum(K.ones_like(y_true) - y_pred) + smooth)

    diceloss = d_dice
    return K.mean(diceloss)

def difficult_dice_loss_v7(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = K.cast(0.25, dtype='float32')
    gamma1 = K.cast(20, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)
    d_dice = (gamma1*(gamma*b_1 + b_2*(1-gamma) + smooth))/(K.sum(K.ones_like(y_true) - y_pred) + smooth)

    diceloss = d_dice
    return K.mean(diceloss)

def difficult_dice_loss_v8(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')

    alpha = tf.constant(0.25, dtype=tf.float32)
    gamma = tf.constant(2, dtype=tf.float32)

    #gamma = K.cast(100, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)
    d_dice = (10.0*(b_1 + b_2) + smooth)/(K.sum(K.ones_like(y_true) - y_pred) + smooth)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

    diceloss = K.mean(d_dice) + K.mean(focal_loss)
    return diceloss

def new_xy_loss_v12(y_true, y_pred):
    alpha = K.constant(0.25, dtype=tf.float32)
    alpha1 = K.constant(0.75, dtype=tf.float32)
    #alpha2 = K.constant(0.55, dtype=tf.float32)
    gamma = K.constant(0.7, dtype=tf.float32)
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')

    y_t1 = y_true * y_pred
    y_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    y_p1 = (K.ones_like(y_true) - y_true) * y_pred
    y_p2 = (K.ones_like(y_true) - y_true) * y_pred + y_true

    y_t1 = tf.math.greater_equal(y_t1, alpha1)
    y_t2 = tf.math.less(y_t2, alpha)
    y_t1 = tf.where(y_t1, 1, 0)
    y_t2 = tf.where(y_t2, 1, 0)
    y_t1 = tf.cast(y_t1, tf.float32)
    y_t2 = tf.cast(y_t2, tf.float32)
    y_t11 = y_t1
    y_t22 = y_t2
    y_t1 =y_pred*y_t1
    y_t2 =  y_t2*y_pred
    y_t2 = (y_t22 *K.ones_like(y_true)-y_t2) + y_t22 *K.ones_like(y_true)*gamma

    y_p1 = tf.math.greater_equal(y_p1, alpha)
    y_p2 = tf.math.less(y_p2, alpha1)
    y_p1 = tf.where(y_p1, 1, 0)
    y_p2 = tf.where(y_p2, 1, 0)
    y_p1 = tf.cast(y_p1, tf.float32)
    y_p2 = tf.cast(y_p2, tf.float32)
    y_p11 = y_p1
    y_p22 = y_p2
    y_p1 = y_pred*y_p1
    y_p2 = y_p2*y_pred
    y_p2 = y_p22 *K.ones_like(y_true)-y_p2
    y_p1 = y_p11 *K.ones_like(y_true)*gamma + y_p1

    y_pt1 = y_p1 + y_t2
    y_pt2 = y_p2 + y_t1
    y_pt2 = (K.ones_like(y_true) - (y_p22 + y_t11)) + y_pt2 + K.epsilon()
    y_pt2 = K.ones_like(y_true) / y_pt2

    y_pt = y_pt2 +y_pt1
    y_pt_loss = K.mean(y_pt)
    #print(y_pt_loss)

    return y_pt_loss

def select_difficult_dice_loss_v4(y_true, y_pred):
    global loss1
    if loss1==None:
        loss1=1.0
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    gamma = K.cast(20, dtype='float32')
    smooth = 1
    p_t1 = (K.ones_like(y_true) - y_true) * y_pred
    # p_t2 = (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    p_t2 = y_true * y_pred + (K.ones_like(y_true) - y_true)
    b_1 = tf.math.greater_equal(p_t1, 0.5)
    # b_2 = tf.math.greater_equal(p_t2, 0.7)
    b_2 = tf.math.less(p_t2, 0.5)
    b_1 = tf.where(b_1, 1, 0)
    b_2 = tf.where(b_2, 1, 0)
    b_1 = tf.cast(b_1, tf.float32)
    b_2 = tf.cast(b_2, tf.float32)
    b_1 = b_1*y_pred
    b_2 = b_2*(K.ones_like(y_true) - y_pred)
    b_1 = K.sum(b_1)
    b_2 = K.sum(b_2)
    d_dice = gamma*((b_1 + b_2 + smooth)/(K.sum(K.ones_like(y_true) - y_pred) + smooth))


    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    dice = (2*intersection + smooth)/(sumtotal + smooth)
    diceloss = 1 - dice

    #loss1 = K.cast(loss1,dtype='float32')

    diceloss = tf.cond(loss1>0.3, lambda:  K.cast(diceloss,dtype='float32'), lambda:  K.cast(d_dice,dtype='float32'))
    # if loss1>0.3:
    #     diceloss = diceloss
    # else:
    #     diceloss = d_dice
    loss1 = diceloss
    return K.mean(diceloss)

def dice_loss1(y_true, y_pred):
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    smooth = 1
    y_pred1 = tf.sqrt(y_pred + K.epsilon())
    intersection1 = K.sum(y_true*y_pred1)
    sumtotal1 = K.sum(y_true + y_pred1)
    dice1 = (2*intersection1 + smooth)/(sumtotal1+smooth)
    diceloss1 = 1 - dice1

    return K.mean(diceloss1)

def two_dice_loss(y_true, y_pred):
    threshold = 0.25
    #y_pred = tensorflow.keras.cast(tensorflow.keras.greater(y_pred,threshold), dtype='float32')
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice

    y_pred1 = tf.sqrt(y_pred + K.epsilon())
    intersection1 = K.sum(y_true*y_pred1)
    sumtotal1 = K.sum(y_true + y_pred1)
    dice1  = (2*intersection1 + smooth)/(sumtotal1+smooth)
    diceloss1 = 1 - dice1

    return K.mean(diceloss) +K.mean(diceloss1)

def two_dice_loss_v2(y_true, y_pred):
    threshold = 0.25
    #y_pred = tensorflow.keras.cast(tensorflow.keras.greater(y_pred,threshold), dtype='float32')
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    sumtotal = K.sum(y_true + y_pred)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice

    y_pred1 = tf.sqrt(y_pred + K.epsilon())
    y_pred1 = tf.sqrt(y_pred1 + K.epsilon())
    y_pred1 = tf.sqrt(y_pred1 + K.epsilon())
    intersection1 = K.sum(y_true*y_pred1)
    sumtotal1 = K.sum(y_true + y_pred1)
    dice1 = (2*intersection1 + smooth)/(sumtotal1+smooth)
    diceloss1 = 1 - dice1

    return K.mean(diceloss) +K.mean(diceloss1)

def two_dice_loss_v3(y_true, y_pred):
    threshold = 0.25
    gamma = 2
    #y_pred = tensorflow.keras.cast(tensorflow.keras.greater(y_pred,threshold), dtype='float32')
    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    y_pred0 = K.pow((y_pred + K.epsilon()), gamma)
    intersection = K.sum(y_true*y_pred0)
    sumtotal = K.sum(y_true + y_pred0)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice

    y_pred1 = tf.sqrt(y_pred + K.epsilon())
    y_pred1 = tf.sqrt(y_pred1 + K.epsilon())
    intersection1 = K.sum(y_true*y_pred1)
    sumtotal1 = K.sum(y_true + y_pred1)
    dice1 = (2*intersection1 + smooth)/(sumtotal1+smooth)
    diceloss1 = 1 - dice1

    return K.mean(diceloss) +K.mean(diceloss1)

def binary_focal_loss_fixed(y_true, y_pred):
    """
    y_true shape need be (None,1)
    y_pred need be compute after sigmoid
    """
    alpha = K.constant(0.75, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

    #focal_loss = - alpha * y_true * K.pow((K.ones_like(y_pred) - y_pred),gamma)* K.log(y_pred) - (1-alpha) * (K.ones_like(y_true) - y_true) * K.pow(y_pred,gamma)*K.log(K.ones_like(y_pred) - y_pred)
    return K.mean(focal_loss)

def CB_focal_loss(y_true, y_pred):
    alpha = K.constant(0.999, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    n = K.sum(y_true)
    #alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
    alpha_t = (1-alpha)/(1-K.pow(alpha, n))

    #p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    p_t = y_true * y_pred + K.epsilon()
    #focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    focal_loss = -y_true * K.pow((K.ones_like(y_pred) - y_pred+ K.epsilon()), gamma) * K.log(y_pred)
    focal_loss = K.sum(focal_loss)/n

    #return K.mean(100*focal_loss)
    return focal_loss

def CE_loss(y_true, y_pred):
    alpha = K.constant(0.05, dtype=tf.float32)
    gamma = K.constant(2, dtype=tf.float32)

    y_true = K.cast(y_true,dtype='float32')
    y_pred = K.cast(y_pred,dtype='float32')
    n = K.cast(K.sum(y_true),dtype='float32')

    p_t = y_true * y_pred +(K.ones_like(y_true) - y_true) + K.epsilon()
    #focal_loss = -alpha*K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    focal_loss = -K.log(p_t)
    focal_loss = K.sum(focal_loss)/n

    #focal_loss = - alpha * y_true * K.pow((K.ones_like(y_pred) - y_pred),gamma)* K.log(y_pred) - (1-alpha) * (K.ones_like(y_true) - y_true) * K.pow(y_pred,gamma)*K.log(K.ones_like(y_pred) - y_pred)
    return focal_loss

def focal_dice_loss(y_true, y_pred):
    return binary_focal_loss_fixed(y_true, y_pred) - K.log(dice_loss(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return 0.5 * tensorflow.tensorflow.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true,y_pred)
    return loss

def augmented_focal_loss(y_true, y_pred):
    """
    y_true shape need be (None,1)
    y_pred need be compute after sigmoid
    """
    alpha = K.constant(50, dtype=tf.float32)
    gamma = K.constant(1.5, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * 1.0

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

    #focal_loss = - alpha * y_true * K.pow((K.ones_like(y_pred) - y_pred),gamma)* K.log(y_pred) - (1-alpha) * (K.ones_like(y_true) - y_true) * K.pow(y_pred,gamma)*K.log(K.ones_like(y_pred) - y_pred)
    return K.mean(focal_loss)

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        #K.log是以e为底
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

        #focal_loss = - alpha * y_true * K.pow((K.ones_like(y_pred) - y_pred),gamma)* K.log(y_pred) - (1-alpha) * (K.ones_like(y_true) - y_true) * K.pow(y_pred,gamma)*K.log(K.ones_like(y_pred) - y_pred)
        # K.mean把矩阵里的值全部加起来然后求均值
        return K.mean(focal_loss)
        # return focal_loss


    return binary_focal_loss_fixed

def focal_with_dice_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    #创建一个常数张量
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        #类型转换
        y_true = tf.cast(y_true, tf.float32)
        #print(tf.size(y_pred))
        #print(tf.size(y_true))
        #损失函数
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)


        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        #y_true = K.cast(y_true,dtype='float32')
        y_pred = K.cast(y_pred,dtype='float32')
        # y_true = K.flatten(y_true)
        # y_pred = K.flatten(y_pred)
        #k.sum是用来取绝对值的
        intersection = K.sum(y_true*y_pred)
        sumtotal = K.sum(y_true + y_pred)
        smooth = 1
        dice  = (2*intersection + smooth)/(sumtotal+smooth)
        diceloss = 1 - dice



        #focal_loss = - alpha * y_true * K.pow((K.ones_like(y_pred) - y_pred),gamma)* K.log(y_pred) - (1-alpha) * (K.ones_like(y_true) - y_true) * K.pow(y_pred,gamma)*K.log(K.ones_like(y_pred) - y_pred)
        return K.mean(focal_loss)+K.mean(diceloss)
        #return K.mean(focal_loss)-K.log(K.mean(diceloss))
        #return K.mean(focal_loss)

    return binary_focal_loss_fixed



    #把一个函数作为参数传给另一个函数，第一个函数称为回调函数：tensorflow.keras.callbacks.Callback

class LossHistory(tensorflow.keras.callbacks.Callback):
    def __init__(self):

        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        log_dir = "logs/"
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        self.accuracy   = []
        self.val_acc    = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('main_loss'))
        self.val_loss.append(logs.get('val_main_loss'))
        self.accuracy.append(logs.get('main__f_score'))
        self.val_acc.append(logs.get('val_main__f_score'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main_loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main_loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_accuracy_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main__f_score')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_acc_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main__f_score')))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        plt.plot(iters,self.accuracy, 'black', linewidth = 2, label='accuracy')
        plt.plot(iters,self.val_acc, 'blue', linewidth = 2, label='val_acc')
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss-accuracy')
        plt.title('A Loss and accuracy Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

if __name__ == "__main__":
    log_dir = "logs/"
    log_dir1 = "logs/"
    #tensorflow.tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    # 获取model
    model = MLBSNet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=False,aux_branch=True)
    #model = pspnet(n_classes=NCLASSES, input_height=HEIGHT,input_width=WIDTH, backbone="resnet50", aux_branch=True)
    #model = FCN_Vgg16_32s(input_shape = (HEIGHT,WIDTH,1), weight_decay=1e-4, classes=NCLASSES)
    #model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    #model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,1),deploy=False)
    #model = UNetPlusPlus(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES,deep_supervision=True)
    #model = U_Net(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES)
    #model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,1))

    #Output the parameter status of each layer of the model
    model.summary()
    # WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    # model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    # weights_path = 'logs/ep035-loss0.126-val_loss0.231.h5'
    # model.load_weights(weights_path, by_name=True, skip_mismatch=False)

    #Open dataset file and read paths of all images into line_list and shuffle
    with open(r"./gw_904/train.txt", "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None) #Reset the randomization algorithm to always arrange data in same manner every run

    #divide data into training and validation (90:10)
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    history = LossHistory()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    checkpoint_period = ModelCheckpoint(log_dir1 + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1
    )
    model.compile(#loss = 'binary_crossentropy',
        #loss = bce_dice_loss,
        #loss = tversky_loss,
        #loss = jaccard,
        #loss = jaccard_v1,
        #loss = dice_loss,
        #loss = dice_loss2,
        #loss = difficult_dice_loss,
        #loss = difficult_dice_loss_v1,
        #loss=difficult_dice_loss_v2,
        #loss=difficult_dice_loss_v3,
        #loss=difficult_dice_loss_v4,
        #loss = difficult_dice_loss_v5,
        #loss=difficult_dice_loss_v6,
        #loss=difficult_dice_loss_v7,
        #loss=difficult_dice_loss_v8,
        #loss = select_difficult_dice_loss_v4,
        #loss = dice_loss1,
        #loss =two_dice_loss,
        #loss = two_dice_loss_v1,
        #loss=two_dice_loss_v2,
        #loss=two_dice_loss_v3,
        #loss = dice_loss_v1,
        #loss=new_dice_loss,
        #loss = new_dice_loss_v1,
        #loss = new_dice_loss_v1_1,
        #loss=new_dice_loss_v2,
        #loss=new_dice_loss_v3,
        #loss = new_dice_loss_v4,
        #loss = new_dice_loss_v5,
        #loss = new_dice_loss_v6,
        #loss = new_dice_loss_v7,
        #loss=new_dice_loss_v8,
        #loss=new_dice_loss_v9,
        #loss = binary_focal_loss_fixed_v1,
        #loss = binary_focal_loss_fixed_v2,
        #loss=new_binary_focal_loss,
        #loss = focal_with_dice_loss(),
        #loss = new_focal_with_dice_loss_v1,
        #loss = new_focal_with_dice_loss_v_1,
        #loss = new_focal_with_dice_loss_v2,
        #loss =  new_focal_with_dice_loss_v3,
        #loss= test_new_focal_with_dice_loss(),
        #loss=new_dice_loss,
        #loss = binary_focal_loss(),
        #loss= new_focal_with_dice_loss(),
        #loss= binary_focal_loss_fixed,
        #loss = new_xy_loss_v1,
        #loss = new_xy_loss_v2,
        #loss = new_xy_loss_v3,
        #loss = new_xy_loss_v4,
        #loss = new_dice_and_focal_loss_v4,
        #loss = new_dice_and_focal_and_xy_loss_v4,
        #loss = new_dice_and_focal_loss_v5,
        #loss = new_xy_loss_v5,
        #loss = new_xy_loss_v6,
        #loss=new_xy_loss_v7,
        #loss=new_xy_loss_v8,
        #loss = new_xy_loss_v9,
        #loss = new_xy_loss_v10,
        #loss=new_xy_loss_v11,
        #loss = new_xy_loss_v12,
        #loss = new_xy_loss_v13,
        #loss =new_xy_loss_v14,
        #loss=new_xy_loss_v16,
        #loss = my_huber_loss,
        #loss = CE_loss,
        #loss = new_sqrt,
        #loss = augmented_focal_loss,
        #loss = CB_focal_loss,
        loss=new_xy_loss_v44,
        optimizer = Adam(lr=0.0001),
        metrics = [f_score()])
        #metrics = ['accuracy'])
        #metrics = ["binary_crossentropy", mean_iou, dice_coef])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # Start training, aug for data increase
    # steps_per_epoch: Total number of steps (batch samples) from generator before declaring an epoch to complete and starting the next epoch.
    # It should normally be equal to the sample size of your dataset divided by the batch size.
    #validation_data: Similar to our generator, only this one is used for verification and does not participate in training.
    # callbacks: A series of callback functions called at training time.
    # initial_epoch: The number of rounds to start training (helps to resume previous training).
    # validation_data: Assess the loss and any model metrics at the end of each epoch. The model is not trained on this data.
    # validation_steps: Available only if validation_data is a generator.
    # The total number of steps generated by the generator before stopping (number of sample batches).

    model.fit(generate_arrays_from_file(lines[:num_train], batch_size,aug),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val//batch_size),
                        epochs=50, initial_epoch=0,  callbacks=[checkpoint_period, reduce_lr, history])

    model.save_weights(log_dir + 'last1-MLBS-Net-Pig-NoTransfer.h5')

    history.loss_plot()