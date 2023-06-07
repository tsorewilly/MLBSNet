#from nets.Repunet import convnet_unet
#from nets.fcn import FCN_Vgg16_32s
from nets.unetplusplus import UNetPlusPlus
from nets.deeplab import Deeplabv3
from nets.Repconvnet import Rep_get_convnet_encoder
from nets.pspnet import pspnet  #Change the code and data to return, change the accuracy evaluation, change pr[0]
from nets.repvgg import repvgg_model_convert
from nets.helper_functions import U_Net#, UNetPlusPlus
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
from lossFcns import *
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

def generate_arrays_from_file(lines, batch_size, aug = None):    
    n, i = len(lines), 0    
    while 1:
        X_train, Y_train = [], []
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)                
            name = lines[i].split(';')[0]
            
            # read images in the training set from jpg folder
            img = Image.open(r"./gw_TMI/jpg" + '/' + name)
            img = img.resize((WIDTH,HEIGHT))
            tmp_im_array = np.array(img)[:,:,0]
            tmp_im_array = tmp_im_array/255
            tmp_im_array = tmp_im_array[np.newaxis,:,:]

            #Split file entries, trim components to read masks from png folder
            name = (lines[i].split(';')[1]).replace("\n", "") 
            img = Image.open(r"./gw_TMI/png" + '/' + name)
            img = img.resize((int(WIDTH1),int(HEIGHT1)))
            tmp_lb_array = np.array(img)[:,:,0]
            tmp_lb_array = tmp_lb_array[np.newaxis,:,:] #Class #1

            if len(X_train) == 0:
                X_train = tmp_im_array
                Y_train = tmp_lb_array
            else:
                X_train = np.concatenate((X_train,tmp_im_array),axis=0)
                Y_train = np.concatenate((Y_train,tmp_lb_array),axis=0)

            i = (i+1) % n        

        #Store images and masks. PS: one more dimension is added
        X_train = X_train[:,:,:,np.newaxis]
        Y_train = Y_train[:,:,:,np.newaxis] 

        if (aug is not None):# and (random.random() > 0.5):
            aug.fit(X_train)
            for X_batch, y_batch in aug.flow(X_train, Y_train, batch_size=batch_size):
                for i in range(0, batch_size):
                    #printaxsi, X_batch[0].shape)                    
                    fig = plt.figure()                    
                    fig.add_subplot(2,2,i+1).imshow(X_batch[i].reshape(256, 256, 1))
                    plt.show()
                break
            #X_train = next(aug.flow(X_train,batch_size = batch_size,shuffle=False,seed = i))
            #Y_train = next(aug.flow(Y_train,batch_size = batch_size,shuffle=False,seed = i))
        #print(X_train.shape, Y_train.shape)
        
        # Return data in batchsize, reshuffle after each return round
        yield(X_train, Y_train)
        #return X_train, Y_train

aug = ImageDataGenerator( #define a data augmentation generator
     rotation_range = 45,       #define the rotation range
     #zoom_range = 0.2,          #Randomly scale the image size proportionally
     width_shift_range = 0.2,   #image horizontal shift range
     height_shift_range = 0.2,  #vertical shift range of the image
     brightness_range=(0,0.9),
     #shear_range = 0.05,       #horizontal or vertical projection transformation
     horizontal_flip = True,    #Flip the image horizontally
     fill_mode = "reflect",      #fill pixels, appear after rotation or translation
     validation_split=0.2
)

"""
datagen = ImageDataGenerator(rotation_range=90)
datagen.fit(x_train)


for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(img_rows, img_cols, 3))
        pyplot.show()
    break


data_augmentation = tf.keras.Sequential([
     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
X_train = generate_arrays_from_file(lines[:num_train], 10)
for i in range(9):
  augmented_image = next(aug.flow(X_train[0], 10))
  #data_augmentation(X_train)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
  """


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
        self.binary_crossentropy = []
        self.val_binary_crossentropy = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('main_loss'))
        self.val_loss.append(logs.get('val_main_loss'))
        self.accuracy.append(logs.get('main__f_score'))
        self.val_acc.append(logs.get('val_main__f_score'))
        self.binary_crossentropy.append('main_binary_crossentropy')
        self.binary_crossentropy.append('val_binary_crossentropy')
        self.binary_crossentropy.append('main_dice_coef')
        self.binary_crossentropy.append('val_main_dice_coef')        
        self.binary_crossentropy.append('main_mean_io_u_1')
        self.binary_crossentropy.append('val_main_mean_io_u_1')
        
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
        with open(os.path.join(self.save_path, "epoch_binary_crossentropy_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main_binary_crossentropy')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_binary_crossentropy_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main_binary_crossentropy')))
            f.write("\n")            
        with open(os.path.join(self.save_path, "epoch_main_dice_coef_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main_dice_coef')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_main_dice_coef_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main_dice_coef')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_main_mean_io_u_1_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('main_mean_io_u_1')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_main_mean_io_u_1_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_main_mean_io_u_1')))
            f.write("\n")            

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        plt.plot(iters,self.accuracy, 'black', linewidth = 2, label='accuracy')
        plt.plot(iters,self.val_acc, 'blue', linewidth = 2, label='val_acc')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss-accuracy')
        plt.title('A Loss and accuracy Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

def get_d_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops  # Prints the "flops" of the model.


def get_d2_flops(model):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            #model = tf.keras.models.load_model(model)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops

if __name__ == "__main__":
    log_dir = "logs/"    
    #tensorflow.tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    #model = MLBSNet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=False,aux_branch=True)
    #model = pspnet(n_classes=NCLASSES, input_height=HEIGHT,input_width=WIDTH, backbone="resnet50", aux_branch=True)
    #model = FCN_Vgg16_32s(input_shape = (HEIGHT,WIDTH,1), weight_decay=1e-4, classes=NCLASSES)
    #model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    #model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,1),deploy=False)
    #model = UNetPlusPlus(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES,deep_supervision=True)
    model = U_Net(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES)
    #model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,1))

    #Output the parameter status of each layer of the model
    model.summary()    

    #Open dataset file and read paths of all images into line_list and shuffle
    with open(r"./gw_TMI/train.txt", "r") as f:
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
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1
    )
    model.compile(
        loss = dice_loss, #new_xy_loss_v44, 347,521,281
        optimizer = Adam(lr=0.0001),
        #metrics = [f_score()])
        #metrics = ['accuracy'])
        metrics = [f_score(), "binary_crossentropy", tf.keras.metrics.MeanIoU(num_classes=2), dice_coef, dice_loss])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    from keras_flops import get_flops
    #flops = get_flops(model, batch_size=batch_size)	# Calculae FLOPS
    flops = get_d2_flops(model)

    print(f"FLOPS: {flops} G")# / 10 ** 9:.03} G")

    # Start training, aug for data increase
    # steps_per_epoch: Total number of steps (batch samples) from generator before declaring an epoch to complete and starting the next epoch.
    # It should normally be equal to the sample size of your dataset divided by the batch size.
    # validation_data: Similar to our generator, only this one is used for verification and does not participate in training.
    # callbacks: A series of callback functions called at training time.
    # initial_epoch: The number of rounds to start training (helps to resume previous training).
    # validation_data: Assess the loss and any model metrics at the end of each epoch. The model is not trained on this data.
    # validation_steps: Available only if validation_data is a generator.
    # The total number of steps generated by the generator before stopping (number of sample batches).

    model.fit(generate_arrays_from_file(lines[:num_train], batch_size,aug),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val//batch_size),
                        epochs=5, initial_epoch=0,  callbacks=[checkpoint_period, reduce_lr, history])

    model.save_weights(log_dir + 'last1-CPU-UNet.h5')# 'last1-MLBS-Rab-Base-Ep_50-CmpLoss-90%.h5')
#    model.save(log_dir + 'MLBS-Rabbit-Base-Model.h5') 
    #Already saved per generation with checkpoint_period

    history.loss_plot()

# In[] Analysis and plotting     
    iters = range(len(history.losses))

    plt.figure()
    plt.plot(iters, history.losses, 'red', linewidth = 2, label='train loss')
    plt.plot(iters, history.val_loss, 'coral', linewidth = 2, label='val loss')    
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Plot')
    plt.legend(loc="upper right")
    
    plt.figure()
    plt.plot(iters,history.accuracy, 'black', linewidth = 2, label='accuracy')
    plt.plot(iters,history.val_acc, 'blue', linewidth = 2, label='val_acc')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Accuracy Curve')
    plt.legend(loc="lower right")
    
    """
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()
    """