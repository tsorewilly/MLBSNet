import os
from random import shuffle
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K




# 公式：L(pt) = -αt(1-pt)^γ log(pt)，
# pt=p and αt=α  when y=1 ,pt=1-p and αt=1-α when y=-1或者0 视情况而定
def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    print('*' * 20, 'alpha={}, gamma={}'.format(alpha, gamma))

    def focal_loss_calc(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1 - y_probs, tf.ones_like(y_probs))

        loss = -alpha * tf.pow(1 - positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
               (1 - alpha) * tf.pow(1 - negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))

        return tf.reduce_sum(loss)

    return focal_loss_calc

def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(n_classes, logits, true_label):
        epsilon = 1.e-8
        # 得到y_true和y_pred
        y_true = tf.one_hot(true_label, n_classes)
        probs = tf.nn.sigmoid(logits)
        y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
        # 得到调节因子weight和alpha
        ## 先得到y_true和1-y_true的概率【这里是正负样本的概率都要计算哦！】
        p_t = y_true * y_pred \
              + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        ## 然后通过p_t和gamma得到weight
        weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
        ## 再得到alpha，y_true的是alpha，那么1-y_true的是1-alpha
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        # 最后就是论文中的公式，相当于：- alpha * (1-p_t)^gamma * log(p_t)
        focal_loss = - alpha_t * weight * tf.log(p_t)
        return tf.reduce_mean(focal_loss)

def new_focal_with_dice_loss(gamma=2, alpha=0.25):
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
    # 创建一个常数张量
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        # 类型转换
        y_true = tf.cast(y_true, tf.float32)
        # 损失函数
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        x = K.ones_like(y_true) - p_t
        # tanh = (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
        dim = [1, 512, 512, 2]
        e = math.exp(1)
        e = tf.fill(dim, e)
        e0 = tf.pow(e, x)
        e1 = tf.pow(e, -x)
        tanh = (e0 - e1) / (e0 + e1)
        #tanh = K.pow(tanh, 2)
        focal_loss = - alpha_t * tanh * K.log(p_t)

        # y_true = K.cast(y_true,dtype='float32')
        y_pred = K.cast(y_pred, dtype='float32')
        # y_true = K.flatten(y_true)
        # y_pred = K.flatten(y_pred)
        # k.sum是用来取绝对值的
        intersection = K.sum(y_true * y_pred)
        sumtotal = K.sum(y_true + y_pred)
        smooth = 1
        dice = (2 * intersection + smooth) / (sumtotal + smooth)
        diceloss = 1 - dice

        # focal_loss = - alpha * y_true * K.pow((K.ones_like(y_pred) - y_pred),gamma)* K.log(y_pred) - (1-alpha) * (K.ones_like(y_true) - y_true) * K.pow(y_pred,gamma)*K.log(K.ones_like(y_pred) - y_pred)
        return K.mean(focal_loss) + K.mean(diceloss)
        # return K.mean(focal_loss)-K.log(K.mean(diceloss))

    return binary_focal_loss_fixed


def new_loss(y_true, y_pred):
    """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
    alpha = K.constant(2, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
    # K.epsilon()以数值形式返回一个（一般来说很小的）数，用以防止除0错误
    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()

    x = K.ones_like(y_true) - p_t
    # tanh = (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    dim = [1, 512, 512, 2]
    e = math.exp(1)
    e = tf.fill(dim, e)
    e0 = tf.pow(e, x)
    e1 = tf.pow(e, -x)
    tanh = (e0 - e1) / (e0 + e1)
    # tanh = K.pow(tanh, 2)

    loss = - alpha_t * tanh * K.log(p_t)

    # focal_loss = - alpha * y_true * K.pow((K.ones_like(y_pred) - y_pred),gamma)* K.log(y_pred) - (1-alpha) * (K.ones_like(y_true) - y_true) * K.pow(y_pred,gamma)*K.log(K.ones_like(y_pred) - y_pred)
    return K.mean(loss)


def focal_loss(y_true, y_pred):
    gamma = 0.75
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)
    focalloss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focalloss

def dice_loss_with_CE(beta=1, smooth = 1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE

def CE():
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))

    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label


class Generator(object):
    def __init__(self,batch_size,train_lines,image_size,num_classes,dataset_path):
        self.batch_size     = batch_size
        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.image_size     = image_size
        self.num_classes    = num_classes
        self.dataset_path   = dataset_path

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")
        
        # flip image or not
        flip = rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data,label
        
    def generate(self, random_data = True):
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            # 从文件中读取图像
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
            png = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".png"))

            if random_data:
                jpg, png = self.get_random_data(jpg,png,(int(self.image_size[1]),int(self.image_size[0])))
            else:
                jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]),int(self.image_size[0])))
            
            inputs.append(np.array(jpg)/255)
            
            png = np.array(png)
            png[png >= self.num_classes] = self.num_classes
            #-------------------------------------------------------#
            #   转化成one_hot的形式
            #   在这里需要+1是因为voc数据集有些标签具有白边部分
            #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
            #-------------------------------------------------------#
            seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes+1))
            
            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets

class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))  
        self.losses     = []
        self.val_loss   = []
        self.accuracy   = []
        self.val_acc    = []
        
        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('_f_score'))
        self.val_acc.append(logs.get('val__f_score'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_accuracy_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('_f_score')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_acc_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val__f_score')))
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
