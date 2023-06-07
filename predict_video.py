#from nets.unet import convnet_unet
# from nets.unetplusplus import UNetPlusPlus
# from nets.Repunet import convnet_unet
# from nets.pspnet import pspnet
# from nets.fcn import FCN_Vgg16_32s
#from nets.helper_functions import UNetPlusPlus,U_Net
#from nets.deeplab import Deeplabv3
from nets.unet import resnet_decoder
from PIL import Image
import numpy as np
import random
import copy
import os
import time
import tensorflow.keras.backend as K
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.font_manager as font_manager
import tensorflow as tf
import skimage.io as io
from nets.MLBSNet import MLBSNet
import cv2

import warnings
warnings.filterwarnings("ignore")


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

def getValue(flies):
    flieslist = []
    for i in range(len(flies)):
        flie = flies[i]
        str = ""
        for j in range(len(flie)):
            #print(flie[j])
            if flie[j].isdigit():
                str = str+ flie[j]
        flieslist.append(str)
    print(flieslist)
    return flieslist
def fileSort(flies):
    flies1 = flies
    flieslist = getValue(flies)
    for i in range(1,len(flies)):
        for j in range(0,len(flies)-i):
            if float(flieslist[j])>float(flieslist[j+1]):
                flies1[j],flies1[j+1] = flies1[j+1],flies1[j]
                flieslist[j],flieslist[j+1] = flieslist[j+1],flieslist[j]
    return flies1

class_colors = [[0,0,0],[0,255,0]]
NCLASSES = 1
HEIGHT = 256
WIDTH = 256
HEIGHT1= 256
WIDTH1= 256
model = resnet_decoder(NCLASSES, HEIGHT, WIDTH)
#model = mobilenet_unet(n_classes=NCLASSES, input_height=416,input_width=416)
# model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=False,aux_branch=False)
#model = UNetPlusPlus(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES,deep_supervision=True)
#model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,1))
#model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=True)
#model = pspnet(n_classes=NCLASSES, input_height=HEIGHT, input_width = WIDTH, backbone="resnet50", aux_branch=True)
#model = FCN_Vgg16_32s(input_shape = (HEIGHT,WIDTH,1), weight_decay=1e-4, classes=NCLASSES)
#model = U_Net(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES)
#model.load_weights("logs/inferlast1.h5")
#model = MLBSNet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=False,aux_branch=True)
model.load_weights("logs/ep048-loss0.691-val_loss0.691.h5")
#model.load_weights("logs/ep050-loss0.257-val_loss0.245.h5")

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

# for jpg in image_ids:
#     img = Image.open(imgs + jpg + '.jpg')
#     orininal_h = np.array(img).shape[0]
#     orininal_w = np.array(img).shape[1]
#     img = img.resize((WIDTH,HEIGHT))
#     img = np.array(img)[:,:,0]  #类别
#     img = img/255
#     img = img[np.newaxis,:,:,np.newaxis]
#
#     pr = model.predict(img)[2]
#
#     pr = pr[0] #pspnet
#
#     pr[pr > 0.5] = 1
#     pr[pr <= 0.5] = 0
#     pr1 = copy.deepcopy(pr)

# videos_src_path = r'F:\yi\MLBSNetgw-Reinforcement-learning\input_video\video6'  # 提取图片的视频文件夹
imgs = r"gw_1650/jpg/"
image_ids = open(r"gw_1650/test1.txt",'r').read().splitlines()
fileSort = fileSort(image_ids)

# 视频列表
# videos_dir = list(filter(lambda x: x.endswith('mp4'), os.listdir(videos_src_path)))  # 筛选出mp4格式

# for video_name in videos_dir:  # 循环读取路径下的文件并筛选输出
#     if os.path.splitext(video_name)[1] == ".mp4":  # 筛选mp4文件
#         print(video_name)  # 输出所有的mp4文件

fps = 30
size = (1560,1440) #图片的分辨率片
#size = (1920,1080)
# size = (850,700)
#size = (720, 480)
file_path = r"F:\yi\JSR_Unet_automatic\out_video\\" + str(int(time.time())) + ".mp4"  # 导出路径
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

video = cv2.VideoWriter(file_path, fourcc, fps, size)


# cap = cv2.VideoCapture(videos_src_path + os.sep + video_name)
# videoFPS = int((cap.get(cv2.CAP_PROP_FPS)))
# print(videoFPS)
count = 1
for jpg in image_ids:
    # ret, frame = cap.read()
    # # show a frame
    # if ret is True:
        # print(frame.shape)
    count = count + 1
        # print(count)
        # if count % 1 == 0:  # 设置抽帧间隔，fps=25，每秒抽2帧
            # cropped = frame[0:700, 250:1100]  # 裁剪坐标为[y0:y1, x0:x1]
            #output_name = os.path.join(output_path, str(int(count / 1)) + ".jpg")
            #cropped=cv2.imwrite(output_name, cropped)
            # cropped = Image.fromarray(cropped.astype('uint8'))
            # cropped = cropped.convert("RGB")
    img = Image.open(imgs + jpg + '.jpg')
    image = img
    old_img = copy.deepcopy(image)

    orininal_h  = np.array(img).shape[0]
    orininal_w  = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)[:,:,0]  #类别
    img = img/255
    img = img[np.newaxis,:,:,np.newaxis]

    pr = model.predict(img)

    pr = pr[0] #pspnet

    pr[pr > 0.5] = 1
    pr[pr <= 0.5] = 0

    #pred = np.array(pr)//255
    #print(pr.shape)
    #pr = pr.reshape((int(256), int(256),2)).argmax(axis=-1)
    #pr = Image.fromarray(np.uint8(pr))
    pr = pr.squeeze()
    #pr = pr.resize(256,256)
    #pr = np.array(pr)[:,:]
    #pr = np.array(pr)
    #pr.transpose(2,1,0)
    colors = [(0, 0, 0),   (0, 255, 0),  (0, 255, 0),(255, 255, 0), (100, 0, 100),(0, 0, 255),(0, 255, 255),
              (255, 255, 255),
              (255, 0, 255), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]
    #print(pr.shape)
    seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    #seg_img = np.zeros((256, 256, 3))
    for c in range(2):
        seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    #imgb = Image.fromarray(np.uint8(pr*255))
    #image = Image.fromarray(np.uint8(seg_img))
    image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
    image = Image.blend(old_img,image,0.5)
    #r_image = copy.deepcopy(pr)
    #t_image = io.imsave("./image/" + str(count) + ".png",pr)
    #print(t_image)
    #r_image.show()
    r_image = np.array(image)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
    #r_image = r_image[...,::-1]

    print(count)
    video.write(r_image)  # 把图片写进视频


    # else:
    #     break
video.release()  # 释放
print("finished")
# cap.release()
# print("done!")