#from nets.unet import convnet_unet
# from nets.unetplusplus import UNetPlusPlus
# from nets.Repunet import convnet_unet
# from nets.pspnet import pspnet
# from nets.fcn import FCN_Vgg16_32s
#from nets.helper_functions import UNetPlusPlus,U_Net
#from nets.deeplab import Deeplabv3
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

import warnings
warnings.filterwarnings("ignore")

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

class_colors = [[0,0,0],[0,255,0]]
NCLASSES = 1
HEIGHT = 256
WIDTH = 256
HEIGHT1= 256
WIDTH1= 256
#model = mobilenet_unet(n_classes=NCLASSES, input_height=416,input_width=416)
#model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=False,aux_branch=False)
#model = UNetPlusPlus(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES,deep_supervision=True)
#model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,1))
#model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=True)
#model = pspnet(n_classes=NCLASSES, input_height=HEIGHT, input_width = WIDTH, backbone="resnet50", aux_branch=True)
#model = FCN_Vgg16_32s(input_shape = (HEIGHT,WIDTH,1), weight_decay=1e-4, classes=NCLASSES)
#model = U_Net(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES)
#model.load_weights("logs/inferlast1.h5")
model = MLBSNet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=False,aux_branch=True)
model.load_weights("logs/ep044-loss0.004-val_loss0.004.h5")
#model.load_weights("logs/ep050-loss0.257-val_loss0.245.h5")

imgs = r"new dataset\jpg/"
labelimgs = r"new dataset\png/"
image_ids = open(r"new dataset\test.txt",'r').read().splitlines()
print(image_ids)
if not os.path.exists("./miou_pr_dir"):
    os.makedirs("./miou_pr_dir")
fps = 0
i = 0
for jpg in image_ids:
    #t1 = time.time()
    print(imgs + jpg + '.jpg')
    img = Image.open(imgs + jpg + '.jpg')
    labelimg = Image.open(labelimgs + jpg + '.png')
    #old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]
    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)[:,:,0]  #类别
    img = img/255
    #img = img[:,:,0]
    #img = img.reshape(-1,HEIGHT,WIDTH,3)
    img = img[np.newaxis,:,:,np.newaxis]

    labelimg = labelimg.resize((WIDTH1,HEIGHT1))
    labelimg = np.array(labelimg)
    seg_labels = np.zeros((int(HEIGHT1), int(WIDTH1), NCLASSES))
    for c in range(NCLASSES):
        #seg_labels[:, :, c] = (labelimg[:, :] == c).astype(int)
        seg_labels[:, :, c] = (labelimg[:, :,0] == c+1).astype(int)  #可针对多类
    #按类别数进行自动调整，将标签打平成一列
    seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
    t1 = time.time()
    pr = model.predict(img)[2]

    pr = pr[0] #pspnet

    fps += (1. / (time.time() - t1))
    #print(pr.shape)
    #pr1 = copy.deepcopy(pr)
    #print(np.max(pr1))

    pr[pr > 0.5] = 1
    pr[pr <= 0.5] = 0

    pr1 = copy.deepcopy(pr)

    if i == 0:
        sample = pr1
        samplelabel = seg_labels
    else:
        sample = np.concatenate([sample,pr1],axis=0)
        #print(sample.shape)
        samplelabel = np.concatenate([samplelabel,seg_labels],axis=0)
    # print(sample.shape)
    # print(samplelabel.shape)

    # print(pr)
    # print(seg_labels)

    #pr = pr.reshape((int(HEIGHT1), int(HEIGHT1),NCLASSES)).argmax(axis=-1)
    # seg = pr.argmax(axis=-1)+1
    # print(pr.max(axis=-1).shape)
    # for i in  pr.max(axis=-1):
    #     for j in i:
    #         if j <= 0.5:
    #             seg[i][j] = 0
    # seg = np.where(pr.max(axis=-1) <= 0.5, 0, seg)
    # pr = seg

    #print(pr.shape)
    #seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
    #colors = class_colors

    # for c in range(NCLASSES):

    #     seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
    #     seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
    #     seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    #io.imsave("./miou_pr_dir/" + jpg + ".png",((pr*255).astype(np.uint8)))
    #pr = 255-pr
    io.imsave("./miou_pr_dir/" + jpg + ".png",pr)  #灰度预测图

    # seg_img = Image.fromarray(np.uint8(pr))
    # seg_img.save("./miou_pr_dir/" + jpg + ".png")

    # image = Image.blend(old_img,seg_img,0.3)

    # if fps1 == 0:
    #     fps = 1. / (time.time() - t1)
    #     fps1 = fps
    # else:
    #     fps1 = (fps1 + (1. / (time.time() - t1))) / 2
    #     print("fps1 = %.2f"%fps1)
    #fps += (1. / (time.time() - t1))
    i = 1
print("fps = %.2f"%(fps/73))

sample = np.reshape(sample,(-1,NCLASSES))
# for i in sample:
#     print(np.around(i, 2))
print(sample.shape)
print(samplelabel.shape)
precision = dict()
recall = dict()
pr_ap = dict()

fpr = dict()
tpr = dict()
roc_auc = dict()
#print(sample.ravel())
for i in range(NCLASSES):
    precision[i], recall[i], _ = metrics.precision_recall_curve(samplelabel[:, i], sample[:, i]) #precision_recall_curve roc_curve
    pr_ap[i] = metrics.average_precision_score(samplelabel[:, i], sample[:, i],average='weighted')
    # fpr[i], tpr[i], _ = metrics.roc_curve(samplelabel[:, i], sample[:, i])
    # #roc_auc[i] = metrics.roc_auc_score(samplelabel[:, i], sample[:, i])
    # roc_auc[i] = metrics.auc(fpr[i], tpr[i])
#print(roc_auc)
area = np.trapz(y=precision[i][::-1], x=recall[i][::-1])
print(area)
print(precision[i],recall[i])
print(precision[i].shape)
print(recall[i].shape)
print(pr_ap[i])
# print(fpr[i],tpr[i])
# print(fpr[i].shape)
# print(tpr[i].shape)
# print(sample.shape)
# print(samplelabel.shape)
#print(np.array(sample))
# precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(samplelabel.ravel(),sample.ravel())
# pr_ap["micro"] = metrics.average_precision_score(samplelabel.ravel(),sample.ravel())
#roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# all_recall = np.unique(np.concatenate([recall[i] for i in range(2)]))  #该函数是去除数组中的重复数字，并进行排序之后输出。
# mean_precision = np.zeros_like(all_recall)
# for i in range(2):
#     mean_precision += np.interp(all_recall, recall[i], precision[i])
# mean_precision /= 2
# recall["macro"] = all_recall
# precision["macro"] = mean_precision
# pr_ap["macro"] = metrics.average_precision_score(fpr["macro"], tpr["macro"])

#print(roc_auc["micro"])
lw=2
plt.figure()
# plt.plot(precision["micro"], recall["micro"],
#          label='micro-average PR curve (area = {0:0.2f})'
#                ''.format(pr_ap["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          # label='macro-average ROC curve (area = {0:0.2f})'
#          #       ''.format(roc_auc["macro"]),
#         label = "2",
#          color='navy', linestyle=':', linewidth=4)

color = cycle(['red','black','gold','green','purple','hotpink','aqua', 'darkorange', 'cornflowerblue'])
#linestyle = cycle(['-', '--', '-.',':','solid','o',' ','dashed','dashdot'])
linestyle = cycle([(0, ()), (0, (1, 1)), (0, (5,5)),'-.',(0, (5, 1)),(0, (3, 1, 1, 1)),(0, (3, 1, 1, 1, 1, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1,1,1))])
#marker =  cycle(['*',',','o','v','^','<','>','s','p','h','H','+','x','D','d','|','_','.',','])
#linestyle = cycle()
#linestyle = cycle([':'])
# for i, color in zip(range(NCLASSES), colors):
#     plt.plot(recall[i], precision[i], color=color, lw=lw,
#              label='PR curve of class {0} (area = {1:0.2f})'
#              ''.format(i, pr_ap[i]))
#classes = ["背景","肝","肾","胰腺","血管","肾上腺","胆囊","骨头","脾脏"]
#classes = ["background", "liver", "kidney", "pancreas", "vessel", "adrenals", "gallbladder", "bone", "spleen"]
classes = ["background", "guidewire"]
#classes = ["ship"]
#classes = ["background", "lung"]
# for i,cla, marker in zip(range(NCLASSES), classes ,marker):
#     plt.plot(recall[i], precision[i], marker=marker, lw=lw,
#              label='PR curve of {0} (area = {1:0.2f})'
#              ''.format(cla, pr_ap[i]))
x_major_locator = plt.MultipleLocator(0.2)
y_major_locator = plt.MultipleLocator(0.2)  #把y轴的刻度间隔设置为0.2
ax=plt.gca() #ax为两条坐标轴的实例
ax.yaxis.set_major_locator(y_major_locator) #把y轴的主刻度设置为0.2的倍数
ax.xaxis.set_major_locator(x_major_locator)
plt.ylim(0,1.05)
plt.yticks(fontproperties = 'Times New Roman',fontsize=20)
plt.xticks(fontproperties = 'Times New Roman',fontsize=20)

for i,cla, linestyle, color in zip(range(NCLASSES), classes ,linestyle,color):
    plt.plot(recall[i], precision[i],
             #plt.plot(fpr[i], tpr[i],
             linestyle=linestyle, color =color,  lw=lw,
             label='PR curve of {0} (area = {1:0.3f})'
                   ''.format(cla, pr_ap[i]))
    # label='ROC curve of {0} (area = {1:0.2f})'
    #       ''.format(cla, roc_auc[i]))
#print(fpr, tpr, _)
print(_.shape)
print(_)
#plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
font = font_manager.FontProperties(family='Times New Roman', size=20)
#legend_font = {"family" : "Times New Roman"}
plt.xlabel('Recall',fontproperties = 'Times New Roman',fontsize=20)
plt.ylabel('Precision',fontproperties = 'Times New Roman',fontsize=20)
# plt.xlabel('False positive rate',fontproperties = 'Times New Roman',fontsize=20)
# plt.ylabel('True positive rate',fontproperties = 'Times New Roman',fontsize=20)
#plt.title('Some extension of Receiver operating characteristic to multi-class')
#plt.legend.get_title().set_fontsize(fontsize = 15)
plt.legend(loc="lower right",prop = font)
plt.show()



