#from nets.unet import convnet_unet
# from nets.Repunet import convnet_unet
# from nets.pspnet import pspnet
# from nets.fcn import FCN_Vgg16_32s

from nets.MLBSNet import MLBSNet
#from nets.unetplusplus import UNetPlusPlus
#from nets.deeplab import Deeplabv3
#from nets.Repconvnet import Rep_get_convnet_encoder
#from nets.pspnet import pspnet  #Change the code and data to return, change the accuracy evaluation, change pr[0]
#from nets.repvgg import repvgg_model_convert
#from nets.helper_functions import U_Net#, UNetPlusPlus


import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
   
import random
import copy
import os
import time
import tensorflow.keras.backend as K
from sklearn import metrics
from itertools import cycle
import tensorflow as tf
import skimage.io as io
from draw_confusion_matrix import plot_confusion_matrix_from_data
import warnings

warnings.filterwarnings("ignore")

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Set the label width W, long H
def fast_hist(a, b, n):
    # a is the label, shape (H×W, converted into a one-dimensional array;
    # b is the label converted into a one-dimensional array, shape (H×W, )
    k = (a >= 0) & (a < n)  # Filter out pixels that are not in the category
    # returning the value shape (n, n)
    print(n, n * a[k], n * a[k].astype(int), n * a[k].astype(int) + b[k])
    count = 0
    # for i in n *a[k]:
    #     if i == 1:
    #         count+=1
    # print(count)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    # Returns 0 for BG as BG; 1 for BG as FG; 2 for FG as BG; 3 for FG as FG;
    # np.bincount counted 0 to 3 appeared several times each,


def per_class_iu(hist):
    # The value on matrix diagonal consists of a one-dimensional array/sum of all elements of the matrix,
    # returning the value shape (n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
def compute_binaryMetrics(sample, samplelabel, NCLASSES):    
    cm1 = metrics.confusion_matrix(sample, samplelabel)
    accu = (cm1[0,0]+cm1[1,1])/sum(sum(cm1))        
    sens = cm1[0,0]/(cm1[0,0]+cm1[0,1])    
    spec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    mcc = metrics.matthews_corrcoef(pred, samplelabel)

    precision = dict()
    recall = dict()
    pr_ap = dict()
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NCLASSES):
        precision[i], recall[i], _ = metrics.precision_recall_curve(samplelabel[:, i], sample[:, i]) #precision_recall_curve roc_curve
        pr_ap[i] = metrics.average_precision_score(samplelabel[:, i], sample[:, i],average='weighted')
        fpr[i], tpr[i], _ = metrics.roc_curve(samplelabel[:, i], sample[:, i])
        #roc_auc[i] = metrics.roc_auc_score(samplelabel[:, i], sample[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #print(roc_auc)
    area = np.trapz(y=precision[i][::-1], x=recall[i][::-1])
    # print(fpr[i],tpr[i])
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(samplelabel.ravel(),sample.ravel())
    pr_ap["micro"] = metrics.average_precision_score(samplelabel.ravel(),sample.ravel())
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(samplelabel.ravel(), sample.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    """
    all_recall = np.unique(np.concatenate([recall[i] for i in range(2)]))  
    mean_precision = np.zeros_like(all_recall)
    for i in range(2):
        mean_precision += np.interp(all_recall, recall[i], precision[i])
    mean_precision /= 2
    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    pr_ap["macro"] = metrics.average_precision_score(fpr["macro"], tpr["macro"])  
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])  
    """    
    return cm1, accu, sens, spec, mcc, area, precision, recall, fpr, tpr, pr_ap, roc_auc


def compute_mIoU(gt_dir, pred_dir, gw_grp, png_name_list, num_classes, name_classes):
    # A function that computes mIoU
    print('Num classes', num_classes)
    ## 1
    hist = np.zeros((num_classes, num_classes))

    # Get a list of tag paths for the validation set for easy direct reading
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]

    # Get the list of verification set image segmentation result paths for direct reading
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # Read each (image-tag) pair
    for ind in range(len(gt_imgs)):
        # Read segmentation result and convert it to numpy array
        pred = np.array(Image.open(pred_imgs[ind])) // 255
        # Read corresponding tag and convert it to a numpy array
        img = Image.open(gt_imgs[ind])
        img = img.resize((int(256), int(256)), resample=Image.BICUBIC)
        label = np.array(img)[:, :, 0]
        imga = Image.fromarray(label * 255)
        imga.save(r"./"+gw_grp+"save_cmpLoss/{}.png".format(png_name_list[ind] + "origin"))
        imgb = Image.fromarray(pred * 255)
        imgb.save(r"./"+gw_grp+"save_cmpLoss/{}.png".format(png_name_list[ind] + "predict"))
        # If segmentation result and label size are not the same, the image is not counted
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue
        # Calculate the histogram matrix of 19×19 on an image and accumulate
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)

        # For 10 images,output average mIoU of all categories
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    return hist

def figPlot(y_true, y_pred, saveAs='foo.png', title=''):
    #print(metrics.classification_report(y_true, y_pred))
    #print("Accuracy: {0}".format(metrics.accuracy_score(y_true, y_pred)))

    plot_confusion_matrix_from_data(y_true, y_pred, columns=['BG', 'GW'], annot=True, cmap='Blues', 
                fmt='.2f', fz=20, lw=0.5, cbar=False, figsize=[6, 6], show_null_values=2, 
                pred_val_axis='y', SaveFig = 1, saveAs=saveAs, title=title)

def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        
    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, 
                                                  cmd='op', options=opts)    
            return flops.total_float_ops 
# In[] DEPLOY MODEL FOR SEGMENTATION AND PERFORMANCE ANALYSIS 
if __name__ == "__main__":
    gw_grp = "gw_TMI/"
    imgs = r""+gw_grp+"jpg/"
    labelimgs = r""+gw_grp+"png/"
    pred_dir = "./miou_pr_dir/"+gw_grp   
    gt_dir = "./"+gw_grp+"png"    
    png_name_list = open(r""+gw_grp+"test.txt", 'r').read().splitlines()
    
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(gw_grp+"save_cmpLoss"):
        os.makedirs(gw_grp+"save_cmpLoss")

    #png_name_list = open(r"VOCdevkit\VOC2007\test_data.txt", 'r').read().splitlines()
    
    num_classes = 2
    name_classes = ["BG", "GW"]
    # Executes a function that computes mIoU
    
# In[] SEGMENTATION DONE HERE
    class_colors = [[0,0,0],[0,255,0]]
    NCLASSES = 1
    HEIGHT, WIDTH, HEIGHT1, WIDTH1 = 256, 256, 256, 256
    
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        
    #with graph.as_default():
    #    with session.as_default():
    model = MLBSNet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH, deploy=False, aux_branch=True)            
    #model = mobilenet_unet(n_classes=NCLASSES, input_height=416,input_width=416)
    #model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=False,aux_branch=False)
    #model = UNetPlusPlus(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES,deep_supervision=True)            
    #model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,1),deploy=False)
    #model = convnet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH,deploy=True)
    #model = pspnet(n_classes=NCLASSES, input_height=HEIGHT, input_width = WIDTH, backbone="resnet50", aux_branch=True)
    #model = FCN_Vgg16_32s(input_shape = (HEIGHT,WIDTH,1), weight_decay=1e-4, classes=NCLASSES)
    #model = U_Net(HEIGHT,WIDTH,color_type=1,num_class=NCLASSES)
            
            #run_meta = tf.compat.v1.RunMetadata()
            #opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            #flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)                
    model.load_weights('logs/last1-MLBS-Rabbit-Base-Weights_ep-50.h5')#('logs/last1-MLBS-Rabbit-Base-Weights.h5')
    #model.load_weights("logs/last1-MLBS-PigTransfer-Weights.h5") #last1-DeepLabV-Net
    #model.save_weights(log_dir + 'last1-MLBS-PigTransfer-Weights.h5')
# In[] EVALUATION DONE HERE    
    tpf, fps, i = 0, 0, 0
    
    for jpg in png_name_list:
        img = Image.open(imgs + jpg + '.jpg')
        labelimg = Image.open(labelimgs + jpg + '.png')
        img = img.resize((WIDTH,HEIGHT))
        img = (np.array(img)[:,:,0])/255
        img = img[np.newaxis,:,:,np.newaxis]
    
        labelimg = labelimg.resize((WIDTH1,HEIGHT1))
        labelimg = np.array(labelimg)
        #if len(labelimg.shape) == 2:
        #    labelimg = labelimg[:,:,np.newaxis]
            
        seg_labels = np.zeros((int(HEIGHT1), int(WIDTH1), NCLASSES))
        for c in range(NCLASSES):
            seg_labels[:, :, c] = (labelimg[:, :,0] == c+1).astype(int)
        
        seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
        t1 = time.time()
        pr = (model.predict(img)[2])[0]
        #t2 = time.time()
        tpf += time.time() - t1
        fps += (1. / (time.time() - t1))
        
        pr[pr > 0.5] = 1
        pr[pr <= 0.5] = 0        
        pr1 = copy.deepcopy(pr)
    
        if i == 0:
            sample = pr1
            samplelabel = seg_labels
        else:
            sample = np.concatenate([sample,pr1],axis=0)
            samplelabel = np.concatenate([samplelabel,seg_labels],axis=0)
        i=1
        io.imsave(pred_dir + jpg + ".png",pr) 
    #fps = fps/len(png_name_list)
    #tpf = tpf/len(png_name_list)
# In[] COMPUTE BINARY METRICS FOR QUANTITATIVE PERFORMANCE ANALYSIS    
    num_classes = 2
    name_classes = ["BG", "GW"]
    
    # Executes a function that computes mIoU
    hist = compute_mIoU(gt_dir, pred_dir, gw_grp, png_name_list, num_classes, name_classes)
        # Calculate the category-by-category mIoU value for all validation set pictures    
    
    pred = np.reshape(sample,(-1,NCLASSES))
    cm1, accu, sens, spec, mcc, area, precision, recall, fpr, tpr, pr_ap, roc_auc = compute_binaryMetrics(pred, samplelabel, NCLASSES)

    mIoUs = per_class_iu(cm1)
    # Output the mIoU values category by category
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + '\t' + str(round(mIoUs[ind_class] * 100, 2)))
    # Average mIoU values for all categories on all validation set images, ignoring NaN values when calculating
    print('===>mIoU\t ' + str(round(np.nanmean(mIoUs) * 100, 2))) 

# In[] Analysis and plotting   
    figPlot(samplelabel, pred, saveAs='CF-TestSet-MLBS-Net.png', title='CF TestSet MLBS Network Model')
    
    TP, FP, FN, TN = (np.reshape(cm1, (4,1)).astype(int))
    DSC = (TP[0]+TP[0])/(TP[0]+TP[0]+FP[0]+FN[0])
    F1s = 2 * (precision['micro'][1] * recall['micro'][1]) / (precision['micro'][1] + recall['micro'][1])
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr['micro'], tpr['micro'])
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('True Positive Rate(Sensitivity)')
    plt.xlabel('False Positive Rate(Specificity)')
    plt.show()
    
    lw=2
    plt.figure()
    plt.plot(precision["micro"], recall["micro"], 
             #label='Micro-avg PR curve (area = {0:0.2f})'.format(pr_ap["micro"]), 
                   color='deeppink', linestyle=':', linewidth=4)
    
    #plt.plot(fpr["macro"], tpr["macro"], label='macro-avg ROC curve (area = {0:0.2f})'
    #         ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
    
    color = cycle(['red','black','gold','green','purple','hotpink','aqua', 'darkorange', 'cornflowerblue'])
    #linestyle = cycle(['-', '--', '-.',':','solid','o',' ','dashed','dashdot'])
    linestyle = cycle([(0, ()), (0, (1, 1)), (0, (5,5)),'-.',(0, (5, 1)),(0, (3, 1, 1, 1)),(0, (3, 1, 1, 1, 1, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1,1,1))])
    #marker =  cycle(['*',',','o','v','^','<','>','s','p','h','H','+','x','D','d','|','_','.',','])
    #linestyle = cycle()
    #linestyle = cycle([':'])
    classes = ["BG", "GW"]
    x_major_locator = plt.MultipleLocator(0.2)
    y_major_locator = plt.MultipleLocator(0.2)  
    ax=plt.gca() 
    ax.yaxis.set_major_locator(y_major_locator) 
    ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(0,1.05)
    plt.yticks(fontproperties = 'Times New Roman',fontsize=20)
    plt.xticks(fontproperties = 'Times New Roman',fontsize=20)
    
    for i, cla, linestyle, color in zip(range(NCLASSES), classes, linestyle, color):
        plt.plot(recall[i], precision[i], linestyle=linestyle, color =color, 
                 lw=lw, label='PR curve of {0} (area = {1:0.3f})'
                       ''.format(cla, pr_ap[i]))
    
    #plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    font = font_manager.FontProperties(family='Times New Roman', size=15)
    #legend_font = {"family" : "Times New Roman"}
    plt.xlabel('Recall',fontproperties = 'Times New Roman',fontsize=20)
    plt.ylabel('Precision',fontproperties = 'Times New Roman',fontsize=20)
    # plt.xlabel('False positive rate',fontproperties = 'Times New Roman',fontsize=20)
    # plt.ylabel('True positive rate',fontproperties = 'Times New Roman',fontsize=20)
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    #plt.legend.get_title().set_fontsize(fontsize = 15)
    plt.legend(loc="lower right",prop = font)
    plt.show()
    
