# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:57:38 2022
@author: Omisore
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
from sklearn import metrics
from draw_confusion_matrix import plot_confusion_matrix_from_data
from os.path import join

HEIGHT = 256
WIDTH = 256
HEIGHT1 = 256
WIDTH1 = 256

def getDataBatch(batchFile):
    with open(r""+batchFile+"/train.txt", "r") as f:
        lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None) #Reset the randomization algorithm to always arrange data in same manner every run
        
        #divide data into training and validation (90:10)
        num_val = int(len(lines)*0.1)
        num_train = len(lines) - num_val
    return lines, num_train, num_val


def generate_arrays_from_file(batchFile, lines, batch_size, aug = None):    
    n, i = len(lines), 0    
    while 1:
        X_train, Y_train = [], []
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)                
            name = lines[i].split(';')[0]
            
            # read images in the training set from jpg folder
            img = Image.open(r"./"+batchFile+"/jpg" + '/' + name)
            img = img.resize((WIDTH,HEIGHT))
            if len((np.array(img)).shape) == 3:
                tmp_im_array = np.array(img)[:,:,0]
            else:
                tmp_im_array = np.array(img)
            
            tmp_im_array = tmp_im_array/255
            tmp_im_array = tmp_im_array[np.newaxis,:,:]

            #Split file entries, trim components to read masks from png folder
            name = (lines[i].split(';')[1]).replace("\n", "") 
            img = Image.open(r"./"+batchFile+"/png" + '/' + name)
            img = img.resize((int(WIDTH1),int(HEIGHT1)))
            if len((np.array(img)).shape) == 3:
                tmp_lb_array = np.array(img)[:,:,0]
            else:
                tmp_lb_array = np.array(img)
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
            #aug.fit(X_train)
            X_train = next(aug.flow(X_train,batch_size = batch_size,shuffle=False,seed = i))
            Y_train = next(aug.flow(Y_train,batch_size = batch_size,shuffle=False,seed = i))

        # Return data in batchsize, reshuffle after each return round
        yield(X_train, Y_train)

def get_flops(model):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        
    with graph.as_default():
        with session.as_default():
            #model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, 
                                                  cmd='op', options=opts)    
            return flops.total_float_ops  
        
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
        
#from keras_flops import get_flops
#flops = get_flops(model, batch_size=batch_size)	# Calculae FLOPS

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
    mcc = metrics.matthews_corrcoef(sample, samplelabel)

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
        imga.save(r"./"+gw_grp+"/save/{}.png".format(png_name_list[ind] + "origin"))
        imgb = Image.fromarray(pred * 255)
        imgb.save(r"./"+gw_grp+"/save/{}.png".format(png_name_list[ind] + "predict"))
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
          
