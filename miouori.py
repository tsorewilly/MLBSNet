import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
# 设标签宽W，长H
def fast_hist(a, b, n):
    # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)
    k = (a >= 0) & (a < n) #把不在种类内的像素点筛掉
    #print(k) # k为布尔值
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    # 返回中，写对角线上的为分类正确的像素点
    print(n,n * a[k],n * a[k].astype(int),n * a[k].astype(int) + b[k])
    count = 0
    # for i in n *a[k]:
    #     if i == 1:
    #         count+=1
    # print(count)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  # (label)n * a[k].astype(int)=0或2，(pre)b[k]为0或1，结果为1则
    #为背景判为前景，结果为0则为背景判为背景，结果为2则为前景判为背景，结果为3则为前景判为前景，np.bincount分别统计了0到3各出现了几次，
def per_class_iu(hist):
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    # 计算mIoU的函数
    print('Num classes', num_classes)
    ## 1
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]  # 获得验证集图像分割结果路径列表，方便直接读取

    # 读取每一个（图片-标签）对
    for ind in range(len(gt_imgs)):
        # 读取一张图像分割结果，转化成numpy数组
        print(pred_imgs[ind])
        pred = np.array(Image.open(pred_imgs[ind]))//255
        # 读取一张对应的标签，转化成numpy数组
        print(gt_imgs[ind])
        img = Image.open(gt_imgs[ind])
        img = img.resize((int(256), int(256)), resample=Image.BICUBIC)
        #img = img.convert("L")
        label = np.array(img)[:,:,0]
        #imga = Image.fromarray(235-(label*235 ))
        imga = Image.fromarray(label*255)
        imga.save(r".\new dataset\save\{}.png".format(png_name_list[ind]+"origin"))
        #imgb = Image.fromarray(235-(pred*235 ))
        imgb = Image.fromarray(pred*255)
        imgb.save(r".\new dataset\save\{}.png".format(png_name_list[ind] + "predict"))
        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        print(len(label.flatten()),len(pred.flatten()))
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue
        # 对一张图片计算19×19的hist矩阵，并累加
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)

        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs),
                                                100 * np.mean(
                                                    per_class_iu(hist))))

    # #画混淆矩阵
    # confusion = hist
    # # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    # plt.imshow(confusion, cmap=plt.cm.Blues)
    # # ticks 坐标轴的坐标点
    # # label 坐标轴标签说明
    # indices = range(len(confusion))
    # # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # # plt.xticks(indices, [0, 1, 2])
    # # plt.yticks(indices, [0, 1, 2])
    # plt.xticks(indices, ['background', 'guidewire'])
    # plt.yticks(indices, ['background', 'guidewire'])
    #
    # plt.colorbar()
    #
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.title('confusion matrix')
    #
    # # plt.rcParams两行是用于解决标签不能显示汉字的问题
    # # plt.rcParams['font.sans-serif']=['SimHei']
    # # plt.rcParams['axes.unicode_minus'] = False
    #
    # # 显示数据
    # for first_index in range(len(confusion)):  # 第几行
    #     for second_index in range(len(confusion[first_index])):  # 第几列
    #         plt.text(first_index, second_index, confusion[first_index][second_index])
    # # 在matlab里面可以对矩阵直接imagesc(confusion)
    # # 显示
    # plt.savefig('confusion_matrix')
    # plt.show()


    # confusion = np.array((a+[float("%0.4f"%(a[0]/sum(a)))*100],b+[float("%0.4f"%(b[1]/sum(b)))*100],[float("%0.4f"%(a[0]/c.sum(axis=0)[0]))*100]+[float("%0.2f"%(b[1]/c.sum(axis=0)[1]))*100]+[float("%0.2f"%((a[0]+b[1])/c.sum(axis=(0,1))))]))
    # deep 前后端 confusion = np.array(([2998895,407776,88.03],[593080,737312249,99.92],[83.49,99.94,0]))
    # deep 后端 confusion = np.array(([2499620,381872,86.75],[452772,594208136,99.92],[84.66,99.94,0]))
    # deep 前端 confusion = np.array(([397786,55367,87.78],[89985,145472862,99.94],[81.55,99.96,0]))
    # segnet 前后端 confusion = np.array(([1774646,1614456,52.36],[754199,737168699,99.9],[70.18,99.78,0]))
    # segnet 后端 confusion = np.array(([616559,2301436,21.13],[482593,594141812,99.92],[56.09,99.61,0]))
    #print("guidewire recall=%0.4f" % ((a) / (a + c)))
    #hist *= 14.94666666666666666666
    gp = round((hist[1][1]/ sum(hist[:,1]))*100,2)
    #gp = 0
    bp = round((hist[0][0]/ sum(hist[:,0]))*100,2)
    gr = round((hist[1][1]/ sum(hist[1,:]))*100,2)
    br = round((hist[0][0]/ sum(hist[0,:]))*100,2)
    gf = 2*(gr*gp)/(gr+gp)
    #gf = 0
    bf = 2*(br*bp)/(br+bp)
    maf1 = round((gf+bf)/2,2)
    gf = round(gf,2)
    confusion = np.array(([hist[1][1], hist[0][1], gp], [hist[1][0]+1, hist[0][0]+1, bp], [gr, br, maf1]))
    a = hist[1][1] + hist[0][1] + hist[1][0] + hist[0][0]
    b = hist[1][1] + hist[1][0]
    c = hist[0][1] + hist[0][0]
    print(a,b,c)
    #confusion = np.array(([hist[1][1], hist[0][1], 0], [hist[1][0], hist[0][0]+1, bp], [gr, br, 0]))
    print(confusion)
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # plt.xticks(indices, [0, 1, 2])
    # plt.yticks(indices, [0, 1, 2])
    plt.xticks(indices, ['S', 'B', 'R'], y=1.1, fontproperties='Times New Roman', fontsize=20)
    plt.yticks(indices, ['S', 'B', 'P'], fontproperties='Times New Roman', fontsize=20)

    # plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=15)
    # cb.ax.yaxis.get_offset_text().set_fontsize(15)

    plt.xlabel('Predicted label', fontproperties='Times New Roman', fontsize=20)
    plt.ylabel('True label', fontproperties='Times New Roman', fontsize=20)
    # plt.title('Confusion matrix',fontproperties = 'Times New Roman',fontsize=17)

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # ind_array = np.arange(len(confusion))
    # x, y = np.meshgrid(ind_array, ind_array)
    # 显示数据
    fmt = 'd'
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            # if second_index == 2:
            #     plt.text(first_index, second_index, format(confusion[]), color='red', va='center',
            #              ha='center', fontproperties='Times New Roman', fontsize=20)
            if first_index == 2 and second_index == 2:
                plt.text(first_index, second_index, "Macro-F1" + "\n" + format(confusion[first_index][second_index]) + "%",
                         color='black',
                         va='center', ha='center', fontproperties='Times New Roman', fontsize=20)
                # plt.text(first_index, second_index, format(confusion[first_index][second_index]*100)+"%",color='red',va='center', ha='center',fontproperties = 'Times New Roman',fontsize=20)
                continue
            if first_index == 2 or second_index == 2:
                plt.text(first_index, second_index, format(confusion[first_index][second_index]) + "%", color='red',
                         va='center', ha='center', fontproperties='Times New Roman', fontsize=20)
                continue
            plt.text(first_index, second_index, "%d" % confusion[first_index][second_index], color='red',
                     va='center', ha='center', fontproperties='Times New Roman', fontsize=20)
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示

    tick_marks = np.array(range(len(confusion))) + 0.5
    # Recall:0 B:1 G:2
    print(tick_marks)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')

    plt.show()

    # 计算所有验证集图片的逐类别mIoU值
    mIoUs = per_class_iu(hist)
    # 逐类别输出一下mIoU值
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


if __name__ == "__main__":
    gt_dir = "./new dataset/png"
    pred_dir = "./miou_pr_dir"
    png_name_list = open(r"new dataset\test.txt",'r').read().splitlines()
    #png_name_list = open(r"VOCdevkit\VOC2007\test_data.txt", 'r').read().splitlines()
    
    num_classes = 2
    name_classes = ["background","guidewire"]
    compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)  # 执行计算mIoU的函数
