from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from nets.repvgg import RepVGGBlock
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)

def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)


def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels / 4))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x

def SepConv_BN(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    
    # 如果需要激活函数
    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

# def return_activation(x, nl):
#     # 用于判断使用哪个激活函数
#     if nl == 'HS':
#         x = Activation(hard_swish)(x)
#     if nl == 'RE':
#         x = Activation(relu6)(x)
#
#     return x
#
#
# def conv_block(inputs, filters, kernel, strides, nl):
#     # 一个卷积单元，也就是conv2d + batchnormalization + activation
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#
#     x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
#     x = BatchNormalization(axis=channel_axis)(x)
#
#     return return_activation(x, nl)
#
# def bottleneck(inputs, filters, kernel, up_dim, stride, sq, nl,rate=1):
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#
#     input_shape = K.int_shape(inputs)
#
#     tchannel = int(up_dim)
#     cchannel = int(alpha * filters)
#
#     r = stride == 1 and input_shape[3] == filters
#     # 1x1卷积调整通道数，通道数上升
#     x = conv_block(inputs, tchannel, (1, 1), (1, 1), nl)
#     # 进行3x3深度可分离卷积
#     x = DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same',dilation_rate=rate)(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#     x = return_activation(x, nl)
#     # 引入注意力机制
#     # if sq:
#     #     x = squeeze(x)
#     # 下降通道数
#     x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#
#     return x

def _xception_block(inputs, depth_list,  skip_connection_type, stride, rate=1, depth_activation=False, return_skip=False, deploy=True):
    residual = inputs
    residual = Conv2D(int(depth_list[-1]*1.2), (1, 1), strides=(1, 1), padding='same')(residual)
    residual = BatchNormalization()(residual)
    residual = SepConv_BN(residual, depth_list[-1], stride=1, rate=rate, depth_activation=depth_activation) #128，128，128
    skip = residual

    if skip_connection_type == 'conv':
        residual = MaxPooling2D((2, 2), strides=(2, 2))(residual)
        #if i == 1:

    if skip_connection_type == 'conv':
        shortcut = RepVGGBlock(in_channels=depth_list[-1],out_channels=depth_list[-1],kernel_size=3,strides=2,deploy=deploy)(inputs)
        outputs = layers.add([residual, shortcut])  #1*1和3*3分支
    elif skip_connection_type == 'sum':
        shortcut = RepVGGBlock(in_channels=depth_list[-1],out_channels=depth_list[-1],kernel_size=3,deploy=deploy)(inputs)
        outputs = layers.add([residual, shortcut])   #1*1、3*3和identity分支
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

def _xception_block1(inputs, depth_list, skip_connection_type, stride, rate=1, depth_activation=False, return_skip=False, deploy=True):
    residual = inputs
    residual = Conv2D(int(depth_list[-1]*1.2), (1, 1), strides=(1, 1), padding='same')(residual)
    residual = BatchNormalization()(residual)
    residual = SepConv_BN(residual, depth_list[-1], stride=1, rate=rate, depth_activation=depth_activation)#128，128，128
    skip = residual

    if skip_connection_type == 'conv':
        residual = MaxPooling2D((2, 2), strides=(2, 2))(residual)
    if skip_connection_type == 'conv':
        shortcut = RepVGGBlock(in_channels=depth_list[-1],out_channels=depth_list[-1],kernel_size=3,strides=2,deploy=deploy)(inputs)
        outputs = layers.add([residual, shortcut])  #1*1和3*3分支
    elif skip_connection_type == 'sum':
        shortcut = RepVGGBlock(in_channels=depth_list[-1],out_channels=depth_list[-1],kernel_size=3,deploy=deploy)(inputs)
        outputs = layers.add([residual, shortcut])   #1*1、3*3和identity分支
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

def MLBDNet(inputs,alpha=1,OS=16,deploy=True):
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides=(1, 1), name='entry_flow_conv1_1', use_bias=False, padding='same')(inputs) #32, #256,256,32
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)  #64, # 256,256,64
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)
    f1 = x
    #print(f1)
    # 256,256,128 -> 256,256,128 -> 128,128,128
    x = _xception_block(x, [96, 96, 96], skip_connection_type='conv', stride=entry_block3_stride, #2,
                        depth_activation=False, deploy=deploy)#'entry_flow_block1',  #128, 128, 128
    f2 = x
    # 128,128,256 -> 128,128,256 -> 64,64,256
    # skip = 128,128,256
    x, skip1 = _xception_block(x, [128, 128, 128], #'entry_flow_block2', #256, 256, 256
                                skip_connection_type='conv', stride=entry_block3_stride, #2,
                                depth_activation=False, return_skip=True,deploy=deploy)
    f3 = x

    x = _xception_block(x, [192, 192, 192], #'entry_flow_block3',  #728,728,728
                        skip_connection_type='conv', stride=entry_block3_stride,  #2
                        depth_activation=False,deploy=deploy)

    for i in range(16):
        x = _xception_block(x, [192, 192, 192], #'middle_flow_unit_{}'.format(i + 1),  #728,728,728
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False,deploy=deploy)
    f4 = x
    x = _xception_block(x, [192, 384, 384], #'exit_flow_block1',  #728, 1024, 1024
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0], #1
                        depth_activation=False,deploy=deploy)

    #print(f4.shape)
    x = _xception_block(x, [512, 512, 728], #'exit_flow_block2',#1536, 1536, 2048
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1], #2
                        depth_activation=True,deploy=deploy)
    f5 = x
    #print(f5.shape)
    #return x,atrous_rates,skip1
    return inputs, [f1, f2, f3, f4, f5]