from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Input
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras.models import Model
from keras import backend as K

alpha = 1


def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)


def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def return_activation(x, nl):
    # 用于判断使用哪个激活函数
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x


def conv_block(inputs, filters, kernel, strides, nl):
    # 一个卷积单元，也就是conv2d + batchnormalization + activation
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)

    return return_activation(x, nl)


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


def bottleneck(inputs, filters, kernel, up_dim, stride, sq, nl,rate=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    input_shape = K.int_shape(inputs)

    tchannel = int(up_dim)
    cchannel = int(alpha * filters)

    r = stride == 1 and input_shape[3] == filters
    # 1x1卷积调整通道数，通道数上升
    x = conv_block(inputs, tchannel, (1, 1), (1, 1), nl)
    # 进行3x3深度可分离卷积
    x = DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same',dilation_rate=rate)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl)
    # 引入注意力机制
    if sq:
        x = squeeze(x)
    # 下降通道数
    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def get_mobilenet_encoder(input_height=224 ,  input_width=224 , pretrained='imagenet'):
    #inputs = Input(shape)
    # 416,416,3 -> 208,208,16
    img_input = Input(shape=(input_height, input_width, 3))
    x = conv_block(img_input, 16, (3, 3), strides=(2, 2), nl='HS')
    f1 = x
    # 208,208,16 -> 104,104,24
    x = bottleneck(x, 16, (3, 3), up_dim=16, stride=1, sq=False, nl='RE')

    x = bottleneck(x, 24, (3, 3), up_dim=64, stride=2, sq=False, nl='RE')
    f2 = x
    x = bottleneck(x, 24, (3, 3), up_dim=72, stride=1, sq=False, nl='RE')



    # 104,104,24 -> 52,52,40
    x = bottleneck(x, 40, (5, 5), up_dim=72, stride=2, sq=True, nl='RE')
    f3 = x
    x = bottleneck(x, 40, (5, 5), up_dim=120, stride=1, sq=True, nl='RE')
    x = bottleneck(x, 40, (5, 5), up_dim=120, stride=1, sq=True, nl='RE')

    # 52,52,40 -> 28,28,80
    x = bottleneck(x, 80, (3, 3), up_dim=240, stride=2, sq=False, nl='HS')
    f4 = x
    x = bottleneck(x, 80, (3, 3), up_dim=200, stride=1, sq=False, nl='HS')
    x = bottleneck(x, 80, (3, 3), up_dim=184, stride=1, sq=False, nl='HS')
    x = bottleneck(x, 80, (3, 3), up_dim=184, stride=1, sq=False, nl='HS')
    f4 = x
    #14,14,80 -> 14,14,112
    # x = bottleneck(x, 112, (3, 3), up_dim=480, stride=1, sq=True, nl='HS')
    # x = bottleneck(x, 112, (3, 3), up_dim=672, stride=1, sq=True, nl='HS')
    # # f5 = x
    # #
    # # # 14,14,112 -> 7,7,160
    # x = bottleneck(x, 160, (5, 5), up_dim=672, stride=2, sq=True, nl='HS')
    # f5 = x
    # x = bottleneck(x, 160, (5, 5), up_dim=960, stride=1, sq=True, nl='HS')
    # x = bottleneck(x, 160, (5, 5), up_dim=960, stride=1, sq=True, nl='HS')

    # 7,7,160 -> 7,7,960
    # x = conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, 960))(x)
    #
    # x = Conv2D(1280, (1, 1), padding='same')(x)
    # x = return_activation(x, 'HS')
    #
    # x = Conv2D(n_class, (1, 1), padding='same', activation='softmax')(x)
    # x = Reshape((n_class,))(x)
    #
    # model = Model(inputs, x)

    return img_input , [f1 , f2 , f3 , f4 ] #, f5
