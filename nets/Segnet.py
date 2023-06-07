from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
import tensorflow.keras.backend as K
import tensorflow.keras
import tensorflow as tf
#from nets.pooling_indices import MaxPoolingWithArgmax2D

IMAGE_ORDERING = 'channels_last'

def mish(x):
    return x * K.tanh(K.softplus(x))

def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

# def squeeze(inputs):
#     # 注意力机制单元
#     input_channels = int(inputs.shape[-1])
#
#     x = GlobalAveragePooling2D()(inputs)
#     x = Dense(int(input_channels/4))(x)
#     x = Activation(mish)(x)
#     x = Dense(input_channels)(x)
#     x = Activation(relu6)(x)
#     x = Reshape((1, 1, input_channels))(x)
#     x = Multiply()([inputs, x])
#
#     return x

def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels//16))(x)
    x = Activation('relu')(x)
    x = Dense(input_channels)(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x

def MaxUnpooling2D(pool, ind, ksize=[1,2,2,1], scope='MaxUnpooling2D'):

    """
    Unpooling layer after max_pool_with_argmax.
    Args:
    pool:  max pooled output tensor
    ind:   argmax indices
    ksize: ksize is the same as for the pool
    Return:
      ret: unpooling tensor
    """
    with tf.compat.v1.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]


        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        #print(set_output_shape)
        return ret

def Segnet( n_classes=3,input_height=416,  input_width=416, pretrained='imagenet' ):

    img_input = Input(shape=(input_height,input_width , 1 ))

    # 416,416,3 -> 208,208,64
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    #x = MaxPoolingWithArgmax2D()(x)
    x1,argmax1 = tf.nn.max_pool_with_argmax(input=x, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)a = x[1]
	#x = squeeze(x)
    # f1 = x[0]
    # a = x[1]
    # 208,208,64 -> 128,128,128
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    #x = MaxPoolingWithArgmax2D()(x)
    x2,argmax2 = tf.nn.max_pool_with_argmax(input=x, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #x = squeeze(x)
    # f2 = x[0]
    # b = x[1]
    # 104,104,128 -> 52,52,256
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    #x = MaxPoolingWithArgmax2D()(x)
    x3,argmax3 = tf.nn.max_pool_with_argmax(input=x, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #x = squeeze(x)
    # f3 = x[0]
    # c = x[1]
    # 52,52,256 -> 26,26,512
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    #x = MaxPoolingWithArgmax2D()(x)
    x4,argmax4 = tf.nn.max_pool_with_argmax(input=x, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #x = squeeze(x)
    # f4 = x[0]
    # d = x[1]

    # 26,26,512 -> 13,13,512
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    #x = MaxPoolingWithArgmax2D()(x)
    x5,argmax5 = tf.nn.max_pool_with_argmax(input=x, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #x = squeeze(x)
    #f5 = x

    o = MaxUnpooling2D(x5,argmax5,scope="MaxUnpooling1")

    o = (Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = MaxUnpooling2D(o,argmax4,scope="MaxUnpooling2")

    o = (Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = MaxUnpooling2D(o,argmax3,scope="MaxUnpooling3")

    o = (Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = MaxUnpooling2D(o,argmax2,scope="MaxUnpooling4")

    o = (Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = MaxUnpooling2D(o,argmax1,scope="MaxUnpooling5")

    o = (Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(n_classes, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Activation('sigmoid')(o)
    model = Model(img_input, o)
    model.model_name = "SegNet"
    
    return model

