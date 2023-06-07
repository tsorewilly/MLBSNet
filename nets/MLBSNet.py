from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow
import tensorflow as tf
from nets.Repconvnet import Rep_get_convnet_encoder
#from nets.convnet import get_convnet_encoder
# from nets.convnet import get_convnet_encoder2,get_convnet_encoder
# from nets.resnet50 import get_resnet50_encoder
#from nets.Xception import Xception
from nets.MLBDNet import MLBDNet
# from nets.resnet50 import get_resnet50_encoder
#from nets.Repconvnet import Rep_get_convnet_encoder
from tensorflow.keras import backend as K

IMAGE_ORDERING = 'channels_last'
from tensorflow.keras.regularizers import l2

def one_side_pad( x ):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x

def one_side_del( x ):
    #x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x

def relu6(x):
    # relu function
    return K.relu(x, max_value=6.0)


def hard_swish(x):
    # Use relu function to multiply 'x' to simulate sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def squeeze(inputs):
    # Attention Mechanism Unit
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels / 4))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x

def JSunet_decoder(f, n_classes):
#    assert n_up >= 2
    [f1, f2, f3, f4, f5] = f
    o = f5
    # 26,26,512
    #o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    #o = (Conv2D(512, (3, 3), padding='valid', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (Conv2D(512, (3, 3), padding='same', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,256
    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    # o = bicubic_upsampling(o)
    o = (concatenate([o, f4], axis=-1))

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # o = (Conv2D(256, (3, 3), padding='valid', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (Conv2D(256, (3, 3), padding='same', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    # for _ in range(n_up - 2):
    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
#    o = (concatenate([one_side_pad( o ), f3], axis=-1))
    o = (concatenate([ o , f3], axis=-1))

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # o = (Conv2D(128, (3, 3), padding='valid',activation='relu', data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (Conv2D(128, (3, 3), padding='same', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=-1))

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # o = (Conv2D(64, (3, 3), padding='valid',activation='relu', data_format=IMAGE_ORDERING))(o) #改 #activation='relu',
    o = (Conv2D(64, (3, 3), padding='same', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/2
    # 208,208,64

    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f1], axis=-1))
    # o = bicubic_upsampling(o)
    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (Conv2D(64, (3, 3), padding='same', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 此时输出为h_input/2,w_input/2,nclasses
    # 208,208,2
    o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o) #activation='relu',

    # o = Conv2D(n_classes, 1, activation="softmax")(o)
    return o

def centre_crop(layer, target_size):
    # print(K.int_shape(layer))
    # _,  layer_height, layer_width ,_ = K.int_shape(layer)
    #print(layer.shape)
    _, layer_height, layer_width, _ = layer.shape
    #print(layer_height, layer_width)
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    #(diff_y, diff_x)

    tensorflow.keras_layer = Lambda(lambda x: x[:, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1]), :])
    res_get = tensorflow.keras_layer(layer)
    return res_get


def unet_decoder(f, n_classes, n_up=3):
    weight_decay = 1E-4
    MERGE_AXIS = -1
    assert n_up >= 2
#    initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    [f1, f2, f3, f4, f5] = f
    # 28,28,1024 -> 56,56,512
    o = f5
    #print(f4.shape)

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)

    o = (Conv2D(512, (3, 3),  padding='same', data_format=IMAGE_ORDERING))(o) #无
    o = (BatchNormalization())(o)

    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    # 56,56,512 -> 56,56,1024
    crop1 = centre_crop(f4, o.shape[1:3])
    o = (concatenate([o, crop1], axis=MERGE_AXIS))
    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # 52,52,256
    o = (Conv2D(512, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o) #无
    # 56,56,512 -> 56,56,1024
    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o) #无

    o = (Conv2D(256, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)  #无
    o = (BatchNormalization())(o)
    # 104,104,256
    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    o = one_side_del(o) #104->103
    # 104,104,384
    crop2 = centre_crop(f3, o.shape[1:3])
    o = (concatenate([o, crop2], axis=MERGE_AXIS))

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # 104,104,128
    o = (Conv2D(256, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o) #无
    # 208,208,128
    o = (Conv2D(256, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o) #无

    o = (Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o) #无
    o = (BatchNormalization())(o)

    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)

    crop3 = centre_crop(f2, o.shape[1:3])
    o = (concatenate([o, crop3], axis=MERGE_AXIS))

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)

    o = (Conv2D(128, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(128, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    #o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)

    crop4 = centre_crop(f1, o.shape[1:3])
    o = (concatenate([o, crop4], axis=MERGE_AXIS))

    o = (Conv2D(64, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(64, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING)(o)
    # 将结果进行reshape
    # o = Reshape((o.shape[1] * o.shape[2], -1))(o)
    #
    # o = Softmax()(o)
    # model = Model(img_input, o)

    return o

def resnet_decoder(f, n_classes):
#    assert n_up >= 2
    [f1, f2, f3, f4, f5] = f
    o = f5
    # 26,26,512
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,256
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    # o = bicubic_upsampling(o)
#    o = (concatenate([one_side_del(o), f4], axis=-1))
    o = (concatenate([o, f4], axis=-1))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    # for _ in range(n_up - 2):
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    #o = (concatenate([ one_side_del(o) , f3], axis=-1))
    o = (concatenate([ o , f3], axis=-1))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid',activation='relu', data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=-1))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid',activation='relu', data_format=IMAGE_ORDERING))(o) #改 #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/2
    # 208,208,64

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f1], axis=-1))
    # o = bicubic_upsampling(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid',activation='relu', data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 此时输出为h_input/2,w_input/2,nclasses
    # 208,208,2
    o = Conv2D(n_classes, (3, 3), padding='same', activation='relu',data_format=IMAGE_ORDERING)(o) #activation='relu',

    # o = Conv2D(n_classes, 1, activation="softmax")(o)
    return o

def pyramid_decoder(f, n_classes):
    [f1, f2, f3, f4, f5] = f
    o = f5
    #o1 = Lambda(lambda xx: tf.image.resize(o, f4.shape[1:3]))(o)
    o1 = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    o1 = (concatenate([o1, f4], axis=-1))
    o1 = squeeze(o1)
    o1 = Conv2D(128, (1, 1), activation='relu',padding='same')(o1) #o1 = Conv2D(128, (1, 1), padding='same')(o1)
    o1 = (BatchNormalization())(o1)
    o1 = Dropout(0.5)(o1)
    #o1 = Lambda(lambda xx: tf.image.resize(o1, f1.shape[1:3]))(o1)
    o1 = (UpSampling2D((8, 8), interpolation='bilinear', data_format=IMAGE_ORDERING))(o1)

    #o2 = Lambda(lambda xx: tf.image.resize(o, f3.shape[1:3]))(o)
    o2 = (UpSampling2D((4, 4), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    o2 = (concatenate([o2, f3], axis=-1))
    o2 = squeeze(o2)
    o2 = Conv2D(128, (1, 1), activation='relu',padding='same')(o2)
    o2 = (BatchNormalization())(o2)
    o2 = Dropout(0.5)(o2)
    #o2 = Lambda(lambda xx: tf.image.resize(o2, f1.shape[1:3]))(o2)
    o2 = (UpSampling2D((4, 4), interpolation='bilinear', data_format=IMAGE_ORDERING))(o2)

    #o3 = Lambda(lambda xx: tf.image.resize(o, f2.shape[1:3]))(o)
    o3 = (UpSampling2D((8, 8), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    o3 = (concatenate([o3, f2], axis=-1))
    o3 = squeeze(o3)
    o3 = Conv2D(128, (1, 1), activation='relu',padding='same')(o3)
    o3 = (BatchNormalization())(o3)
    o3 = Dropout(0.5)(o3)
    #o3 = Lambda(lambda xx: tf.image.resize(o3, f1.shape[1:3]))(o3)
    o3 = (UpSampling2D((2, 2), interpolation='bilinear', data_format=IMAGE_ORDERING))(o3)

    #o4 = Lambda(lambda xx: tf.image.resize(o, f1.shape[1:3]))(o)
    o4 = (UpSampling2D((16, 16), interpolation='bilinear', data_format=IMAGE_ORDERING))(o)
    o4 = (concatenate([o4, f1], axis=-1))
    o4 = squeeze(o4)
    o4 = Conv2D(128, (1, 1), activation='relu',padding='same')(o4)
    o4 = (BatchNormalization())(o4)
    o4 = Dropout(0.5)(o4)
    x = (concatenate([o1, o2, o3, o4], axis=-1))
    #x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(n_classes, (3, 3), padding='same')(x)
    return x


def _pyramid(n_classes, encoder, input_height = 416, input_width = 416, 
             encoder_level = 3, deploy = True, aux_branch = True):
    input_shape=(input_height, input_width, 1)
    img_input = Input(shape=input_shape)
    img_input, levels = encoder(img_input, alpha=1., OS=16, deploy=deploy)
    
    feat = levels
    #o = unet_decoder(feat, n_classes, n_up=3)
    #o = JSunet_decoder(feat, n_classes)
    #o = resnet_decoder(feat, n_classes)
    o = pyramid_decoder(feat, n_classes) # Returns output of a Conv2D layer with same padding
    o = Activation('sigmoid',name="main")(o)
    #o = Activation('sigmoid',name="main")(o)
    if aux_branch:
        # f2 = feat[1]
        # f3 = Conv2D(128, (3,3), padding='same', use_bias=False)(f2)
        # f3 = BatchNormalization()(f3)
        # f3 = Activation('relu')(f3)
        # f3 = Dropout(0.1)(f3)
        #
        # f3 = UpSampling2D((4,4),interpolation='bilinear')(f3)
        # f3 = Conv2D(n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same')(f3)
        # f3 = Activation("sigmoid", name="aux1")(f3)

        f3 = feat[2]
        f3 = Conv2D(128, (3,3), padding='same', use_bias=False)(f3)
        f3 = BatchNormalization()(f3)
        f3 = Activation('relu')(f3)
        f3 = Dropout(0.1)(f3)

        f3 = UpSampling2D((4,4),interpolation='bilinear')(f3)
        f3 = Conv2D(n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same')(f3)
        f3 = Activation("sigmoid", name="aux1")(f3)

        f4 = feat[3]
        f4 = Conv2D(128, (3,3), padding='same', use_bias=False)(f4)
        f4 = BatchNormalization()(f4)
        f4 = Activation('relu')(f4)
        f4 = Dropout(0.1)(f4)

        f4 = UpSampling2D((8,8),interpolation='bilinear')(f4)
        f4 = Conv2D(n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same')(f4)
        f4 = Activation("sigmoid", name="aux2")(f4)
        model = Model(img_input,[f3,f4,o])

    else:
        model = Model(img_input, o)

    return model


def MLBSNet(n_classes, input_height=224, input_width=224, encoder_level=3,deploy=False,aux_branch=True):
    model = _pyramid(n_classes, MLBDNet, input_height=input_height, input_width=input_width,
                    encoder_level=encoder_level,deploy=deploy,aux_branch=aux_branch)
    model.model_name = "MLBSNet"
    return model
