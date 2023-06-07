from tensorflow.keras.models import *
from tensorflow.keras.layers import *
#from nets.convnet import get_convnet_encoder
#from nets.convnet import get_convnet_encoder
from nets.resnet50 import get_resnet50_encoder
IMAGE_ORDERING = 'channels_last'
from tensorflow.keras.regularizers import l2
# import sys
# sys.setrecursionlimit(100000) #例如这里设置为十万

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

def segnet_decoder(f, n_classes):
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
    o = (concatenate([o, f4], axis=-1))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    # for _ in range(n_up - 2):
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([one_side_pad( o ), f3], axis=-1))
    #o = (concatenate([ o , f3], axis=-1))

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

def centre_crop(layer, target_size):
    # print(K.int_shape(layer))
    # _,  layer_height, layer_width ,_ = K.int_shape(layer)
    #print(layer.shape)
    _, layer_height, layer_width, _ = layer.shape
    #print(layer_height, layer_width)
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    #(diff_y, diff_x)

    keras_layer = Lambda(lambda x: x[:, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1]), :])
    res_get = keras_layer(layer)
    return res_get


def unet_decoder(f, n_classes, n_up=3):
    weight_decay = 1E-4
    MERGE_AXIS = -1
    assert n_up >= 2
    #initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    [f1, f2, f3, f4, f5] = f
    # 28,28,1024 -> 56,56,512
    o = f5
    #print(f4.shape)

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3),  padding='same', data_format=IMAGE_ORDERING))(o) #无
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
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
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
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

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    crop3 = centre_crop(f2, o.shape[1:3])
    o = (concatenate([o, crop3], axis=MERGE_AXIS))

    # o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)

    o = (Conv2D(128, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = (Conv2D(128, (3, 3), activation='relu', padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

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

def resnet_decoder(n_classes,input_height,  input_width):
#    assert n_up >= 2
    #img_input = (512, 512, 1)
    img_input,[f1, f2, f3, f4, f5] = get_resnet50_encoder(input_height,  input_width )
    o = f5

    # 26,26,512
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,256
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    print(one_side_del(o))
    # o = bicubic_upsampling(o)
    o = (concatenate([o, f4], axis=-1))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu',data_format=IMAGE_ORDERING))(o) #activation='relu',
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    # for _ in range(n_up - 2):
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o , f3], axis=-1))
    #o = (concatenate([ o , f3], axis=-1))

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
    o = Conv2D(16, (3, 3), padding='same', activation='relu',data_format=IMAGE_ORDERING)(o) #activation='relu',
    o = Conv2D(n_classes, 1, activation="sigmoid")(o)
    #o = Activation('sigmoid',name="main")(o)
    model = Model(img_input, o)

    return model

def _segnet(n_classes, encoder, input_height=416, input_width=416, encoder_level=3):
    # encoder通过主干网络
    img_input, levels = encoder(input_height=input_height, input_width=input_width)

    # 获取hw压缩四次后的结果
    feat = levels

    # 将特征传入segnet网络
    o = unet_decoder(feat, n_classes, n_up=3)
    #o = segnet_decoder(feat, n_classes)
    #o = resnet_decoder(feat, n_classes)

    # 将结果进行reshape
    o = Reshape((o.shape[1] * o.shape[2], -1))(o)
    o = Softmax()(o)
    model = Model(img_input, o)

    return model


def convnet_unet(n_classes, input_height=224, input_width=224, encoder_level=3):
    model = resnet_decoder(get_resnet50_encoder, n_classes)
    #model.model_name = "convnet_segnet"
    return model
