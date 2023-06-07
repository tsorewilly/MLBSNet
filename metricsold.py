from tensorflow.keras import backend
import tensorflow as tf
def Iou_score(smooth = 1e-5, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())
        intersection = backend.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        union = backend.sum(y_true[...,:-1] + y_pred, axis=[0,1,2]) - intersection
        
        score = (intersection + smooth) / (union + smooth)
        # score = tf.reduce_sum(score)/tf.reduce_sum(weights)
        return score
    return _Iou_score

def f_score(beta=1, smooth = 1e-5, threhold = 0.5):
    def _f_score(y_true, y_pred):
        y_pred = backend.greater(y_pred, threhold) #逐个比对y_pred > threhold的真值,返回布尔张量
        #print(y_pred)
        y_pred = backend.cast(y_pred, backend.floatx()) #执行张量数据类型转换,返回默认浮点类型float32,小于0.5为False(0),大于0.5为True(1)
        #print(y_pred)
        #print(y_pred.shape)
        tp = backend.sum(y_true * y_pred)
        #print(tp.shape)
        fp = backend.sum(y_pred)-tp
        fn = backend.sum(y_true)-tp
        # tp = backend.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        # fp = backend.sum(y_pred         , axis=[0,1,2]) - tp
        # fn = backend.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        # score = tf.reduce_sum(score)/tf.reduce_sum(weights)
        return score
    return _f_score