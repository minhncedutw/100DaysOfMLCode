import tensorflow as tf
from keras.losses import categorical_crossentropy
def pointnet_loss(xtran, reg_weight=0.001):
    """
    Description:
    :param NAME: TYPE, MEAN
    :return: TYPE, MEAN
    Implemented from: https://github.com/charlesq34/pointnet/blob/master/part_seg/pointnet_part_seg.py
        Source to reference more: https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py; https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_segmentation.py
    Usage: model.compile(loss=[pointnet_loss(xtran=learner.model.get_layer(name='xtran2'), reg_weight=0.001)], metrics=["accuracy"], optimizer=adam)
    """
    def pointnet_loss_fixed(y_true, y_pred):
        loss = categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        seg_loss = tf.reduce_mean(loss)

        xtran_shape = xtran.output_shape[1]
        # mat_diff = Lambda(matmul, output_shape=matmul_out_shape)([xtran.output, K.permute_dimensions(xtran.output, pattern=(0, 2, 1))])
        # mat_diff -= K.constant(np.eye(xtran_shape), dtype='float32')
        mat_diff = tf.matmul(xtran.output, tf.transpose(xtran.output, perm=[0, 2, 1])) - tf.constant(np.eye(xtran_shape), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)

        return seg_loss + mat_diff_loss * reg_weight

    return pointnet_loss_fixed