import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from keras.models import Model
from keras_radam import RAdam
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model


# address some interface discrepancies when using tensorflow.keras
# hack to use the load_from_json in tf otherwise we get an exception
# adapted/modified hack from here:
# https://github.com/keras-team/keras-contrib/issues/488
if "slice" not in keras.backend.__dict__:
    # this is a good indicator that we are using tensorflow.keras
    print('using tensorflow, need to monkey patch')
    try:
        # at first try to monkey patch what we need

        try:
            tf.python.keras.backend.__dict__.update(
                is_tensor=tf.is_tensor,
                slice=tf.slice,
            )
        finally:
            print('tf.python.backend.slice overwritten by monkey patch')
    except Exception:
        print('monkey patch failed, override methods')
        # if that doesn't work we do a dirty copy of the code required
        import tensorflow as tf
        from tensorflow.python.framework import ops as tf_ops


        def is_tensor(x):
            return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)


        def slice(x, start, size):
            x_shape = keras.int_shape(x)
            if (x_shape is not None) and (x_shape[0] is not None):
                len_start = keras.int_shape(start)[0] if is_tensor(start) else len(start)
                len_size = keras.int_shape(size)[0] if is_tensor(size) else len(size)
                if not (len(keras.int_shape(x)) == len_start == len_size):
                    raise ValueError('The dimension and the size of indices should match.')
            return tf.slice(x, start, size)


def conv_layer(inputs, filters=16, f_size=(3, 3, 3), activation='elu', batch_norm=True, kernel_init='he_normal',
               pad='same', bn_first=False, ndims=2):
    """
    Wrapper for a 2/3D-conv layer + batchnormalisation
    Either with Conv,BN,activation or Conv,activation,BN

    :param inputs: numpy or tensor object batchsize,z,x,y,channels
    :param filters: int, number of filters
    :param f_size: tuple of int, filterzise per axis
    :param activation: string, which activation function should be used
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras conv block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[:ndims]

    if bn_first:

        conv1 = Conv(filters, f_size, kernel_initializer=kernel_init, padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1
        conv1 = Activation(activation)(conv1)

    else:
        conv1 = Conv(filters, f_size, activation=activation, kernel_initializer=kernel_init, padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1

    return conv1


def downsampling_block(inputs, filters=16, f_size=(3, 3, 3), activation='elu', drop=0.3, batch_norm=True,
                       kernel_init='he_normal', pad='same', m_pool=(2, 2), bn_first=False, ndims=2):
    """
    Create an 2D/3D-downsampling block for the u-net architecture
    :param inputs: numpy or tensor input with batchsize,z,x,y,channels
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    """
    m_pool = m_pool[:ndims]
    pool = getattr(kl, 'MaxPooling{}D'.format(ndims))

    conv1 = conv_layer(inputs=inputs, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                       kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    conv1 = Dropout(drop)(conv1)
    conv1 = conv_layer(inputs=conv1, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                       kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    p1 = pool(m_pool)(conv1)

    return (conv1, p1)


def upsampling_block(lower_input, conv_input, use_upsample=True, filters=16, f_size=(3, 3, 3), activation='elu',
                     drop=0.3, batch_norm=True, kernel_init='he_normal', pad='same', up_size=(2, 2), bn_first=False,
                     ndims=2):
    """
    Create an upsampling block for the u-net architecture
    Each blocks consists of these layers: upsampling/transpose,concat,conv,dropout,conv
    Either with "upsampling,conv" or "transpose" upsampling
    :param lower_input: numpy input from the lower block: batchsize,z,x,y,channels
    :param conv_input: numpy input from the skip layers: batchsize,z,x,y,channels
    :param use_upsample: bool, whether to use upsampling or not
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[:ndims]

    # use upsample&conv or transpose layer
    if use_upsample:
        UpSampling = getattr(kl, 'UpSampling{}D'.format(ndims))
        deconv1 = UpSampling(size=up_size)(lower_input)
        deconv1 = Conv(filters, f_size, padding=pad, kernel_initializer=kernel_init, activation=activation)(deconv1)
    else:
        ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(ndims))
        deconv1 = ConvTranspose(filters, up_size, strides=up_size, padding=pad, kernel_initializer=kernel_init,
                                activation=activation)(lower_input)

    deconv1 = concatenate([deconv1, conv_input])

    deconv1 = conv_layer(inputs=deconv1, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                         kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    deconv1 = Dropout(drop)(deconv1)
    deconv1 = conv_layer(inputs=deconv1, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                         kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)

    return deconv1


# Build U-Net model
def create_unet(config, metrics=None):
    """
    create a 2D/3D u-net for image segmentation
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :returns compiled keras model
    """

    inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))

    # define standard values according to convention over configuration paradigm
    metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
    activation = config.get('ACTIVATION', 'elu')
    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
    batch_norm = config.get('BATCH_NORMALISATION', False)
    use_upsample = config.get('USE_UPSAMPLE', 'False')  # use upsampling + conv3D or transpose layer
    gpu_ids = config.get('GPU_IDS', '1').split(',') # used in previous tf versions
    pad = config.get('PAD', 'same')
    kernel_init = config.get('KERNEL_INIT', 'he_normal')
    mask_channels = config.get("MAKS_Values", 4)
    m_pool = config.get('M_POOL', (2, 2, 2))
    f_size = config.get('F_SIZE', (3, 3, 3))
    filters = config.get('FILTERS', 16)
    drop_1 = config.get('DROPOUT_L1_L2', 0.3)
    drop_3 = config.get('DROPOUT_L5', 0.5)
    bn_first = config.get('BN_FIRST', False)
    ndims = len(config.get('DIM', [224, 224]))
    depth = config.get('DEPTH', 4)
    # define two layers for the middle part and the final  by  layer
    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    one_by_one = (1, 1, 1)[:ndims]

    encoder = list()
    decoder = list()
    # increase the dropout through the layer depth
    dropouts = list(np.linspace(drop_1, drop_3, depth))

    # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
    strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # strategy for multi-GPU usage not necessary for single GPU usage
    with strategy.scope():
        # build the encoder
        for l in range(depth):

            if len(encoder) == 0:
                # first block
                input_tensor = inputs
            else:
                # all other blocks, use the max-pooled output of the previous encoder block
                # remember the max-pooled output from the previous layer
                input_tensor = encoder[-1][1]
            encoder.append(
                downsampling_block(inputs=input_tensor,
                                   filters=filters,
                                   f_size=f_size,
                                   activation=activation,
                                   drop=dropouts[l],
                                   batch_norm=batch_norm,
                                   kernel_init=kernel_init,
                                   pad=pad,
                                   m_pool=m_pool,
                                   bn_first=bn_first,
                                   ndims=ndims))
            filters *= 2

        # middle part
        input_tensor = encoder[-1][1]
        fully = conv_layer(inputs=input_tensor, filters=filters, f_size=f_size,
                           activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                           pad=pad, bn_first=bn_first, ndims=ndims)
        fully = Dropout(drop_3)(fully)
        fully = conv_layer(inputs=fully, filters=filters, f_size=f_size,
                           activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                           pad=pad, bn_first=bn_first, ndims=ndims)

        # build the decoder
        decoder.append(fully)
        for l in range(depth):
            # take the output of the previous decoder block and the output of the corresponding
            # encoder block
            input_lower = decoder[-1]
            input_skip = encoder.pop()[0]
            filters //= 2
            decoder.append(
                upsampling_block(lower_input=input_lower,
                                 conv_input=input_skip,
                                 use_upsample=use_upsample,
                                 filters=filters,
                                 f_size=f_size,
                                 activation=activation,
                                 drop=dropouts.pop(),
                                 batch_norm=batch_norm,
                                 up_size=m_pool,
                                 bn_first=bn_first,
                                 ndims=ndims))

        outputs = Conv(config.get('MASK_CLASSES', mask_channels), one_by_one, activation='softmax')(decoder[-1])
        outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs)

        print('Outputs dtype: %s' % outputs.dtype.name)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=get_optimizer(config), loss=loss_f, metrics=metrics)

    return model


def create_unet_new(config, metrics=None):
    """
    Does not work so far!!!
    :param config:
    :param metrics:
    :return:
    """
    unet = Unet(config, metrics)
    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
    unet.compile(optimizer=get_optimizer(config), loss=loss_f, metrics=metrics)
    input_shape = config.get('DIM', [224, 224])
    unet.build(input_shape)
    return unet


class Unet(tf.keras.Model):

    def __init__(self, config, metrics=None):
        name = "Unet"
        super(Unet, self).__init__(name=name)

        self.inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))

        self.config = config
        # self.metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        self.activation = config.get('ACTIVATION', 'elu')
        self.loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        self.batch_norm = config.get('BATCH_NORMALISATION', False)
        self.use_upsample = config.get('USE_UPSAMPLE', 'False')  # use upsampling + conv3D or transpose layer
        self.gpu_ids = config.get('GPU_IDS', '1').split(',')
        self.pad = config.get('PAD', 'same')
        self.kernel_init = config.get('KERNEL_INIT', 'he_normal')
        self.m_pool = config.get('M_POOL', (2, 2, 2))
        self.f_size = config.get('F_SIZE', (3, 3, 3))
        self.filters = config.get('FILTERS', 16)
        self.drop_1 = config.get('DROPOUT_L1_L2', 0.3)
        self.drop_2 = config.get('DROPOUT_L3_L4', 0.4)
        self.drop_3 = config.get('DROPOUT_L5', 0.5)
        self.bn_first = config.get('BN_FIRST', False)
        self.ndims = len(config.get('DIM', [224, 224]))
        self.depth = config.get('DEPTH', 4)
        self.Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        self.one_by_one = (1, 1, 1)[:self.ndims]

        self.encoder = list()
        self.decoder = list()
        # increase the dropout through the layer depth
        self.dropouts = list(np.linspace(self.drop_1, self.drop_3, self.depth))

        self.downsampling = downsampling_block
        self.upsampling = upsampling_block

    def __call__(self, inputs):
        strategy = tf.distribute.MirroredStrategy(devices=self.config.get('GPUS', ["/gpu:0"]))
        tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # build the encoder
            for l in range(self.depth):

                if len(self.encoder) == 0:
                    input_tensor = inputs
                else:
                    # take the max-pooled output from the previous layer
                    input_tensor = self.encoder[-1][1]
                self.encoder.append(
                    downsampling_block(inputs=input_tensor, filters=self.filters, f_size=self.f_size,
                                       activation=self.activation,
                                       drop=self.dropouts[l],
                                       batch_norm=self.batch_norm, kernel_init=self.kernel_init, pad=self.pad,
                                       m_pool=self.m_pool,
                                       bn_first=self.bn_first,
                                       ndims=self.ndims))
                self.filters *= 2

            # middle part of the U-net
            input_tensor = self.encoder[-1][1]
            fully = conv_layer(inputs=input_tensor, filters=self.filters, f_size=self.f_size,
                               activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)
            fully = Dropout(self.drop_3)(fully)
            fully = conv_layer(inputs=fully, filters=self.filters, f_size=self.f_size, activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

            # build the decoder
            self.decoder.append(fully)
            for l in range(self.depth):
                input_lower = self.decoder[-1]
                input_skip = self.encoder.pop()[0]
                self.filters //= 2
                self.decoder.append(
                    upsampling_block(lower_input=input_lower, conv_input=input_skip, use_upsample=self.use_upsample,
                                     filters=self.filters,
                                     f_size=self.f_size, activation=self.activation, drop=self.dropouts.pop(),
                                     batch_norm=self.batch_norm,
                                     up_size=self.m_pool, bn_first=self.bn_first, ndims=self.ndims))

            outputs = self.Conv(self.config.get('MASK_CLASSES', 4), self.one_by_one, activation='softmax')(
                self.decoder[-1])

            return outputs


def get_optimizer(config):
    """
    Returns a keras.optimizer
    default is an Adam optimizer
    :param config: Key, value dict, Keys in upper letters
    :return: tf.keras.optimizer
    """

    opt = config.get('OPTIMIZER', 'Adam')
    lr = config.get('LEARNING_RATE', 0.001)
    ep = config.get('EPSILON', 1e-08)
    de = config.get('DECAY', 0.0)

    optimizer = None

    if opt == 'Adagrad':
        optimizer = keras.optimizers.Adagrad(lr=lr, epsilon=ep, decay=de)
    elif opt == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=ep, decay=de)
    elif opt == 'Adadelta':
        optimizer = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=ep, decay=de)
    elif opt == 'Radam':
        optimizer = RAdam()
    elif opt == 'Adam':
        optimizer = keras.optimizers.Adam(lr=lr)
    else:
        optimizer = keras.optimizers.Adam(lr=lr)

    logging.info('Optimizer: {}'.format(opt))
    return optimizer


def get_model(config=dict(), metrics=None):
    """
    create a new model or load a pre-trained model
    :param config: json file
    :param metrics: list of tensorflow or keras metrics with gt,pred
    :return: returns a compiled keras model
    """

    # load a pre-trained model with config
    if config.get('LOAD', False):
        return load_pretrained_model(config, metrics)

    # create a new 2D or 3D model with given config params
    return create_unet(config, metrics)


def load_pretrained_model(config=None, metrics=None, comp=True, multigpu=False):
    """
    Load a pre-trained keras model
    for a given model.json file and the weights as h5 file

    :param config: dict
    :param metrics: keras or tensorflow loss function in a list
    :param comp: bool, compile the model or not
    :multigpu: wrap model in multi gpu wrapper
    :return:
    """

    if config is None:
        config = {}
    if metrics is None:
        metrics = [keras.metrics.binary_accuracy]

    gpu_ids = config.get('GPU_IDS', '1').split(',')

    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)

    # load model
    logging.info('loading model from: {} .'.format(os.path.join(config.get('MODEL_PATH', './'), 'model.json')))
    json = open(os.path.join(config.get('MODEL_PATH', './'), 'model.json')).read()
    model = tf.keras.models.model_from_json(json)
    #model = model_from_json(open(os.path.join(config.get('MODEL_PATH', './'), 'model.json')).read())
    logging.info('loading model description')
    try:
        model.load_weights(os.path.join(config.get('MODEL_PATH', './'), 'checkpoint.h5'))
        # make sure to work with wrapped multi-gpu models
        if multigpu:
            logging.info('multi GPU model, try to unpack the model and load weights again')
            model = model.layers[-2]
            model = multi_gpu_model(model, gpus=len(gpu_ids), cpu_merge=False) if (len(gpu_ids) > 1) else model
    except Exception as e:
        # some models are wrapped two times into a keras multi-gpu model, so we need to unpack it - hack
        logging.info(str(e))
        logging.info('multi GPU model, try to unpack the model and load weights again')
        model = model.layers[-2]
        model.load_weights(os.path.join(config.get('MODEL_PATH', './'), 'checkpoint.h5'))
    logging.info('loading model weights')

    if comp:
        try:
            # try to compile with given params, else use fallback parameters
            model.compile(optimizer=get_optimizer(config), loss=loss_f, metrics=metrics)

        except Exception as e:
            logging.error('Failed to compile with given parameters, use default vaules: {}'.format(str(e)))
            model.compile(optimizer='adam', loss=loss_f, metrics=metrics)
    logging.info('model {} loaded'.format(os.path.join(config.get('MODEL_PATH', './'), 'model.json')))
    return model


def test_unet():
    """
    Create a keras unet with a pre-configured config
    :return: prints model summary file
    """
    try:
        from src.utils.utils_io import Console_and_file_logger
        Console_and_file_logger('test 2d network')
    except Exception as e:
        print("no logger defined, use print")

    config = {'GPU_IDS': '0', 'GPUS': ['/gpu:0'], 'EXPERIMENT': '2D/tf2/temp', 'ARCHITECTURE': '2D',
              'DIM': [224, 224], 'DEPTH': 4, 'SPACING': [1.0, 1.0], 'M_POOL': [2, 2], 'F_SIZE': [3, 3],
              'IMG_CHANNELS': 1, 'MASK_VALUES': [0, 1, 2, 3], 'MASK_CLASSES': 4, 'AUGMENT': False, 'SHUFFLE': True,
              'AUGMENT_GRID': True, 'RESAMPLE': False, 'DATASET': 'GCN_2nd', 'TRAIN_PATH': 'data/raw/GCN_2nd/2D/train/',
              'VAL_PATH': 'data/raw/GCN_2nd/2D/val/', 'TEST_PATH': 'data/raw/GCN_2nd/2D/val/',
              'DF_DATA_PATH': 'data/raw/GCN_2nd/2D/df_kfold.csv', 'MODEL_PATH': 'models/2D/tf2/gcn/2020-03-26_17_25',
              'TENSORBOARD_LOG_DIR': 'reports/tensorboard_logs/2D/tf2/gcn/2020-03-26_17_25',
              'CONFIG_PATH': 'reports/configs/2D/tf2/gcn/2020-03-26_17_25',
              'HISTORY_PATH': 'reports/history/2D/tf2/gcn/2020-03-26_17_25', 'GENERATOR_WORKER': 32, 'BATCHSIZE': 32,
              'INITIAL_EPOCH': 0, 'EPOCHS': 150, 'EPOCHS_BETWEEN_CHECKPOINTS': 5, 'MONITOR_FUNCTION': 'val_loss',
              'MONITOR_MODE': 'min', 'SAVE_MODEL_FUNCTION': 'val_loss', 'SAVE_MODEL_MODE': 'min', 'BN_FIRST': False,
              'OPTIMIZER': 'Adam', 'ACTIVATION': 'elu', 'LEARNING_RATE': 0.001, 'DECAY_FACTOR': 0.5, 'MIN_LR': 1e-10,
              'DROPOUT_L1_L2': 0.3, 'DROPOUT_L3_L4': 0.4, 'DROPOUT_L5': 0.5, 'BATCH_NORMALISATION': True,
              'USE_UPSAMPLE': True, 'LOSS_FUNCTION': keras.losses.binary_crossentropy}
    metrics = [tf.keras.losses.categorical_crossentropy]

    model = get_model(config, metrics)
    model.summary()


if __name__ == '__main__':
    test_unet()
