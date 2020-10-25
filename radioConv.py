#!/usr/bin/python3

import os
import sys
import argparse
import pdb

from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D, Conv1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
import keras.backend as K
from keras.utils import np_utils

import numpy as np
import pandas as pd

ROOT_DIR = os.getenv('ROOT_DIR')
resDir = os.path.join(ROOT_DIR, 'resDir')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def generate_default_params(dataType):
    return {
            'depth': 3,
            'optimizer': 'Adam',
            'learning_rate': 0.01,
            'act': 'selu',
            'drop_rate1': 0.3,
            'drop_rate2': 0.1,
            'drop_rate3': 0.3,
            'drop_rate4': 0.5,
            'drop_rate5': 0.5,
            'batch_size': 70,
            'data_dim': 5000,
            'epochs': 500,
            'conv1': 64,
            'conv2': 128,
            'conv3': 256,
            'conv4': 128,
            'conv5': 128,
            'pool1': 5,
            'pool2': 3,
            'pool3': 1,
            'pool4': 3,
            'pool5': 3,
            'kernel_size1': 15,
            'kernel_size2': 21,
            'kernel_size3': 15,
            'kernel_size4': 11,
            'kernel_size5': 11,
            'dense': 130,
            'dense_act': 'softsign'
            }


def baselineBlock():
    pass


def createHomegrown():
def DF(input_shape=None, emb_size=None, Classification=False):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = ELU(alpha=1.0, name='block1_adv_act1')(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2')(model)
    model = ELU(alpha=1.0, name='block1_adv_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool')(model)
    model = Dropout(0.1, name='block1_dropout')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation('relu', name='block2_act1')(model)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2')(model)
    model = Activation('relu', name='block2_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool')(model)
    model = Dropout(0.1, name='block2_dropout')(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1')(model)
    model = Activation('relu', name='block3_act1')(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2')(model)
    model = Activation('relu', name='block3_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool')(model)
    model = Dropout(0.1, name='block3_dropout')(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1')(model)
    model = Activation('relu', name='block4_act1')(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2')(model)
    model = Activation('relu', name='block4_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool')(model)

    output = Flatten()(model)

    if Classification:
        dense_layer = Dense(emb_size, name='FeaturesVec', activation='softmax')(output)
    else:
        dense_layer = Dense(emb_size, name='FeaturesVec')(output)
    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def createBaseline():
    pass


def createResnet():
    pass


def create_model(opts, NUM_CLASS):
    if 'homegrown' == opts.modelType:
        model = createHomegrown()
    elif 'baseline' == opts.modelType:
        model = createBaseline()
    elif 'resnet' == opts.modelType:
        model = createResnet()
    else:
        raise ValueError('model type {} not support yet'.format(opts.modelType))

    return model


def test(opts):
    modelTypes = ['homegrown', 'baseline', 'resnet']
    for modelType in modelTypes:
        model = create_model()
        try:
            flag = test_run(model)
        except Exception as e:
            print(e.message)

    print('all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelType', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    test(opts)
