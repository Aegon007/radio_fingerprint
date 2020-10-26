#! /usr/bin/python3

import os
import sys
import argparse
import pdb

from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import MaxPooling2D, BatchNormalization
from keras.layers import Activation, GlobalAveragePooling2D, Conv2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import EarlyStopping, Callback
import keras.backend as K
from keras.utils import np_utils

import numpy as np
import pandas as pd

import resnet50_2D

ROOT_DIR = os.getenv('ROOT_DIR')
resDir = os.path.join(ROOT_DIR, 'resDir')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def generate_default_params(dataType):
    pass


def createHomegrown(inp_shape, emb_size):
    # -----------------Entry flow -----------------
    input_data = Input(shape=inp_shape)

    filter_num = ['None', 50, 50]
    kernel_size = ['None', (2, 7), (2, 7)]
    conv_stride_size = ['None', 1, 1]
    pool_stride_size = ['None', 1, 1]
    activation_func = ['None', 'relu', 'relu']
    dense_layer_size = ['None', 256, 80]

    model = Conv2D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = Activation(activation_func[1], name='block1_act1')(model)
    model = Dropout(0.5, name='block1_dropout')(model)

    model = Conv2D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation(activation_func[2], name='block2_act1')(model)
    model = Dropout(0.5, name='block2_dropout')(model)

    output = Flatten()(model)

    dense_layer = Dense(dense_layer_size[1], name='dense1', activation='relu')(output)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation='relu')(dense_layer)
    dense_layer = Dense(emb_size, name='dense3', activation='softmax')(dense_layer)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def baselineBlock(input, block_idx):
    filter_num = ['None', 128, 128]
    kernel_size = ['None', (7, 2), (5, 2)]
    conv_stride = ['None', 1, 1]
    pool_size = ['None', (2, 2)]
    pool_stride = ['None', 1]
    act_func = 'relu'

    model = Conv2D(filters=filter_num[1], kernel_size=kernel_size[1], name='conv1_{}'.format(block_idx),
                   strides=conv_stride[1], padding='same', activation=act_func)(input)
    model = Conv2D(filters=filter_num[2], kernel_size=kernel_size[2], name='conv2_{}'.format(block_idx),
                   strides=conv_stride[2], padding='same', activation=act_func)(model)
    output = MaxPooling2D(pool_size=pool_size[1], strides=pool_stride[1], padding='same',
                         name='pool_{}'.format(block_idx))(model)

    return output


def createBaseline(inp_shape, emb_size):
    dense_layer_size = ['None', 256, 256, 128]
    act_func = ['None', 'relu', 'relu', 'relu']

    blockNum = 4
    input_data = Input(shape=inp_shape)
    model = baselineBlock(input_data, 1)
    for i in range(blockNum):
        model = baselineBlock(model, i+2)

    dense_layer = Dense(dense_layer_size[1], name='dense1', activation=act_func[1])(model)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation=act_func[2])(dense_layer)
    dense_layer = Dense(dense_layer_size[3], name='dense3', activation=act_func[3])(dense_layer)
    dense_layer = Dense(emb_size, name='dense4', activation='softmax')(dense_layer)

    conv_model = Model(inputs=input_data, outputs=dense_layer)
    return conv_model


def createResnet(inp_shape, emb_size):
    return resnet50_2D.create_model(inp_shape, emb_size)


def create_model(opts, inp_shape, NUM_CLASS):
    emb_size = NUM_CLASS
    if 'homegrown' == opts.modelType:
        model = createHomegrown(inp_shape, emb_size)
    elif 'baseline' == opts.modelType:
        model = createBaseline(inp_shape, emb_size)
    elif 'resnet' == opts.modelType:
        model = createResnet(inp_shape, emb_size)
    else:
        raise ValueError('model type {} not support yet'.format(opts.modelType))

    return model


def test_run(model):
    model.compile(optimizer='adam', loss='mse')


def test(opts):
    modelTypes = ['homegrown', 'baseline', 'resnet']
    NUM_CLASS = 10
    inp_shape = (288, 2, 1)
    signal = True
    for modelType in modelTypes:
        opts.modelType = modelType
        model = create_model(opts, inp_shape, NUM_CLASS)
        try:
            flag = test_run(model)
        except Exception as e:
            print(e)

    print('all done!') if signal else print('test failed')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modelType', default='homegrown', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    test(opts)
