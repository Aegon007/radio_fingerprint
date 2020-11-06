#!/usr/bin/python3

import os
import sys
import argparse
import pdb

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils import np_utils
import numpy as np

import radioConv
import readSigmf2 as readSigmf
import load_slice_IQ
import config

ROOT_DIR = os.getenv('ROOT_DIR')
resDir = os.path.join(ROOT_DIR, 'resDir')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def main(opts):
    # setup params
    Batch_Size = 128
    Epoch_Num = 100
    saveModelPath = os.path.join(modelDir, 'best_model_{}.h5'.format(opts.modelType))
    checkpointer = ModelCheckpoint(filepath=saveModelPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    callBackList = [checkpointer, earlyStopper]

    print('loading data...')
    x_day_dir = opts.input
    #train_x, train_y, val_x, val_y, test_x, test_y = readSigmf.getData(opts, x_day_dir)
    dataOpts = load_slice_IQ.loadDataOpts(opts.input, num_slice=100000)
    train_x, train_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts)

    print('get the model and compile it...')
    inp_shape = (train_x.shape[1], train_x.shape[2])
    model = radioConv.create_model(opts, inp_shape, NUM_CLASS, channel='last')
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    print('fit the model with data...')
    model.fit(x=train_x, y=train_y,
              batch_size=Batch_Size,
              epochs=Epoch_Num,
              verbose=opts.verbose,
              callbacks=callBackList,
              validation_split=0.1,
              shuffle=True)

    print('test the trained model...')
    score, acc = model.evaluate(test_x, test_y, batch_size=Batch_Size, verbose=0)
    print('test acc is: ', acc)

    print('all test done!')


if __name__ == "__main__":
    opts = config.parse_args(sys.argv)
    main(opts)
