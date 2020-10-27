#! /usr/bin python3.6
import os
import sys
import pdb
import argparse
import random

import json
import numpy as np
import itertools
from sigmf import SigMFFile, sigmffile


def loadData(metafile, binfile):
    # Load a dataset
    with open(metafile, 'r') as f:
        metadata = json.loads(f.read())
    signal = SigMFFile(metadata=metadata['_metadata'], data_file=binfile)
    return signal


def get_one_sample(raw_data, start, length):
    if not isinstance(raw_data, np.ndarray):
        raise
    return raw_data[start: start+length]


def convert2IQdata(one_raw_data):
    rtnList = []
    for item in one_raw_data:
        tmp = [np.real(item), np.imag(item)]
        rtnList.append(tmp)

    rtnMat = np.array(rtnList)
    return rtnMat


def divideIntoChucks(raw_data, chuckNum):
    dataLen = len(raw_data)
    sliceLen = dataLen // chuckNum

    chuckList = []
    start = 0
    for i in range(chuckNum):
        end = start + sliceLen
        oneSlice = raw_data[start: end]
        start = end
        chuckList.append(oneSlice)

    return chuckList


def formInpData(raw_data, sample_length, selectedNum):
    print('raw data length is: ', len(raw_data))
    start_range = len(raw_data) - sample_length
    print(start_range)
    raw_samples = []
    for i in range(start_range):
        tmp_sample = get_one_sample(raw_data, i, sample_length)
        raw_samples.append(tmp_sample)

    selectedSamples = random.sample(raw_samples, selectedNum)
    rtn_samples = []
    for tmp_sample in selectedSamples:
        tmp_sample = convert2IQdata(tmp_sample)
        rtn_samples.append(tmp_sample)
    rtn_samples = np.array(rtn_samples)
    return rtn_samples


def splitData(allData, splitRatio):
    np.random.seed(42)
    allDataSize = len(allData)
    shuffled_indices = np.random.permutation(len(allData))
    allData = np.array(allData)

    train_set_size = int(allDataSize * splitRatio['train'])
    val_set_size = int(allDataSize * splitRatio['val'])
    test_set_size = int(allDataSize * splitRatio['test'])

    start, end = 0, train_set_size
    train_indices = shuffled_indices[start: end]

    start, end = train_set_size, train_set_size + val_set_size
    val_indices = shuffled_indices[start: end]

    start, end = train_set_size + val_set_size, train_set_size + val_set_size + test_set_size
    test_indices = shuffled_indices[start: end]

    return allData[train_indices], allData[val_indices], allData[test_indices]


def splitChuckData(opts, chuckList, splitRatio, sample_length, selectedNum, label):
    if opts.splitType == 'random':
        allData = []
        for chuck in chuckList:
            oneChuck = formInpData(chuck, sample_length, selectedNum)
            allData.append(oneChuck)

        trainData, valData, testData = splitData(allData, splitRatio)

        trainLabels = np.ones(trainData.shape[1]) * label
        valLabels = np.ones(valData.shape[1]) * label
        testLabels = np.ones(testData.shape[1]) * label

    elif opts.splitType == 'order':
        pass
    else:
        raise

    return trainData, trainLabels, valData, valLabels, testData, testLabels


def get_samples(opts, signal, label):
    # Get some metadata and all annotations
    sample_length = 288
    selectedNum = 10000
    chuckNum = 10
    splitRatio = {'train': 0.7, 'val': 0.2, 'test': 0.1}

    raw_data = signal.read_samples(0, -1)

    chuckList = divideIntoChucks(raw_data, chuckNum)

    trainData, trainLabels, valData, valLabels, testData, testLabels = splitChuckData(opts, chuckList, splitRatio, sample_length, selectedNum, label)

    return trainData, trainLabels, valData, valLabels, testData, testLabels


def getFilesAndLabels(opts):
    


def test_read_one_data(opts):
    # Load a dataset
    file_idx = 'A_1_1'
    label = 0

    binfile = os.path.join(opts.input, file_idx+'.bin')
    metafile = os.path.join(opts.input, file_idx+'.sigmf-meta')

    signal = loadData(metafile, binfile)
    trainData, trainLabels, valData, valLabels, testData, testLabels = get_samples(opts, signal, label)
    print(trainData.shape, valData.shape, testData.shape)
    print(trainLabels)



def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-sp', '--splitType', help='choose from random/order')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    test_read_one_data(opts)
    print('all test passed!')
