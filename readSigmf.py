#! /usr/bin python3.6
import os
import sys
import pdb
import argparse
import random

from collections import defaultdict
import json
import numpy as np
import itertools
from sigmf import SigMFFile, sigmffile

import mytools.tools as mytools


def createSignal(metafile, binfile):
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


def splitData(allData, allLabel, splitRatio):
    np.random.seed(42)
    allDataSize = len(allData)
    shuffledind = np.random.permutation(len(allData))
    allData = np.array(allData)

    train_set_size = int(allDataSize * splitRatio['train'])
    val_set_size = int(allDataSize * splitRatio['val'])
    test_set_size = int(allDataSize * splitRatio['test'])

    start, end = 0, train_set_size
    train_ind = shuffledind[start: end]

    start, end = train_set_size, train_set_size + val_set_size
    val_ind = shuffledind[start: end]

    start, end = train_set_size + val_set_size, train_set_size + val_set_size + test_set_size
    test_ind = shuffledind[start: end]

    return allData[train_ind], allLabel[traini_ind], allData[val_ind], allLabel[val_ind], allData[test_ind], allLabel[test_ind]


def readDataFile(chuckList, sample_length, selectedNum):
    allData = []
    for chuck in chuckList:
        oneChuck = formInpData(chuck, sample_length, selectedNum)
        allData.append(oneChuck)
    return allData


def splitChuckData(opts, splitRatio, allData, allLabel):
    if opts.splitType == 'random':
        trainData, trainLabels, valData, valLabels, testData, testLabels = splitData(allData, allLabel, splitRatio)
    elif opts.splitType == 'order':
        pass
    else:
        raise
    return trainData, trainLabels, valData, valLabels, testData, testLabels


def get_oneDay_samples(signal, label, params):
    # Get some metadata and all annotations
    chuckNum = params['chuckNum']
    sample_length = params['sample_length']
    selectedNum = params['selectedNum']

    raw_data = signal.read_samples(0, -1)
    chuckList = divideIntoChucks(raw_data, chuckNum)

    oneData = readDataFile(chuckList, sample_length, selectedNum)
    oneLabel = np.ones(oneData.shape[0], dtype=np.int) * label
    return oneData, oneLabel


def searchFp(fname, metaFileList):
    for mfp in metaFileList:
        mfn = os.path.basename(mfp).split(mfp)
        if mfn == fname:
            return mfp
    return ''


def getSignalList(fpTuple):
    binFileList, metaFileList = fpTuple
    signalList = []
    for bfp in binFileList:
        fname = os.path.basename(bfp).split('.')[0]
        mfp = searchFp(fname, metaFileList)
        if not mfp:
            raise ValueError('binfile {} does not have a match'.format(bfp))
        signal = createSignal(mfp, bfp)
        signalList.append(signal)
    return signalList


def getOneDevData(fpTuple, label, params):
    signalList = getSignalList(fpTuple)
    allData, allLabel = [], []
    for signal in signalList:
        oneData, oneLabel = get_oneDay_samples(signal, label, params)
        allData.extend(oneData)
        allLabel.extend(allLabel)

    return allData, allLabel


def getfpTuple(strLabel, x_day_dir):
    dayDevDir = os.path.join(x_day_dir, strLabel)
    fList = os.listdir(dayDevDir)
    binFileList, metaFileList = [], []
    for fname in fList:
        fp = os.path.join(dayDevDir, fname)
        if fp.endswith('bin'):
            binFileList.append(fp)
        elif fp.endswith('sigmf-meta'):
            metaFileList.append(fp)
        else:
            raise
    return (binFileList, metaFileList)


def getDataAndLabels(opts, x_day_dir):
    '''this is made to read one day data'''
    params = {
            'sample_length': 288,
            'selectedNum': 10000,
            'chuckNum': 10,
            'splitRatio': {'train': 0.7, 'val': 0.2, 'test': 0.1}
            }

    devList = os.listdir(x_day_dir)
    label2Data = defaultdict()

    allData, allLabel = [], []
    for i in range(len(devList)):
        strLabel = devList[i]
        fpTuple = getfpTuple(strLabel, x_day_dir)
        label2Data[i] = fpTuple

        oneData, oneLabel = getOneDevData(fpTuple, i)
        allData.extend(oneData)
        allLabel.extend(oneLabel)

    trainData, trainLabels, valData, valLabels, testData, testLabels = splitChuckData(opts, splitRatio, allData, allLabel)
    return trainData, trainLabels, valData, valLabels, testData, testLabels


def test_read_one_data(opts):
    # Load a dataset
    file_idx = 'A_1_1'
    label = 0

    binfile = os.path.join(opts.input, file_idx+'.bin')
    metafile = os.path.join(opts.input, file_idx+'.sigmf-meta')

    signal = createSignal(metafile, binfile)
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
