import os
import sys
import glob
import argparse
import pdb

import numpy as np
from keras.utils import np_utils
import utils


def read_f32_bin(filename, start_ix):
    bin_f = open(filename, 'rb')
    iq_seq = np.fromfile(bin_f, dtype='<f4')
    n_samples = iq_seq.shape[0] // 2

    IQ_data = np.zeros((2, n_samples))

    IQ_data[0, :] = iq_seq[range(0, iq_seq.shape[0]-1, 2)]
    IQ_data[1, :] = iq_seq[range(1, iq_seq.shape[0], 2)]
    bin_f.close()
    del iq_seq
    return IQ_data[:, start_ix:]


def dev_bin_dataset(glob_dat_path, n_samples, start_ix=0):
    filelist = sorted(glob.glob(glob_dat_path))
    all_IQ_data = read_f32_bin(filelist[0], start_ix)
    isEnoughSample = False

    if all_IQ_data.shape[1] < n_samples:
        for f in filelist[1:]:
            all_IQ_data = np.concatenate((all_IQ_data, read_f32_bin(f, start_ix)), axis=1)
            if all_IQ_data.shape[1] >= n_samples:
                isEnoughSample = True
                break
    else:
        isEnoughSample = True

    if not isEnoughSample:
        print("ERROR! There are not enough samples to satisfy dataset parameters. Aborting...")
        sys.exit(-1)

    all_IQ_data = all_IQ_data[:, 0:n_samples]
    return all_IQ_data


def loadData(args):
    n_slices_per_dev = args.num_slice
    start_ix = args.start_ix
    file_key = args.file_key

    dev_dir_list = []
    dev_dir_names = os.listdir(args.root_dir)
    for n in dev_dir_names:
        tmp = os.path.join(args.root_dir, n)
        dev_dir_list.append(tmp)

    n_devices = len(dev_dir_list)
    slice_dims = (2, args.slice_len)
    stride = args.stride
    samps_to_retrieve = (n_slices_per_dev - 1) * stride + slice_dims[1]

    x_train, y_train, x_test, y_test = [], [], [], []
    split_ratio = {'train': 0.8, 'val': 0.2}
    for i, d in enumerate(dev_dir_list):
        pre_X_data = dev_bin_dataset(os.path.join(d, file_key), samps_to_retrieve, start_ix)

        X_data_pd = []
        count_s = 0
        for j in range(0, samps_to_retrieve, stride):
            X_data_pd.append(pre_X_data[:, j:j+slice_dims[1]])
            count_s += 1
            if count_s == n_slices_per_dev:
                break
        X_data_pd = np.array(X_data_pd)
        y_data_pd = i * np.ones(n_slices_per_dev, )

        # split one class data
        x_train_pd, y_train_pd, x_test_pd, y_test_pd = utils.splitData(split_ratio, X_data_pd, y_data_pd)

        if i == 0:
            x_train, x_test = x_train_pd, x_test_pd
            y_train, y_test = y_train_pd, y_test_pd
        else:
            x_train = np.concatenate((x_train, x_train_pd), axis=0)
            x_test = np.concatenate((x_test, x_test_pd), axis=0)
            y_train = np.concatenate((y_train, y_train_pd), axis=0)
            y_test = np.concatenate((y_test, y_test_pd), axis=0)

        del pre_X_data
        del X_data_pd

    y_train = np_utils.to_categorical(y_train, n_devices)
    y_test = np_utils.to_categorical(y_test, n_devices)
    return x_train, y_train, x_test, y_test, n_devices


class loadDataOpts():
    def __init__(self, root_dir, file_key='*.bin', num_slice=100000, start_ix=0, slice_len=288, stride=1):
        self.root_dir = root_dir
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key


def parseArgs(argv):
    Desc = 'Read and slice the collected I/Q samples'
    parser = argparse.ArgumentParser(description=Desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--root_dir', required=True, help='Root directory for the devices\' folders.')
    parser.add_argument('-n', '--num_slice', required=True, type=int, help='Number of slices to be generated for each device.')
    parser.add_argument('-i', '--start_ix', type=int, default=0, help='Starting read index in .bin files.')
    parser.add_argument('-l', '--slice_len', type=int, default=288, help='Lenght of slices.')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Stride used for windowing.')
    parser.add_argument('-f', '--file_key', default='*.bin', help='used to choose different filetype, choose from *.bin/*.sigmf-meta')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    x_train, y_train, x_test, y_test, NUM_CLASS = loadData(opts)
    print('train data shape: ', x_train.shape, 'train label shape: ', y_train.shape)
    print('test data shape: ', x_test.shape, 'test label shape: ', y_test.shape)
    print('all test done!')
