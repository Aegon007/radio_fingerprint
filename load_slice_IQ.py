import os
import sys
import glob
import argparse

import numpy as np
from keras.utils import np_utils


def read_f32_bin(filename, start_ix):
    bin_f = open(filename, 'rb')
    iq_seq = np.fromfile(bin_f, dtype='<f4')
    n_samples = iq_seq.shape[0]/2

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
    dev_dir_list = [os.path.join(args.root_dir, a) for a in args.antenna_dir]

    n_devices = len(dev_dir_list)
    slice_dims = (2, args.slice_len)
    stride = args.stride
    samps_to_retrieve = (n_slices_per_dev - 1) * stride + slice_dims[1]

    X_data, y_data = [], []

    for i, d in enumerate(dev_dir_list):
        pre_X_data = dev_bin_dataset(d, samps_to_retrieve, start_ix)
        y_data = np.concatenate((y_data, i * np.ones(n_slices_per_dev, )), axis=0)
        X_data_pd = []
        count_s = 0
        for j in range(0, samps_to_retrieve, stride):
            X_data_pd.append(pre_X_data[:, j:j+slice_dims[1]])
            count_s += 1
            if count_s == n_slices_per_dev:
                break

        X_data_pd = np.array(X_data_pd)
        del pre_X_data

        if i == 0:
            X_data = X_data_pd
        else:
            X_data = np.concatenate((X_data, X_data_pd), axis=0)
        del X_data_pd

    y_data = np_utils.to_categorical(y_data, n_devices)
    return X_data, y_data


class loadDataOpts():
    def __init__(self, root_dir, antenna_dir, num_slice=100000, start_ix=0, slice_len=288, stride=1):
        self.root_dir = root_dir
        self.antenna_dir = antenna_dir
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride


def parseArgs(argv):
    Desc = 'Read and slice the collected I/Q samples'
    parser = argparse.ArgumentParser(description=Desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--root_dir', required=True, help='Root directory for the devices\' folders.')
    parser.add_argument('-a', '--antenna_dir', required=True, nargs='+', help='A list of devices as \' folders.')
    parser.add_argument('-n', '--num_slice', required=True, type=int, help='Number of slices to be generated for each device.')
    parser.add_argument('-i', '--start_ix', type=int, default=0, help='Starting read index in .bin files.')
    parser.add_argument('-l', '--slice_len', type=int, required=True, help='Lenght of slices.')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Stride used for windowing.')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    loadData(opts)
